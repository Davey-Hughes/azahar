// Copyright 2026 Citra Emulator Project / Azahar Emulator Project
// Licensed under GPLv2 or any later version
// Refer to the license.txt file included.

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>
#include <limits>
#include <numbers>
#include "audio_core/output_pipeline.h"
#include "common/logging/log.h"

namespace AudioCore {

namespace {

s16 ToSample(float v) {
    return static_cast<s16>(std::clamp(std::lround(v * 32768.0f), -32768L, 32767L));
}

/// How long JumpBegin() gives the tail to reach the device before going ahead without it.
/// Comfortably over StreamRamp's tail and the mute behind it, so it is a backstop against a
/// sink that never calls back rather than a budget the ramp is expected to fit in.
constexpr auto kJumpSettleTimeout = std::chrono::milliseconds(50);

const char* ModeName(StretchGate::Mode mode) {
    switch (mode) {
    case StretchGate::Mode::Bypass:
        return "bypass";
    case StretchGate::Mode::Warming:
        return "warming";
    case StretchGate::Mode::Stretch:
        return "stretch";
    case StretchGate::Mode::Drain:
        return "drain";
    }
    return "?";
}

} // namespace

OutputPipeline::OutputPipeline() {
    Reset();
}

void OutputPipeline::SetOutputSampleRate(unsigned rate) {
    time_stretcher.SetOutputSampleRate(rate);
    Reset();
}

void OutputPipeline::Reset() {
    // Whatever the emulation thread pushed before belongs to the old stream.
    while (fifo.Pop(pop_scratch.data(), kMaxCallbackFrames) > 0) {
    }
    stash.Clear();
    history.Clear();
    time_stretcher.Clear();
    EnterStretch();
    gate.Reset();
    speed = 1.0;
    fifo_left = 0;
    prefilling = true;
    depth_window.fill(0);
    cut_window.fill(0);
    window_pos = 0;
    warm_raw_pos = 0;
    warm_lag = 0;
    last = {};
    fade_out_frames = 0;
    fade_out_from = {};
    fade_last_out = {};
    // A new stream: nothing of the old one to continue, and it opens on a ramp.
    ramp = StreamRamp{};
    ramp_was_enabled = true;
    stream_settled.store(true, std::memory_order_release);
}

std::size_t OutputPipeline::Push(const void* frames, std::size_t num_frames) {
    return fifo.Push(frames, num_frames);
}

void OutputPipeline::SetStretching(bool enable) {
    enable_stretching = enable;
}

void OutputPipeline::SetRamp(bool enable) {
    enable_ramp = enable;
}

void OutputPipeline::StreamEnd() {
    core_silenced.store(true, std::memory_order_release);
}

void OutputPipeline::StreamBegin() {
    core_silenced.store(false, std::memory_order_release);
}

bool OutputPipeline::JumpBegin() {
    if (core_silenced.exchange(true, std::memory_order_acq_rel)) {
        return false;
    }
    // The tail is the audio thread's to play. A load replaces this object and its sink, and a
    // reset closes the sink outright, so unless the tail has reached the device by then it is
    // cut off like the audio it was to replace: wait for a callback to report the stream
    // settled, down with its tail out. A source that had already stopped, its tail long gone,
    // is settled as it stands, and there is nothing to wait for. Deadline-bounded, since a sink
    // that never calls back (null, or the libretro sink's immediate submission) settles nothing;
    // a wakeup lost between the predicate and the wait costs the deadline, not the tail.
    const auto deadline = std::chrono::steady_clock::now() + kJumpSettleTimeout;
    std::unique_lock lock{settled_mutex};
    settled_cv.wait_until(lock, deadline,
                          [this] { return stream_settled.load(std::memory_order_acquire); });
    return true;
}

void OutputPipeline::JumpEnd(bool ramped) {
    if (ramped) {
        StreamBegin();
    }
}

std::size_t OutputPipeline::FillTarget(std::size_t num_frames) const {
    return static_cast<std::size_t>(native_sample_rate / 60) + num_frames;
}

void OutputPipeline::Render(s16* out, std::size_t num_frames) {
    while (num_frames > 0) {
        const std::size_t n = std::min(num_frames, kMaxCallbackFrames);
        RenderChunk(out, n);
        out += n * 2;
        num_frames -= n;
    }
}

void OutputPipeline::RenderChunk(s16* out, std::size_t num_frames) {
    const bool silenced = core_silenced.load(std::memory_order_acquire);
    const std::size_t fifo_now = fifo.Size();
    const std::size_t depth = fifo_now + stash.Size();

    if (silenced) {
        // Taken down on purpose. Nothing is measured or decided; what was produced before is
        // discarded below, and the ramp fills the buffer with the tail. Bypass refills before
        // it plays again.
        prefilling = true;
    } else {
        // Frames that arrived since the last callback, over the frames asked for, smoothed
        // over kSpeedTimeConstant. Arrivals come a video frame at a time, so a single reading
        // is 0 or 2 as often as 1; the smoothing is what makes it a speed.
        const double arrived =
            fifo_now >= fifo_left ? static_cast<double>(fifo_now - fifo_left) : 0.0;
        const double alpha = std::min(1.0, static_cast<double>(num_frames) /
                                               (kSpeedTimeConstant * native_sample_rate));
        speed += ((arrived / static_cast<double>(num_frames)) - speed) * alpha;
    }

    const StretchGate::Mode before = gate.CurrentMode();
    const bool synced = before == StretchGate::Mode::Warming &&
                        time_stretcher.OutputBacklog() >= WarmDiscardNeeded() + num_frames;
    const StretchGate::Edge edge = gate.Update({
        .speed = speed,
        .buffered = depth,
        .backlog = time_stretcher.OutputBacklog(),
        .ratio = time_stretcher.Ratio(),
        .enabled = enable_stretching.load(),
        .prefilling = prefilling,
        .silenced = silenced,
        .synced = synced,
        .num_frames = num_frames,
    });
    const StretchGate::Mode mode = gate.CurrentMode();

    RenderStats stats{};
    stats.edge = edge;
    stats.depth = depth;
    switch (edge) {
    case StretchGate::Edge::Engage:
        Engage();
        break;
    case StretchGate::Edge::Sync:
        Sync(stats);
        break;
    case StretchGate::Edge::Abort:
        time_stretcher.Clear();
        break;
    case StretchGate::Edge::Handover:
        Handover(stats);
        break;
    case StretchGate::Edge::None:
        break;
    }
    if (before == StretchGate::Mode::Stretch && mode == StretchGate::Mode::Drain) {
        EnterDrain(num_frames);
    } else if (before == StretchGate::Mode::Drain && mode == StretchGate::Mode::Stretch) {
        EnterStretch();
    }
    if (mode != before) {
        LOG_DEBUG(Audio, "{} -> {}: speed {:.3f}, {} frames buffered, {} in the stretcher",
                  ModeName(before), ModeName(mode), speed, depth, time_stretcher.OutputBacklog());
    }

    std::size_t written = 0;
    if (silenced) {
        DiscardPending();
    } else {
        switch (mode) {
        case StretchGate::Mode::Bypass:
            written = RenderBypass(out, num_frames, stats);
            break;
        case StretchGate::Mode::Warming:
            written = RenderWarming(out, num_frames, stats);
            break;
        case StretchGate::Mode::Stretch:
        case StretchGate::Mode::Drain:
            written = RenderStretch(out, num_frames);
            break;
        }
    }
    stats.written = written;
    last = stats;
    fifo_left = fifo.Size();

    // What the source did not fill is the ramp's to fill, below.
    if (written < num_frames) {
        std::memset(out + (written * 2), 0, (num_frames - written) * 2 * sizeof(s16));
    }

    ApplyHandoverFade(out, num_frames);

    // Last before the volume, so the history it keeps is what was played and the tail it
    // synthesizes from that history is scaled like everything else. Where the source stopped
    // short, deliberately or not, the tail takes over from the frame it stopped on; when it
    // returns, the first frames ramp in.
    const bool ramp_enabled = enable_ramp.load();
    if (ramp_was_enabled && !ramp_enabled) {
        // Turned off mid-stream, possibly mid-tail. Drop that state rather than freeze it: a
        // tail left pending would hold stream_settled false for good, and every JumpBegin()
        // after it would wait out its whole deadline for a tail that will never play.
        ramp = StreamRamp{};
    }
    ramp_was_enabled = ramp_enabled;
    if (ramp_enabled) {
        ramp.Process(out, num_frames, written);
    }

    // Signaled on the rising edge alone: a notify every callback would put a futex wake on the
    // audio thread once a buffer, where this fires only at a takedown, which is rare and is the
    // only time anyone is waiting. Never takes settled_mutex, so the callback cannot be made to
    // wait on the thread that is waiting on it.
    const bool settled = ramp.Down() && !ramp.InTail();
    if (settled) {
        if (!stream_settled.exchange(true, std::memory_order_acq_rel)) {
            settled_cv.notify_all();
        }
    } else {
        stream_settled.store(false, std::memory_order_release);
    }
}

std::size_t OutputPipeline::RenderBypass(s16* out, std::size_t num_frames, RenderStats& stats) {
    stash.PullFrom(fifo, num_frames + PeriodSplicer::kMaxPeriod);
    if (prefilling) {
        if (stats.depth < FillTarget(num_frames)) {
            return 0;
        }
        prefilling = false;
    }
    depth_window[window_pos] = stats.depth;
    cut_window[window_pos] = 0;
    const auto r = splicer.Cut(out, num_frames, stash, Excess(num_frames));
    cut_window[window_pos] = r.spliced;
    window_pos = (window_pos + 1) % kLowWaterWindow;
    history.Record(out, r.written);
    stats.consumed = r.consumed;
    stats.cut = r.spliced;
    return r.written;
}

std::size_t OutputPipeline::RenderWarming(s16* out, std::size_t num_frames, RenderStats& stats) {
    // Pull all the fifo will give: the sooner the stretcher has it, the sooner it can run a
    // round, and the stash is what the raw path plays from meanwhile.
    const std::size_t pulled = stash.PullFrom(fifo, FrameStash::kCapacity);
    if (pulled > 0) {
        time_stretcher.Feed(stash.Data() + ((stash.Size() - pulled) * 2), pulled);
    }
    // Insert() falls back to a plain copy on its own when the stash is too short to repeat
    // from; the ramp covers whatever that leaves short.
    const auto r = splicer.Insert(out, num_frames, stash);
    warm_raw_pos += r.consumed;
    history.Record(out, r.written);
    stats.consumed = r.consumed;
    stats.inserted = r.spliced;
    return r.written;
}

std::size_t OutputPipeline::RenderStretch(s16* out, std::size_t num_frames) {
    // The whole FIFO every callback, as master did; the stash is empty in these modes.
    const std::size_t pulled = fifo.Pop(pop_scratch.data(), kMaxCallbackFrames);
    return time_stretcher.Process(pop_scratch.data(), pulled, out, num_frames);
}

void OutputPipeline::Engage() {
    // Prime from what was just played, so the stretcher's look-ahead is already full when
    // the first new frames arrive. Its output index tracks its input index one for one on
    // average from the first frame (see TimeStretcher::BeginPrime(), audio_core/time_stretch.h),
    // so after feeding `have` frames of history and dropping `dropped` of output, its next
    // output frame is source frame `dropped` counted from where the history began, while the
    // raw path stands `have` frames past that same origin. The difference is what Sync()
    // discards, on top of whatever the raw path plays in the meantime.
    const std::size_t need = time_stretcher.BeginPrime();
    const std::size_t have =
        history.Last(std::min({need, history.Size(), kMaxCallbackFrames}), pop_scratch.data());
    time_stretcher.Feed(pop_scratch.data(), have);
    const std::size_t dropped = time_stretcher.Discard(std::numeric_limits<std::size_t>::max());
    warm_lag = static_cast<long>(have) - static_cast<long>(dropped);
    warm_raw_pos = 0;
    // The stash is ahead of the history and belongs to the stretcher next; the raw path plays
    // on from it meanwhile.
    if (stash.Size() > 0) {
        time_stretcher.Feed(stash.Data(), stash.Size());
    }
    EnterStretch();
}

std::size_t OutputPipeline::WarmDiscardNeeded() const {
    const long needed = static_cast<long>(warm_raw_pos) + warm_lag;
    return needed > 0 ? static_cast<std::size_t>(needed) : 0;
}

void OutputPipeline::Sync(RenderStats& stats) {
    const std::size_t needed = WarmDiscardNeeded();
    const std::size_t discard = std::min(needed, time_stretcher.OutputBacklog());
    time_stretcher.Discard(discard);
    stats.replayed = needed - discard;
    // The stash's frames are inside the stretcher now.
    stash.Clear();
    ArmHandoverFade();
    LOG_DEBUG(Audio, "stretcher synced: skipped {} of {} frames, {} replayed", discard, needed,
              stats.replayed);
}

void OutputPipeline::Handover(RenderStats& stats) {
    const std::size_t room = std::min(stash.Room(), flush_scratch.size() / 2);
    const std::size_t got = time_stretcher.FlushInto(flush_scratch.data(), room);
    stats.flushed = stash.Append(flush_scratch.data(), got);
    prefilling = false;
    ArmHandoverFade();
    LOG_DEBUG(Audio, "stretcher handed over {} frames", stats.flushed);
}

void OutputPipeline::EnterStretch() {
    time_stretcher.SetTargetBacklog(kStretchTargetBacklog);
    time_stretcher.SetRatioBounds(kStretchMinRatio, std::numeric_limits<double>::infinity());
}

void OutputPipeline::EnterDrain(std::size_t num_frames) {
    // Toward one callback of backlog, no faster than 10%: about 150 ms of backlog drains in a
    // second and a half, which is not heard as a speed-up.
    time_stretcher.SetTargetBacklog(static_cast<double>(num_frames) / native_sample_rate);
    time_stretcher.SetRatioBounds(StretchGate::kDrainMinRatio, StretchGate::kDrainMaxRatio);
}

void OutputPipeline::DiscardPending() {
    // Produced before the stream was taken down; behind the tail it would only splice in.
    while (fifo.Pop(pop_scratch.data(), kMaxCallbackFrames) > 0) {
    }
    stash.Clear();
}

std::size_t OutputPipeline::Excess(std::size_t num_frames) const {
    std::size_t low_water = std::numeric_limits<std::size_t>::max();
    std::size_t cuts = 0;
    for (std::size_t i = 0; i < kLowWaterWindow; i++) {
        low_water = std::min(low_water, depth_window[i]);
        cuts += cut_window[i];
    }
    const std::size_t floor = FillTarget(num_frames) + cuts;
    return low_water > floor ? low_water - floor : 0;
}

void OutputPipeline::ArmHandoverFade() {
    // Neither side of a handover continues the other exactly, so dip through a short
    // equal-power cross-fade from the level the stream stopped at rather than splice. Not
    // re-armed mid-fade: the edge can arrive twice in quick succession.
    if (fade_out_frames != 0) {
        return;
    }
    fade_out_frames = kHandoverFadeFrames;
    fade_out_from = fade_last_out;
}

void OutputPipeline::ApplyHandoverFade(s16* buffer, std::size_t num_frames) {
    if (num_frames == 0) {
        return;
    }

    const std::size_t n = std::min<std::size_t>(fade_out_frames, num_frames);
    for (std::size_t j = 0; j < n; j++) {
        const unsigned done = kHandoverFadeFrames - fade_out_frames + static_cast<unsigned>(j) + 1;
        // Smoothstep the progress so the gain leaves and arrives with zero slope; a step in
        // rate of change is heard as a blip at each end of the window.
        const float u = static_cast<float>(done) / kHandoverFadeFrames;
        const float th = 0.5f * std::numbers::pi_v<float> * (u * u * (3.0f - 2.0f * u));
        const float g = std::cos(th);
        const float gn = std::sin(th);
        const float env = 1.0f - (1.0f - kHandoverDipGain) *
                                     std::sin(std::numbers::pi_v<float> * static_cast<float>(done) /
                                              kHandoverFadeFrames);
        for (std::size_t ch = 0; ch < 2; ch++) {
            const float in = buffer[(j * 2) + ch] / 32768.0f;
            buffer[(j * 2) + ch] = ToSample(((fade_out_from[ch] * g) + (in * gn)) * env);
        }
    }
    fade_out_frames -= static_cast<unsigned>(n);

    fade_last_out[0] = buffer[((num_frames - 1) * 2) + 0] / 32768.0f;
    fade_last_out[1] = buffer[((num_frames - 1) * 2) + 1] / 32768.0f;
}

} // namespace AudioCore
