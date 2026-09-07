// Copyright 2017-2026 Citra Emulator Project / Azahar Emulator Project
// Licensed under GPLv2 or any later version
// Refer to the license.txt file included.

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <numbers>
#include "audio_core/dsp_interface.h"
#include "audio_core/sink.h"
#include "audio_core/sink_details.h"
#include "common/assert.h"
#include "common/settings.h"
#include "core/core.h"
#include "core/dumping/backend.h"

namespace AudioCore {

namespace {

s16 ToSample(float v) {
    return static_cast<s16>(std::clamp(std::lround(v * 32768.0f), -32768L, 32767L));
}

/// How long JumpBegin() gives the tail to reach the device before going ahead without it.
/// Comfortably over StreamRamp's tail and the mute behind it, so it is a backstop against a
/// sink that never calls back rather than a budget the ramp is expected to fit in.
constexpr auto kJumpSettleTimeout = std::chrono::milliseconds(50);

} // namespace

DspInterface::DspInterface(Core::System& system_) : system(system_) {}

DspInterface::~DspInterface() = default;

void DspInterface::SetSink(AudioCore::SinkType sink_type, std::string_view audio_device) {
    // Dispose of the current sink first to avoid contention.
    sink.reset();

    sink = AudioCore::GetSinkDetails(sink_type).create_sink(audio_device);
    // Primed before SetCallback(): the SDL2 sink may call back immediately, already unpaused.
    sink_sample_rate = static_cast<double>(sink->GetNativeSampleRate());
    time_stretcher.SetOutputSampleRate(sink->GetNativeSampleRate());
    low_pass.Init(sink_sample_rate);
    achieved_speed = 0.0;
    // A new sink is a new stream: nothing of the old one to continue, and it opens on a ramp.
    ramp = StreamRamp{};
    silenced_seen = false;
    stream_settled.store(true, std::memory_order_release);
    sink->SetCallback(
        [this](s16* buffer, std::size_t num_frames) { OutputCallback(buffer, num_frames); });
}

Sink& DspInterface::GetSink() {
    ASSERT(sink);
    return *sink.get();
}

void DspInterface::EnableStretching(bool enable) {
    enable_time_stretching = enable;
}

void DspInterface::SetSpeedupAudio(bool enable, u16 lowpass_reference) {
    enable_speedup_audio = enable;
    speedup_lowpass_reference = lowpass_reference;
}

void DspInterface::SetAudioRamp(bool enable) {
    enable_audio_ramp = enable;
}

void DspInterface::StreamEnd() {
    core_silenced.store(true, std::memory_order_release);
}

void DspInterface::StreamBegin() {
    core_silenced.store(false, std::memory_order_release);
}

bool DspInterface::JumpBegin() {
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

void DspInterface::JumpEnd(bool ramped) {
    if (ramped) {
        StreamBegin();
    }
}

void DspInterface::DiscardPending() {
    // Produced before the stream was taken down; behind the tail it would only splice in.
    while (fifo.Pop(pop_scratch.data(), kPopChunkFrames) > 0) {
    }
    if (!silenced_seen) {
        silenced_seen = true;
        // The stretcher's reserve and whatever it has synthesised are older still, and the
        // frames that arrive when the core resumes are not continuous with them.
        if (wsola_engaged) {
            wsola.BeginSession();
        }
    }
}

void DspInterface::OutputFrame(StereoFrame16 frame) {
    if (!sink) {
        return;
    }

    if (sink->ImmediateSubmission()) {
        sink->PushSamples(frame.data(), frame.size());
    } else {
        fifo.Push(frame.data(), frame.size());
    }

    auto video_dumper = system.GetVideoDumper();
    if (video_dumper && video_dumper->IsDumping()) {
        video_dumper->AddAudioFrame(std::move(frame));
    }
}

void DspInterface::OutputSample(std::array<s16, 2> sample) {
    if (!sink) {
        return;
    }

    if (sink->ImmediateSubmission()) {
        sink->PushSamples(&sample, 1);
    } else {
        fifo.Push(&sample, 1);
    }

    auto video_dumper = system.GetVideoDumper();
    if (video_dumper && video_dumper->IsDumping()) {
        video_dumper->AddAudioSample(std::move(sample));
    }
}

void DspInterface::DrainFifoIntoWsola() {
    // The FIFO can hold more than the ring accepts at once, so loop rather than pop once.
    while (true) {
        const std::size_t popped = fifo.Pop(pop_scratch.data(), kPopChunkFrames);
        if (popped == 0) {
            break;
        }
        if (wsola.Write(pop_scratch.data(), static_cast<int>(popped)) < static_cast<int>(popped)) {
            // The ring saturated and the popped frames are gone, so resync rather than splice.
            wsola.Resync();
            break;
        }
    }
}

std::size_t DspInterface::FillFromWsola(s16* buffer, std::size_t num_frames) {
    DrainFifoIntoWsola();

    // Arrival is lumpy, since the emu thread delivers in bursts, so smooth it into a rate.
    const s64 written = wsola.TotalWritten();
    const double delta = static_cast<double>(std::max<s64>(0, written - last_written));
    last_written = written;
    arrival_avg += (delta - arrival_avg) * 0.05;

    const double ratio = SpeedupStretchRatio(arrival_avg, num_frames, wsola.InputFill(),
                                             WsolaStretcher::TargetInputFill(arrival_avg));
    return static_cast<std::size_t>(wsola.Read(buffer, static_cast<int>(num_frames), ratio));
}

void DspInterface::ArmHandoverFade() {
    // At engage the stretcher opens behind the FIFO by the reserve it builds; at release that
    // reserve is dropped and the FIFO takes over past it. Neither side continues the other, so
    // dip through a short equal-power cross-fade from the level the stream stopped at rather
    // than splice. Not re-armed mid-fade: the edge can arrive twice in quick succession.
    if (fade_out_frames != 0) {
        return;
    }
    fade_out_frames = kHandoverFadeFrames;
    fade_out_from = fade_last_out;
}

void DspInterface::ApplyHandoverFade(s16* buffer, std::size_t num_frames) {
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

void DspInterface::OutputCallback(s16* buffer, std::size_t num_frames) {
    // Latch the stretching setting for this callback, arming a flush if it has just been
    // turned off so the stretcher's remainder is played out rather than stranded.
    // TODO: Only activate audio stretching when emulation speed goes below 95% threshold
    //       (see #2487) -OS
    if (performing_time_stretching && !enable_time_stretching) {
        // If we just stopped stretching, flush the stretcher before returning to normal output.
        flushing_time_stretcher = true;
    }
    performing_time_stretching = enable_time_stretching.load();

    // Read once: it can flip mid-callback, and both branches below must agree on it.
    const bool speedup_enabled = enable_speedup_audio.load();
    const auto frame_limit = Settings::GetFrameLimit();
    const double speed = SpeedupSpeedFromFrameLimit(frame_limit);
    const bool off_speed = speedup_enabled && SpeedupIsOffSpeed(speed);

    // Taken down on purpose: nothing is popped, so the stretcher's bookkeeping stands still
    // and picks up where it left off, and the ramp below fills the buffer with the tail.
    const bool silenced = core_silenced.load(std::memory_order_acquire);
    if (!silenced) {
        silenced_seen = false;
    }

    std::size_t frames_written = 0;
    if (silenced) {
        DiscardPending();
    } else if (off_speed) {
        if (!wsola_engaged) {
            wsola_engaged = true;
            ArmHandoverFade();
            // Not Reset(): that memsets 256 KB, unfit for a realtime callback.
            wsola.BeginSession();
            // The requested speed is a ceiling the host may not reach, so a run seeded from it
            // starts wrong and spends the run converging. Seed from what the last run reached
            // instead, still capped at the request while the limiter is on; the first run after
            // a sink change has nothing banked and seeds from the request as before.
            double anchor = speed;
            if (speed > 1.05 && achieved_speed > 1.05) {
                anchor = achieved_speed;
                if (frame_limit != 0 && anchor > speed) {
                    anchor = speed;
                }
            }
            arrival_avg = static_cast<double>(num_frames) * anchor;
            // Drain first, so last_written's delta reflects what arrives after engaging, not the
            // whole backlog in one lump.
            DrainFifoIntoWsola();
            last_written = wsola.TotalWritten();
            // Clears the stretcher so engaging mid-stream doesn't strand samples in it, and
            // cancels the flush whose output Clear() just discarded.
            time_stretcher.Clear();
            flushing_time_stretcher = false;
        }
        frames_written = FillFromWsola(buffer, num_frames);
    } else {
        if (wsola_engaged && num_frames > 0) {
            // Bank the speed this run actually reached, before the next engage seeds from it.
            const double reached = arrival_avg / static_cast<double>(num_frames);
            if (std::isfinite(reached) && reached > 1.05) {
                achieved_speed = reached;
            }
        }
        if (wsola_engaged) {
            ArmHandoverFade();
        }
        wsola_engaged = false;
        if (performing_time_stretching) {
            // Not a bare Pop(): that value-inits a vector to the FIFO's whole capacity every
            // callback. Sized to Size() instead, so a racing push just waits for the next one.
            const std::vector<s16> in{fifo.Pop(fifo.Size())};
            const std::size_t num_in{in.size() / 2};
            frames_written = time_stretcher.Process(in.data(), num_in, buffer, num_frames);
        } else {
            if (flushing_time_stretcher) {
                time_stretcher.Flush();
                frames_written = time_stretcher.Process(nullptr, 0, buffer, num_frames);
                flushing_time_stretcher = false;

                // Make sure any frames that did not fit are cleared from the time stretcher,
                // so that they do not bleed into the next time the stretcher is enabled.
                time_stretcher.Clear();
            }
            frames_written += fifo.Pop(buffer, num_frames - frames_written);
        }
    }

    // What the source did not fill is the ramp's to fill, below; the low-pass sees silence
    // there in the meantime, which is what the tail decays to.
    if (frames_written < num_frames) {
        std::memset(buffer + (frames_written * 2), 0,
                    (num_frames - frames_written) * 2 * sizeof(s16));
    }

    // The stretcher keys off requested speed; the filter keys off achieved speed, since it exists
    // to soften audibly fast playback. This matters at an unlimited frame limit, where the
    // request pins at kUnlimitedSpeed even though the host may only manage normal speed.
    double filter_speed = 1.0;
    if (off_speed && num_frames > 0) {
        filter_speed =
            std::clamp(arrival_avg / static_cast<double>(num_frames), 1.0, kUnlimitedSpeed);
    }

    // Skipped when it can't engage (off, or at the sentinel): the cutoff is wide_open anyway.
    const u16 lowpass_reference = speedup_lowpass_reference.load();
    const bool lowpass_active = speedup_enabled && (lowpass_reference < kSpeedupLowPassOff);
    if (lowpass_active) {
        if (!lowpass_was_active) {
            // Newly able to engage: reset to wide open rather than resume stale state.
            low_pass.Init(sink_sample_rate);
        }
        // Pre-volume; runs even while bypassed so the cutoff smoother stays in step with the
        // signal.
        low_pass.Process(
            buffer, num_frames,
            SpeedupLowPassCutoff(filter_speed, lowpass_reference, low_pass.WideOpenCutoff()),
            static_cast<double>(num_frames) / sink_sample_rate);
    }
    lowpass_was_active = lowpass_active;

    ApplyHandoverFade(buffer, num_frames);

    // Last before the volume, so the history it keeps is what was played and the tail it
    // synthesises from that history is scaled like everything else. Where the source stopped
    // short, deliberately or not, the tail takes over from the frame it stopped on; when it
    // returns, the first frames ramp in.
    const bool ramp_enabled = enable_audio_ramp.load();
    if (ramp_was_enabled && !ramp_enabled) {
        // Turned off mid-stream, possibly mid-tail. Drop that state rather than freeze it: a
        // tail left pending would hold stream_settled false for good, and every JumpBegin()
        // after it would wait out its whole deadline for a tail that will never play.
        ramp = StreamRamp{};
    }
    ramp_was_enabled = ramp_enabled;
    if (ramp_enabled) {
        ramp.Process(buffer, num_frames, frames_written);
    }

    // Signalled on the rising edge alone: a notify every callback would put a futex wake on the
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

    // Implementation of the hardware volume slider
    // A cubic curve is used to approximate a linear change in human-perceived loudness
    const float linear_volume = std::clamp(Settings::Volume(), 0.0f, 1.0f);
    if (linear_volume != 1.0) {
        const float volume_scale_factor = linear_volume * linear_volume * linear_volume;
        for (std::size_t i = 0; i < num_frames; i++) {
            buffer[i * 2 + 0] = static_cast<s16>(buffer[i * 2 + 0] * volume_scale_factor);
            buffer[i * 2 + 1] = static_cast<s16>(buffer[i * 2 + 1] * volume_scale_factor);
        }
    }
}

} // namespace AudioCore
