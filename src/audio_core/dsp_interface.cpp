// Copyright 2017-2026 Citra Emulator Project / Azahar Emulator Project
// Licensed under GPLv2 or any later version
// Refer to the license.txt file included.

#include <chrono>
#include <cstddef>
#include <cstring>
#include "audio_core/dsp_interface.h"
#include "audio_core/sink.h"
#include "audio_core/sink_details.h"
#include "common/assert.h"
#include "common/settings.h"
#include "core/core.h"
#include "core/dumping/backend.h"

namespace AudioCore {

namespace {

/// How long JumpBegin() gives the tail to reach the device before going ahead without it.
/// Comfortably over StreamRamp's tail and the mute behind it, so it is a backstop against a
/// sink that never calls back rather than a budget the ramp is expected to fit in.
constexpr auto kJumpSettleTimeout = std::chrono::milliseconds(50);

} // namespace

DspInterface::DspInterface(Core::System& system_) : system(system_) {}

DspInterface::~DspInterface() = default;

void DspInterface::SetSink(AudioCore::SinkType sink_type, std::string_view audio_device) {
    // Every settings apply comes through here, and the teardown below cuts the stream wherever
    // the waveform stands. Nothing to do for the sink already playing what was asked for.
    if (sink && sink_type == current_sink_type && audio_device == current_audio_device) {
        return;
    }

    // The sink really is going: take the stream down on its tail rather than cutting it. Here
    // rather than in the caller, so an apply that changes nothing pays neither the teardown nor
    // the fade, and so every caller gets it without having to remember.
    const bool ramped = JumpBegin();

    // Dispose of the current sink first to avoid contention.
    sink.reset();

    sink = AudioCore::GetSinkDetails(sink_type).create_sink(audio_device);
    current_sink_type = sink_type;
    current_audio_device = audio_device;
    // Primed before SetCallback(): the SDL2 sink may call back immediately, already unpaused.
    sink_sample_rate = static_cast<double>(sink->GetNativeSampleRate());
    time_stretcher.SetOutputSampleRate(sink->GetNativeSampleRate());
    low_pass.Init(sink_sample_rate);
    // A new sink is a new stream: nothing of the old one to continue, and it opens on a ramp.
    ramp = StreamRamp{};
    stream_settled.store(true, std::memory_order_release);
    sink->SetCallback(
        [this](s16* buffer, std::size_t num_frames) { OutputCallback(buffer, num_frames); });
    JumpEnd(ramped);
}

Sink& DspInterface::GetSink() {
    ASSERT(sink);
    return *sink.get();
}

void DspInterface::EnableStretching(bool enable) {
    enable_time_stretching = enable;
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
    const bool was_silenced = core_silenced.exchange(true, std::memory_order_acq_rel);
    // The tail is the audio thread's to play. A load replaces this object and its sink, and a
    // reset closes the sink outright, so unless the tail has reached the device by then it is
    // cut off like the audio it was to replace: wait for a callback to report the stream
    // settled, down with its tail out. A stream someone else took down is waited for just the
    // same, since a pause is followed by a load often enough and its tail is no further along
    // for having been armed by the pause; a source that had already stopped, its tail long
    // gone, is settled as it stands and the wait returns at once. Deadline-bounded, since a
    // sink that never calls back (null, or the libretro sink's immediate submission) settles
    // nothing; a wakeup lost between the predicate and the wait costs the deadline, not the
    // tail.
    const auto deadline = std::chrono::steady_clock::now() + kJumpSettleTimeout;
    std::unique_lock lock{settled_mutex};
    settled_cv.wait_until(lock, deadline,
                          [this] { return stream_settled.load(std::memory_order_acquire); });
    return !was_silenced;
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
    // The stretcher holds a backlog of its own, an eighth of a second of it by design and up to
    // a second before it stops taking input, which a resume would play behind the tail. Nothing
    // refills it while the stream is down, so one clear per takedown does.
    if (!stretcher_discarded) {
        stretcher_discarded = true;
        time_stretcher.Clear();
        flushing_time_stretcher = false;
    }
}

void DspInterface::SetSpeedupAudio(bool enable, u16 lowpass_reference) {
    enable_speedup_audio = enable;
    speedup_lowpass_reference = lowpass_reference;
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

void DspInterface::OutputCallback(s16* buffer, std::size_t num_frames) {
    // Determine if we should stretch based on the current emulation speed.
    // TODO: Only activate audio stretching when emulation speed goes below 95% threshold
    //       (see #2487) -OS
    if (performing_time_stretching && !enable_time_stretching) {
        // If we just stopped stretching, flush the stretcher before returning to normal output.
        flushing_time_stretcher = true;
    }
    performing_time_stretching = enable_time_stretching.load();

    // Taken down on purpose: nothing is popped, so the stretcher's bookkeeping stands still
    // and picks up where it left off, and the ramp below fills the buffer with the tail.
    const bool silenced = core_silenced.load(std::memory_order_acquire);
    if (!silenced) {
        stretcher_discarded = false;
    }

    // Read once: it can flip mid-callback, and both branches below must agree on it.
    const bool speedup_enabled = enable_speedup_audio.load();
    const double speed = SpeedupSpeedFromFrameLimit(Settings::GetFrameLimit());
    const bool off_speed = speedup_enabled && SpeedupIsOffSpeed(speed);

    std::size_t frames_written = 0;
    if (silenced) {
        DiscardPending();
    } else if (off_speed) {
        if (!wsola_engaged) {
            wsola_engaged = true;
            // Not Reset(): that memsets 256 KB, unfit for a realtime callback.
            wsola.BeginSession();
            arrival_avg = static_cast<double>(num_frames) * speed;
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

    // What the source did not fill is the ramp's to fill, below.
    if (frames_written < num_frames) {
        std::memset(buffer + (frames_written * 2), 0,
                    (num_frames - frames_written) * 2 * sizeof(s16));
    }

    // Last before the volume, so the history it keeps is what was played and the tail it
    // synthesizes from that history is scaled like everything else. Where the source stopped
    // short, deliberately or not, the tail takes over from the frame it stopped on; when it
    // returns, the first frames ramp in.
    const bool ramp_enabled = enable_audio_ramp.load();
    if (ramp_was_enabled && !ramp_enabled) {
        // Turned off mid-stream, possibly mid-tail. Drop that state rather than freeze it: a
        // tail left pending would hold stream_settled false for good, and every JumpBegin()
        // after it would wait out its whole deadline for a tail that will never play. Down,
        // since with the ramp gone from the path there is nothing left to settle.
        ramp.Reset(true);
    } else if (!ramp_was_enabled && ramp_enabled) {
        // Turned back on over a stream that is already at level: adopt it where it stands
        // rather than fade it in from zero over audio that is playing.
        ramp.Reset(frames_written == 0);
    }
    ramp_was_enabled = ramp_enabled;
    if (ramp_enabled) {
        ramp.Process(buffer, num_frames, frames_written);
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
