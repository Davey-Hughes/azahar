// Copyright 2026 Citra Emulator Project / Azahar Emulator Project
// Licensed under GPLv2 or any later version
// Refer to the license.txt file included.

#pragma once

#include <array>
#include <atomic>
#include <condition_variable>
#include <cstddef>
#include <mutex>
#include "audio_core/audio_types.h"
#include "audio_core/frame_buffers.h"
#include "audio_core/period_splicer.h"
#include "audio_core/stream_ramp.h"
#include "audio_core/stretch_gate.h"
#include "audio_core/time_stretch.h"
#include "common/common_types.h"
#include "common/ring_buffer.h"

namespace AudioCore {

/// Everything between the DSP's output and the sink's callback: the FIFO the emulation thread
/// fills, the time stretcher and the bypass around it, the handover cross-fade and the stream
/// ramp. Render() is the sink's callback minus the volume, which DspInterface::OutputCallback()
/// (audio_core/dsp_interface.cpp) applies after it. Push() is the only entry from another
/// thread; the stream and setting calls flip atomics the audio thread reads.
///
/// StretchGate (audio_core/stretch_gate.h) picks the mode; this class carries it out. In
/// Bypass, frames go FIFO -> stash -> PeriodSplicer::Cut() -> out, and the splicer trims any
/// depth above the fill target one period at a time. In Warming the same path runs through
/// PeriodSplicer::Insert(), so it falls behind real time, while every frame pulled is also fed
/// to the stretcher; once the stretcher's output covers the raw play point, Sync discards up to
/// that point and Stretch reads from the stretcher, as master did. Drain lowers the stretcher's
/// target at a bounded tempo, and Handover flushes it exactly into the stash, where the cut
/// trims the excess away.
class OutputPipeline {
public:
    /// The most a single Render() serves at once; a sink asking for more is served in pieces.
    /// Also the FIFO's capacity, as on master.
    static constexpr std::size_t kMaxCallbackFrames = 0x2000;

    /// What the last Render() did, for tests and the edge log.
    struct RenderStats {
        std::size_t written = 0;  // frames the source filled, before the ramp
        std::size_t consumed = 0; // source frames taken from the stash (Bypass and Warming)
        std::size_t cut = 0;      // frames the splicer removed
        std::size_t inserted = 0; // frames the splicer repeated
        std::size_t flushed = 0;  // frames the stretcher handed to the stash at Handover
        std::size_t replayed = 0; // frames of stretcher output a forced Sync could not skip
        std::size_t depth = 0;    // fifo + stash at the start of the callback
        StretchGate::Edge edge = StretchGate::Edge::None;
    };

    OutputPipeline();

    /// A new sink: sets the stretcher's output rate and resets the stream. Not safe against a
    /// concurrent Render(); call before the sink's callback is installed.
    void SetOutputSampleRate(unsigned rate);
    /// Back to a fresh stream: empty, in Bypass, prefilling, opening on the ramp.
    void Reset();

    /// Emulation thread. Returns the frames accepted; the rest are dropped, as on master.
    std::size_t Push(const void* frames, std::size_t num_frames);
    /// Audio thread. Fills `out` completely.
    void Render(s16* out, std::size_t num_frames);

    void SetStretching(bool enable);
    void SetRamp(bool enable);
    /// The core has stopped producing audio on purpose: end the stream on a ramp rather than
    /// wherever the waveform happens to be, and discard whatever it had already produced.
    /// Any thread.
    void StreamEnd();
    /// The core is producing again: the next frames ramp back in. Any thread.
    void StreamBegin();
    /// Bracket a jump the frontend makes in the game's state, a load or a reset, so the splice
    /// lands in silence: takes the stream down and waits, bounded, for the tail to reach the
    /// sink. Returns false and does nothing if the stream is already down, so the ramp back up
    /// stays with whatever took it down. Emulation thread.
    bool JumpBegin();
    void JumpEnd(bool ramped);

    /// Bypass's fill target: one video frame of audio plus one callback, the least that keeps
    /// the raw path fed between the emulator's bursts.
    std::size_t FillTarget(std::size_t num_frames) const;

    // Audio-thread state, read between Render() calls: for tests and logging.
    StretchGate::Mode CurrentMode() const {
        return gate.CurrentMode();
    }
    double Speed() const {
        return speed;
    }
    std::size_t Buffered() const {
        return fifo.Size() + stash.Size();
    }
    RenderStats LastRender() const {
        return last;
    }

private:
    void RenderChunk(s16* out, std::size_t num_frames);
    std::size_t RenderBypass(s16* out, std::size_t num_frames, RenderStats& stats);
    std::size_t RenderWarming(s16* out, std::size_t num_frames, RenderStats& stats);
    std::size_t RenderStretch(s16* out, std::size_t num_frames);
    void Engage();
    void Sync(RenderStats& stats);
    void Handover(RenderStats& stats);
    void EnterStretch();
    void EnterDrain(std::size_t num_frames);
    void DiscardPending();
    std::size_t WarmDiscardNeeded() const;
    std::size_t Excess(std::size_t num_frames) const;
    void ArmHandoverFade();
    void ApplyHandoverFade(s16* buffer, std::size_t num_frames);

    static constexpr double kSpeedTimeConstant = 0.3;      // seconds
    static constexpr double kStretchTargetBacklog = 0.125; // seconds, master's servo target
    static constexpr double kStretchMinRatio = 0.05;
    static constexpr std::size_t kLowWaterWindow = 8;

    // Filled by DspInterface::OutputFrame() on the emulation thread, drained here.
    Common::RingBuffer<s16, kMaxCallbackFrames, 2> fifo;
    FrameStash stash;
    // Frames the raw path emitted, post-splice and pre-fade: what the stretcher is primed from.
    FrameHistory history;
    TimeStretcher time_stretcher;
    StretchGate gate;
    PeriodSplicer splicer;
    std::array<s16, kMaxCallbackFrames * 2> pop_scratch{};
    std::array<s16, FrameStash::kCapacity * 2> flush_scratch{};

    std::atomic<bool> enable_stretching{false};
    std::atomic<bool> enable_ramp{true};
    std::atomic<bool> core_silenced{false};

    // Arrival over request, smoothed: the speed the emulation actually runs at, as the audio
    // thread sees it. fifo_left is the FIFO's depth after the previous callback's pops, so
    // the difference at the next is what arrived in between.
    double speed = 1.0;
    std::size_t fifo_left = 0;
    // Bypass emits nothing until the buffer first reaches the fill target: at reset, and after
    // each silence.
    bool prefilling = true;
    // Depth at the start of each of the last kLowWaterWindow callbacks, and what the splicer
    // cut in each: excess is judged on the window's minimum less those cuts, since arrivals
    // come a burst at a time and an instantaneous depth would trigger cuts at the right
    // average depth.
    std::array<std::size_t, kLowWaterWindow> depth_window{};
    std::array<std::size_t, kLowWaterWindow> cut_window{};
    std::size_t window_pos = 0;
    // Warm-up: source frames the raw path consumed since Engage, and how far the stretcher's
    // next output frame sits behind the frame the history ended on (history fed minus output
    // dropped at priming).
    std::size_t warm_raw_pos = 0;
    long warm_lag = 0;
    RenderStats last{};

    // Cross-fade across a handover, in output frames. Kept short: both sides are the same
    // material at different points, so a long overlap is heard for itself. The gain at its
    // midpoint: lower attenuates the discontinuity the handover carries, at the cost of a
    // deeper notch. Audio thread only.
    static constexpr unsigned kHandoverFadeFrames = 256;
    static constexpr float kHandoverDipGain = 0.15f;
    unsigned fade_out_frames = 0;
    std::array<float, 2> fade_out_from{};
    // Last frame handed on before the volume, normalized: the level a handover fades from.
    std::array<float, 2> fade_last_out{};

    // Ends the stream on a ramp and brings it back on one, on the last buffer before the sink.
    // Audio thread only; core_silenced is how the other threads reach it.
    StreamRamp ramp;
    // Whether the ramp ran on the previous callback, so it can be dropped on the edge rather
    // than frozen mid-tail. Audio thread only.
    bool ramp_was_enabled = true;
    // Whether the last callback found the stream settled: down, with its tail fully out. What
    // JumpBegin() waits on, since it cannot read the ramp from its own thread. A fresh stream
    // is settled, having nothing to take down. The audio thread signals the condition variable
    // on the rising edge alone; the mutex is the waiter's, and the callback never takes it.
    std::atomic<bool> stream_settled{true};
    std::mutex settled_mutex;
    std::condition_variable settled_cv;
};

} // namespace AudioCore
