// Copyright 2026 Citra Emulator Project / Azahar Emulator Project
// Licensed under GPLv2 or any later version
// Refer to the license.txt file included.

#pragma once

#include "common/common_types.h"

namespace AudioCore {

/// WSOLA time-stretcher: preserves pitch across playback-rate changes by overlap-adding
/// similarity-matched frames. Not thread-safe: DspInterface::OutputCallback()
/// (audio_core/dsp_interface.cpp) calls Write() then Read() back to back on the audio thread.
class WsolaStretcher {
public:
    static constexpr int kFrameSize = 256;               // analysis window, frames
    static constexpr int kSynthesisHop = kFrameSize / 2; // periodic Hann at 50% sums to unity
    static constexpr int kSearchRadius = 1024;           // frames either side of nominal
    static constexpr int kCoarseStride = 4;
    static constexpr int kFineRadius = 3;
    static constexpr int kInputCapacity = 32768; // must be a power of two
    static constexpr int kOutputCapacity = 8192; // must be a power of two

    static constexpr int kMinTargetInputFill = 4096; // buffered-input floor, frames

    /// Buffered-input target, in frames, scaled to the measured arrival rate rather than fixed.
    static int TargetInputFill(double arrival_per_callback);

    WsolaStretcher();

    /// Full clear, including the input ring; the audio thread's Resync() is the cheap alternative.
    void Reset();

    /// Restarts synthesis at the newest input rather than splicing across a gap in it. Frames
    /// already synthesized are kept: only the analysis state went stale.
    void Resync();

    /// Resync(), and also drop anything already synthesized. For starting a fresh run, where the
    /// output ring may still hold frames from the previous one.
    void BeginSession();

    /// Appends interleaved stereo frames, returning how many were accepted.
    int Write(const s16* samples, int num_frames);

    /// Emit up to num_frames of interleaved stereo, synthesizing as needed.
    int Read(s16* samples, int num_frames, double ratio);

    int InputFill() const;
    int OutputFill() const;
    s64 TotalWritten() const;

private:
    static s16 Saturate(float v);

    bool CanSynthesise() const;
    void SynthesiseHop(double ratio);
    s64 FindBestOffset() const;
    double Energy(s64 pos) const;
    double Score(s64 pos, double ref_energy) const;

    float window[kFrameSize];

    s16 in_l[kInputCapacity];
    s16 in_r[kInputCapacity];
    float in_mono[kInputCapacity];
    s64 write_pos = 0;

    s64 analysis_pos = 0;
    s64 natural_pos = 0;
    // Oldest frame the search may look at. Bounds it to the current run, so a splice cannot
    // reach back across a discontinuity into audio from before it.
    s64 session_start = 0;
    bool primed = false;

    float acc_l[kFrameSize];
    float acc_r[kFrameSize];

    s16 out_l[kOutputCapacity];
    s16 out_r[kOutputCapacity];
    s64 out_read_pos = 0;
    s64 out_write_pos = 0;
};

} // namespace AudioCore
