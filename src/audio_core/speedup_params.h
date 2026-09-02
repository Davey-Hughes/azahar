// Copyright 2026 Citra Emulator Project / Azahar Emulator Project
// Licensed under GPLv2 or any later version
// Refer to the license.txt file included.

#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>
#include "common/common_types.h"

namespace AudioCore {

/// Sentinel meaning "do not filter"; taken literally, 24000 would still filter at 12 kHz at 2x.
constexpr u16 kSpeedupLowPassOff = 24000;

/// Speed used when the frame limiter is off; Settings::GetFrameLimit() returns 0 for unlimited.
constexpr double kUnlimitedSpeed = 10.0;

constexpr double kStretchTrimGain = 0.25; // gain of the input-fill correction on the ratio

/// Below 1 the stretcher expands (slow motion), above it compresses (fast-forward).
constexpr double kMinStretchRatio = 0.25;
constexpr double kMaxStretchRatio = 32.0;

/// Lowest cutoff the low-pass will be asked for, in Hz.
constexpr double kMinLowPassCutoff = 200.0;

inline double SpeedupSpeedFromFrameLimit(double frame_limit_percent) {
    if (frame_limit_percent <= 0.0) {
        return kUnlimitedSpeed;
    }
    return frame_limit_percent / 100.0;
}

/// Whether a speed other than normal was requested. Keyed off the requested speed, not the
/// achieved one: the host falling behind is TimeStretcher's job (audio_core/time_stretch.h).
inline bool SpeedupIsOffSpeed(double speed) {
    return std::fabs(speed - 1.0) > 0.01;
}

/// Input consumed per output frame, tracked from measured arrival rather than requested speed.
inline double SpeedupStretchRatio(double arrival_per_callback, std::size_t output_per_callback,
                                  int input_fill, int target_fill) {
    if (output_per_callback == 0) {
        return 1.0;
    }

    double ratio = arrival_per_callback / static_cast<double>(output_per_callback);

    if (target_fill > 0) {
        double err = (input_fill - static_cast<double>(target_fill)) / target_fill;
        err = std::clamp(err, -1.0, 1.0);
        ratio *= 1.0 + (kStretchTrimGain * err);
    }

    return std::clamp(ratio, kMinStretchRatio, kMaxStretchRatio);
}

/// Low-pass cutoff, in Hz (wide_open = transparent); reference/speed gives the applied cutoff.
inline double SpeedupLowPassCutoff(double speed, u16 reference, double wide_open) {
    if (wide_open <= kMinLowPassCutoff) {
        return wide_open;
    }
    if (speed <= 1.0) {
        return wide_open;
    }
    if (reference >= kSpeedupLowPassOff) {
        return wide_open;
    }

    return std::clamp(reference / speed, kMinLowPassCutoff, wide_open);
}

} // namespace AudioCore
