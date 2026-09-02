// Copyright 2026 Citra Emulator Project / Azahar Emulator Project
// Licensed under GPLv2 or any later version
// Refer to the license.txt file included.

#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <numbers>
#include "common/common_types.h"

namespace AudioCore {

/// Fourth-order Butterworth low-pass: two cascaded RBJ biquads, stereo with independent state
/// per channel. The cutoff is smoothed rather than jumped, so engaging fast-forward slides the
/// filter shut instead of stepping the coefficients, which would click. At wide open the output
/// is left untouched.
class SpeedupLowPass {
public:
    // Time constant of the cutoff smoother, in seconds.
    static constexpr double kSmoothingTau = 0.05;
    // Fraction of wide-open at which the filter stops touching the output.
    static constexpr double kBypassThreshold = 0.995;
    // Section Q's for a fourth-order Butterworth cascade.
    static constexpr double kSectionQ[2] = {0.54119610014619698, 1.3065629648763766};
    // Lowest cutoff the coefficient design will accept.
    static constexpr double kMinCutoff = 20.0;

    void Init(double sample_rate_) {
        sample_rate = sample_rate_;
        wide_open = std::max(0.45 * sample_rate, kMinCutoff);
        for (int s = 0; s < 2; s++) {
            stages[s].z1[0] = stages[s].z1[1] = 0.0;
            stages[s].z2[0] = stages[s].z2[1] = 0.0;
        }
        SetCutoffNow(wide_open);
    }

    double WideOpenCutoff() const {
        return wide_open;
    }
    double Cutoff() const {
        return cur_cutoff;
    }
    bool Bypassed() const {
        return cur_cutoff >= (wide_open * kBypassThreshold);
    }

    /// Advance the smoothed cutoff by one block, then filter in place.
    void Process(s16* samples, std::size_t num_frames, double target_hz, double block_seconds) {
        Smooth(target_hz, block_seconds);
        const bool bypass = Bypassed();

        for (std::size_t i = 0; i < num_frames; i++) {
            for (std::size_t ch = 0; ch < 2; ch++) {
                // Runs even when bypassed: its state must stay in step with the signal, or
                // re-engaging would click.
                const double y = ProcessSample(samples[(i * 2) + ch], ch);
                if (!bypass) {
                    samples[(i * 2) + ch] = Saturate(y);
                }
            }
        }
    }

    void Smooth(double target_hz, double block_seconds) {
        target_hz = std::clamp(target_hz, kMinCutoff, wide_open);
        const double a = 1.0 - std::exp(-block_seconds / kSmoothingTau);
        SetCutoffNow(cur_cutoff + ((target_hz - cur_cutoff) * a));
    }

    void SetCutoffNow(double cutoff_hz) {
        cur_cutoff = std::clamp(cutoff_hz, kMinCutoff, wide_open);
        for (int s = 0; s < 2; s++) {
            stages[s].Design(cur_cutoff, sample_rate, kSectionQ[s]);
        }
    }

    double ProcessSample(double x, std::size_t ch) {
        double y = x;
        for (int s = 0; s < 2; s++) {
            y = stages[s].Run(y, ch);
        }
        return y;
    }

private:
    static s16 Saturate(double y) {
        long v = std::lround(y);
        if (v > 32767) {
            v = 32767;
        }
        if (v < -32768) {
            v = -32768;
        }
        return static_cast<s16>(v);
    }

    struct Biquad {
        double b0 = 1.0, b1 = 0.0, b2 = 0.0, a1 = 0.0, a2 = 0.0;
        double z1[2] = {0.0, 0.0};
        double z2[2] = {0.0, 0.0};

        void Design(double cutoff_hz, double sample_rate, double q) {
            const double w0 = 2.0 * std::numbers::pi * (cutoff_hz / sample_rate);
            const double cw = std::cos(w0);
            const double alpha = std::sin(w0) / (2.0 * q);
            const double a0 = 1.0 + alpha;

            b0 = ((1.0 - cw) * 0.5) / a0;
            b1 = (1.0 - cw) / a0;
            b2 = b0;
            a1 = (-2.0 * cw) / a0;
            a2 = (1.0 - alpha) / a0;
        }

        // Transposed direct form II.
        double Run(double x, std::size_t ch) {
            const double y = (b0 * x) + z1[ch];
            z1[ch] = (b1 * x) - (a1 * y) + z2[ch];
            z2[ch] = (b2 * x) - (a2 * y);
            return y;
        }
    };

    double sample_rate = 48000.0;
    double wide_open = 21600.0;
    double cur_cutoff = 21600.0;
    Biquad stages[2];
};

} // namespace AudioCore
