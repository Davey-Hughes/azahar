// Copyright 2026 Citra Emulator Project / Azahar Emulator Project
// Licensed under GPLv2 or any later version
// Refer to the license.txt file included.

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include "audio_core/speedup_params.h"

using Catch::Matchers::WithinRel;

TEST_CASE("SpeedupSpeedFromFrameLimit", "[audio_core][speedup]") {
    REQUIRE_THAT(AudioCore::SpeedupSpeedFromFrameLimit(100.0), WithinRel(1.0, 1e-9));
    REQUIRE_THAT(AudioCore::SpeedupSpeedFromFrameLimit(400.0), WithinRel(4.0, 1e-9));
    REQUIRE_THAT(AudioCore::SpeedupSpeedFromFrameLimit(50.0), WithinRel(0.5, 1e-9));

    // 0 means "unlimited" in Settings::GetFrameLimit(), not "stopped".
    REQUIRE_THAT(AudioCore::SpeedupSpeedFromFrameLimit(0.0),
                 WithinRel(AudioCore::kUnlimitedSpeed, 1e-9));
    REQUIRE_THAT(AudioCore::SpeedupSpeedFromFrameLimit(-5.0),
                 WithinRel(AudioCore::kUnlimitedSpeed, 1e-9));
}

TEST_CASE("SpeedupIsOffSpeed", "[audio_core][speedup]") {
    REQUIRE_FALSE(AudioCore::SpeedupIsOffSpeed(1.0));
    REQUIRE_FALSE(AudioCore::SpeedupIsOffSpeed(1.005));
    REQUIRE(AudioCore::SpeedupIsOffSpeed(1.5));
    REQUIRE(AudioCore::SpeedupIsOffSpeed(0.5));
}

TEST_CASE("SpeedupStretchRatio tracks arrival rate", "[audio_core][speedup]") {
    // With the fill already on target, the ratio is exactly arrival / output.
    REQUIRE_THAT(AudioCore::SpeedupStretchRatio(2048.0, 512, 4096, 4096), WithinRel(4.0, 1e-9));

    // A full input buffer must drain faster, so the ratio goes up.
    REQUIRE(AudioCore::SpeedupStretchRatio(2048.0, 512, 8192, 4096) > 4.0);
    // A starved input buffer must drain slower.
    REQUIRE(AudioCore::SpeedupStretchRatio(2048.0, 512, 1024, 4096) < 4.0);

    // Degenerate inputs must not divide by zero or escape the clamp.
    REQUIRE_THAT(AudioCore::SpeedupStretchRatio(2048.0, 0, 4096, 4096), WithinRel(1.0, 1e-9));
    REQUIRE(AudioCore::SpeedupStretchRatio(1.0e9, 512, 4096, 4096) <= AudioCore::kMaxStretchRatio);
    REQUIRE(AudioCore::SpeedupStretchRatio(0.0, 512, 0, 4096) >= AudioCore::kMinStretchRatio);
}

TEST_CASE("SpeedupLowPassCutoff", "[audio_core][speedup]") {
    // wide_open for azahar's 32728 Hz sink: 0.45 * 32728.
    constexpr double wide_open = 14727.6;

    // At or below normal speed the filter is transparent.
    REQUIRE_THAT(AudioCore::SpeedupLowPassCutoff(1.0, 12000, wide_open),
                 WithinRel(wide_open, 1e-9));
    REQUIRE_THAT(AudioCore::SpeedupLowPassCutoff(0.5, 12000, wide_open),
                 WithinRel(wide_open, 1e-9));

    // The setting is a reference, divided by speed to get the applied cutoff.
    REQUIRE_THAT(AudioCore::SpeedupLowPassCutoff(2.0, 12000, wide_open), WithinRel(6000.0, 1e-9));
    REQUIRE_THAT(AudioCore::SpeedupLowPassCutoff(4.0, 24000 - 1, wide_open),
                 WithinRel(5999.75, 1e-9));

    // The sentinel means off, at any speed.
    REQUIRE_THAT(AudioCore::SpeedupLowPassCutoff(4.0, AudioCore::kSpeedupLowPassOff, wide_open),
                 WithinRel(wide_open, 1e-9));

    // Never above wide_open, never below the 200 Hz floor.
    REQUIRE_THAT(AudioCore::SpeedupLowPassCutoff(1.1, 20000, wide_open),
                 WithinRel(wide_open, 1e-9));
    REQUIRE(AudioCore::SpeedupLowPassCutoff(100.0, 1000, wide_open) >= 200.0);
}
