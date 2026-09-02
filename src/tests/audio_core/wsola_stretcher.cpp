// Copyright 2026 Citra Emulator Project / Azahar Emulator Project
// Licensed under GPLv2 or any later version
// Refer to the license.txt file included.

#include <algorithm>
#include <cmath>
#include <numbers>
#include <vector>
#include <catch2/catch_test_macros.hpp>
#include "audio_core/speedup_params.h"
#include "audio_core/wsola_stretcher.h"
#include "common/common_types.h"

namespace {

constexpr double kSampleRate = 32728.0;

/// SDL2Sink::SDL2Sink()'s callback size, and CubebSink::CubebSink()'s floor.
constexpr int kCallbackFrames = 512;
/// DspInterface::kPopChunkFrames, audio_core/dsp_interface.h.
constexpr int kPopChunkFrames = 2048;
/// Long enough that cycling it is rare, short enough to stay cheap.
constexpr int kPoolFrames = 1 << 16;

std::vector<s16> MakeSine(double freq, int num_frames) {
    std::vector<s16> out(static_cast<std::size_t>(num_frames) * 2);
    for (int i = 0; i < num_frames; i++) {
        const double t = static_cast<double>(i) / kSampleRate;
        const auto v = static_cast<s16>(10000.0 * std::sin(2.0 * std::numbers::pi * freq * t));
        out[(static_cast<std::size_t>(i) * 2) + 0] = v;
        out[(static_cast<std::size_t>(i) * 2) + 1] = v;
    }
    return out;
}

/// Zero crossings per second over the back half, which for a clean tone is 2x its frequency.
double EstimateFrequency(const std::vector<s16>& samples, int num_frames) {
    int crossings = 0;
    const int start = num_frames / 2;
    for (int i = start + 1; i < num_frames; i++) {
        const s16 prev = samples[(static_cast<std::size_t>(i - 1) * 2)];
        const s16 cur = samples[(static_cast<std::size_t>(i) * 2)];
        if ((prev < 0 && cur >= 0) || (prev >= 0 && cur < 0)) {
            crossings++;
        }
    }
    const double seconds = (num_frames - start) / kSampleRate;
    return crossings / (2.0 * seconds);
}

struct ConvergenceResult {
    /// First callback whose input fill was within tolerance of the target, or -1.
    int converged_at = -1;
    /// Callbacks at or after that point which could not fill the output buffer.
    int short_after_converged = 0;
    int final_fill = 0;
    int final_target = 0;
};

/// Runs the ratio controller and the stretcher as the closed loop they form in
/// DspInterface::FillFromWsola() (audio_core/dsp_interface.cpp): arrival is written in
/// kPopChunkFrames chunks with a Resync() if the ring saturates, the arrival estimate uses the
/// same smoother, and the ratio that falls out drives one Read() of a whole callback.
ConvergenceResult RunRatioLoop(double speed, int callbacks, double tolerance) {
    AudioCore::WsolaStretcher stretcher;
    const int arrival = static_cast<int>(std::lround(kCallbackFrames * speed));

    // Cycled rather than regenerated, so the similarity search always has real signal to lock on.
    const auto pool = MakeSine(440.0, kPoolFrames);
    int pool_pos = 0;

    std::vector<s16> out(static_cast<std::size_t>(kCallbackFrames) * 2);

    // Seeded the way DspInterface::OutputCallback() seeds it when it engages the off-speed path.
    double arrival_avg = kCallbackFrames * speed;
    s64 last_written = 0;

    ConvergenceResult result;
    for (int c = 0; c < callbacks; c++) {
        int remaining = arrival;
        while (remaining > 0) {
            const int chunk = std::min(remaining, kPopChunkFrames);
            if (pool_pos + chunk > kPoolFrames) {
                pool_pos = 0;
            }
            const int accepted =
                stretcher.Write(&pool[static_cast<std::size_t>(pool_pos) * 2], chunk);
            pool_pos += chunk;
            remaining -= chunk;
            if (accepted < chunk) {
                stretcher.Resync();
                break;
            }
        }

        const s64 written = stretcher.TotalWritten();
        const auto delta = static_cast<double>(std::max<s64>(0, written - last_written));
        last_written = written;
        arrival_avg += (delta - arrival_avg) * 0.05;

        const int fill = stretcher.InputFill();
        const int target = AudioCore::WsolaStretcher::TargetInputFill(arrival_avg);
        const double ratio =
            AudioCore::SpeedupStretchRatio(arrival_avg, kCallbackFrames, fill, target);
        const int got = stretcher.Read(out.data(), kCallbackFrames, ratio);

        const double error = std::fabs(fill - static_cast<double>(target)) / target;
        if (result.converged_at < 0 && error <= tolerance) {
            result.converged_at = c;
        }
        if (result.converged_at >= 0 && got < kCallbackFrames) {
            result.short_after_converged++;
        }
        result.final_fill = fill;
        result.final_target = target;
    }
    return result;
}

} // namespace

TEST_CASE("WsolaStretcher reports what it holds", "[audio_core][speedup]") {
    AudioCore::WsolaStretcher stretcher;

    REQUIRE(stretcher.InputFill() == 0);
    REQUIRE(stretcher.OutputFill() == 0);
    REQUIRE(stretcher.TotalWritten() == 0);

    const auto input = MakeSine(440.0, 1024);
    REQUIRE(stretcher.Write(input.data(), 1024) == 1024);
    REQUIRE(stretcher.InputFill() == 1024);
    REQUIRE(stretcher.TotalWritten() == 1024);
}

TEST_CASE("WsolaStretcher refuses to overrun its ring", "[audio_core][speedup]") {
    AudioCore::WsolaStretcher stretcher;
    const auto input = MakeSine(440.0, AudioCore::WsolaStretcher::kInputCapacity);

    const int first = stretcher.Write(input.data(), AudioCore::WsolaStretcher::kInputCapacity);
    REQUIRE(first == AudioCore::WsolaStretcher::kInputCapacity);

    // Nothing has been read, so there is no room and the write must be refused, not wrapped.
    REQUIRE(stretcher.Write(input.data(), 512) == 0);
}

TEST_CASE("WsolaStretcher reopens ring space as it consumes", "[audio_core][speedup]") {
    AudioCore::WsolaStretcher stretcher;
    const auto input = MakeSine(440.0, AudioCore::WsolaStretcher::kInputCapacity);
    REQUIRE(stretcher.Write(input.data(), AudioCore::WsolaStretcher::kInputCapacity) ==
            AudioCore::WsolaStretcher::kInputCapacity);

    // Ring full, floor still at 0: nothing can be accepted.
    REQUIRE(stretcher.Write(input.data(), 512) == 0);

    std::vector<s16> out(2048 * 2);
    REQUIRE(stretcher.Read(out.data(), 2048, 4.0) == 2048);

    // Consuming advanced the floor, so space must have reopened. This is the substance of
    // computing the floor inline rather than publishing it from the consumer side.
    REQUIRE(stretcher.Write(input.data(), 512) == 512);
}

TEST_CASE("WsolaStretcher compresses at ratio > 1", "[audio_core][speedup]") {
    AudioCore::WsolaStretcher stretcher;
    const int num_frames = AudioCore::WsolaStretcher::kInputCapacity / 2;
    const auto input = MakeSine(440.0, num_frames);
    REQUIRE(stretcher.Write(input.data(), num_frames) == num_frames);

    std::vector<s16> out(2048 * 2);
    const int got = stretcher.Read(out.data(), 2048, 4.0);
    REQUIRE(got == 2048);

    // Compressing 4:1 must have consumed roughly four input frames per output frame.
    const int consumed = num_frames - stretcher.InputFill();
    REQUIRE(consumed > 2048 * 3);
    REQUIRE(consumed < 2048 * 5);
}

TEST_CASE("WsolaStretcher preserves pitch", "[audio_core][speedup]") {
    AudioCore::WsolaStretcher stretcher;
    const int num_frames = AudioCore::WsolaStretcher::kInputCapacity / 2;
    const auto input = MakeSine(440.0, num_frames);
    REQUIRE(stretcher.Write(input.data(), num_frames) == num_frames);

    std::vector<s16> out(4096 * 2);
    REQUIRE(stretcher.Read(out.data(), 4096, 3.0) == 4096);

    // The whole point: 3x faster playback, same tone. A resampler would report ~1320 Hz.
    const double freq = EstimateFrequency(out, 4096);
    REQUIRE(freq > 380.0);
    REQUIRE(freq < 500.0);
}

TEST_CASE("WsolaStretcher yields nothing when starved", "[audio_core][speedup]") {
    AudioCore::WsolaStretcher stretcher;
    std::vector<s16> out(512 * 2, 1234);

    // Not even one analysis frame available, so it must report 0 rather than emit garbage.
    REQUIRE(stretcher.Read(out.data(), 512, 1.0) == 0);
    REQUIRE(out[0] == 1234);
}

TEST_CASE("WsolaStretcher Resync drops stale input", "[audio_core][speedup]") {
    AudioCore::WsolaStretcher stretcher;
    const auto input = MakeSine(440.0, 8192);
    REQUIRE(stretcher.Write(input.data(), 8192) == 8192);
    REQUIRE(stretcher.InputFill() == 8192);

    stretcher.Resync();

    // Resync abandons everything but the newest analysis frame.
    REQUIRE(stretcher.InputFill() <= AudioCore::WsolaStretcher::kFrameSize);
    REQUIRE(stretcher.OutputFill() == 0);
}

TEST_CASE("WsolaStretcher BeginSession does not replay the previous run", "[audio_core][speedup]") {
    AudioCore::WsolaStretcher stretcher;

    // A run of loud audio, partly consumed.
    const auto loud = MakeSine(440.0, 8192);
    REQUIRE(stretcher.Write(loud.data(), 8192) == 8192);
    std::vector<s16> out(2048 * 2);
    REQUIRE(stretcher.Read(out.data(), 2048, 2.0) == 2048);

    // The user drops back to normal speed, where nothing writes here, then engages again. The
    // input ring still holds the run above, so starting anywhere inside it splices arbitrarily
    // old audio into the new run - feed silence, and any non-zero sample out is leaked.
    stretcher.BeginSession();
    const std::vector<s16> silence(static_cast<std::size_t>(8192) * 2, 0);
    REQUIRE(stretcher.Write(silence.data(), 8192) == 8192);

    std::vector<s16> out2(2048 * 2);
    const int got = stretcher.Read(out2.data(), 2048, 2.0);
    REQUIRE(got > 0);

    s16 peak = 0;
    for (int i = 0; i < got * 2; i++) {
        peak = std::max(peak, static_cast<s16>(std::abs(out2[i])));
    }
    REQUIRE(peak == 0);
}

TEST_CASE("WsolaStretcher Resync keeps synthesized output, BeginSession drops it",
          "[audio_core][speedup]") {
    AudioCore::WsolaStretcher stretcher;
    const auto input = MakeSine(440.0, AudioCore::WsolaStretcher::kInputCapacity / 2);
    REQUIRE(stretcher.Write(input.data(), AudioCore::WsolaStretcher::kInputCapacity / 2) ==
            AudioCore::WsolaStretcher::kInputCapacity / 2);

    // 200 is not a multiple of kSynthesisHop, so synthesis overshoots and leaves a remainder in
    // the output ring - the state a sink whose callback size is not hop-aligned sits in.
    std::vector<s16> out(200 * 2);
    REQUIRE(stretcher.Read(out.data(), 200, 2.0) == 200);
    const int pending = stretcher.OutputFill();
    REQUIRE(pending > 0);

    // Resync drops stale input only. Those frames are fully overlap-added and still good, and
    // discarding them would just move the discontinuity earlier.
    stretcher.Resync();
    REQUIRE(stretcher.OutputFill() == pending);

    // BeginSession starts a fresh run, so they go.
    stretcher.BeginSession();
    REQUIRE(stretcher.OutputFill() == 0);
}

TEST_CASE("WsolaStretcher Resync recovers after input is consumed to exhaustion",
          "[audio_core][speedup]") {
    AudioCore::WsolaStretcher stretcher;
    const int num_frames = AudioCore::WsolaStretcher::kInputCapacity / 2;
    const auto input = MakeSine(440.0, num_frames);
    REQUIRE(stretcher.Write(input.data(), num_frames) == num_frames);

    // Read in chunks until starved.
    std::vector<s16> out(256 * 2);
    int got = 256;
    int iterations = 0;
    while (got == 256 && iterations < 200) {
        got = stretcher.Read(out.data(), 256, 4.0);
        iterations++;
    }
    REQUIRE(got < 256);

    stretcher.Resync();

    // Resync() is a pure function of write_pos alone - it does not read the prior analysis_pos
    // or natural_pos - so there is no separate "overshot" case to cover here.
    REQUIRE(stretcher.InputFill() <= AudioCore::WsolaStretcher::kFrameSize);
    REQUIRE(stretcher.OutputFill() == 0);

    // The stretcher must be fully usable again: not wedged with a phantom reservation.
    REQUIRE(stretcher.Write(input.data(), 1024) == 1024);
}

TEST_CASE("WsolaStretcher write floor follows natural_pos at high ratio", "[audio_core][speedup]") {
    AudioCore::WsolaStretcher stretcher;
    const auto input = MakeSine(440.0, AudioCore::WsolaStretcher::kInputCapacity);
    REQUIRE(stretcher.Write(input.data(), AudioCore::WsolaStretcher::kInputCapacity) ==
            AudioCore::WsolaStretcher::kInputCapacity);

    // At ratio 20, hop 2560 exceeds kSearchRadius + kSynthesisHop, so natural_pos trails
    // analysis_pos - kSearchRadius for every possible search result (below about ratio 9 the two
    // terms can swap) - hence the floor takes their minimum, not just the search window.
    std::vector<s16> out(256 * 2);
    REQUIRE(stretcher.Read(out.data(), 256, 20.0) == 256);

    const s64 analysis_pos = stretcher.TotalWritten() - stretcher.InputFill();
    REQUIRE(analysis_pos == 5120);

    // The ring is full, so Write can accept exactly the floor. Were the floor computed from
    // analysis_pos - kSearchRadius alone it would be 4096; natural_pos binds it strictly lower.
    const int accepted = stretcher.Write(input.data(), AudioCore::WsolaStretcher::kInputCapacity);
    REQUIRE(accepted > 0);
    REQUIRE(accepted < analysis_pos - AudioCore::WsolaStretcher::kSearchRadius);
}

TEST_CASE("WsolaStretcher TargetInputFill scales with arrival", "[audio_core][speedup]") {
    // Floor applies when arrival is small.
    REQUIRE(AudioCore::WsolaStretcher::TargetInputFill(64.0) ==
            AudioCore::WsolaStretcher::kMinTargetInputFill);

    // Grows with arrival, because a callback consumes arrival frames.
    REQUIRE(AudioCore::WsolaStretcher::TargetInputFill(4096.0) >
            AudioCore::WsolaStretcher::kMinTargetInputFill);

    // Capped at half the ring, and an absurd arrival must not overflow the int cast.
    REQUIRE(AudioCore::WsolaStretcher::TargetInputFill(2.5e9) ==
            AudioCore::WsolaStretcher::kInputCapacity / 2);
}

TEST_CASE("Speed-up ratio control converges and keeps the callback fed", "[audio_core][speedup]") {
    // Whether the controller settles instead of hunting is the one property a user would notice
    // failing, and nothing else covers it; Pipeline() in audio_core/tools/audiobench.cpp drives
    // the same loop but only prints. Convergence is monotonic at every speed, so the budgets only
    // need headroom over what was measured: 188 callbacks at 0.5x, 43 at 2x, 25 at 4x, 21 at 10x.
    // 0.5x is slowest because the loop closes the fill deficit at a rate proportional to arrival.
    //
    // Per speed rather than one flat figure, because cost per callback scales with arrival: a flat
    // budget spends most of its time at the fastest speed for no extra coverage. Worth keeping
    // lean - this is by far the heaviest test here, and at 800 callbacks a speed it slowed the
    // process enough to upset AudioTest-BiquadFilter, which polls for the DSP frame it compares.
    constexpr double kTolerance = 0.05;

    struct Case {
        double speed;
        int budget; ///< Convergence has to happen inside this many callbacks.
        int total;  ///< And hold for the rest of them.
    };

    for (const Case c : {Case{0.5, 280, 360}, Case{2.0, 90, 140}, Case{4.0, 60, 100},
                         Case{AudioCore::kUnlimitedSpeed, 60, 100}}) {
        DYNAMIC_SECTION("speed " << c.speed << "x") {
            const ConvergenceResult result = RunRatioLoop(c.speed, c.total, kTolerance);

            REQUIRE(result.converged_at >= 0);
            REQUIRE(result.converged_at < c.budget);

            // Converged is not enough: it has to stay there for the rest of the run.
            const double final_error =
                std::fabs(result.final_fill - static_cast<double>(result.final_target)) /
                result.final_target;
            REQUIRE(final_error <= kTolerance);

            // And a settled controller must never leave the callback padding with a held frame.
            REQUIRE(result.short_after_converged == 0);
        }
    }
}
