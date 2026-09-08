// Copyright 2026 Citra Emulator Project / Azahar Emulator Project
// Licensed under GPLv2 or any later version
// Refer to the license.txt file included.

#include <algorithm>
#include <cstddef>
#include <cstdlib>
#include <functional>
#include <limits>
#include <memory>
#include <optional>
#include <utility>
#include <vector>
#include <catch2/catch_test_macros.hpp>
#include "audio_core/audio_types.h"
#include "audio_core/output_pipeline.h"
#include "audio_core/period_splicer.h"
#include "audio_core/stretch_gate.h"
#include "common/common_types.h"

namespace {

using AudioCore::OutputPipeline;
using AudioCore::PeriodSplicer;
using Edge = AudioCore::StretchGate::Edge;
using Mode = AudioCore::StretchGate::Mode;

constexpr double kFs = AudioCore::native_sample_rate;
constexpr std::size_t kCallback = 512;
constexpr double kCallbackSeconds = kCallback / kFs;
constexpr double kFrameSeconds = 1.0 / 60.0;
constexpr double kFramesPerBurst = kFs / 60.0; // one video frame of audio, 545.47
constexpr std::size_t kKey = 8;
/// A seek window plus an overlap of SoundTouch jitter at a seam, rounded up: ~900 frames.
constexpr long kSeamBound = 900;

/// Source frame i: a hash of its index on both channels.
s16 Src(std::size_t i) {
    u32 h = static_cast<u32>(i) * 2654435761u;
    h ^= h >> 15;
    h *= 0x846ca68bu;
    h ^= h >> 16;
    return static_cast<s16>(static_cast<int>((h >> 17) & 0x3fff) - 8192);
}

u64 Key(const s16* frames) {
    u64 k = 1469598103934665603ull;
    for (std::size_t m = 0; m < kKey; m++) {
        k ^= static_cast<u16>(frames[m * 2]);
        k *= 1099511628211ull;
    }
    return k;
}

/// Every eight-frame window of the source, sorted by key.
class SourceIndex {
public:
    explicit SourceIndex(std::size_t total_frames) {
        entries.reserve(total_frames);
        std::vector<s16> win(kKey * 2);
        for (std::size_t i = 0; i + kKey <= total_frames; i++) {
            for (std::size_t m = 0; m < kKey; m++) {
                win[m * 2] = Src(i + m);
                win[(m * 2) + 1] = win[m * 2];
            }
            entries.emplace_back(Key(win.data()), i);
        }
        std::sort(entries.begin(), entries.end());
    }

    /// The source index of the window starting at `frames`, if it is a verbatim copy.
    std::optional<std::size_t> Locate(const s16* frames) const {
        const u64 k = Key(frames);
        const auto it =
            std::lower_bound(entries.begin(), entries.end(), std::make_pair(k, std::size_t{0}));
        if (it == entries.end() || it->first != k) {
            return std::nullopt;
        }
        return it->second;
    }

private:
    std::vector<std::pair<u64, std::size_t>> entries;
};

struct Callback {
    Mode mode; // the mode that rendered it
    OutputPipeline::RenderStats stats;
};

struct Run {
    std::vector<s16> out;
    std::vector<Callback> callbacks;
    std::size_t pushed = 0;
    std::size_t dropped = 0;

    std::size_t FirstFilled() const {
        for (std::size_t c = 0; c < callbacks.size(); c++) {
            if (callbacks[c].stats.written > 0) {
                return c;
            }
        }
        return callbacks.size();
    }
    std::size_t FirstSilenced() const {
        for (std::size_t c = 0; c < callbacks.size(); c++) {
            if (callbacks[c].stats.silenced) {
                return c;
            }
        }
        return callbacks.size();
    }
    std::size_t FirstEdge(Edge edge, std::size_t from = 0) const {
        for (std::size_t c = from; c < callbacks.size(); c++) {
            if (callbacks[c].stats.edge == edge) {
                return c;
            }
        }
        return callbacks.size();
    }
    std::size_t CountEdges(Edge edge) const {
        std::size_t n = 0;
        for (const auto& cb : callbacks) {
            n += cb.stats.edge == edge ? 1 : 0;
        }
        return n;
    }
    static double TimeOf(std::size_t callback) {
        return static_cast<double>(callback) * kCallbackSeconds;
    }
    /// Frames the splicer cut from callback `from` on.
    std::size_t Cuts(std::size_t from = 0) const {
        std::size_t n = 0;
        for (std::size_t c = from; c < callbacks.size(); c++) {
            n += callbacks[c].stats.cut;
        }
        return n;
    }
    /// Minimum depth at callback start over [first, last).
    std::size_t LowWater(std::size_t first, std::size_t last) const {
        std::size_t low = std::numeric_limits<std::size_t>::max();
        for (std::size_t c = first; c < last && c < callbacks.size(); c++) {
            low = std::min(low, callbacks[c].stats.depth);
        }
        return low;
    }
};

/// The emulation thread and the sink's callback as one event loop in wall time: one video
/// frame of audio pushed as a burst every 1/60 s of emulated time, at the speed `speed_at`
/// gives for that moment, and kCallback frames rendered every kCallback / kFs seconds. `at`
/// runs before each callback, for scripting a setting change.
Run Simulate(OutputPipeline& pipeline, double seconds,
             const std::function<double(double)>& speed_at,
             const std::function<void(double, OutputPipeline&)>& at = {}) {
    Run run;
    double t = 0.0;
    double next_burst = 0.0;
    double next_callback = 0.0;
    double burst_acc = 0.0;
    std::vector<s16> burst(1024 * 2);
    std::vector<s16> buffer(kCallback * 2);
    while (t < seconds) {
        if (next_burst <= next_callback) {
            t = next_burst;
            burst_acc += kFramesPerBurst;
            const auto n = static_cast<std::size_t>(burst_acc);
            burst_acc -= static_cast<double>(n);
            for (std::size_t i = 0; i < n; i++) {
                burst[i * 2] = Src(run.pushed + i);
                burst[(i * 2) + 1] = burst[i * 2];
            }
            const std::size_t accepted = pipeline.Push(burst.data(), n);
            run.dropped += n - accepted;
            run.pushed += n;
            next_burst += kFrameSeconds / speed_at(t);
        } else {
            t = next_callback;
            if (at) {
                at(t, pipeline);
            }
            pipeline.Render(buffer.data(), kCallback);
            run.out.insert(run.out.end(), buffer.begin(), buffer.end());
            run.callbacks.push_back({pipeline.CurrentMode(), pipeline.LastRender()});
            next_callback += kCallbackSeconds;
        }
    }
    return run;
}

/// Every callback in [first, last) was filled in full. Two exceptions: an Engage on an
/// empty buffer, where the raw path may come up short on that callback and the two after it
/// while the repeat catches up, and callbacks the core had silenced, which render nothing by
/// design. The ramp covers both.
void RequireFilled(const Run& run, std::size_t first, std::size_t last) {
    std::size_t exempt_until = 0;
    for (std::size_t c = first; c < last && c < run.callbacks.size(); c++) {
        if (run.callbacks[c].stats.edge == Edge::Engage) {
            exempt_until = c + 3;
        }
        if (c < exempt_until || run.callbacks[c].stats.silenced) {
            continue;
        }
        INFO("callback " << c << " at " << Run::TimeOf(c) << " s, mode "
                         << static_cast<int>(run.callbacks[c].mode));
        REQUIRE(run.callbacks[c].stats.written == kCallback);
    }
}

/// Walks the output locating each frame in the source.
///
/// In Bypass and Warming, consecutive located frames advance by one, except across the one
/// splice a callback may carry, where the jump is the splice's size in its direction. The
/// first located frame after a Sync or a Handover is a seam, allowed kSeamBound either way.
/// Between a Handover and the join that ends its flushed run, the frames are the stretcher's
/// own output and may carry its round jitter, again within kSeamBound; the join callback
/// must show exactly one jump matching what it dropped, within kSeamBound, under a blended
/// gap of at least kJoinFrames. Stretch and Drain output is WSOLA and is not checked.
///
/// A callback the source could not fill, or that the core had silenced, carries the ramp's
/// synthesized tail, whose repeat of recent audio can locate; the tail can run on into the
/// next callback, and the ramp then mutes a callback's worth and ramps another in. Tracking
/// starts over at the next located frame, and the located-fraction check waits four
/// callbacks.
///
/// So that a pipeline emitting silence or garbage cannot pass by locating nothing: a raw
/// callback must locate at least 95% of its frames, 70% when it carries a splice or a join
/// (128 blended frames), 40% when it carries an edge (a 256-frame cross-fade), and every
/// seam callback must locate at least one frame.
void CheckTimeline(const Run& run, const SourceIndex& index, std::size_t first_callback) {
    const std::size_t total = run.out.size() / 2;
    std::optional<std::size_t> last_j;
    std::optional<std::size_t> last_p;
    std::size_t skip_until = 0;
    bool in_flush = false;
    for (std::size_t c = first_callback; c < run.callbacks.size(); c++) {
        const auto& cb = run.callbacks[c];
        if (cb.stats.silenced || cb.stats.written < kCallback) {
            last_j.reset();
            last_p.reset();
            skip_until = c + 5;
            continue;
        }
        if (cb.stats.edge == Edge::Handover) {
            in_flush = true;
        }
        const bool raw = cb.mode == Mode::Bypass || cb.mode == Mode::Warming;
        const bool join = cb.stats.joined > 0;
        bool seam = cb.stats.edge == Edge::Sync || cb.stats.edge == Edge::Handover;
        long allowed = 0;
        if (cb.stats.cut > 0) {
            allowed = static_cast<long>(cb.stats.cut);
        }
        if (cb.stats.inserted > 0) {
            allowed = -static_cast<long>(cb.stats.inserted);
        }
        bool jumped = false;
        std::size_t join_jumps = 0;
        std::size_t located = 0;
        for (std::size_t j = c * kCallback; j < (c + 1) * kCallback && j + kKey <= total; j++) {
            const auto p = index.Locate(&run.out[j * 2]);
            if (!p) {
                continue;
            }
            located++;
            if (last_j) {
                const long gap = static_cast<long>(j - *last_j);
                const long jump = static_cast<long>(*p) - static_cast<long>(*last_p) - gap;
                INFO("callback " << c << " at " << Run::TimeOf(c) << " s, frame " << j << ", jump "
                                 << jump << ", gap " << gap);
                if (seam) {
                    REQUIRE(std::abs(jump) <= kSeamBound);
                    seam = false;
                } else if (join && jump != 0 &&
                           std::abs(jump - static_cast<long>(cb.stats.joined)) <= kSeamBound) {
                    REQUIRE(gap >= static_cast<long>(PeriodSplicer::kJoinFrames));
                    join_jumps++;
                } else if (raw && (in_flush || join)) {
                    REQUIRE(std::abs(jump) <= kSeamBound);
                } else if (raw && jump != 0) {
                    REQUIRE(jump == allowed);
                    REQUIRE(!jumped);
                    jumped = true;
                }
            }
            last_j = j;
            last_p = p;
        }
        INFO("callback " << c << " at " << Run::TimeOf(c) << " s located " << located << " of "
                         << kCallback);
        if (join) {
            REQUIRE(join_jumps == 1);
            in_flush = false;
        }
        if (cb.stats.edge == Edge::Sync || cb.stats.edge == Edge::Handover) {
            REQUIRE(located >= 1);
        }
        if (raw && !in_flush && c >= skip_until) {
            const bool spliced = cb.stats.cut > 0 || cb.stats.inserted > 0 || join;
            const double threshold = cb.stats.edge != Edge::None ? 0.40 : spliced ? 0.70 : 0.95;
            REQUIRE(static_cast<double>(located) >= threshold * static_cast<double>(kCallback));
        }
    }
}

double Steady(double) {
    return 1.0;
}
double Dip(double t) {
    return (t >= 4.0 && t < 8.0) ? 0.85 : 1.0;
}
double Hover(double t) {
    return (static_cast<int>(t) % 2 == 0) ? 0.93 : 0.97;
}
double FastForward(double t) {
    return (t >= 3.0 && t < 8.0) ? 3.0 : 1.0;
}
double Creep(double) {
    return 1.001;
}

/// On the heap: the pipeline carries its buffers inline, a few hundred KB, and it owns a mutex
/// and a condition variable, so it is neither small nor movable.
std::unique_ptr<OutputPipeline> Fresh() {
    auto pipeline = std::make_unique<OutputPipeline>();
    pipeline->SetOutputSampleRate(AudioCore::native_sample_rate);
    pipeline->SetStretching(true);
    return pipeline;
}

} // namespace

TEST_CASE("OutputPipeline plays a full-speed stream straight through", "[audio_core][bypass]") {
    auto pipeline = Fresh();
    const Run run = Simulate(*pipeline, 12.0, Steady);
    const SourceIndex index(run.pushed);

    const std::size_t first = run.FirstFilled();
    REQUIRE(Run::TimeOf(first) < 0.1);
    RequireFilled(run, first, run.callbacks.size());
    REQUIRE(run.dropped == 0);
    REQUIRE(run.CountEdges(Edge::Engage) == 0);
    // Prefill may overshoot by up to a burst, which the splicer trims once the low-water
    // window has seen a whole beat cycle; after that a steady stream is never cut.
    REQUIRE(run.Cuts(128) == 0);
    for (const auto& cb : run.callbacks) {
        REQUIRE(cb.mode == Mode::Bypass);
    }
    CheckTimeline(run, index, first + 2);

    // Depth sits where prefill left it: at least a callback, at most the target plus a burst.
    const std::size_t target = pipeline->FillTarget(kCallback);
    const std::size_t low = run.LowWater(run.callbacks.size() - 64, run.callbacks.size());
    REQUIRE(low >= kCallback);
    REQUIRE(low <= target + 600);
}

TEST_CASE("OutputPipeline stretches through a slowdown and hands back", "[audio_core][bypass]") {
    auto pipeline = Fresh();
    const Run run = Simulate(*pipeline, 16.0, Dip);
    const SourceIndex index(run.pushed);

    const std::size_t first = run.FirstFilled();
    RequireFilled(run, first, run.callbacks.size());
    REQUIRE(run.dropped == 0);

    const std::size_t engage = run.FirstEdge(Edge::Engage);
    const std::size_t sync = run.FirstEdge(Edge::Sync);
    const std::size_t handover = run.FirstEdge(Edge::Handover);
    REQUIRE(engage < run.callbacks.size());
    REQUIRE(Run::TimeOf(engage) >= 4.0);
    REQUIRE(Run::TimeOf(engage) < 4.6);
    REQUIRE(sync > engage);
    REQUIRE(Run::TimeOf(sync) - Run::TimeOf(engage) < 0.5);
    REQUIRE(run.callbacks[sync].stats.replayed == 0);
    REQUIRE(handover > sync);
    REQUIRE(Run::TimeOf(handover) > 10.0);
    REQUIRE(Run::TimeOf(handover) < 14.0);
    REQUIRE(run.CountEdges(Edge::Engage) == 1);
    REQUIRE(run.CountEdges(Edge::Handover) == 1);
    REQUIRE(run.callbacks.back().mode == Mode::Bypass);

    CheckTimeline(run, index, first + 2);

    // Within a second of the handover the excess is trimmed to the target plus the slack the
    // splicer leaves, under a period over that, and the raw path never runs dry.
    const std::size_t target = pipeline->FillTarget(kCallback);
    const std::size_t low = run.LowWater(handover + 64, handover + 80);
    REQUIRE(low >= kCallback);
    REQUIRE(low < target + OutputPipeline::kTrimSlack + PeriodSplicer::kMinPeriod);
}

TEST_CASE("OutputPipeline stays stretched while speed hovers below full", "[audio_core][bypass]") {
    auto pipeline = Fresh();
    const Run run = Simulate(*pipeline, 12.0, Hover);

    RequireFilled(run, run.FirstFilled(), run.callbacks.size());
    REQUIRE(run.dropped == 0);
    REQUIRE(run.CountEdges(Edge::Engage) == 1);
    REQUIRE(run.CountEdges(Edge::Sync) == 1);
    REQUIRE(run.CountEdges(Edge::Handover) == 0);
    REQUIRE(run.callbacks.back().mode == Mode::Stretch);
}

TEST_CASE("OutputPipeline stretches through fast-forward and hands back", "[audio_core][bypass]") {
    auto pipeline = Fresh();
    const Run run = Simulate(*pipeline, 16.0, FastForward, [](double t, OutputPipeline& p) {
        // The frontend's frame limit follows the scripted speed.
        p.SetRequestedSpeed(FastForward(t));
    });
    const SourceIndex index(run.pushed);

    const std::size_t first = run.FirstFilled();
    RequireFilled(run, first, run.callbacks.size());
    REQUIRE(run.dropped == 0);

    const std::size_t engage = run.FirstEdge(Edge::Engage);
    const std::size_t sync = run.FirstEdge(Edge::Sync);
    const std::size_t handover = run.FirstEdge(Edge::Handover);
    REQUIRE(engage < run.callbacks.size());
    REQUIRE(sync < run.callbacks.size());
    REQUIRE(handover < run.callbacks.size());
    REQUIRE(Run::TimeOf(engage) >= 3.0);
    REQUIRE(Run::TimeOf(engage) < 3.6);
    REQUIRE(run.callbacks[sync].stats.replayed == 0);
    REQUIRE(Run::TimeOf(handover) > 10.0);
    REQUIRE(Run::TimeOf(handover) < 14.0);
    REQUIRE(run.callbacks.back().mode == Mode::Bypass);
    CheckTimeline(run, index, first + 2);
}

TEST_CASE("OutputPipeline trims a slow creep in Bypass", "[audio_core][bypass]") {
    auto pipeline = Fresh();
    const Run run = Simulate(*pipeline, 20.0, Creep);
    const SourceIndex index(run.pushed);

    const std::size_t first = run.FirstFilled();
    RequireFilled(run, first, run.callbacks.size());
    REQUIRE(run.CountEdges(Edge::Engage) == 0);
    // 0.1% over 20 s is ~650 frames of creep: more than a period, so cuts happen, each at most
    // one period, and the depth stays near the target.
    REQUIRE(run.Cuts() > 0);
    for (const auto& cb : run.callbacks) {
        REQUIRE(cb.stats.cut <= PeriodSplicer::kMaxPeriod);
    }
    CheckTimeline(run, index, first + 2);
    const std::size_t target = pipeline->FillTarget(kCallback);
    const std::size_t low = run.LowWater(run.callbacks.size() - 32, run.callbacks.size());
    REQUIRE(low >= kCallback);
    REQUIRE(low < target + OutputPipeline::kTrimSlack + PeriodSplicer::kMinPeriod +
                      PeriodSplicer::kMaxPeriod);
}

TEST_CASE("OutputPipeline drains and hands over when stretching is turned off",
          "[audio_core][bypass]") {
    auto pipeline = Fresh();
    const Run run = Simulate(*pipeline, 12.0, Dip, [](double t, OutputPipeline& p) {
        if (t >= 6.0) {
            p.SetStretching(false);
        }
    });

    const std::size_t handover = run.FirstEdge(Edge::Handover);
    REQUIRE(handover < run.callbacks.size());
    REQUIRE(Run::TimeOf(handover) >= 6.0);
    REQUIRE(Run::TimeOf(handover) < 9.0);
    // Full up to and including the handover; after it the host is slow with no stretcher,
    // and underruns are what was asked for.
    RequireFilled(run, run.FirstFilled(), handover + 1);
    for (std::size_t c = handover; c < run.callbacks.size(); c++) {
        REQUIRE(run.callbacks[c].mode == Mode::Bypass);
    }
    REQUIRE(run.FirstEdge(Edge::Engage, handover) == run.callbacks.size());
    const SourceIndex index(run.pushed);
    CheckTimeline(run, index, run.FirstFilled() + 2);
}

TEST_CASE("OutputPipeline resumes a warm-up across a pause without replaying",
          "[audio_core][bypass]") {
    // The dip engages by ~4.35 s and syncs around 4.6 s; the core takes the stream down for
    // half a second in the middle of the warm-up. The stash it discards is already inside the
    // stretcher, and the Sync discard must count those frames as played, or they come back
    // as a replay.
    auto pipeline = Fresh();
    const Run run = Simulate(*pipeline, 12.0, Dip, [](double t, OutputPipeline& p) {
        if (t >= 4.45 && t < 4.95) {
            p.StreamEnd();
        } else {
            p.StreamBegin();
        }
    });
    const SourceIndex index(run.pushed);

    const std::size_t engage = run.FirstEdge(Edge::Engage);
    const std::size_t sync = run.FirstEdge(Edge::Sync);
    REQUIRE(engage < run.callbacks.size());
    REQUIRE(sync < run.callbacks.size());
    REQUIRE(Run::TimeOf(engage) < 4.45);
    REQUIRE(Run::TimeOf(sync) >= 4.95);
    REQUIRE(run.callbacks[sync].stats.replayed == 0);
    // Full up to the pause; after it the warm-up refills on the ramp before it plays, so
    // the requirement resumes at the switch.
    RequireFilled(run, run.FirstFilled(), run.FirstSilenced());
    RequireFilled(run, sync, run.callbacks.size());
    CheckTimeline(run, index, run.FirstFilled() + 2);
}
