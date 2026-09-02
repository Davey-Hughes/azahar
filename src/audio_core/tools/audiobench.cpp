// Copyright 2026 Citra Emulator Project / Azahar Emulator Project
// Licensed under GPLv2 or any later version
// Refer to the license.txt file included.

// Measurement tool for the off-speed audio path. Not part of the normal build - gated behind
// the ENABLE_AUDIOBENCH CMake option, which defaults off.
//
//   cmake -S . -B build -DENABLE_AUDIOBENCH=ON
//   cmake --build build --target audiobench
//   ./build/bin/Release/audiobench [stressSeconds]
//
// For the race check, configure a separate build directory with
// -DCMAKE_CXX_FLAGS=-fsanitize=thread and rerun; the stress section is what exercises the
// producer/consumer boundary.
//
// Four sections:
//   pipeline        where audio is lost, and to what, at each speed
//   cost            CPU the stretcher and low-pass actually consume
//   search activity what WsolaStretcher's similarity search actually does: the best_k
//                    distribution FindBestOffset() (audio_core/wsola_stretcher.cpp) picks, and
//                    off-partial energy with the search on versus forced off. Built with
//                    -fno-access-control (this target only, see CMakeLists.txt) to read the
//                    search's private state directly - WsolaStretcher itself carries no test
//                    hooks.
//   stress          two real threads across Common::RingBuffer (common/ring_buffer.h), with a
//                    tearing detector
//
// stress drives the one boundary in this path that is still genuinely cross-thread: the FIFO
// DspInterface::OutputFrame() writes from the emulator thread and OutputCallback() drains from
// the audio thread (audio_core/dsp_interface.cpp). WsolaStretcher itself runs single-threaded
// downstream and is not part of that race, so the tearing detector just compares L against R in
// the frames popped from the FIFO, catching a torn read of a stereo slot.

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <complex>
#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <numbers>
#include <thread>
#include <vector>
#include "audio_core/speedup_lowpass.h"
#include "audio_core/speedup_params.h"
#include "audio_core/wsola_stretcher.h"
#include "common/common_types.h"
#include "common/ring_buffer.h"

static constexpr double kFs = 32728.0; // AudioCore::native_sample_rate, audio_core/audio_types.h

// Mirrors of constants owned elsewhere, so this tool stays standalone.
static constexpr std::size_t kFifoCapacity =
    0x2000; // DspInterface::fifo, audio_core/dsp_interface.h
static constexpr std::size_t kPopChunkFrames = 2048; // DspInterface::kPopChunkFrames, same file
static constexpr int kDspFrameSize = 160; // AudioCore::samples_per_frame, audio_core/audio_types.h

// smoothing 0.0 gives white noise, a flat spectrum whose block-to-block energy is stationary, so
// any periodic modulation the search introduces or removes is not confounded with envelope
// wander already present in the source. Above 0.0 gives colored noise that cannot be predicted
// far ahead but is still smooth from one sample to the next - the worst case for the similarity
// search's CPU cost, since it can't be short-circuited the way silence or a pure tone can.
static std::vector<s16> GenerateNoisePool(int frames, unsigned seed, double smoothing) {
    std::vector<s16> pool(static_cast<std::size_t>(frames) * 2);
    srand(seed);
    double y = 0.0;
    for (int i = 0; i < frames; i++) {
        const double n = ((rand() / static_cast<double>(RAND_MAX)) * 2.0 - 1.0) * 9000.0;
        y = (smoothing * y) + ((1.0 - smoothing) * n);
        const s16 q = static_cast<s16>(std::lround(y));
        pool[(i * 2) + 0] = q;
        pool[(i * 2) + 1] = q;
    }
    return pool;
}

// ---------------------------------------------------------------- pipeline --

struct PipeResult {
    long long callbacks, underruns, silent;
    double meanFill, realFrac, survived;
};

// Models the two buffering stages between the DSP HLE and the audio callback: DspInterface::fifo,
// filled by OutputFrame() once per DSP tick, and drained into the stretcher's own ring by
// DrainFifoIntoWsola() once per callback - the only drain mode azahar has, since OutputFrame()
// always runs on the emu thread and there is no second thread to drain it early.
static PipeResult Pipeline(double speed, int audio_buf_size, double seconds) {
    AudioCore::WsolaStretcher ts;
    ts.Reset();
    Common::RingBuffer<s16, kFifoCapacity, 2> fifo;

    const int len = audio_buf_size;
    const double callback_period = len / kFs;
    const double dsp_tick_period = kDspFrameSize / (kFs * speed);

    std::vector<s16> tick(static_cast<std::size_t>(kDspFrameSize) * 2);
    std::vector<s16> pop(kPopChunkFrames * 2);
    std::vector<s16> out(static_cast<std::size_t>(len) * 2);
    for (int i = 0; i < kDspFrameSize; i++) {
        tick[(i * 2) + 0] = static_cast<s16>(i);
        tick[(i * 2) + 1] = static_cast<s16>(i);
    }

    long long total_made = 0, real_frames = 0;
    double fill_sum = 0.0;
    double arrival_avg = len * speed;
    s64 last_written = 0;
    double t_emu = 0.0, t_cb = 0.0;
    PipeResult r{0, 0, 0, 0.0, 0.0, 0.0};

    auto drain = [&] {
        while (true) {
            const std::size_t popped = fifo.Pop(pop.data(), kPopChunkFrames);
            if (popped == 0) {
                break;
            }
            if (ts.Write(pop.data(), static_cast<int>(popped)) < static_cast<int>(popped)) {
                // The stretcher's own ring saturated: resync rather than splice across the gap,
                // matching DspInterface::DrainFifoIntoWsola().
                ts.Resync();
                break;
            }
        }
    };

    while (t_cb < seconds) {
        while (t_emu + dsp_tick_period <= t_cb + callback_period) {
            t_emu += dsp_tick_period;
            fifo.Push(tick.data(), static_cast<std::size_t>(kDspFrameSize));
            total_made += kDspFrameSize;
        }

        t_cb += callback_period;
        drain();

        const s64 written = ts.TotalWritten();
        const s64 delta = std::max<s64>(0, written - last_written);
        last_written = written;
        arrival_avg += (delta - arrival_avg) * 0.05;

        const int fill = ts.InputFill();
        const double ratio =
            AudioCore::SpeedupStretchRatio(arrival_avg, static_cast<std::size_t>(len), fill,
                                           AudioCore::WsolaStretcher::TargetInputFill(arrival_avg));
        const int n = ts.Read(out.data(), len, ratio);

        r.callbacks++;
        if (n < len) {
            r.underruns++;
        }
        if (n < 1) {
            r.silent++;
        }
        fill_sum += fill;
        real_frames += n;
    }

    r.meanFill = fill_sum / static_cast<double>(r.callbacks);
    r.realFrac = static_cast<double>(real_frames) / static_cast<double>(r.callbacks * len);
    r.survived = total_made ? (static_cast<double>(ts.TotalWritten()) / total_made) : 1.0;
    return r;
}

static void RunPipeline() {
    printf("== pipeline ==\n");
    printf("output%% is how much of each buffer was real audio rather than padding.\n"
           "input%% is how much of what the DSP produced survived the FIFO and the stretcher's "
           "own ring.\n\n");

    // 512 is SDL2Sink's fixed callback size (audio_core/sdl2_sink.cpp); CubebSink asks for
    // std::max(512u, minimum_latency) (audio_core/cubeb_sink.cpp), so 512 is a floor there rather
    // than a fixed size. 1024 stands in for a device that negotiated a larger latency.
    // AudioCore::kUnlimitedSpeed is what SpeedupSpeedFromFrameLimit() reports for an unlimited
    // frame limiter; Settings::values.frame_limit otherwise maxes out at 1000% (10x).
    const double speeds[] = {0.5, 1.5, 2.0, 3.0, 4.0, 8.0, AudioCore::kUnlimitedSpeed};
    for (int buf_size : {512, 1024}) {
        printf("audioBufSize %4d:\n", buf_size);
        for (double s : speeds) {
            PipeResult r = Pipeline(s, buf_size, 8.0);
            printf("  %5.1fx  underruns %5.1f%%  output %5.1f%%  input %5.1f%%\n", s,
                   100.0 * static_cast<double>(r.underruns) / static_cast<double>(r.callbacks),
                   100.0 * r.realFrac, 100.0 * r.survived);
        }
        printf("\n");
    }
}

// -------------------------------------------------------------------- cost --

static volatile long long g_sink = 0;

static void RunCost() {
    printf("== cost ==\n");
    printf("Noise input, the worst case for the similarity search.\n\n");

    const int len = 256;
    const double seconds = 15.0;
    const int callbacks = static_cast<int>(seconds * kFs / len);
    const int pool_frames = 1 << 20;

    const std::vector<s16> pool = GenerateNoisePool(pool_frames, 7, 0.6);

    for (double ratio : {1.0, 2.0, 3.0, 4.0, 8.0}) {
        AudioCore::WsolaStretcher ts;
        ts.Reset();
        AudioCore::SpeedupLowPass lp;
        lp.Init(kFs);
        std::vector<s16> out(len * 2);
        double carry = 0.0;
        int pos = 0;

        const auto t0 = std::chrono::steady_clock::now();
        for (int c = 0; c < callbacks; c++) {
            carry += len * ratio;
            int got = static_cast<int>(carry);
            carry -= got;
            if (got > static_cast<int>(kPopChunkFrames)) {
                got = static_cast<int>(kPopChunkFrames);
            }
            if (pos + got > pool_frames) {
                pos = 0;
            }
            ts.Write(&pool[static_cast<std::size_t>(pos) * 2], got);
            pos += got;
            const int n = ts.Read(out.data(), len, ratio);
            lp.Process(out.data(), static_cast<std::size_t>(len), 6000.0, len / kFs);
            g_sink += n + out[0];
        }
        const auto t1 = std::chrono::steady_clock::now();

        const double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
        printf("  ratio %4.1f : %6.1f ms per %.0f s audio = %.2f%% of one core\n", ratio, ms,
               seconds, 100.0 * (ms / 1000.0) / seconds);
    }
    printf("\n");
}

// --------------------------------------------------------- search activity --

// Three tones, not one: a stretcher that splices at an arbitrary phase still emits something
// close to a pure sine, since a single frequency has no relative phase between partials to get
// wrong. A chord is where a bad splice shows up as energy at frequencies the input never had.
static std::vector<s16> GenerateChord(int frames) {
    std::vector<s16> v(static_cast<std::size_t>(frames) * 2);
    for (int i = 0; i < frames; i++) {
        const double t = i / kFs;
        const double x = 4000.0 * std::sin(2.0 * std::numbers::pi * 220.0 * t) +
                         4000.0 * std::sin(2.0 * std::numbers::pi * 329.63 * t) +
                         4000.0 * std::sin(2.0 * std::numbers::pi * 440.0 * t);
        const s16 q = static_cast<s16>(std::lround(x));
        v[(i * 2) + 0] = q;
        v[(i * 2) + 1] = q;
    }
    return v;
}

// MAINTENANCE: SynthesiseHopSearchOff() and ClampBounds() below are hand-maintained mirrors of
// AudioCore::WsolaStretcher::SynthesiseHop() and AudioCore::WsolaStretcher::FindBestOffset()
// (audio_core/wsola_stretcher.cpp). They are faithful today, but nothing enforces that: if either
// real method changes and these are not updated to match, this tool will keep printing a
// confident "search off" baseline and pinned-bound check that no longer reflect shipped behavior.
//
// Reimplements SynthesiseHop() (audio_core/wsola_stretcher.cpp) with the search bypassed: chosen
// is always the nominal position, never FindBestOffset()'s pick. The shipped class has no public
// way to ask for this - the search isn't optional in production - so this exists purely to
// measure what forcing it off sounds like. It changes no shipped behavior: WsolaStretcher's own
// Read()/SynthesiseHop() path is untouched and never calls this function.
static void SynthesiseHopSearchOff(AudioCore::WsolaStretcher& w, double ratio) {
    using AZ = AudioCore::WsolaStretcher;
    int hop = static_cast<int>(std::lround(AZ::kSynthesisHop * ratio));
    if (hop < 1) {
        hop = 1;
    }

    const s64 chosen = w.analysis_pos;
    w.primed = true;

    for (int i = 0; i < AZ::kFrameSize; i++) {
        const int idx = static_cast<int>((chosen + i) & (AZ::kInputCapacity - 1));
        const float wgt = w.window[i];
        w.acc_l[i] += wgt * static_cast<float>(w.in_l[idx]);
        w.acc_r[i] += wgt * static_cast<float>(w.in_r[idx]);
    }
    for (int i = 0; i < AZ::kSynthesisHop; i++) {
        const int idx = static_cast<int>(w.out_write_pos & (AZ::kOutputCapacity - 1));
        w.out_l[idx] = AZ::Saturate(w.acc_l[i]);
        w.out_r[idx] = AZ::Saturate(w.acc_r[i]);
        w.out_write_pos++;
    }
    std::memmove(w.acc_l, w.acc_l + AZ::kSynthesisHop, AZ::kSynthesisHop * sizeof(float));
    std::memmove(w.acc_r, w.acc_r + AZ::kSynthesisHop, AZ::kSynthesisHop * sizeof(float));
    std::memset(w.acc_l + AZ::kSynthesisHop, 0, AZ::kSynthesisHop * sizeof(float));
    std::memset(w.acc_r + AZ::kSynthesisHop, 0, AZ::kSynthesisHop * sizeof(float));

    w.natural_pos = chosen + AZ::kSynthesisHop;
    w.analysis_pos += hop;
}

// Mirrors FindBestOffset()'s clamp math (audio_core/wsola_stretcher.cpp) so the pinned-to-bound
// check in ReportBestK() does not need to call it a second time.
static void ClampBounds(const AudioCore::WsolaStretcher& w, s64 analysis_pos, int* lowest_k,
                        int* highest_k) {
    using AZ = AudioCore::WsolaStretcher;
    const s64 oldest = std::max<s64>(0, w.write_pos - AZ::kInputCapacity);
    int lo = -AZ::kSearchRadius;
    if ((analysis_pos + lo) < oldest) {
        lo = static_cast<int>(oldest - analysis_pos);
    }
    int hi = AZ::kSearchRadius;
    const s64 latest = w.write_pos - AZ::kFrameSize;
    if ((analysis_pos + hi) > latest) {
        hi = static_cast<int>(latest - analysis_pos);
    }
    if (hi < lo) {
        hi = lo;
    }
    *lowest_k = lo;
    *highest_k = hi;
}

struct HopStat {
    int best_k;
    int lowest_k;
    int highest_k;
};

struct ActivityResult {
    std::vector<HopStat> hops;
    std::vector<double> out_mono; // steady-state output, one sample per frame (L+R averaged)
};

// Drives the real Write()/Read() path - search compiled in, nothing bypassed - and recovers each
// measured hop's best_k after the fact: SynthesiseHop() sets natural_pos to chosen +
// kSynthesisHop, so reading analysis_pos before the call and natural_pos after reproduces
// FindBestOffset()'s answer for that hop without calling it separately.
static ActivityResult DriveOn(const std::vector<s16>& sig, double ratio, int warmup_hops,
                              int measure_hops) {
    using AZ = AudioCore::WsolaStretcher;
    constexpr int kTargetFill = 8192;

    AZ w;
    w.Reset();
    ActivityResult result;
    std::size_t sp = 0;
    const std::size_t sig_frames = sig.size() / 2;

    auto top_up = [&] {
        while (w.InputFill() < kTargetFill) {
            int want = kTargetFill - w.InputFill();
            if ((sp + static_cast<std::size_t>(want)) > sig_frames) {
                printf("  WARNING: search-activity signal pool wrapped; lengthen it\n");
                sp = 0;
            }
            const int got = w.Write(&sig[sp * 2], want);
            if (got <= 0) {
                break;
            }
            sp += static_cast<std::size_t>(got);
        }
    };
    top_up();
    w.Resync();
    top_up();

    std::vector<s16> buf(AZ::kSynthesisHop * 2);
    for (int h = 0; h < (warmup_hops + measure_hops); h++) {
        top_up();
        if (!w.CanSynthesise()) {
            break;
        }
        const s64 ap_before = w.analysis_pos;
        const bool measure = (h >= warmup_hops) && w.primed;
        int lowest_k = 0, highest_k = 0;
        if (measure) {
            ClampBounds(w, ap_before, &lowest_k, &highest_k);
        }
        const int n = w.Read(buf.data(), AZ::kSynthesisHop, ratio);
        if (measure && (n == AZ::kSynthesisHop)) {
            const s64 chosen = w.natural_pos - AZ::kSynthesisHop;
            result.hops.push_back(
                HopStat{static_cast<int>(chosen - ap_before), lowest_k, highest_k});
        }
        if (h >= warmup_hops) {
            for (int i = 0; i < n; i++) {
                result.out_mono.push_back(0.5 * (buf[(i * 2) + 0] + buf[(i * 2) + 1]));
            }
        }
    }
    return result;
}

// Same drive loop as DriveOn(), but each hop goes through SynthesiseHopSearchOff() instead of the
// real Read()-driven SynthesiseHop(); Read() is still called afterward to drain the hop that
// produced through the normal out_read_pos/out_write_pos bookkeeping, so this stays a thin layer
// over the shipped ring rather than a second output path.
static std::vector<double> DriveOff(const std::vector<s16>& sig, double ratio, int warmup_hops,
                                    int measure_hops) {
    using AZ = AudioCore::WsolaStretcher;
    constexpr int kTargetFill = 8192;

    AZ w;
    w.Reset();
    std::vector<double> out_mono;
    std::size_t sp = 0;
    const std::size_t sig_frames = sig.size() / 2;

    auto top_up = [&] {
        while (w.InputFill() < kTargetFill) {
            int want = kTargetFill - w.InputFill();
            if ((sp + static_cast<std::size_t>(want)) > sig_frames) {
                printf("  WARNING: search-activity signal pool wrapped; lengthen it\n");
                sp = 0;
            }
            const int got = w.Write(&sig[sp * 2], want);
            if (got <= 0) {
                break;
            }
            sp += static_cast<std::size_t>(got);
        }
    };
    top_up();
    w.Resync();
    top_up();

    std::vector<s16> buf(AZ::kSynthesisHop * 2);
    for (int h = 0; h < (warmup_hops + measure_hops); h++) {
        top_up();
        if (!w.CanSynthesise()) {
            break;
        }
        SynthesiseHopSearchOff(w, ratio);
        const int n = w.Read(buf.data(), AZ::kSynthesisHop, ratio);
        if (h >= warmup_hops) {
            for (int i = 0; i < n; i++) {
                out_mono.push_back(0.5 * (buf[(i * 2) + 0] + buf[(i * 2) + 1]));
            }
        }
    }
    return out_mono;
}

static void Fft(std::vector<std::complex<double>>& a) {
    const std::size_t n = a.size();
    for (std::size_t i = 1, j = 0; i < n; i++) {
        std::size_t bit = n >> 1;
        for (; j & bit; bit >>= 1) {
            j ^= bit;
        }
        j ^= bit;
        if (i < j) {
            std::swap(a[i], a[j]);
        }
    }
    for (std::size_t len = 2; len <= n; len <<= 1) {
        const double ang = -2.0 * std::numbers::pi / static_cast<double>(len);
        const std::complex<double> wl(std::cos(ang), std::sin(ang));
        for (std::size_t i = 0; i < n; i += len) {
            std::complex<double> wk(1.0, 0.0);
            for (std::size_t k = 0; k < (len / 2); k++) {
                const std::complex<double> u = a[i + k];
                const std::complex<double> v = a[i + k + (len / 2)] * wk;
                a[i + k] = u + v;
                a[i + k + (len / 2)] = u - v;
                wk *= wl;
            }
        }
    }
}

// Fraction of output energy landing outside +-8 FFT bins of every tone in `tones`. 0.0 = only the
// input's own partials came out; large = splices injected energy the input never had. Fixed FFT
// length regardless of how much output is passed in, so the figure cannot depend on the analysis
// length - a sensitivity that made the deleted RunPumping() unreliable.
static double OutOfBandFraction(const std::vector<double>& x, const std::vector<double>& tones) {
    std::size_t n = 1;
    while ((n * 2) <= x.size()) {
        n *= 2;
    }
    if (n < 4096) {
        return -1.0;
    }
    std::vector<std::complex<double>> a(n);
    for (std::size_t i = 0; i < n; i++) {
        const double win = 0.5 * (1.0 - std::cos(2.0 * std::numbers::pi * static_cast<double>(i) /
                                                 static_cast<double>(n)));
        a[i] = std::complex<double>(x[i] * win, 0.0);
    }
    Fft(a);

    const double bin_hz = kFs / static_cast<double>(n);
    const int half = 8; // +-8 bins; Hann's main lobe is +-2, so this is generous
    double total = 0.0, inband = 0.0;
    for (std::size_t k = 0; k < (n / 2); k++) {
        const double p = std::norm(a[k]);
        total += p;
        for (double f0 : tones) {
            const int c = static_cast<int>(std::lround(f0 / bin_hz));
            if ((static_cast<int>(k) >= (c - half)) && (static_cast<int>(k) <= (c + half))) {
                inband += p;
                break;
            }
        }
    }
    if (total <= 0.0) {
        return -1.0;
    }
    return (total - inband) / total;
}

static void ReportBestK(const std::vector<HopStat>& hops) {
    if (hops.empty()) {
        printf("  best_k: no search hops recorded\n");
        return;
    }
    std::map<int, int> distinct;
    int at_zero = 0, at_bound = 0;
    for (const auto& h : hops) {
        distinct[h.best_k]++;
        if (h.best_k == 0) {
            at_zero++;
        }
        if ((h.best_k == h.lowest_k) || (h.best_k == h.highest_k)) {
            at_bound++;
        }
    }
    const int n = static_cast<int>(hops.size());
    printf("  best_k: %d hops, %zu distinct offsets, k==0 on %.1f%%, pinned to a clamp bound on "
           "%.1f%%\n",
           n, distinct.size(), 100.0 * at_zero / n, 100.0 * at_bound / n);
}

static void RunSearchActivity() {
    printf("== search activity ==\n");
    printf("What FindBestOffset() (audio_core/wsola_stretcher.cpp) actually picks, on a "
           "three-tone chord at ratio 3. \"search on\" drives the real Write()/Read() path; "
           "\"search off\" replicates SynthesiseHop() with FindBestOffset() skipped, since the "
           "shipped class has no way to disable it.\n\n");

    const double ratio = 3.0;
    const int warmup_hops = 200;
    const int measure_hops = 1200;
    const int pool_frames = 1 << 20;

    const std::vector<s16> chord = GenerateChord(pool_frames);

    const ActivityResult on = DriveOn(chord, ratio, warmup_hops, measure_hops);
    ReportBestK(on.hops);

    const std::vector<double> off = DriveOff(chord, ratio, warmup_hops, measure_hops);

    const std::vector<double> tones = {220.0, 329.63, 440.0};
    const double e_on = OutOfBandFraction(on.out_mono, tones);
    const double e_off = OutOfBandFraction(off, tones);
    if ((e_on >= 0.0) && (e_off >= 0.0)) {
        printf("  off-partial energy: search on %7.4f%% (%6.1f dB)   search off %7.4f%% "
               "(%6.1f dB)\n",
               100.0 * e_on, 10.0 * std::log10(e_on + 1e-15), 100.0 * e_off,
               10.0 * std::log10(e_off + 1e-15));
    } else {
        printf("  off-partial energy: insufficient output for the spectral check\n");
    }
    printf("\n");
}

// ------------------------------------------------------------------ stress --

static Common::RingBuffer<s16, kFifoCapacity, 2> g_fifo;
static std::atomic<bool> g_stop{false};
static std::atomic<long long> g_written{0}, g_read{0}, g_refused{0};
static std::atomic<long long> g_torn{0}, g_out_of_range{0};
static constexpr int kAmp = 9000;

// Per-thread state: rand() is neither reentrant nor reproducible when two threads share it, and a
// stress test that cannot be replayed is worth less.
static u32 NextRand(u32& st) {
    st = (st * 1664525u) + 1013904223u;
    return st >> 16;
}

static void Producer() {
    std::vector<s16> buf(static_cast<std::size_t>(kDspFrameSize) * 2);
    u32 rng = 0xA5A5A5A5u;
    long long t = 0;
    while (!g_stop.load(std::memory_order_relaxed)) {
        // A burst of DSP ticks per iteration, mimicking bursty scheduling on the emu thread
        // rather than one fixed-size push at a fixed rate.
        const int ticks = 1 + static_cast<int>(NextRand(rng) % 8);
        for (int k = 0; k < ticks; k++) {
            for (int i = 0; i < kDspFrameSize; i++) {
                const double v = kAmp * std::sin(2.0 * std::numbers::pi * 440.0 * ((t + i) / kFs));
                const s16 s = static_cast<s16>(std::lround(v));
                buf[(i * 2) + 0] = s;
                buf[(i * 2) + 1] = s; // L == R, always
            }
            t += kDspFrameSize;
            const std::size_t pushed =
                g_fifo.Push(buf.data(), static_cast<std::size_t>(kDspFrameSize));
            g_written.fetch_add(static_cast<long long>(pushed), std::memory_order_relaxed);
            if (pushed < static_cast<std::size_t>(kDspFrameSize)) {
                g_refused.fetch_add(kDspFrameSize - static_cast<long long>(pushed),
                                    std::memory_order_relaxed);
            }
        }
        std::this_thread::yield();
    }
}

static void Consumer() {
    const int len = 256;
    u32 rng = 0x5A5A5A5Au;
    std::vector<s16> pop(kPopChunkFrames * 2);
    std::vector<s16> out(len * 2);
    AudioCore::WsolaStretcher ts;
    ts.Reset();
    while (!g_stop.load(std::memory_order_relaxed)) {
        while (true) {
            const std::size_t popped = g_fifo.Pop(pop.data(), kPopChunkFrames);
            if (popped == 0) {
                break;
            }
            g_read.fetch_add(static_cast<long long>(popped), std::memory_order_relaxed);
            for (std::size_t i = 0; i < popped; i++) {
                // Input frames all have L == R, and Push()/Pop() move whole stereo slots, so any
                // popped frame where they differ means this read overlapped a concurrent write.
                if (pop[(i * 2) + 0] != pop[(i * 2) + 1]) {
                    g_torn.fetch_add(1, std::memory_order_relaxed);
                }
                if (std::abs(static_cast<int>(pop[i * 2])) > kAmp + 2000) {
                    g_out_of_range.fetch_add(1, std::memory_order_relaxed);
                }
            }
            if (ts.Write(pop.data(), static_cast<int>(popped)) < static_cast<int>(popped)) {
                ts.Resync();
            }
        }
        const double ratio = 0.5 + (static_cast<double>(NextRand(rng) % 700) / 100.0);
        ts.Read(out.data(), len, ratio);
        std::this_thread::yield();
    }
}

static int RunStress(int seconds) {
    printf("== stress ==\n");

    std::thread p(Producer), c(Consumer);
    std::this_thread::sleep_for(std::chrono::seconds(seconds));
    g_stop.store(true, std::memory_order_relaxed);
    p.join();
    c.join();

    printf("  %ds: written %lld, read %lld, refused %lld\n", seconds, g_written.load(),
           g_read.load(), g_refused.load());
    printf("  torn frames (L != R): %lld\n", g_torn.load());
    printf("  out-of-range samples: %lld\n", g_out_of_range.load());

    const bool ok = (g_torn.load() == 0) && (g_out_of_range.load() == 0) &&
                    (g_written.load() > 0) && (g_read.load() > 0);
    printf("  %s\n\n", ok ? "PASS" : "FAIL");
    return ok ? 0 : 1;
}

int main(int argc, char** argv) {
    const int stress_seconds = (argc > 1) ? atoi(argv[1]) : 5;
    RunPipeline();
    RunCost();
    RunSearchActivity();
    const int rc = RunStress(stress_seconds);
    if (g_sink == 0x7fffffff) {
        printf("unreachable\n");
    }
    return rc;
}
