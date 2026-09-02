// Copyright 2026 Citra Emulator Project / Azahar Emulator Project
// Licensed under GPLv2 or any later version
// Refer to the license.txt file included.

#include <algorithm>
#include <cmath>
#include <cstring>
#include <numbers>
#include "audio_core/wsola_stretcher.h"

namespace AudioCore {

int WsolaStretcher::TargetInputFill(double arrival_per_callback) {
    // Half the ring. Above a ratio of about 9 the oldest frame Write() must preserve is
    // governed by natural_pos rather than the search window, so the usable span grows with the
    // hop; leaving half the ring free keeps that from squeezing Write() down to nothing.
    const double cap = kInputCapacity / 2;

    double need = (2.0 * arrival_per_callback) + kSearchRadius + kFrameSize;
    // Clamped as a double: a near-zero target speed seeds an arrival estimate in the billions
    // (0.0001 gives around 2.5e9), and casting that to int is undefined.
    if (!(need > kMinTargetInputFill)) {
        need = kMinTargetInputFill;
    }
    if (need > cap) {
        need = cap;
    }
    return static_cast<int>(need);
}

WsolaStretcher::WsolaStretcher() {
    for (int i = 0; i < kFrameSize; i++) {
        window[i] =
            static_cast<float>(0.5 * (1.0 - std::cos((2.0 * std::numbers::pi * i) / kFrameSize)));
    }
    Reset();
}

void WsolaStretcher::Reset() {
    write_pos = 0;
    analysis_pos = 0;
    natural_pos = 0;
    session_start = 0;
    out_read_pos = 0;
    out_write_pos = 0;
    primed = false;
    std::memset(acc_l, 0, sizeof(acc_l));
    std::memset(acc_r, 0, sizeof(acc_r));

    // Belt and braces: the search is already bounded to written frames, but this makes any
    // future bound slip degrade to silence rather than noise.
    std::memset(in_l, 0, sizeof(in_l));
    std::memset(in_r, 0, sizeof(in_r));
    std::memset(in_mono, 0, sizeof(in_mono));
}

void WsolaStretcher::Resync() {
    // Cheap by design - the audio callback calls this whenever input is lost, so unlike Reset()
    // it leaves the input ring alone. Synthesized frames stay too: they are complete, and
    // dropping them only moves the discontinuity earlier. The accumulator goes, since it holds
    // the previous frame's falling half and splicing that onto post-gap input is the artifact
    // this exists to avoid; primed stays false so the next hop fades in.
    analysis_pos = std::max<s64>(0, write_pos - kFrameSize);
    natural_pos = analysis_pos;
    // Input either side of the gap is not contiguous, so the search must not span it.
    session_start = analysis_pos;
    primed = false;
    std::memset(acc_l, 0, sizeof(acc_l));
    std::memset(acc_r, 0, sizeof(acc_r));
}

void WsolaStretcher::BeginSession() {
    Resync();
    // Both rings still hold the previous run, which can be arbitrarily old. Start at write_pos
    // rather than Resync()'s write_pos - kFrameSize, so CanSynthesise() waits for genuinely new
    // input instead of splicing the last run into this one.
    analysis_pos = write_pos;
    natural_pos = write_pos;
    session_start = write_pos;
    out_read_pos = 0;
    out_write_pos = 0;
}

int WsolaStretcher::InputFill() const {
    const s64 pending = write_pos - analysis_pos;
    if (pending < 0) {
        return 0;
    }
    if (pending > kInputCapacity) {
        return kInputCapacity;
    }
    return static_cast<int>(pending);
}

int WsolaStretcher::OutputFill() const {
    return static_cast<int>(out_write_pos - out_read_pos);
}

s64 WsolaStretcher::TotalWritten() const {
    return write_pos;
}

int WsolaStretcher::Write(const s16* samples, int num_frames) {
    // Refuse rather than lap the read position: overwriting frames the search may still
    // reference would splice noise into the output. natural_pos has to be in the bound as well
    // as the search floor: it trails the nominal position, and at a large hop it sits well
    // below analysis_pos - kSearchRadius.
    const s64 oldest_needed = std::min(analysis_pos - kSearchRadius, natural_pos);
    s64 space = kInputCapacity - (write_pos - std::max<s64>(0, oldest_needed));
    if (space < 0) {
        space = 0;
    }
    if (num_frames > space) {
        num_frames = static_cast<int>(space);
    }
    if (num_frames <= 0) {
        return 0;
    }

    for (int i = 0; i < num_frames; i++) {
        const int idx = static_cast<int>((write_pos + i) & (kInputCapacity - 1));
        const s16 l = samples[(i * 2) + 0];
        const s16 r = samples[(i * 2) + 1];
        in_l[idx] = l;
        in_r[idx] = r;
        in_mono[idx] = 0.5f * (static_cast<float>(l) + static_cast<float>(r));
    }

    write_pos += num_frames;
    return num_frames;
}

int WsolaStretcher::Read(s16* samples, int num_frames, double ratio) {
    while ((OutputFill() < num_frames) && CanSynthesise()) {
        SynthesiseHop(ratio);
    }

    const int n = std::min(num_frames, OutputFill());
    for (int i = 0; i < n; i++) {
        const int idx = static_cast<int>(out_read_pos & (kOutputCapacity - 1));
        samples[(i * 2) + 0] = out_l[idx];
        samples[(i * 2) + 1] = out_r[idx];
        out_read_pos++;
    }
    return n;
}

s16 WsolaStretcher::Saturate(float v) {
    long s = std::lround(v);
    if (s > 32767) {
        s = 32767;
    }
    if (s < -32768) {
        s = -32768;
    }
    return static_cast<s16>(s);
}

bool WsolaStretcher::CanSynthesise() const {
    if ((kOutputCapacity - OutputFill()) < kSynthesisHop) {
        return false;
    }

    // Only the frame itself and the natural-continuation reference need to be present; the
    // search clamps to whatever else is available. Demanding the full radius here would emit no
    // output at all on a short FIFO.
    const s64 frame_end = analysis_pos + kFrameSize;
    const s64 natural_end = natural_pos + kSynthesisHop;
    return (write_pos >= frame_end) && (write_pos >= natural_end);
}

void WsolaStretcher::SynthesiseHop(double ratio) {
    int hop = static_cast<int>(std::lround(kSynthesisHop * ratio));
    if (hop < 1) {
        hop = 1;
    }

    const s64 chosen = primed ? FindBestOffset() : analysis_pos;
    primed = true;

    for (int i = 0; i < kFrameSize; i++) {
        const int idx = static_cast<int>((chosen + i) & (kInputCapacity - 1));
        const float w = window[i];
        acc_l[i] += w * static_cast<float>(in_l[idx]);
        acc_r[i] += w * static_cast<float>(in_r[idx]);
    }

    for (int i = 0; i < kSynthesisHop; i++) {
        const int idx = static_cast<int>(out_write_pos & (kOutputCapacity - 1));
        out_l[idx] = Saturate(acc_l[i]);
        out_r[idx] = Saturate(acc_r[i]);
        out_write_pos++;
    }

    std::memmove(acc_l, acc_l + kSynthesisHop, kSynthesisHop * sizeof(float));
    std::memmove(acc_r, acc_r + kSynthesisHop, kSynthesisHop * sizeof(float));
    std::memset(acc_l + kSynthesisHop, 0, kSynthesisHop * sizeof(float));
    std::memset(acc_r + kSynthesisHop, 0, kSynthesisHop * sizeof(float));

    // The nominal pointer advances by hop alone; the search only picks which frame to window.
    // Advancing from chosen instead would make consumption hop + E[best_k], which the caller's
    // ratio control cannot see.
    natural_pos = chosen + kSynthesisHop;
    analysis_pos += hop;
}

s64 WsolaStretcher::FindBestOffset() const {
    // Whichever is newer: what the ring still holds, or where this run began. The ring keeps the
    // previous run's frames until they are overwritten, and splicing those in replays old audio.
    const s64 oldest = std::max(std::max<s64>(0, write_pos - kInputCapacity), session_start);

    // natural_pos trails analysis_pos by up to hop + kSearchRadius - kSynthesisHop, which at a
    // high ratio on a near-full ring can fall off the back. The reference would then be
    // overwritten frames, so search nothing instead.
    if (natural_pos < oldest) {
        return analysis_pos;
    }

    const double ref_energy = Energy(natural_pos);

    // Full radius regardless of hop: expansion needs the reach, since the natural continuation
    // sits |kSynthesisHop - hop| ahead of the nominal.
    const int radius = kSearchRadius;

    // Masking a negative position wraps it into frames we never wrote.
    int lowest_k = -radius;
    if ((analysis_pos + lowest_k) < oldest) {
        lowest_k = static_cast<int>(oldest - analysis_pos);
    }

    // Clamp to what has arrived, so a short FIFO narrows the search.
    int highest_k = radius;
    const s64 latest = write_pos - kFrameSize;
    if ((analysis_pos + highest_k) > latest) {
        highest_k = static_cast<int>(latest - analysis_pos);
    }
    if (highest_k < lowest_k) {
        highest_k = lowest_k;
    }

    int best_k = std::min(std::max(lowest_k, 0), highest_k);
    double best_score = -1.0e30;

    for (int k = lowest_k; k <= highest_k; k += kCoarseStride) {
        const double s = Score(analysis_pos + k, ref_energy);
        if (s > best_score) {
            best_score = s;
            best_k = k;
        }
    }

    const int lo = std::max(lowest_k, best_k - kFineRadius);
    const int hi = std::min(highest_k, best_k + kFineRadius);
    for (int k = lo; k <= hi; k++) {
        const double s = Score(analysis_pos + k, ref_energy);
        if (s > best_score) {
            best_score = s;
            best_k = k;
        }
    }

    return analysis_pos + best_k;
}

double WsolaStretcher::Energy(s64 pos) const {
    double e = 0.0;
    for (int i = 0; i < kSynthesisHop; i++) {
        const double v = in_mono[static_cast<int>((pos + i) & (kInputCapacity - 1))];
        e += v * v;
    }
    return e;
}

// Normalized so the search doesn't just latch onto the loudest candidate.
double WsolaStretcher::Score(s64 pos, double ref_energy) const {
    double dot = 0.0;
    double energy = 0.0;
    for (int i = 0; i < kSynthesisHop; i++) {
        const double a = in_mono[static_cast<int>((pos + i) & (kInputCapacity - 1))];
        const double b = in_mono[static_cast<int>((natural_pos + i) & (kInputCapacity - 1))];
        dot += a * b;
        energy += a * a;
    }
    return dot / std::sqrt((energy * ref_energy) + 1.0e-9);
}

} // namespace AudioCore
