// Copyright 2026 Citra Emulator Project / Azahar Emulator Project
// Licensed under GPLv3 or any later version
// Refer to the LICENSE.txt file included.

#include "common/arch.h"
#include "common/thread.h"
#include "video_core/pica/vertex_worker_pool.h"

#if CITRA_ARCH(x86_64)
#include <immintrin.h>
#elif CITRA_ARCH(arm64) && defined(_MSC_VER)
#include <intrin.h>
#endif
#ifdef __linux__
#include <sys/resource.h>
#include <unistd.h>
#endif

namespace Pica {

namespace {
void CpuRelax() {
#if CITRA_ARCH(x86_64)
    _mm_pause();
#elif CITRA_ARCH(arm64) && defined(_MSC_VER)
    __yield();
#elif CITRA_ARCH(arm64)
    asm volatile("yield");
#endif
}

#if CITRA_ARCH(x86_64)
/// MXCSR exception flags; the other bits are control bits.
constexpr u32 MxcsrFlags = 0x3F;
#endif

/// The calling thread's FP control word: FPCR on arm64, the MXCSR control bits on x86-64, 0 on
/// anything else (see VertexWorkerFpControlSupported in vertex_worker_pool.h).
u64 GetFpControl() {
#if CITRA_ARCH(x86_64)
    return _mm_getcsr() & ~MxcsrFlags;
#elif CITRA_ARCH(arm64) && defined(_MSC_VER)
    return static_cast<u64>(_ReadStatusReg(ARM64_FPCR));
#elif CITRA_ARCH(arm64)
    u64 fpcr;
    asm volatile("mrs %0, fpcr" : "=r"(fpcr));
    return fpcr;
#else
    return 0;
#endif
}

/// Sets the calling thread's FP control word, unless it already has that value.
void SetFpControl(u64 control) {
    if (GetFpControl() == control) {
        return;
    }
#if CITRA_ARCH(x86_64)
    _mm_setcsr((_mm_getcsr() & MxcsrFlags) | static_cast<u32>(control));
#elif CITRA_ARCH(arm64) && defined(_MSC_VER)
    _WriteStatusReg(ARM64_FPCR, static_cast<__int64>(control));
#elif CITRA_ARCH(arm64)
    asm volatile("msr fpcr, %0" : : "r"(control) : "memory");
#endif
}
} // Anonymous namespace

VertexWorkerPool::VertexWorkerPool(u32 num_workers, int nice_value, std::chrono::microseconds spin)
    : spin_time{spin} {
    workers.reserve(num_workers);
    for (u32 i = 0; i < num_workers; ++i) {
        workers.emplace_back([this, nice_value] { WorkerLoop(nice_value); });
    }
}

VertexWorkerPool::~VertexWorkerPool() {
    {
        std::scoped_lock lock{mutex};
        stop.store(true, std::memory_order_relaxed);
    }
    cv.notify_all();
    for (std::thread& worker : workers) {
        worker.join();
    }
}

u64 VertexWorkerPool::Run(u32 num_chunks, const std::function<void(u32 chunk)>& fn) {
    if (num_chunks == 0) {
        return 0;
    }
    const u32 job = ++last_job;
    const u64 tag = u64{job} << 32;
    job_fn = &fn;
    job_fp_control = GetFpControl();
    job_size.store(tag | num_chunks, std::memory_order_relaxed);
    done.store(0, std::memory_order_relaxed);
    claim.store(tag, std::memory_order_release);
    if (!workers.empty()) {
        {
            // Under the lock, so a worker can't miss the job between its check and its wait.
            std::scoped_lock lock{mutex};
            posted_job.store(job, std::memory_order_release);
        }
        cv.notify_all();
    }

    Work(job, false);
    if (done.load(std::memory_order_acquire) == num_chunks) {
        return 0;
    }
    const auto start = std::chrono::steady_clock::now();
    while (done.load(std::memory_order_acquire) != num_chunks) {
        CpuRelax();
    }
    const auto waited = std::chrono::steady_clock::now() - start;
    return static_cast<u64>(std::chrono::duration_cast<std::chrono::nanoseconds>(waited).count());
}

void VertexWorkerPool::Work(u32 job, bool adopt_fp_control) {
    u64 current = claim.load(std::memory_order_acquire);
    for (;;) {
        if (static_cast<u32>(current >> 32) != job) {
            return;
        }
        // Read after the claim word. The tag check stops a late worker from pairing its job's
        // final claim with the size of the job the caller is posting next.
        const u64 size = job_size.load(std::memory_order_relaxed);
        if (static_cast<u32>(size >> 32) != job) {
            return;
        }
        const u32 chunk = static_cast<u32>(current);
        if (chunk >= static_cast<u32>(size)) {
            return;
        }
        // Success means the chunk was unclaimed, so the job is still running and fn is alive.
        // Failure reloads current.
        if (claim.compare_exchange_weak(current, current + 1, std::memory_order_acquire,
                                        std::memory_order_acquire)) {
            if (adopt_fp_control) {
                // Read like job_fn: after a successful claim, so it is this job's.
                SetFpControl(job_fp_control);
                adopt_fp_control = false;
            }
            (*job_fn)(chunk);
            done.fetch_add(1, std::memory_order_release);
            current = claim.load(std::memory_order_acquire);
        }
    }
}

void VertexWorkerPool::WorkerLoop(int nice_value) {
    Common::SetCurrentThreadName("VertexWorker");
#ifdef __linux__
    if (nice_value != 0) {
        setpriority(PRIO_PROCESS, static_cast<id_t>(gettid()), nice_value);
    }
#endif
    u32 seen = 0;
    for (;;) {
        u32 job = posted_job.load(std::memory_order_acquire);
        if (job == seen) {
            const auto deadline = std::chrono::steady_clock::now() + spin_time;
            while (job == seen && !stop.load(std::memory_order_relaxed) &&
                   std::chrono::steady_clock::now() < deadline) {
                CpuRelax();
                job = posted_job.load(std::memory_order_acquire);
            }
        }
        if (job == seen) {
            std::unique_lock lock{mutex};
            cv.wait(lock, [&] {
                job = posted_job.load(std::memory_order_acquire);
                return job != seen || stop.load(std::memory_order_relaxed);
            });
        }
        if (stop.load(std::memory_order_relaxed)) {
            return;
        }
        seen = job;
        Work(job, true);
    }
}

} // namespace Pica
