// Copyright 2026 Citra Emulator Project / Azahar Emulator Project
// Licensed under GPLv3 or any later version
// Refer to the LICENSE.txt file included.

#pragma once

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <functional>
#include <mutex>
#include <thread>
#include <vector>
#include "common/arch.h"
#include "common/common_types.h"

namespace Pica {

/// Whether the worker pool can copy the caller's floating-point control state: MXCSR on x86-64,
/// FPCR on arm64. Without it, workers could shade with other denormal/rounding modes.
inline constexpr bool VertexWorkerFpControlSupported =
#if CITRA_ARCH(x86_64) || CITRA_ARCH(arm64)
    true;
#else
    false;
#endif

/// Fork-join pool: Run() splits a job into chunks that the caller and the workers claim.
/// Between jobs the workers spin for a while, then park.
class VertexWorkerPool {
public:
    VertexWorkerPool(u32 num_workers, int nice_value, std::chrono::microseconds spin);
    ~VertexWorkerPool();

    VertexWorkerPool(const VertexWorkerPool&) = delete;
    VertexWorkerPool& operator=(const VertexWorkerPool&) = delete;

    /// Runs fn once for every chunk in [0, num_chunks) and returns when all have finished. Every
    /// fn call runs with the caller's floating-point control state as it was when Run was called.
    /// Returns the nanoseconds the caller waited for workers after it found no chunk left.
    u64 Run(u32 num_chunks, const std::function<void(u32 chunk)>& fn);

    u32 NumWorkers() const {
        return static_cast<u32>(workers.size());
    }

private:
    void WorkerLoop(int nice_value);

    /// Claims and runs chunks of the given job until none is left or the job is over. With
    /// adopt_fp_control (workers), switches to the job's FP control word before the first chunk.
    void Work(u32 job, bool adopt_fp_control);

    /// (job << 32) | next unclaimed chunk.
    alignas(64) std::atomic<u64> claim{0};
    /// (job << 32) | num_chunks, so a claim is never checked against another job's size.
    std::atomic<u64> job_size{0};
    const std::function<void(u32)>* job_fn = nullptr;
    /// The caller's FP control word. Inside guest callbacks on arm64 it is the guest's FPCR
    /// (A32AddressSpace::EmitPrelude(),
    /// externals/dynarmic/src/dynarmic/backend/arm64/a32_address_space.cpp).
    u64 job_fp_control = 0;

    alignas(64) std::atomic<u32> done{0};

    alignas(64) std::atomic<u32> posted_job{0};
    std::atomic<bool> stop{false};
    std::mutex mutex;
    std::condition_variable cv;

    u32 last_job = 0; ///< Caller only.
    std::chrono::microseconds spin_time;
    std::vector<std::thread> workers;
};

} // namespace Pica
