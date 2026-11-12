// Copyright 2026 Azahar Emulator Project
// Licensed under GPLv2 or any later version
// Refer to the misc/licenses/gplv2.txt file included.

#pragma once

#include <array>
#include <limits>
#include <mutex>
#include <string>
#include <vector>
#include "common/common_types.h"
#include "video_core/renderer_vulkan/vk_common.h"

namespace Vulkan {

class Instance;
class Scheduler;

/// PoC profiling, enabled by AZAHAR_GPU_DRAW_TIMING=1: GPU time per draw, by category.
/// A timed draw gets a bottom-of-pipe timestamp just before it and another just after it, so its
/// time is how much later the GPU finished it than the work recorded before it.
class GpuDrawTimer {
public:
    static constexpr std::size_t NumCategories = 10;

    struct Pair {
        vk::QueryPool pool{};
        u32 query{};
    };

    GpuDrawTimer(const Instance& instance, Scheduler& scheduler);
    ~GpuDrawTimer();

    [[nodiscard]] bool IsEnabled() const noexcept {
        return enabled;
    }

    /// Reserves a timestamp pair for one draw. The pool is null when timing is off, no pool is
    /// free, or the current pool is full.
    Pair Reserve(std::size_t category);

    static void WriteBefore(vk::CommandBuffer cmdbuf, const Pair& pair) {
        if (pair.pool) {
            cmdbuf.writeTimestamp(vk::PipelineStageFlagBits::eBottomOfPipe, pair.pool, pair.query);
        }
    }

    static void WriteAfter(vk::CommandBuffer cmdbuf, const Pair& pair) {
        if (pair.pool) {
            cmdbuf.writeTimestamp(vk::PipelineStageFlagBits::eBottomOfPipe, pair.pool,
                                  pair.query + 1);
        }
    }

    /// Called from the scheduler's submit hook, outside a render pass.
    void OnSubmit();

    /// Folds in finished submissions and formats the running totals. The window maximum
    /// restarts after each call.
    std::string Summary(const std::array<const char*, NumCategories>& names);

private:
    static constexpr u32 PoolQueries = 8192;
    static constexpr std::size_t NumPools = 8;
    static constexpr std::size_t NoPool = std::numeric_limits<std::size_t>::max();

    struct Pool {
        vk::UniqueQueryPool pool;
        std::vector<u8> categories; ///< One entry per reserved pair.
        u64 tick{};
        bool submitted{};
    };

    void CollectLocked();

    const Instance& instance;
    Scheduler& scheduler;
    bool enabled{};
    double period_ns{};
    u64 mask{};
    std::mutex mutex;
    std::vector<Pool> pools;
    std::size_t current{NoPool};
    std::array<u64, NumCategories> count{};
    std::array<u64, NumCategories> ns{};
    u64 dropped{};
    u64 submits{};
    u64 window_max_submit_ns{};
};

} // namespace Vulkan
