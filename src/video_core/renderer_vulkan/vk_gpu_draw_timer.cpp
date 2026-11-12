// Copyright 2026 Azahar Emulator Project
// Licensed under GPLv2 or any later version
// Refer to the misc/licenses/gplv2.txt file included.

#include <algorithm>
#include <cstdlib>
#include <fmt/format.h>
#include "common/logging/log.h"
#include "video_core/renderer_vulkan/vk_gpu_draw_timer.h"
#include "video_core/renderer_vulkan/vk_instance.h"
#include "video_core/renderer_vulkan/vk_scheduler.h"

namespace Vulkan {

GpuDrawTimer::GpuDrawTimer(const Instance& instance_, Scheduler& scheduler_)
    : instance{instance_}, scheduler{scheduler_} {
    if (std::getenv("AZAHAR_GPU_DRAW_TIMING") == nullptr) {
        return;
    }
    const vk::PhysicalDevice physical_device = instance.GetPhysicalDevice();
    const auto families = physical_device.getQueueFamilyProperties();
    const u32 valid_bits = families[instance.GetGraphicsQueueFamilyIndex()].timestampValidBits;
    period_ns = physical_device.getProperties().limits.timestampPeriod;
    if (valid_bits == 0 || period_ns <= 0.0) {
        LOG_WARNING(Render_Vulkan, "GPU draw timing: the graphics queue has no timestamps");
        return;
    }
    mask = valid_bits >= 64 ? ~u64{0} : (u64{1} << valid_bits) - 1;

    const vk::Device device = instance.GetDevice();
    pools.resize(NumPools);
    for (Pool& pool : pools) {
        pool.pool = device.createQueryPoolUnique({
            .queryType = vk::QueryType::eTimestamp,
            .queryCount = PoolQueries,
        });
        pool.categories.reserve(PoolQueries / 2);
    }
    enabled = true;
    LOG_INFO(Render_Vulkan, "GPU draw timing: on (period {} ns, {} valid bits)", period_ns,
             valid_bits);
}

GpuDrawTimer::~GpuDrawTimer() = default;

GpuDrawTimer::Pair GpuDrawTimer::Reserve(std::size_t category) {
    if (!enabled) {
        return {};
    }
    std::scoped_lock lock{mutex};
    if (current == NoPool) {
        ++dropped;
        return {};
    }
    Pool& pool = pools[current];
    if (pool.categories.size() >= PoolQueries / 2) {
        ++dropped;
        return {};
    }
    const u32 query = static_cast<u32>(pool.categories.size() * 2);
    pool.categories.push_back(static_cast<u8>(category));
    return {*pool.pool, query};
}

void GpuDrawTimer::OnSubmit() {
    if (!enabled) {
        return;
    }
    std::scoped_lock lock{mutex};
    CollectLocked();
    if (current != NoPool) {
        Pool& pool = pools[current];
        if (pool.categories.empty()) {
            // Still reset and unused: keep it for the next submission.
            return;
        }
        // Scheduler::SubmitExecution() takes the tick before calling the submit hook.
        pool.tick = scheduler.CurrentTick() - 1;
        pool.submitted = true;
        current = NoPool;
    }
    for (std::size_t i = 0; i < pools.size(); ++i) {
        if (!pools[i].submitted) {
            current = i;
            break;
        }
    }
    if (current == NoPool) {
        return;
    }
    // Recorded into the submission being closed, ahead of the next one's timestamps.
    scheduler.Record([pool = *pools[current].pool](vk::CommandBuffer cmdbuf) {
        cmdbuf.resetQueryPool(pool, 0, PoolQueries);
    });
}

void GpuDrawTimer::CollectLocked() {
    const vk::Device device = instance.GetDevice();
    std::vector<u64> data;
    for (Pool& pool : pools) {
        if (!pool.submitted || !scheduler.IsFree(pool.tick)) {
            continue;
        }
        const u32 queries = static_cast<u32>(pool.categories.size() * 2);
        data.resize(queries);
        const vk::Result result = device.getQueryPoolResults(
            *pool.pool, 0, queries, queries * sizeof(u64), data.data(), sizeof(u64),
            vk::QueryResultFlagBits::e64);
        if (result == vk::Result::eSuccess) {
            u64 submit_ns = 0;
            for (std::size_t i = 0; i < pool.categories.size(); ++i) {
                const u64 ticks = (data[i * 2 + 1] - data[i * 2]) & mask;
                const u64 draw_ns = static_cast<u64>(static_cast<double>(ticks) * period_ns);
                const u8 category = pool.categories[i];
                ++count[category];
                ns[category] += draw_ns;
                submit_ns += draw_ns;
            }
            ++submits;
            window_max_submit_ns = std::max(window_max_submit_ns, submit_ns);
        } else {
            dropped += pool.categories.size();
        }
        pool.categories.clear();
        pool.submitted = false;
    }
}

std::string GpuDrawTimer::Summary(const std::array<const char*, NumCategories>& names) {
    std::scoped_lock lock{mutex};
    CollectLocked();
    std::string line;
    u64 total_ns = 0;
    for (std::size_t i = 0; i < NumCategories; ++i) {
        if (count[i] == 0) {
            continue;
        }
        line += fmt::format(" {}={}/{:.1f}ms/{:.2f}us", names[i], count[i], ns[i] / 1e6,
                            ns[i] / 1e3 / count[i]);
        total_ns += ns[i];
    }
    line += fmt::format(" total={:.1f}ms submits={} window_max_submit={:.2f}ms dropped={}",
                        total_ns / 1e6, submits, window_max_submit_ns / 1e6, dropped);
    window_max_submit_ns = 0;
    return line;
}

} // namespace Vulkan
