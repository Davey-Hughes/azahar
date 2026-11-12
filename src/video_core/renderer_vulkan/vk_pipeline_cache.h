// Copyright 2023-2026 Citra Emulator Project / Azahar Emulator Project
// Licensed under GPLv3 or any later version
// Refer to the LICENSE.txt file included.

#pragma once

#include <atomic>
#include <bitset>
#include <memory>
#include <unordered_set>

#include "video_core/rasterizer_interface.h"
#include "video_core/renderer_vulkan/vk_graphics_pipeline.h"
#include "video_core/renderer_vulkan/vk_resource_pool.h"
#include "video_core/renderer_vulkan/vk_shader_disk_cache.h"
#include "video_core/shader/generator/pica_fs_config.h"
#include "video_core/shader/generator/profile.h"
#include "video_core/shader/generator/shader_gen.h"

namespace Pica {
struct RegsInternal;
struct ShaderSetup;
} // namespace Pica

namespace Vulkan {

class Instance;
class Scheduler;
class RenderManager;
class DescriptorUpdateQueue;

enum class DescriptorHeapType : u32 {
    Buffer,
    Texture,
    Utility,
};

/**
 * Stores a collection of rasterizer pipelines used during rendering.
 */
class PipelineCache {
    static constexpr u32 NumRasterizerSets = 3;
    static constexpr u32 NumDescriptorHeaps = 3;
    // Dynamic UBOs in set 0: b0 VS PICA, b1 VS, b2 FS, b6 ubershader fs_config. Vulkan orders
    // dynamic offsets by binding, so fs_config's offset is index 3.
    static constexpr u32 NumDynamicOffsets = 4;
    static constexpr u32 FsConfigOffsetIndex = 3;

public:
    explicit PipelineCache(const Instance& instance, Scheduler& scheduler,
                           RenderManager& renderpass_cache, DescriptorUpdateQueue& update_queue);
    ~PipelineCache();

    /// Acquires and binds a free descriptor set from the appropriate heap.
    vk::DescriptorSet Acquire(DescriptorHeapType type) {
        const u32 index = static_cast<u32>(type);
        const auto descriptor_set = descriptor_heaps[index].Commit();
        bound_descriptor_sets[index] = descriptor_set;
        return descriptor_set;
    }

    /// Sets the dynamic offset for the uniform buffer at binding
    void UpdateRange(u8 binding, u32 offset) {
        offsets[binding] = offset;
    }

    /// Hash used for the ubershader fragment stage. Pipelines using it are never written to the
    /// shader disk cache (ShaderDiskCache::GetPipeline()).
    static constexpr u64 UBERSHADER_FS_HASH = 0x0BE70000'00000001ULL;

    struct WaitStats {
        u64 count;
        u64 ns;
    };

    struct PipelineCounts {
        u64 specialized;
        u64 ubershader;
    };

    /// Sets the dynamic offset for the ubershader fs_config UBO (set 0, binding 6).
    void UpdateFsConfigRange(u32 offset) {
        offsets[FsConfigOffsetIndex] = offset;
    }

    /// True when the ubershader compiled at startup (only attempted with use_ubershaders on)
    bool IsUbershaderAvailable() const {
        return ubershader_fragment_shader && ubershader_fragment_shader->Handle();
    }

    /// Replaces the bound fragment shader with the ubershader
    void UseUbershaderFragmentShader();

    /// Returns true if the pipeline for the bound shaders and @p info is built. Queues the build
    /// like an async BindPipeline() but records nothing.
    bool IsPipelineReady(PipelineInfo& info);

    // PoC: the pipeline the last IsPipelineReady() call looked up.
    [[nodiscard]] GraphicsPipeline* LastReadyCheck() const noexcept {
        return last_ready_check;
    }

    /// Builds the fragment config for @p regs with this device's profile applied
    Pica::Shader::FSConfig MakeFSConfig(const Pica::RegsInternal& regs) const;

    /// Cumulative count and duration of waits on pipelines that were not built yet. The waits are
    /// timed on the scheduler's recording (worker) thread; the emulation thread feels them at the
    /// next Scheduler::WaitWorker().
    WaitStats GetWaitStats() const {
        return {wait_count.load(std::memory_order_relaxed),
                wait_ns.load(std::memory_order_relaxed)};
    }

    /// Cumulative count of pipelines created at runtime (not loaded from the disk cache)
    PipelineCounts GetPipelineCounts() const {
        return {specialized_pipeline_count.load(std::memory_order_relaxed),
                ubershader_pipeline_count.load(std::memory_order_relaxed)};
    }

    /// Loads the driver pipeline cache and the disk shader cache
    void LoadCache(const std::atomic_bool& stop_loading = std::atomic_bool{false},
                   const VideoCore::DiskResourceLoadCallback& callback = {});

    /// Switches the driver pipeline cache and the shader disk cache to the specified title
    void SwitchCache(u64 title_id, const std::atomic_bool& stop_loading = std::atomic_bool{false},
                     const VideoCore::DiskResourceLoadCallback& callback = {});

    /// Returns true when the emulator crashed the last time it loaded the caches of the title
    bool LoadCacheCrashed(u64 title_id) const;

    /// Binds a pipeline using the provided information
    bool BindPipeline(PipelineInfo& info, bool wait_built = false);

    Pica::Shader::Generator::ExtraVSConfig CalcExtraConfig(
        const Pica::Shader::Generator::PicaVSConfig& config);

    /// Binds a PICA decompiled vertex shader
    bool UseProgrammableVertexShader(const Pica::RegsInternal& regs, Pica::ShaderSetup& setup,
                                     const VertexLayout& layout);

    /// Binds a passthrough vertex shader
    void UseTrivialVertexShader();

    /// Binds a PICA decompiled geometry shader
    bool UseFixedGeometryShader(const Pica::RegsInternal& regs);

    /// Binds a passthrough geometry shader
    void UseTrivialGeometryShader();

    /// Binds a fragment shader generated from PICA state
    void UseFragmentShader(const Pica::RegsInternal& regs, const Pica::Shader::UserConfig& user);

    /// Queries which channels need min/max blend emulation in the fragment shader
    void QueryBlendEmulation(const Pica::RegsInternal& regs, bool& rgb_emulation,
                             bool& alpha_emulation) const;

    /// Gets the current program ID
    u64 GetProgramID() const {
        return current_program_id;
    }

    void SetProgramID(u64 program_id) {
        current_program_id = program_id;
    }

    void SetAccurateMul(bool _accurate_mul) {
        profile.enable_accurate_mul = _accurate_mul;
    }

private:
    friend ShaderDiskCache;

    template <typename Work>
    void QueueVertexShaderWork(Work&& work) {
        if (priority_workers.NumWorkers() > 0) {
            priority_workers.QueueWork(std::forward<Work>(work));
        } else if (uber_first_front) {
            shader_workers.QueuePriorityWork(std::forward<Work>(work));
        } else {
            shader_workers.QueueWork(std::forward<Work>(work));
        }
    }
    Common::ThreadWorker& UberPipelineWorkers() {
        return priority_workers.NumWorkers() > 0 ? priority_workers : pipeline_workers;
    }

    /// Loads the driver pipeline cache
    void LoadDriverPipelineDiskCache(const std::atomic_bool& stop_loading = std::atomic_bool{false},
                                     const VideoCore::DiskResourceLoadCallback& callback = {});

    /// Stores the generated pipeline cache
    void SaveDriverPipelineDiskCache();

    /// Loads the shader disk cache
    void LoadDiskCache(const std::atomic_bool& stop_loading = std::atomic_bool{false},
                       const VideoCore::DiskResourceLoadCallback& callback = {});

    /// Switches the disk cache at runtime to use a different title ID
    void SwitchDiskCache(u64 title_id, const std::atomic_bool& stop_loading,
                         const VideoCore::DiskResourceLoadCallback& callback);

    /// Marks the cache of the title as being loaded. Returns true when the flag was created.
    bool SetCacheLoadFlag(u64 title_id);

    /// Clears the load flag of the title once the cached pipelines are built
    void QueueClearCacheLoadFlag(u64 title_id);

    /// Returns the path of the file that marks the caches of the title as being loaded
    std::string GetLoadFlagPath(u64 title_id) const;

    /// Builds the rasterizer pipeline layout
    void BuildLayout();

    /// Returns true when the disk data can be used by the current driver
    bool IsCacheValid(std::span<const u8> cache_data) const;

    /// Create pipeline cache directories. Returns true on success.
    bool EnsureDirectories() const;

    /// Returns the Vulkan shader directory
    std::string GetVulkanDir() const;

    /// Returns the pipeline cache storage dir
    std::string GetPipelineCacheDir() const;

    /// Returns the transferable shader dir
    std::string GetTransferableDir() const;

private:
    const Instance& instance;
    Scheduler& scheduler;
    RenderManager& renderpass_cache;
    DescriptorUpdateQueue& update_queue;

    Pica::Shader::Profile profile{};
    vk::UniquePipelineCache driver_pipeline_cache;
    vk::UniquePipelineLayout pipeline_layout;
    std::size_t num_worker_threads;
    Common::ThreadWorker pipeline_workers;
    Common::ThreadWorker shader_workers;
    // PoC: AZAHAR_UBER_FIRST=N runs vertex shader compiles and ubershader pipeline builds on
    // their own N workers, so they don't queue behind specialized work. AZAHAR_UBER_FIRST=front
    // queues them ahead of other work on the existing workers instead. Unset = upstream.
    Common::ThreadWorker priority_workers;
    bool uber_first_front;
    PipelineInfo current_info{};
    GraphicsPipeline* current_pipeline{};
    GraphicsPipeline* last_ready_check{};
    std::array<DescriptorHeap, NumDescriptorHeaps> descriptor_heaps;
    std::array<vk::DescriptorSet, NumRasterizerSets> bound_descriptor_sets{};
    std::array<u32, NumDynamicOffsets> offsets{};

    std::array<u64, MAX_SHADER_STAGES> shader_hashes;
    std::array<Shader*, MAX_SHADER_STAGES> current_shaders;

    Shader trivial_vertex_shader;
    std::unique_ptr<Shader> ubershader_fragment_shader;
    std::atomic<u64> wait_count{0};
    std::atomic<u64> wait_ns{0};
    std::atomic<u64> specialized_pipeline_count{0};
    std::atomic<u64> ubershader_pipeline_count{0};

    u64 current_program_id{0};
    std::vector<std::shared_ptr<ShaderDiskCache>> disk_caches;
    std::shared_ptr<ShaderDiskCache> curr_disk_cache{};

    std::unordered_set<u64> load_flag_titles;
};

} // namespace Vulkan
