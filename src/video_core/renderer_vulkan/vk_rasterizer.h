// Copyright 2022-2026 Citra Emulator Project / Azahar Emulator Project
// Licensed under GPLv3 or any later version
// Refer to the LICENSE.txt file included.

#pragma once

#include <chrono>
#include <unordered_map>
#include <unordered_set>

#include "video_core/rasterizer_accelerated.h"
#include "video_core/renderer_vulkan/vk_descriptor_update_queue.h"
#include "video_core/renderer_vulkan/vk_gpu_draw_timer.h"
#include "video_core/renderer_vulkan/vk_pipeline_cache.h"
#include "video_core/renderer_vulkan/vk_render_manager.h"
#include "video_core/renderer_vulkan/vk_stream_buffer.h"
#include "video_core/renderer_vulkan/vk_texture_runtime.h"
#include "video_core/shader/generator/fs_config_uniform.h"

namespace Frontend {
class EmuWindow;
}

namespace VideoCore {
class CustomTexManager;
class RendererBase;
} // namespace VideoCore

namespace Pica {
struct DisplayTransferConfig;
struct MemoryFillConfig;
struct FramebufferConfig;
} // namespace Pica

namespace Vulkan {

struct ScreenInfo;

class Instance;
class Scheduler;
class RenderManager;

class RasterizerVulkan : public VideoCore::RasterizerAccelerated {
public:
    explicit RasterizerVulkan(Memory::MemorySystem& memory, Pica::PicaCore& pica,
                              VideoCore::CustomTexManager& custom_tex_manager,
                              VideoCore::RendererBase& renderer, Frontend::EmuWindow& emu_window,
                              const Instance& instance, Scheduler& scheduler,
                              RenderManager& renderpass_cache, DescriptorUpdateQueue& update_queue,
                              u32 image_count);
    ~RasterizerVulkan() override;

    void TickFrame();
    void LoadDefaultDiskResources(const std::atomic_bool& stop_loading,
                                  const VideoCore::DiskResourceLoadCallback& callback) override;

    void DrawTriangles() override;
    void FlushAll() override;
    void FlushRegion(PAddr addr, u32 size) override;
    void InvalidateRegion(PAddr addr, u32 size) override;
    void FlushAndInvalidateRegion(PAddr addr, u32 size) override;
    void ClearAll(bool flush) override;
    bool AccelerateDisplayTransfer(const Pica::DisplayTransferConfig& config) override;
    bool AccelerateTextureCopy(const Pica::DisplayTransferConfig& config) override;
    bool AccelerateFill(const Pica::MemoryFillConfig& config) override;
    bool AccelerateDisplay(const Pica::FramebufferConfig& config, PAddr framebuffer_addr,
                           u32 pixel_stride, ScreenInfo& screen_info);
    bool AccelerateDrawBatch(bool is_indexed) override;

    /// Switches the disk resources to the specified title
    void SwitchDiskResources(u64 title_id) override;

    bool DiskResourcesLoadCrashed(u64 title_id) override;

private:
    /// Syncs pipeline state from PICA registers
    void SyncDrawState();

    /// Syncs and uploads the lighting, fog and proctex LUTs
    void SyncAndUploadLUTs();
    void SyncAndUploadLUTsLF();

    /// Syncs all enabled PICA texture units
    void SyncTextureUnits(const Framebuffer* framebuffer);

    /// Syncs all utility textures in the fragment shader.
    void SyncUtilityTextures(const Framebuffer* framebuffer);

    /// Binds the PICA shadow cube required for shadow mapping
    void BindShadowCube(const Pica::TexturingRegs::FullTextureConfig& texture,
                        vk::DescriptorSet texture_set);

    /// Binds a texture cube to texture unit 0
    void BindTextureCube(const Pica::TexturingRegs::FullTextureConfig& texture,
                         vk::DescriptorSet texture_set);

    /// Upload the uniform blocks to the uniform buffer object
    void UploadUniforms(bool accelerate_draw);

    /// Uploads the ubershader fs_config block for @p config and sets its dynamic offset
    void UploadFsConfig(const Pica::Shader::FSConfig& config);

    /// Logs the cumulative ubershader counters
    void LogUbershaderStats() const;
    void LogDrawTimes() const;

    /// Generic draw function for DrawTriangles and AccelerateDrawBatch
    bool Draw(bool accelerate, bool is_indexed);

    /// Internal implementation for AccelerateDrawBatch
    bool AccelerateDrawBatchInternal(bool is_indexed, bool force_wait);
    void PocNoteVsBools();
    std::unordered_map<u64, std::unordered_set<u32>> poc_vs_bools;

    /// Setup index array for AccelerateDrawBatch
    void SetupIndexArray();

    /// Setup vertex array for AccelerateDrawBatch
    void SetupVertexArray();

    /// Setup the fixed attribute emulation in vulkan
    void SetupFixedAttribs();

    /// Setup vertex shader for AccelerateDrawBatch
    bool SetupVertexShader();

    /// Setup geometry shader for AccelerateDrawBatch
    bool SetupGeometryShader();

    /// Creates the vertex layout struct used for software shader pipelines
    void MakeSoftwareVertexLayout();

private:
    const Instance& instance;
    Scheduler& scheduler;
    RenderManager& renderpass_cache;
    DescriptorUpdateQueue& update_queue;
    PipelineCache pipeline_cache;
    TextureRuntime runtime;
    RasterizerCache res_cache;

    VertexLayout software_layout;
    std::array<u32, 16> binding_offsets{};
    std::array<bool, 16> enable_attributes{};
    std::array<vk::Buffer, 16> vertex_buffers;
    VertexArrayInfo vertex_info;
    PipelineInfo pipeline_info{};

    StreamBuffer stream_buffer;     ///< Vertex+Index buffer
    StreamBuffer uniform_buffer;    ///< Uniform buffer
    StreamBuffer texture_buffer;    ///< Texture buffer
    StreamBuffer texture_lf_buffer; ///< Texture Light-Fog buffer
    vk::UniqueBufferView texture_lf_view;
    vk::UniqueBufferView texture_rg_view;
    vk::UniqueBufferView texture_rgba_view;
    vk::DeviceSize uniform_buffer_alignment;
    u32 uniform_size_aligned_vs_pica;
    u32 uniform_size_aligned_vs;
    u32 uniform_size_aligned_fs;
    bool async_shaders{false};
    bool use_ubershaders{false};
    bool ubershader_cpu_vertex{false};
    // Set from the AZAHAR_UBERSHADER_FORCE environment variable (PoC only).
    bool force_ubershader{false};
    // Cleared by AZAHAR_UBERSHADER_SMALL_DRAWS=0 (PoC only): draws of 6 or fewer vertices then
    // wait for their pipeline, as upstream does, instead of taking the ubershader.
    bool ubershader_small_draws{true};
    // Set by an accelerated draw that hands itself to the CPU vertex path. Read and cleared at the
    // top of the next RasterizerVulkan::Draw(), and also cleared by an empty
    // RasterizerVulkan::DrawTriangles().
    bool ubershader_fallback_pending{false};

    struct UbershaderStats {
        u64 frames{};
        u64 slow_frames{};
        u64 skipped{};
        u64 uber_gpu{};
        u64 uber_cpu{};
        u64 unsupported{};
        // unsupported, split by the first missing feature (Pica::Shader::FindUbershaderGap())
        std::array<u64, static_cast<std::size_t>(Pica::Shader::UbershaderGap::Count)>
            unsupported_by_gap{};
        // PoC profiling: wall time spent in Draw(), by DrawPath and by size (index 1 = 6 or fewer
        // vertices), plus the time PICA spends on the CPU vertex shader between a hand-off and
        // its fallback draw.
        std::array<std::array<u64, 2>, 5> draw_ns{};
        std::array<std::array<u64, 2>, 5> draw_count{};
        u64 cpu_vs_ns{};
        // PoC: hand-offs by what the GPU-VS ubershader pipeline was waiting for
        // (GraphicsPipeline::PocWaitState).
        std::array<u64, static_cast<std::size_t>(GraphicsPipeline::PocWaitState::Count)>
            handoff_wait{};
    } uber_stats;
    enum class DrawPath : u8 { Specialized, UberGpu, Handoff, Fallback, Software };
    std::chrono::steady_clock::time_point handoff_time{};
    std::chrono::steady_clock::time_point last_frame_time{};
    // PoC profiling, AZAHAR_SLOW_FRAME_MS=N: log every frame slower than N ms, split into CPU
    // time, Draw() time and waits (RasterizerVulkan::LogSlowFrame()). 0 = off.
    u32 slow_frame_ms{};
    struct FrameStats {
        u64 draw_ns{};
        u64 draws{};
        u64 uber_draws{};
        u64 wait_worker_ns{};
    } frame_stats;
    u64 last_frame_cpu_ns{};
    u64 last_pipeline_wait_ns{};
    u64 last_runq_ns{};
    u64 last_vcsw{};
    u64 last_ivcsw{};
    void LogSlowFrame(std::chrono::steady_clock::time_point now);
    // GpuDrawTimer category: DrawPath * 2, plus 1 for 6 or fewer vertices.
    [[nodiscard]] std::size_t GpuDrawCategory(DrawPath path) const {
        return static_cast<std::size_t>(path) * 2 + (regs.pipeline.num_vertices <= 6 ? 1 : 0);
    }
    mutable GpuDrawTimer gpu_draw_timer{instance, scheduler};
};

} // namespace Vulkan
