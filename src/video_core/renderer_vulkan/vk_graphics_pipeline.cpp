// Copyright 2023-2026 Citra Emulator Project / Azahar Emulator Project
// Licensed under GPLv3 or any later version
// Refer to the LICENSE.txt file included.

#include <algorithm>
#include <memory>
#include <string>
#include <unordered_map>

#include <boost/container/static_vector.hpp>

#include "common/alignment.h"
#include "common/hash.h"
#include "common/microprofile.h"
#include "video_core/renderer_vulkan/pica_to_vk.h"
#include "video_core/renderer_vulkan/vk_graphics_pipeline.h"
#include "video_core/renderer_vulkan/vk_instance.h"
#include "video_core/renderer_vulkan/vk_render_manager.h"
#include "video_core/renderer_vulkan/vk_shader_util.h"

namespace Vulkan {

MICROPROFILE_DEFINE(Vulkan_Pipeline, "Vulkan", "Pipeline Building", MP_RGB(0, 192, 32));

vk::ShaderStageFlagBits MakeShaderStage(std::size_t index) {
    switch (index) {
    case 0:
        return vk::ShaderStageFlagBits::eVertex;
    case 1:
        return vk::ShaderStageFlagBits::eFragment;
    case 2:
        return vk::ShaderStageFlagBits::eGeometry;
    default:
        LOG_CRITICAL(Render_Vulkan, "Invalid shader stage index!");
        UNREACHABLE();
    }
    return vk::ShaderStageFlagBits::eVertex;
}

u64 StaticPipelineInfo::OptimizedHash(const Instance& instance) const {
    // With dynamic blend, only blend_enable and logic_op are baked into the pipeline.
    const u64 blend_hash = instance.IsExtendedDynamicState3Supported()
                               ? Common::HashCombine(static_cast<u64>(blending.blend_enable),
                                                     static_cast<u64>(blending.logic_op))
                               : Common::ComputeStructHash64(blending);
    u64 info_hash = Common::HashCombine(
        shader_ids[0], shader_ids[1], shader_ids[2], Common::ComputeStructHash64(vertex_layout),
        Common::ComputeStructHash64(attachments), blend_hash);

    if (!instance.IsExtendedDynamicStateSupported()) {
        info_hash = Common::HashCombine(info_hash, Common::ComputeStructHash64(rasterization),
                                        Common::ComputeStructHash64(depth_stencil));
    }

    return info_hash;
}

u16 PipelineInfo::GetFinalColorWriteMask(const Instance& instance) {
    u16 color_write_mask = state.blending.color_write_mask;
    const bool is_logic_op_emulated =
        instance.NeedsLogicOpEmulation() && !state.blending.blend_enable;
    const bool is_logic_op_noop = state.blending.logic_op == Pica::FramebufferRegs::LogicOp::NoOp;
    if (is_logic_op_emulated && is_logic_op_noop) {
        // Color output is disabled by logic operation. We use color write mask to skip
        // color but allow depth write.
        color_write_mask = 0;
    }
    return color_write_mask;
}

vk::ColorBlendEquationEXT MakeBlendEquation(const BlendingState& blending) {
    vk::ColorBlendEquationEXT eq = {
        .srcColorBlendFactor = PicaToVK::BlendFunc(blending.src_color_blend_factor),
        .dstColorBlendFactor = PicaToVK::BlendFunc(blending.dst_color_blend_factor),
        .colorBlendOp = PicaToVK::BlendEquation(blending.color_blend_eq),
        .srcAlphaBlendFactor = PicaToVK::BlendFunc(blending.src_alpha_blend_factor),
        .dstAlphaBlendFactor = PicaToVK::BlendFunc(blending.dst_alpha_blend_factor),
        .alphaBlendOp = PicaToVK::BlendEquation(blending.alpha_blend_eq),
    };
    if (blending.rgb_blend_emulation) {
        eq.srcColorBlendFactor = vk::BlendFactor::eOne;
        eq.dstColorBlendFactor = vk::BlendFactor::eZero;
        eq.colorBlendOp = vk::BlendOp::eAdd;
    }
    if (blending.alpha_blend_emulation) {
        eq.srcAlphaBlendFactor = vk::BlendFactor::eOne;
        eq.dstAlphaBlendFactor = vk::BlendFactor::eZero;
        eq.alphaBlendOp = vk::BlendOp::eAdd;
    }
    return eq;
}

boost::container::static_vector<vk::DynamicState, 16> MakeDynamicStates(const Instance& instance) {
    boost::container::static_vector<vk::DynamicState, 16> dynamic_states = {
        vk::DynamicState::eViewport,           vk::DynamicState::eScissor,
        vk::DynamicState::eStencilCompareMask, vk::DynamicState::eStencilWriteMask,
        vk::DynamicState::eStencilReference,   vk::DynamicState::eBlendConstants,
    };

    if (instance.IsExtendedDynamicStateSupported()) {
        constexpr std::array extended = {
            vk::DynamicState::eCullModeEXT,        vk::DynamicState::eDepthCompareOpEXT,
            vk::DynamicState::eDepthTestEnableEXT, vk::DynamicState::eDepthWriteEnableEXT,
            vk::DynamicState::eFrontFaceEXT,       vk::DynamicState::ePrimitiveTopologyEXT,
            vk::DynamicState::eStencilOpEXT,       vk::DynamicState::eStencilTestEnableEXT,
        };
        dynamic_states.insert(dynamic_states.end(), extended.begin(), extended.end());
    }

    if (instance.IsExtendedDynamicState3Supported()) {
        constexpr std::array dynamic_blend = {
            vk::DynamicState::eColorBlendEquationEXT,
            vk::DynamicState::eColorWriteMaskEXT,
        };
        dynamic_states.insert(dynamic_states.end(), dynamic_blend.begin(), dynamic_blend.end());
    }
    return dynamic_states;
}

Shader::Shader(const Instance& instance) : device{instance.GetDevice()} {}

Shader::Shader(const Instance& instance, vk::ShaderStageFlagBits stage, std::string code)
    : Shader{instance} {
    module = Compile(code, stage, instance.GetDevice());
    MarkDone();
}

Shader::~Shader() {
    if (device && module) {
        device.destroyShaderModule(module);
    }
}

GraphicsPipeline::GraphicsPipeline(const Instance& instance_, RenderManager& renderpass_cache_,
                                   const PipelineInfo& info_, vk::PipelineCache pipeline_cache_,
                                   vk::PipelineLayout layout_, std::array<Shader*, 3> stages_,
                                   Common::ThreadWorker* worker_)
    : instance{instance_}, renderpass_cache{renderpass_cache_}, worker{worker_},
      pipeline_layout{layout_}, pipeline_cache{pipeline_cache_}, info{info_}, stages{stages_} {}

GraphicsPipeline::~GraphicsPipeline() = default;

bool GraphicsPipeline::TryBuild(bool wait_built) {
    // The pipeline is currently being compiled. We can either wait for it
    // or skip the draw.
    if (is_pending) {
        return wait_built;
    }

    // If the shaders haven't been compiled yet, we cannot proceed.
    const bool shaders_pending = std::any_of(
        stages.begin(), stages.end(), [](Shader* shader) { return shader && !shader->IsDone(); });
    if (!wait_built && shaders_pending) {
        return false;
    }

    // Ask the driver if it can give us the pipeline quickly.
    if (!shaders_pending && instance.IsPipelineCreationCacheControlSupported()) {
        const auto start = PocClock::now();
        if (Build(true)) {
            poc_queued = start;
            PocLogBuild(start, PocClock::now(), true);
            return true;
        }
    }

    // Fallback to (a)synchronous compilation
    poc_queued = PocClock::now();
    auto build = [this] {
        ApplyCompileWorkerNice();
        const auto start = PocClock::now();
        poc_started.store(true, std::memory_order_relaxed);
        Build();
        PocLogBuild(start, PocClock::now(), false);
    };
    if (priority) {
        worker->QueuePriorityWork(std::move(build));
    } else {
        worker->QueueWork(std::move(build));
    }
    is_pending = true;
    return wait_built;
}

bool GraphicsPipeline::Build(bool fail_on_compile_required) {
    MICROPROFILE_SCOPE(Vulkan_Pipeline);

    const u32 stride_alignment = instance.GetMinVertexStrideAlignment();
    std::array<vk::VertexInputBindingDescription, MAX_VERTEX_BINDINGS> bindings;
    for (u32 i = 0; i < info.state.vertex_layout.binding_count; i++) {
        const auto& binding = info.state.vertex_layout.bindings[i];
        bindings[i] = vk::VertexInputBindingDescription{
            .binding = binding.binding,
            .stride = Common::AlignUp(binding.byte_count.Value(), stride_alignment),
            .inputRate = binding.fixed.Value() ? vk::VertexInputRate::eInstance
                                               : vk::VertexInputRate::eVertex,
        };
    }

    std::array<vk::VertexInputAttributeDescription, MAX_VERTEX_ATTRIBUTES> attributes;
    for (u32 i = 0; i < info.state.vertex_layout.attribute_count; i++) {
        const auto& attr = info.state.vertex_layout.attributes[i];
        const FormatTraits& traits = instance.GetTraits(attr.type, attr.size);
        attributes[i] = vk::VertexInputAttributeDescription{
            .location = attr.location,
            .binding = attr.binding,
            .format = traits.native,
            .offset = attr.offset,
        };

        // At the end there's always the fixed binding which takes up
        // at least 16 bytes so we should always be able to alias.
        if (traits.needs_emulation) {
            const FormatTraits& comp_four_traits = instance.GetTraits(attr.type, 4);
            attributes[i].format = comp_four_traits.native;
        }
    }

    const vk::PipelineVertexInputStateCreateInfo vertex_input_info = {
        .vertexBindingDescriptionCount = info.state.vertex_layout.binding_count,
        .pVertexBindingDescriptions = bindings.data(),
        .vertexAttributeDescriptionCount = info.state.vertex_layout.attribute_count,
        .pVertexAttributeDescriptions = attributes.data(),
    };

    const vk::PipelineInputAssemblyStateCreateInfo input_assembly = {
        .topology = PicaToVK::PrimitiveTopology(info.state.rasterization.topology),
        .primitiveRestartEnable = false,
    };

    const vk::PipelineRasterizationStateCreateInfo raster_state = {
        .depthClampEnable = false,
        .rasterizerDiscardEnable = false,
        .cullMode = PicaToVK::CullMode(info.state.rasterization.cull_mode,
                                       info.state.rasterization.flip_viewport),
        .frontFace = PicaToVK::FrontFace(info.state.rasterization.cull_mode),
        .depthBiasEnable = false,
        .lineWidth = 1.0f,
    };

    const vk::PipelineMultisampleStateCreateInfo multisampling = {
        .rasterizationSamples = vk::SampleCountFlagBits(info.state.attachments.sample_count),
        .sampleShadingEnable = true,
        .minSampleShading = 1.0f,
    };

    const auto& blending = info.state.blending;
    const vk::ColorBlendEquationEXT blend_eq = MakeBlendEquation(blending);
    const vk::PipelineColorBlendAttachmentState colorblend_attachment = {
        .blendEnable = blending.blend_enable,
        .srcColorBlendFactor = blend_eq.srcColorBlendFactor,
        .dstColorBlendFactor = blend_eq.dstColorBlendFactor,
        .colorBlendOp = blend_eq.colorBlendOp,
        .srcAlphaBlendFactor = blend_eq.srcAlphaBlendFactor,
        .dstAlphaBlendFactor = blend_eq.dstAlphaBlendFactor,
        .alphaBlendOp = blend_eq.alphaBlendOp,
        .colorWriteMask =
            static_cast<vk::ColorComponentFlags>(info.GetFinalColorWriteMask(instance)),
    };

    const vk::PipelineColorBlendStateCreateInfo color_blending = {
        .logicOpEnable = !info.state.blending.blend_enable && !instance.NeedsLogicOpEmulation(),
        .logicOp = PicaToVK::LogicOp(info.state.blending.logic_op),
        .attachmentCount = 1,
        .pAttachments = &colorblend_attachment,
        .blendConstants = std::array{1.0f, 1.0f, 1.0f, 1.0f},
    };

    const vk::Viewport viewport = {
        .x = 0.0f,
        .y = 0.0f,
        .width = 1.0f,
        .height = 1.0f,
        .minDepth = 0.0f,
        .maxDepth = 1.0f,
    };

    const vk::Rect2D scissor = {
        .offset = {0, 0},
        .extent = {1, 1},
    };

    const vk::PipelineViewportStateCreateInfo viewport_info = {
        .viewportCount = 1,
        .pViewports = &viewport,
        .scissorCount = 1,
        .pScissors = &scissor,
    };

    const auto dynamic_states = MakeDynamicStates(instance);
    const vk::PipelineDynamicStateCreateInfo dynamic_info = {
        .dynamicStateCount = static_cast<u32>(dynamic_states.size()),
        .pDynamicStates = dynamic_states.data(),
    };

    const vk::StencilOpState stencil_op_state = {
        .failOp = PicaToVK::StencilOp(info.state.depth_stencil.stencil_fail_op),
        .passOp = PicaToVK::StencilOp(info.state.depth_stencil.stencil_pass_op),
        .depthFailOp = PicaToVK::StencilOp(info.state.depth_stencil.stencil_depth_fail_op),
        .compareOp = PicaToVK::CompareFunc(info.state.depth_stencil.stencil_compare_op),
    };

    const vk::PipelineDepthStencilStateCreateInfo depth_info = {
        .depthTestEnable = static_cast<u32>(info.state.depth_stencil.depth_test_enable.Value()),
        .depthWriteEnable = static_cast<u32>(info.state.depth_stencil.depth_write_enable.Value()),
        .depthCompareOp = PicaToVK::CompareFunc(info.state.depth_stencil.depth_compare_op),
        .depthBoundsTestEnable = false,
        .stencilTestEnable = static_cast<u32>(info.state.depth_stencil.stencil_test_enable.Value()),
        .front = stencil_op_state,
        .back = stencil_op_state,
    };

    u32 shader_count = 0;
    std::array<vk::PipelineShaderStageCreateInfo, MAX_SHADER_STAGES> shader_stages;
    for (std::size_t i = 0; i < stages.size(); i++) {
        Shader* shader = stages[i];
        if (!shader) {
            continue;
        }

        shader->WaitDone();
        shader_stages[shader_count++] = vk::PipelineShaderStageCreateInfo{
            .stage = MakeShaderStage(i),
            .module = shader->Handle(),
            .pName = "main",
        };
    }

    const vk::RenderPass render_pass =
        renderpass_cache.GetRenderpass(info.state.attachments.color, info.state.attachments.depth,
                                       false, info.state.attachments.sample_count);

    if (poc_uber && instance.IsPocUberGplEnabled()) {
        // PoC: the ubershader FS is the expensive part and the same for every ubershader
        // pipeline, so build it once per fragment state as a library and fast-link it with a
        // per-pipeline vertex library. The vertex library still compiles the game's VS.
        // The fail_on_compile_required (emu thread) attempt succeeds only when the fragment
        // library exists and the driver has the vertex library cached, as without GPL.
        using Ms = std::chrono::duration<double, std::milli>;
        const vk::Device device = instance.GetDevice();
        const auto t0 = PocClock::now();

        // Everything the fragment library bakes in. Depth/stencil, blend equations and the
        // color write mask are dynamic (IsPocUberGplEnabled() requires EDS and EDS3).
        const u64 fragment_key = Common::HashCombine(
            reinterpret_cast<u64>(static_cast<VkShaderModule>(stages[1]->Handle())),
            Common::ComputeStructHash64(info.state.attachments),
            static_cast<u64>(blending.blend_enable), static_cast<u64>(blending.logic_op));
        PocFragmentLibrary& fragment = PocGetFragmentLibrary(fragment_key);
        {
            // On the emu-thread attempt, don't wait for another thread's fragment build, and
            // only take a fragment library the driver can return without compiling.
            std::unique_lock lock{fragment.mutex, std::defer_lock};
            if (fail_on_compile_required) {
                if (!lock.try_lock()) {
                    return false;
                }
            } else {
                lock.lock();
            }
            if (!fragment.pipeline) {
                const auto f0 = PocClock::now();
                const vk::GraphicsPipelineLibraryCreateInfoEXT library_info = {
                    .flags = vk::GraphicsPipelineLibraryFlagBitsEXT::eFragmentShader |
                             vk::GraphicsPipelineLibraryFlagBitsEXT::eFragmentOutputInterface,
                };
                const vk::PipelineShaderStageCreateInfo fs_stage = {
                    .stage = vk::ShaderStageFlagBits::eFragment,
                    .module = stages[1]->Handle(),
                    .pName = "main",
                };
                // Depth-only render passes have no color attachment to describe.
                vk::PipelineColorBlendStateCreateInfo fragment_blending = color_blending;
                if (info.state.attachments.color == VideoCore::PixelFormat::Invalid) {
                    fragment_blending.attachmentCount = 0;
                }
                vk::GraphicsPipelineCreateInfo fragment_info = {
                    .pNext = &library_info,
                    .flags = vk::PipelineCreateFlagBits::eLibraryKHR,
                    .stageCount = 1,
                    .pStages = &fs_stage,
                    .pMultisampleState = &multisampling,
                    .pDepthStencilState = &depth_info,
                    .pColorBlendState = &fragment_blending,
                    .pDynamicState = &dynamic_info,
                    .layout = pipeline_layout,
                    .renderPass = render_pass,
                };
                if (fail_on_compile_required) {
                    fragment_info.flags |=
                        vk::PipelineCreateFlagBits::eFailOnPipelineCompileRequiredEXT;
                }
                auto result = device.createGraphicsPipeline(pipeline_cache, fragment_info);
                if (result.result == vk::Result::eErrorPipelineCompileRequiredEXT) {
                    return false;
                }
                ASSERT_MSG(result.result == vk::Result::eSuccess,
                           "PoC GPL fragment library creation failed");
                // Never destroyed: shared by every ubershader pipeline for the device's life.
                fragment.pipeline = result.value;
                LOG_INFO(Render_Vulkan,
                         "PoC GPL fragment library: key={:016x} color={} depth={} samples={} "
                         "blend={} logic_op={} ms={:.1f}",
                         fragment_key, static_cast<u32>(info.state.attachments.color),
                         static_cast<u32>(info.state.attachments.depth),
                         info.state.attachments.sample_count, blending.blend_enable,
                         static_cast<u32>(blending.logic_op),
                         Ms(PocClock::now() - f0).count());
            }
        }
        const auto t1 = PocClock::now();
        if (poc_fragment_only) {
            // Startup prewarm (PipelineCache::PocPrewarmFragmentLibraries()): library only.
            MarkDone();
            return true;
        }

        std::array<vk::PipelineShaderStageCreateInfo, MAX_SHADER_STAGES> vertex_stages;
        u32 vertex_stage_count = 0;
        for (const std::size_t i : {std::size_t{0}, std::size_t{2}}) {
            if (stages[i]) {
                vertex_stages[vertex_stage_count++] = vk::PipelineShaderStageCreateInfo{
                    .stage = MakeShaderStage(i),
                    .module = stages[i]->Handle(),
                    .pName = "main",
                };
            }
        }
        const vk::GraphicsPipelineLibraryCreateInfoEXT vertex_library_info = {
            .flags = vk::GraphicsPipelineLibraryFlagBitsEXT::eVertexInputInterface |
                     vk::GraphicsPipelineLibraryFlagBitsEXT::ePreRasterizationShaders,
        };
        vk::GraphicsPipelineCreateInfo vertex_info = {
            .pNext = &vertex_library_info,
            .flags = vk::PipelineCreateFlagBits::eLibraryKHR,
            .stageCount = vertex_stage_count,
            .pStages = vertex_stages.data(),
            .pVertexInputState = &vertex_input_info,
            .pInputAssemblyState = &input_assembly,
            .pViewportState = &viewport_info,
            .pRasterizationState = &raster_state,
            .pDynamicState = &dynamic_info,
            .layout = pipeline_layout,
            .renderPass = render_pass,
        };
        if (fail_on_compile_required) {
            vertex_info.flags |= vk::PipelineCreateFlagBits::eFailOnPipelineCompileRequiredEXT;
        }
        auto vertex_result = device.createGraphicsPipelineUnique(pipeline_cache, vertex_info);
        if (vertex_result.result == vk::Result::eErrorPipelineCompileRequiredEXT) {
            return false;
        }
        ASSERT_MSG(vertex_result.result == vk::Result::eSuccess,
                   "PoC GPL vertex library creation failed");
        poc_vertex_library = std::move(vertex_result.value);
        const auto t2 = PocClock::now();

        const std::array libraries{*poc_vertex_library, fragment.pipeline};
        const vk::PipelineLibraryCreateInfoKHR link_info = {
            .libraryCount = static_cast<u32>(libraries.size()),
            .pLibraries = libraries.data(),
        };
        const vk::GraphicsPipelineCreateInfo linked_info = {
            .pNext = &link_info,
            .layout = pipeline_layout,
        };
        auto linked = device.createGraphicsPipelineUnique(pipeline_cache, linked_info);
        ASSERT_MSG(linked.result == vk::Result::eSuccess, "PoC GPL link failed");
        pipeline = std::move(linked.value);
        poc_gpl_ms = {Ms(t1 - t0).count(), Ms(t2 - t1).count(), Ms(PocClock::now() - t2).count()};
        MarkDone();
        return true;
    }

    vk::GraphicsPipelineCreateInfo pipeline_info = {
        .stageCount = shader_count,
        .pStages = shader_stages.data(),
        .pVertexInputState = &vertex_input_info,
        .pInputAssemblyState = &input_assembly,
        .pViewportState = &viewport_info,
        .pRasterizationState = &raster_state,
        .pMultisampleState = &multisampling,
        .pDepthStencilState = &depth_info,
        .pColorBlendState = &color_blending,
        .pDynamicState = &dynamic_info,
        .layout = pipeline_layout,
        .renderPass = render_pass,
    };

    if (fail_on_compile_required) {
        pipeline_info.flags |= vk::PipelineCreateFlagBits::eFailOnPipelineCompileRequiredEXT;
    }
    if (instance.IsPocPipelineStatsEnabled()) {
        pipeline_info.flags |= vk::PipelineCreateFlagBits::eCaptureStatisticsKHR;
    }

    auto result = instance.GetDevice().createGraphicsPipelineUnique(pipeline_cache, pipeline_info);
    if (result.result == vk::Result::eSuccess) {
        pipeline = std::move(result.value);
    } else if (result.result == vk::Result::eErrorPipelineCompileRequiredEXT) {
        return false;
    } else {
        UNREACHABLE_MSG("Graphics pipeline creation failed!");
    }

    if (instance.IsPocPipelineStatsEnabled()) {
        PocLogStats();
    }
    MarkDone();
    return true;
}

void GraphicsPipeline::PocLogStats() const {
    const vk::Device device = instance.GetDevice();
    const auto executables = device.getPipelineExecutablePropertiesKHR({.pipeline = *pipeline});
    const auto no_spaces = [](const char* text) {
        std::string out{text};
        std::replace(out.begin(), out.end(), ' ', '_');
        return out;
    };
    for (u32 i = 0; i < static_cast<u32>(executables.size()); ++i) {
        const auto stats = device.getPipelineExecutableStatisticsKHR(
            {.pipeline = *pipeline, .executableIndex = i});
        std::string line;
        for (const auto& stat : stats) {
            const std::string name = no_spaces(stat.name.data());
            switch (stat.format) {
            case vk::PipelineExecutableStatisticFormatKHR::eBool32:
                line += fmt::format(" {}={}", name, stat.value.b32);
                break;
            case vk::PipelineExecutableStatisticFormatKHR::eInt64:
                line += fmt::format(" {}={}", name, stat.value.i64);
                break;
            case vk::PipelineExecutableStatisticFormatKHR::eUint64:
                line += fmt::format(" {}={}", name, stat.value.u64);
                break;
            case vk::PipelineExecutableStatisticFormatKHR::eFloat64:
                line += fmt::format(" {}={:.2f}", name, stat.value.f64);
                break;
            }
        }
        LOG_INFO(Render_Vulkan, "PoC pipeline stats: uber={} vs={:016x} exec={}{}", poc_uber,
                 info.state.shader_ids[0], no_spaces(executables[i].name.data()), line);
    }
}

GraphicsPipeline::PocWaitState GraphicsPipeline::PocNoteHandoff() noexcept {
    poc_handoffs.fetch_add(1, std::memory_order_relaxed);
    if (is_pending) {
        return poc_started.load(std::memory_order_relaxed) ? PocWaitState::Building
                                                           : PocWaitState::Queued;
    }
    // stages[] is indexed by Pica::Shader::Generator::ProgramType: VS, FS, GS.
    static constexpr std::array pending_state{PocWaitState::VsPending, PocWaitState::FsPending,
                                              PocWaitState::GsPending};
    for (std::size_t i = 0; i < stages.size(); ++i) {
        if (stages[i] && !stages[i]->IsDone()) {
            return pending_state[i];
        }
    }
    return PocWaitState::Queued;
}

GraphicsPipeline::PocFragmentLibrary& GraphicsPipeline::PocGetFragmentLibrary(u64 key) {
    // Leaked on purpose (PoC): the libraries live as long as the process.
    static std::mutex map_mutex;
    static std::unordered_map<u64, std::unique_ptr<PocFragmentLibrary>> libraries;
    std::scoped_lock lock{map_mutex};
    auto& entry = libraries[key];
    if (!entry) {
        entry = std::make_unique<PocFragmentLibrary>();
    }
    return *entry;
}

void GraphicsPipeline::PocLogBuild(PocClock::time_point start, PocClock::time_point end,
                                   bool fast) const {
    using Ms = std::chrono::duration<double, std::milli>;
    const auto& state = info.state;
    LOG_INFO(Render_Vulkan,
             "PoC pipeline built: uber={} fast={} vs={:016x} wait_shaders_ms={:.0f} "
             "queued_ms={:.0f} build_ms={:.1f} handoffs={} gpl_frag_ms={:.0f} gpl_vert_ms={:.1f} "
             "gpl_link_ms={:.1f} layout={:016x} color={} depth={} samples={} blend={} "
             "logic_op={} gs={:016x}",
             poc_uber, fast, state.shader_ids[0], Ms(poc_queued - poc_created).count(),
             Ms(start - poc_queued).count(), Ms(end - start).count(),
             poc_handoffs.load(std::memory_order_relaxed), poc_gpl_ms[0], poc_gpl_ms[1],
             poc_gpl_ms[2], Common::ComputeStructHash64(state.vertex_layout),
             static_cast<u32>(state.attachments.color), static_cast<u32>(state.attachments.depth),
             state.attachments.sample_count, state.blending.blend_enable,
             static_cast<u32>(state.blending.logic_op), state.shader_ids[2]);
}

} // namespace Vulkan
