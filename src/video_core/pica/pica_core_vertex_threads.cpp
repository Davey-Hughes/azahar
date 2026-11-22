// Copyright 2026 Citra Emulator Project / Azahar Emulator Project
// Licensed under GPLv3 or any later version
// Refer to the LICENSE.txt file included.

#include <array>
#include <chrono>
#include <initializer_list>
#include <string>
#include <vector>
#include <fmt/format.h>
#include "common/logging/log.h"
#include "core/memory.h"
#include "video_core/pica/pica_core.h"
#include "video_core/pica/vertex_dedupe.h"
#include "video_core/pica/vertex_loader.h"
#include "video_core/pica/vertex_threads.h"
#include "video_core/pica/vertex_worker_pool.h"
#include "video_core/rasterizer_interface.h"
#include "video_core/shader/shader.h"

namespace Pica {

namespace {
constexpr u32 MaxMismatchLines = 50;
constexpr u64 CheckSummaryEvery = 1000;

/// The field's components as floats, for the check-mode mismatch line.
std::string FieldText(const OutputVertex& v, OutputField field) {
    const auto join = [](std::initializer_list<f24> values) {
        std::string text;
        for (const f24 value : values) {
            if (!text.empty()) {
                text += ',';
            }
            text += fmt::format("{}", value.ToFloat32());
        }
        return text;
    };
    switch (field) {
    case FieldPos:
        return join({v.pos.x, v.pos.y, v.pos.z, v.pos.w});
    case FieldQuat:
        return join({v.quat.x, v.quat.y, v.quat.z, v.quat.w});
    case FieldColor:
        return join({v.color.x, v.color.y, v.color.z, v.color.w});
    case FieldTc0:
        return join({v.tc0.x, v.tc0.y});
    case FieldTc1:
        return join({v.tc1.x, v.tc1.y});
    case FieldTc0W:
        return join({v.tc0_w});
    case FieldView:
        return join({v.view.x, v.view.y, v.view.z});
    case FieldTc2:
        return join({v.tc2.x, v.tc2.y});
    }
    return {};
}
} // Anonymous namespace

struct PicaCore::VertexThreadsState {
    explicit VertexThreadsState(const VertexThreadsConfig& config_)
        : config{config_},
          pool{config.threads - 1, config.nice, std::chrono::microseconds{config.spin_us}} {}

    ~VertexThreadsState() {
        if (config.check) {
            LOG_INFO(HW_GPU, "VertexThreads check: checked_draws={} mismatched_draws={}",
                     checked_draws, mismatched_draws);
        }
    }

    VertexThreadsConfig config;
    VertexWorkerPool pool;
    VertexDedupe dedupe;
    std::vector<OutputVertex> outputs;       ///< One per unique vertex
    std::vector<OutputVertex> check_outputs; ///< One per index, from ShadeSerialReference()
    u64 checked_draws = 0;
    u64 mismatched_draws = 0;
    u32 mismatch_lines = 0;
};

void PicaCore::VertexThreadsDeleter::operator()(VertexThreadsState* state) const {
    delete state;
}

void PicaCore::InitVertexThreads() {
    const VertexThreadsConfig config = ReadVertexThreadsConfig();
    if (config.threads < 2) {
        return;
    }
    if constexpr (!VertexWorkerFpControlSupported) {
        LOG_WARNING(HW_GPU,
                    "VertexThreads: this build can't copy the floating-point mode to workers, "
                    "leaving parallel vertex shading off");
        return;
    }
    LOG_INFO(HW_GPU, "VertexThreads: threads={} min={} check={} nice={} spin_us={}", config.threads,
             config.min_unique, config.check ? 1 : 0, config.nice, config.spin_us);
    vertex_threads.reset(new VertexThreadsState(config));
}

bool PicaCore::LoadVerticesParallel(bool is_indexed) {
    auto& state = *vertex_threads;
    const auto& pipeline = regs.internal.pipeline;
    // use_gs is a 2-bit field; only No makes GeometryPipeline::Reconfigure() drop its backend.
    if (pipeline.use_gs != PipelineRegs::UseGS::No || debug_context) {
        return false;
    }
    const PAddr base_address = pipeline.vertex_attributes.GetPhysicalBaseAddress();
    const auto& index_info = pipeline.index_array;
    const u8* index_address_8 = memory.GetPhysicalPointer(base_address + index_info.offset);
    if (index_address_8 == nullptr) {
        return false;
    }

    // Vertex IDs as LoadVertices() computes them: non-indexed IDs may exceed 65535. A range that
    // wraps past u32 would leave IDs above MaxVertex(), which ResolveArrays() never checked.
    const u32 num_vertices = pipeline.num_vertices;
    if (!is_indexed && u64{pipeline.vertex_offset} + num_vertices > 0x1'0000'0000ull) {
        return false;
    }
    auto& dedupe = state.dedupe;
    if (!is_indexed) {
        dedupe.BuildRange(pipeline.vertex_offset, num_vertices);
    } else if (index_info.format != 0) {
        dedupe.BuildIndexed(std::span{reinterpret_cast<const u16*>(index_address_8), num_vertices});
    } else {
        dedupe.BuildIndexed(std::span{index_address_8, num_vertices});
    }
    const std::span<const u32> unique = dedupe.Unique();
    if (unique.size() < state.config.min_unique) {
        return false;
    }
    const auto loader = VertexLoader(memory, pipeline);
    VertexArrays arrays;
    if (!loader.ResolveArrays(base_address, dedupe.MaxVertex(), arrays)) {
        return false;
    }

    // From here on as LoadVertices() before its loop.
    regs.internal.rasterizer.ValidateSemantics();
    shader_engine->SetupBatch(vs_setup, regs.internal.vs.main_offset);
    geometry_pipeline.Reconfigure();
    geometry_pipeline.Setup(shader_engine.get());

    // Workers read memory only through arrays, and nothing they read changes until Run() returns.
    if (state.outputs.size() < unique.size()) {
        state.outputs.resize(unique.size());
    }
    const std::span<OutputVertex> outputs{state.outputs.data(), unique.size()};
    const ShadeDraw draw{shader_engine.get(),
                         &vs_setup,
                         &regs.internal.vs,
                         &regs.internal.rasterizer,
                         unique,
                         outputs,
                         [&](u32 vertex, AttributeBuffer& input) {
                             loader.LoadVertex(arrays, vertex, input, input_default_attributes);
                         }};
    state.pool.Run(NumVertexChunks(static_cast<u32>(unique.size())),
                   [&draw](u32 chunk) { ShadeChunk(draw, chunk); });

    const std::span<const u32> slots = dedupe.Slots();
    if (state.config.check) {
        if (state.check_outputs.size() < num_vertices) {
            state.check_outputs.resize(num_vertices);
        }
        const std::span<OutputVertex> serial{state.check_outputs.data(), num_vertices};
        ShadeSerialReference(is_indexed, serial);
        const u32 used = UsedOutputFields(regs.internal);
        const u64 draw_id = state.checked_draws++;
        for (u32 index = 0; index < num_vertices; ++index) {
            const OutputVertex& parallel = outputs[slots[index]];
            const auto field = FirstMismatch(serial[index], parallel, used);
            if (!field) {
                continue;
            }
            ++state.mismatched_draws;
            if (state.mismatch_lines < MaxMismatchLines) {
                ++state.mismatch_lines;
                LOG_WARNING(HW_GPU,
                            "VertexThreads check: draw={} index={} vertex={} field={} serial={} "
                            "parallel={} vs_hash={:016x}",
                            draw_id, index, unique[slots[index]], OutputFieldName(*field),
                            FieldText(serial[index], *field), FieldText(parallel, *field),
                            vs_setup.GetProgramCodeHash());
            }
            break;
        }
        if (state.checked_draws % CheckSummaryEvery == 0) {
            LOG_INFO(HW_GPU, "VertexThreads check: checked_draws={} mismatched_draws={}",
                     state.checked_draws, state.mismatched_draws);
        }
    }

    // The calls the submit_vertex handler in PicaCore::PicaCore() makes, in index order.
    const PrimitiveAssembler::TriangleHandler add_triangle =
        [this](const OutputVertex& v0, const OutputVertex& v1, const OutputVertex& v2) {
            rasterizer->AddTriangle(v0, v1, v2);
        };
    for (u32 index = 0; index < num_vertices; ++index) {
        primitive_assembler.SubmitVertex(outputs[slots[index]], add_triangle);
    }
    return true;
}

// A copy of the PicaCore::LoadVertices() loop. LoadVerticesParallel() has already validated the
// semantics, checked the index pointer and set up the shader and the geometry pipeline (no
// geometry shader, no debug context). After a rebase, diff it against LoadVertices() again:
// check mode is only as good as this copy.
void PicaCore::ShadeSerialReference(bool is_indexed, std::span<OutputVertex> out) {
    // Read and validate vertex information from the loaders
    const auto& pipeline = regs.internal.pipeline;
    const PAddr base_address = pipeline.vertex_attributes.GetPhysicalBaseAddress();
    const auto loader = VertexLoader(memory, pipeline);

    // Locate index buffer.
    const auto& index_info = pipeline.index_array;
    const u8* index_address_8 = memory.GetPhysicalPointer(base_address + index_info.offset);
    const u16* index_address_16 = reinterpret_cast<const u16*>(index_address_8);
    const bool index_u16 = index_info.format != 0;

    // Simple circular-replacement vertex cache
    const std::size_t VERTEX_CACHE_SIZE = 64;
    std::array<bool, VERTEX_CACHE_SIZE> vertex_cache_valid{};
    std::array<u16, VERTEX_CACHE_SIZE> vertex_cache_ids;
    std::array<AttributeBuffer, VERTEX_CACHE_SIZE> vertex_cache;
    u32 vertex_cache_pos = 0;

    ShaderUnit shader_unit;
    AttributeBuffer vs_output;

    for (u32 index = 0; index < pipeline.num_vertices; ++index) {
        // Indexed rendering doesn't use the start offset
        const u32 vertex = is_indexed
                               ? (index_u16 ? index_address_16[index] : index_address_8[index])
                               : (index + pipeline.vertex_offset);

        bool vertex_cache_hit = false;
        if (is_indexed) {
            for (u32 i = 0; i < VERTEX_CACHE_SIZE; ++i) {
                if (vertex_cache_valid[i] && vertex == vertex_cache_ids[i]) {
                    vs_output = vertex_cache[i];
                    vertex_cache_hit = true;
                    break;
                }
            }
        }

        if (!vertex_cache_hit) {
            // Initialize data for the current vertex
            AttributeBuffer input;
            loader.LoadVertex(base_address, index, vertex, input, input_default_attributes);

            // Invoke the vertex shader for this vertex.
            shader_unit.LoadInput(regs.internal.vs, input);
            shader_engine->Run(vs_setup, shader_unit);
            shader_unit.WriteOutput(regs.internal.vs, vs_output);

            // Cache the vertex when doing indexed rendering.
            if (is_indexed) {
                vertex_cache[vertex_cache_pos] = vs_output;
                vertex_cache_valid[vertex_cache_pos] = true;
                vertex_cache_ids[vertex_cache_pos] = vertex;
                vertex_cache_pos = (vertex_cache_pos + 1) % VERTEX_CACHE_SIZE;
            }
        }

        // Today's loop submits vs_output here; the submit handler builds this OutputVertex.
        out[index] = OutputVertex(regs.internal.rasterizer, vs_output);
    }
}

} // namespace Pica
