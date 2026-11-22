// Copyright 2026 Citra Emulator Project / Azahar Emulator Project
// Licensed under GPLv3 or any later version
// Refer to the LICENSE.txt file included.

#pragma once

#include <functional>
#include <optional>
#include <span>
#include "common/common_types.h"
#include "video_core/pica/output_vertex.h"

namespace Pica {

struct RasterizerRegs;
struct RegsInternal;
struct ShaderRegs;
struct ShaderSetup;
class ShaderEngine;

/// Parallel CPU vertex shading settings, read from the environment.
struct VertexThreadsConfig {
    u32 threads = 0;     ///< Worker threads; 0 shades on the emu thread only
    u32 min_unique = 64; ///< Fewest unique vertices worth a parallel run
    bool check = false;  ///< Compare each parallel draw against the serial loop
    int nice = 0;
    u32 spin_us = 50;
};

/// Null or empty strings keep the defaults. threads 1 means 0, and threads is capped at 4.
VertexThreadsConfig ParseVertexThreadsConfig(const char* threads, const char* min_unique,
                                             const char* check, const char* nice,
                                             const char* spin_us);

/// Calls ParseVertexThreadsConfig() with the five environment variables.
VertexThreadsConfig ReadVertexThreadsConfig();

inline constexpr u32 VertexChunkSize = 32;

inline u32 NumVertexChunks(u32 num_unique) {
    return (num_unique + VertexChunkSize - 1) / VertexChunkSize;
}

/// Everything ShadeChunk() may touch. Workers never reach into Memory::MemorySystem.
struct ShadeDraw {
    const ShaderEngine* engine;
    const ShaderSetup* setup;
    const ShaderRegs* vs_regs;
    const RasterizerRegs* rasterizer_regs;
    std::span<const u32> unique;     ///< Unique vertex ids, in first-use order
    std::span<OutputVertex> outputs; ///< One per unique vertex
    std::function<void(u32 vertex, AttributeBuffer& input)> load;
};

/// Shades the unique vertices [chunk * VertexChunkSize, +VertexChunkSize) with a fresh shader
/// unit, so the result doesn't depend on which thread runs which chunk.
void ShadeChunk(const ShadeDraw& draw, u32 chunk);

enum OutputField : u32 {
    FieldPos = 1,
    FieldQuat = 2,
    FieldColor = 4,
    FieldTc0 = 8,
    FieldTc1 = 16,
    FieldTc0W = 32,
    FieldView = 64,
    FieldTc2 = 128,
};

/// The OutputVertex fields the rasterizer reads under the current registers.
u32 UsedOutputFields(const RegsInternal& regs);

/// First used field whose bytes differ, if any.
std::optional<OutputField> FirstMismatch(const OutputVertex& serial, const OutputVertex& parallel,
                                         u32 used);

const char* OutputFieldName(OutputField field);

} // namespace Pica
