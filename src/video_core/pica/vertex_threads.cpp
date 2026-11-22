// Copyright 2026 Citra Emulator Project / Azahar Emulator Project
// Licensed under GPLv3 or any later version
// Refer to the LICENSE.txt file included.

#include <algorithm>
#include <charconv>
#include <cstdlib>
#include <cstring>
#include "video_core/pica/regs_internal.h"
#include "video_core/pica/shader_setup.h"
#include "video_core/pica/shader_unit.h"
#include "video_core/pica/vertex_threads.h"
#include "video_core/shader/shader.h"

namespace Pica {

namespace {
/// The whole string must be a decimal number that fits T; anything else (empty, trailing junk,
/// a sign the type can't hold, overflow) gives the fallback.
template <typename T>
T ParseNumber(const char* s, T fallback) {
    if (s == nullptr || *s == '\0') {
        return fallback;
    }
    const char* const end = s + std::strlen(s);
    T value{};
    const auto [ptr, ec] = std::from_chars(s, end, value, 10);
    return ec == std::errc{} && ptr == end ? value : fallback;
}

u32 ParseU32(const char* s, u32 fallback) {
    return ParseNumber<u32>(s, fallback);
}
} // namespace

VertexThreadsConfig ParseVertexThreadsConfig(const char* threads, const char* min_unique,
                                             const char* check, const char* nice,
                                             const char* spin_us) {
    VertexThreadsConfig config;
    config.threads = std::min(ParseU32(threads, 0), 4u);
    if (config.threads == 1) {
        config.threads = 0;
    }
    config.min_unique = ParseU32(min_unique, config.min_unique);
    config.check = check != nullptr && std::strcmp(check, "1") == 0;
    config.nice = ParseNumber<int>(nice, config.nice);
    config.spin_us = ParseU32(spin_us, config.spin_us);
    return config;
}

VertexThreadsConfig ReadVertexThreadsConfig() {
    return ParseVertexThreadsConfig(
        std::getenv("AZAHAR_VERTEX_THREADS"), std::getenv("AZAHAR_VERTEX_THREADS_MIN"),
        std::getenv("AZAHAR_VERTEX_THREADS_CHECK"), std::getenv("AZAHAR_VERTEX_THREADS_NICE"),
        std::getenv("AZAHAR_VERTEX_THREADS_SPIN_US"));
}

void ShadeChunk(const ShadeDraw& draw, u32 chunk) {
    ShaderUnit shader_unit;
    AttributeBuffer vs_output{};
    const u32 begin = chunk * VertexChunkSize;
    const u32 end = std::min<u32>(begin + VertexChunkSize, static_cast<u32>(draw.unique.size()));
    for (u32 k = begin; k < end; ++k) {
        AttributeBuffer input{};
        draw.load(draw.unique[k], input);
        shader_unit.LoadInput(*draw.vs_regs, input);
        draw.engine->Run(*draw.setup, shader_unit);
        shader_unit.WriteOutput(*draw.vs_regs, vs_output);
        draw.outputs[k] = OutputVertex(*draw.rasterizer_regs, vs_output);
    }
}

u32 UsedOutputFields(const RegsInternal& regs) {
    u32 used = FieldPos | FieldColor;
    if (!regs.lighting.disable) {
        used |= FieldQuat | FieldView;
    }
    const auto& main = regs.texturing.main_config;
    if (main.texture0_enable) {
        used |= FieldTc0 | FieldTc0W;
    }
    if (main.texture3_enable) {
        // Procedural texturing reads only abs(texcoord{texture3_coordinates}), a vec2 (see
        // AppendProcTexSampler() in glsl_fs_shader_gen.cpp), so tc0_w is not implied.
        switch (main.texture3_coordinates) {
        case 0:
            used |= FieldTc0;
            break;
        case 1:
            used |= FieldTc1;
            break;
        case 2:
            used |= FieldTc2;
            break;
        default:
            // The shader generator falls back to texcoord0.
            used |= FieldTc0;
            break;
        }
    }
    if (main.texture1_enable || (main.texture2_enable && main.texture2_use_coord1)) {
        used |= FieldTc1;
    }
    if (main.texture2_enable) {
        used |= FieldTc2;
    }
    return used;
}

std::optional<OutputField> FirstMismatch(const OutputVertex& serial, const OutputVertex& parallel,
                                         u32 used) {
    const auto differs = [&](const auto& a, const auto& b) {
        return std::memcmp(&a, &b, sizeof(a)) != 0;
    };
    if ((used & FieldPos) && differs(serial.pos, parallel.pos)) {
        return FieldPos;
    }
    if ((used & FieldQuat) && differs(serial.quat, parallel.quat)) {
        return FieldQuat;
    }
    if ((used & FieldColor) && differs(serial.color, parallel.color)) {
        return FieldColor;
    }
    if ((used & FieldTc0) && differs(serial.tc0, parallel.tc0)) {
        return FieldTc0;
    }
    if ((used & FieldTc1) && differs(serial.tc1, parallel.tc1)) {
        return FieldTc1;
    }
    if ((used & FieldTc0W) && differs(serial.tc0_w, parallel.tc0_w)) {
        return FieldTc0W;
    }
    if ((used & FieldView) && differs(serial.view, parallel.view)) {
        return FieldView;
    }
    if ((used & FieldTc2) && differs(serial.tc2, parallel.tc2)) {
        return FieldTc2;
    }
    return std::nullopt;
}

const char* OutputFieldName(OutputField field) {
    switch (field) {
    case FieldPos:
        return "pos";
    case FieldQuat:
        return "quat";
    case FieldColor:
        return "color";
    case FieldTc0:
        return "tc0";
    case FieldTc1:
        return "tc1";
    case FieldTc0W:
        return "tc0_w";
    case FieldView:
        return "view";
    case FieldTc2:
        return "tc2";
    }
    return "?";
}

} // namespace Pica
