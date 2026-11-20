// SPDX-FileCopyrightText: Azahar Emulator Project
// Copyright(c) 2026: PalindromicBreadLoaf(palindromicbreadloaf@tuta.com)
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <array>
#include "common/common_types.h"
#include "common/vector_math.h"

namespace Pica::Shader {

struct FSConfig;
union UserConfig;

/// Where a TEV input comes from in the ubershader. The packed stages name these instead of PICA's
/// TevStageConfig::Source values so the shader selects with one small switch.
enum UberTevInput : u32 {
    UberTevPrimaryColor = 0,
    UberTevPrimaryFragmentColor = 1,
    UberTevSecondaryFragmentColor = 2,
    UberTevTexture0 = 3, // Texture1-3 follow
    UberTevPreviousBuffer = 7,
    UberTevConstant = 8,
    UberTevPrevious = 9,
    UberTevZero = 10, // a Source value the hardware doesn't define
};

/// Bits 0-3 of a packed TEV stage's z word (see FSConfigUniformData::tev).
enum UberTevFlags : u32 {
    UberTevActive = 1 << 0,      // not a passthrough stage (IsPassthroughTevStage())
    UberTevDot3Rgba = 1 << 1,    // the color result also becomes the alpha result
    UberTevBufferColor = 1 << 2, // the stage writes its color to the combiner buffer
    UberTevBufferAlpha = 1 << 3, // the stage writes its alpha to the combiner buffer
};

/**
 * CPU-side mirror of the `fs_config` std140 uniform block read by the PICA fragment ubershader
 * (src/video_core/host_shaders/vulkan_ubershader.frag). One generic pipeline can then emulate any
 * fragment state, so a draw whose specialized pipeline is still compiling can render right away.
 */
struct FSConfigUniformData {
    u32 alpha_test_func;      // FramebufferRegs::CompareFunc
    u32 scissor_mode;         // RasterizerRegs::ScissorMode
    u32 depthmap_enable;      // 0 = Z-buffering, 1 = W-buffering
    u32 texture0_type;        // TexturingRegs::TextureConfig::TextureType
    u32 texture2_use_coord1;
    u32 combiner_buffer_mask; // tev_combiner_buffer_input (8 bits)
    u32 fog_mode;             // TexturingRegs::FogMode
    u32 fog_flip;
    u32 lighting;             // LightConfig.raw
    u32 proctex_enable;
    u32 tex_border;           // 3 units x (enable_s | enable_t << 1)
    u32 shadow_flags;         // bit 0 = shadow_texture_orthographic, bit 1 = shadow_rendering
    u32 tex_used;             // bit n = texture unit n is read by an active TEV stage or lighting
    // The TEV stages, decoded on the CPU so the ubershader's per-fragment loop does no PICA
    // register unpacking, only one bitfield extract per field. Input j of a stage is 0-2.
    // x: bits 4j-4j+3 = UberTevInput of color input j, 12-15 = color op,
    //    16+4j-19+4j = UberTevInput of alpha input j, 28-31 = alpha op
    // y: bits 8j-8j+5 = the source components color input j reads into r, g, b (2 bits each),
    //    24+2j-25+2j = the component alpha input j reads
    // z: bits 0-3 = UberTevFlags, 4-5 = log2 of the color multiplier, 6-7 = same for alpha,
    //    8+2j-9+2j = color input j's modifier as a 2-bit signed int: 1 keeps the component, -1
    //    takes one minus it, 0 (a modifier the hardware doesn't define) gives 0,
    //    14+2j-15+2j = same for alpha input j, 20-21 / 22-23 = how many color / alpha inputs
    //    the ops read
    alignas(16) std::array<Common::Vec4u, 6> tev;
    // 8 packed Light.raw words: light[i >> 2][i & 3]
    alignas(16) std::array<Common::Vec4u, 2> light;
    // 7 packed LutConfig.raw words (d0, d1, sp, fr, rr, rg, rb)
    alignas(16) std::array<Common::Vec4u, 2> lut_raw;
    // Matching 7 LUT scales
    alignas(16) std::array<Common::Vec4f, 2> lut_scale;
};

static_assert(sizeof(FSConfigUniformData) == 256,
              "FSConfigUniformData does not match the fs_config block in the ubershader");

/// Packs a fragment configuration into the fs_config block. Call FSConfig::ApplyProfile() first.
[[nodiscard]] FSConfigUniformData BuildFSConfigUniform(const FSConfig& config);

/// A fragment feature the ubershader does not implement. A draw that needs one must wait for its
/// specialized shader.
enum class UbershaderGap : u8 {
    None,
    CubeTexture,
    Shadow2DTexture,
    ShadowCubeTexture,
    ProcTex,
    LightingShadow,
    ShadowRendering,
    MinMaxBlend,
    LogicOp,
    CustomNormal,
    Count,
};

/// Returns the first feature in the config that the ubershader lacks, or UbershaderGap::None.
/// Call FSConfig::ApplyProfile() first.
[[nodiscard]] UbershaderGap FindUbershaderGap(const FSConfig& config, const UserConfig& user);

/// Short name for logs, such as "proctex".
[[nodiscard]] const char* UbershaderGapName(UbershaderGap gap);

/// Returns false when the config uses a feature the ubershader does not implement, in which case
/// the draw must wait for its specialized shader. Call FSConfig::ApplyProfile() first.
[[nodiscard]] inline bool UbershaderSupports(const FSConfig& config, const UserConfig& user) {
    return FindUbershaderGap(config, user) == UbershaderGap::None;
}

} // namespace Pica::Shader
