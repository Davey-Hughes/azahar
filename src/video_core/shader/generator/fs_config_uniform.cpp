// SPDX-FileCopyrightText: Azahar Emulator Project
// Copyright(c) 2026: PalindromicBreadLoaf(palindromicbreadloaf@tuta.com)
// SPDX-License-Identifier: GPL-2.0-or-later

#include <utility>
#include "video_core/pica/regs_rasterizer.h"
#include "video_core/shader/generator/fs_config_uniform.h"
#include "video_core/shader/generator/pica_fs_config.h"

namespace Pica::Shader {

namespace {

using TevStageConfig = Pica::TexturingRegs::TevStageConfig;

u32 UberTevInputOf(TevStageConfig::Source source) {
    using Source = TevStageConfig::Source;
    switch (source) {
    case Source::PrimaryColor:
        return UberTevPrimaryColor;
    case Source::PrimaryFragmentColor:
        return UberTevPrimaryFragmentColor;
    case Source::SecondaryFragmentColor:
        return UberTevSecondaryFragmentColor;
    case Source::Texture0:
    case Source::Texture1:
    case Source::Texture2:
    case Source::Texture3:
        return UberTevTexture0 + static_cast<u32>(source) - static_cast<u32>(Source::Texture0);
    case Source::PreviousBuffer:
        return UberTevPreviousBuffer;
    case Source::Constant:
        return UberTevConstant;
    case Source::Previous:
        return UberTevPrevious;
    }
    return UberTevZero;
}

/// The source components a color modifier reads into r, g and b (2 bits each), and its 2-bit
/// signed modifier (see FSConfigUniformData::tev).
std::pair<u32, u32> PackColorModifier(TevStageConfig::ColorModifier modifier) {
    using ColorModifier = TevStageConfig::ColorModifier;
    constexpr auto splat = [](u32 component) { return component * 0b010101u; };
    constexpr u32 keep = 0b01;
    constexpr u32 one_minus = 0b11;
    switch (modifier) {
    case ColorModifier::SourceColor:
        return {0 | 1 << 2 | 2 << 4, keep};
    case ColorModifier::OneMinusSourceColor:
        return {0 | 1 << 2 | 2 << 4, one_minus};
    case ColorModifier::SourceAlpha:
        return {splat(3), keep};
    case ColorModifier::OneMinusSourceAlpha:
        return {splat(3), one_minus};
    case ColorModifier::SourceRed:
        return {splat(0), keep};
    case ColorModifier::OneMinusSourceRed:
        return {splat(0), one_minus};
    case ColorModifier::SourceGreen:
        return {splat(1), keep};
    case ColorModifier::OneMinusSourceGreen:
        return {splat(1), one_minus};
    case ColorModifier::SourceBlue:
        return {splat(2), keep};
    case ColorModifier::OneMinusSourceBlue:
        return {splat(2), one_minus};
    }
    return {0, 0};
}

/// The source component an alpha modifier reads, and its 2-bit signed modifier.
std::pair<u32, u32> PackAlphaModifier(TevStageConfig::AlphaModifier modifier) {
    // SourceAlpha, SourceRed, SourceGreen, SourceBlue, each followed by its one-minus form.
    constexpr std::array<u32, 4> components{3, 0, 1, 2};
    const u32 value = static_cast<u32>(modifier);
    return {components[value >> 1], (value & 1) != 0 ? 0b11u : 0b01u};
}

/// log2 of the multiplier a 2-bit TEV scale selects (scale 3 means 1, like 0).
u32 MultiplierShift(u32 scale) {
    return scale < 3 ? scale : 0;
}

Common::Vec4u PackTevStage(const TevStageConfig& stage, u32 index, u32 buffer_mask) {
    using Source = TevStageConfig::Source;
    std::array color_sources{stage.color_source1.Value(), stage.color_source2.Value(),
                             stage.color_source3.Value()};
    std::array alpha_sources{stage.alpha_source1.Value(), stage.alpha_source2.Value(),
                             stage.alpha_source3.Value()};
    // In the first stage, Previous in inputs 1 and 2 reads what input 3 reads.
    if (index == 0) {
        for (u32 j = 0; j < 2; ++j) {
            if (color_sources[j] == Source::Previous) {
                color_sources[j] = color_sources[2];
            }
            if (alpha_sources[j] == Source::Previous) {
                alpha_sources[j] = alpha_sources[2];
            }
        }
    }
    const std::array color_modifiers{stage.color_modifier1.Value(), stage.color_modifier2.Value(),
                                     stage.color_modifier3.Value()};
    const std::array alpha_modifiers{stage.alpha_modifier1.Value(), stage.alpha_modifier2.Value(),
                                     stage.alpha_modifier3.Value()};

    Common::Vec4u out{};
    for (u32 j = 0; j < 3; ++j) {
        out.x |= UberTevInputOf(color_sources[j]) << (4 * j);
        out.x |= UberTevInputOf(alpha_sources[j]) << (16 + 4 * j);
        const auto [color_components, color_modifier] = PackColorModifier(color_modifiers[j]);
        const auto [alpha_component, alpha_modifier] = PackAlphaModifier(alpha_modifiers[j]);
        out.y |= color_components << (8 * j);
        out.y |= alpha_component << (24 + 2 * j);
        out.z |= color_modifier << (8 + 2 * j);
        out.z |= alpha_modifier << (14 + 2 * j);
    }
    out.x |= static_cast<u32>(stage.color_op.Value()) << 12;
    out.x |= static_cast<u32>(stage.alpha_op.Value()) << 28;

    u32 flags = 0;
    if (!IsPassthroughTevStage(stage)) {
        flags |= UberTevActive;
    }
    if (stage.color_op == TevStageConfig::Operation::Dot3_RGBA) {
        flags |= UberTevDot3Rgba;
    }
    // Only stages 0-3 can write the combiner buffer: bits 0-3 color, bits 4-7 alpha.
    if (index < 4 && (buffer_mask & (1u << index)) != 0) {
        flags |= UberTevBufferColor;
    }
    if (index < 4 && (buffer_mask & (1u << (index + 4))) != 0) {
        flags |= UberTevBufferAlpha;
    }
    out.z |= flags;
    out.z |= MultiplierShift(stage.color_scale) << 4;
    out.z |= MultiplierShift(stage.alpha_scale) << 6;
    out.z |= TevOperationInputs(stage.color_op) << 20;
    out.z |= TevAlphaInputs(stage) << 22;
    return out;
}

/// Texture units the ubershader samples: those an active TEV stage reads, and the bump and shadow
/// maps of lighting.
u32 UsedTextureUnits(const FSConfig& config) {
    using Source = TevStageConfig::Source;
    u32 used = 0;
    for (const TevStageConfig stage : config.texture.tev_stages) {
        if (IsPassthroughTevStage(stage)) {
            continue;
        }
        for (const Source source : {stage.color_source1.Value(), stage.color_source2.Value(),
                                    stage.color_source3.Value(), stage.alpha_source1.Value(),
                                    stage.alpha_source2.Value(), stage.alpha_source3.Value()}) {
            if (source >= Source::Texture0 && source <= Source::Texture3) {
                used |= 1u << (static_cast<u32>(source) - static_cast<u32>(Source::Texture0));
            }
        }
    }
    const auto& lighting = config.lighting;
    if (lighting.enable) {
        using BumpMode = Pica::LightingRegs::LightingBumpMode;
        if (lighting.bump_mode == BumpMode::NormalMap ||
            lighting.bump_mode == BumpMode::TangentMap) {
            used |= 1u << lighting.bump_selector;
        }
        if (lighting.enable_shadow) {
            used |= 1u << lighting.shadow_selector;
        }
    }
    return used;
}

} // namespace

FSConfigUniformData BuildFSConfigUniform(const FSConfig& config) {
    const auto& fb = config.framebuffer;
    const auto& tex = config.texture;
    const auto& lighting = config.lighting;

    FSConfigUniformData data{};
    data.alpha_test_func = static_cast<u32>(fb.alpha_test_func.Value());
    data.scissor_mode = static_cast<u32>(fb.scissor_test_mode.Value());
    data.depthmap_enable =
        fb.depthmap_enable == Pica::RasterizerRegs::DepthBuffering::WBuffering ? 1u : 0u;
    data.texture0_type = static_cast<u32>(tex.texture0_type.Value());
    data.texture2_use_coord1 = tex.texture2_use_coord1.Value();
    data.combiner_buffer_mask = tex.combiner_buffer_input.Value();
    data.fog_mode = static_cast<u32>(tex.fog_mode.Value());
    data.fog_flip = tex.fog_flip.Value();
    data.lighting = lighting.raw;
    data.proctex_enable = config.proctex.enable.Value();
    data.shadow_flags = tex.shadow_texture_orthographic.Value() |
                        (fb.shadow_rendering.Value() << 1);

    // bit (unit * 2) = enable_s, bit (unit * 2 + 1) = enable_t.
    data.tex_border = 0;
    for (u32 unit = 0; unit < 3; ++unit) {
        if (tex.texture_border_color[unit].enable_s) {
            data.tex_border |= 1u << (unit * 2);
        }
        if (tex.texture_border_color[unit].enable_t) {
            data.tex_border |= 1u << (unit * 2 + 1);
        }
    }

    for (u32 i = 0; i < tex.tev_stages.size(); ++i) {
        data.tev[i] = PackTevStage(tex.tev_stages[i], i, tex.combiner_buffer_input);
    }
    data.tex_used = UsedTextureUnits(config);

    for (u32 i = 0; i < lighting.lights.size(); ++i) {
        data.light[i >> 2][i & 3] = lighting.lights[i].raw;
    }

    const std::array luts = {&lighting.lut_d0, &lighting.lut_d1, &lighting.lut_sp, &lighting.lut_fr,
                             &lighting.lut_rr, &lighting.lut_rg, &lighting.lut_rb};
    for (u32 i = 0; i < luts.size(); ++i) {
        data.lut_raw[i >> 2][i & 3] = luts[i]->raw;
        data.lut_scale[i >> 2][i & 3] = luts[i]->GetScale();
    }

    return data;
}

UbershaderGap FindUbershaderGap(const FSConfig& config, const UserConfig& user) {
    using TextureType = Pica::TexturingRegs::TextureConfig::TextureType;

    switch (config.texture.texture0_type.Value()) {
    case TextureType::TextureCube:
        return UbershaderGap::CubeTexture;
    case TextureType::ShadowCube:
        return UbershaderGap::ShadowCubeTexture;
    default:
        break;
    }
    if (config.proctex.enable.Value()) {
        return UbershaderGap::ProcTex;
    }
    if (config.framebuffer.requested_rgb_blend.RequiresMinMaxEmulation() ||
        config.framebuffer.requested_alpha_blend.RequiresMinMaxEmulation()) {
        return UbershaderGap::MinMaxBlend;
    }
    // FramebufferConfig::ApplyProfile() only moves the requested op into logic_op when the shader
    // has to emulate it, so anything other than Copy means emulation.
    if (config.framebuffer.logic_op.Value() != Pica::FramebufferRegs::LogicOp::Copy) {
        return UbershaderGap::LogicOp;
    }
    if (user.use_custom_normal.Value()) {
        return UbershaderGap::CustomNormal;
    }
    return UbershaderGap::None;
}

const char* UbershaderGapName(UbershaderGap gap) {
    switch (gap) {
    case UbershaderGap::None:
        return "none";
    case UbershaderGap::CubeTexture:
        return "cube_texture";
    case UbershaderGap::Shadow2DTexture:
        return "shadow_2d_texture";
    case UbershaderGap::ShadowCubeTexture:
        return "shadow_cube_texture";
    case UbershaderGap::ProcTex:
        return "proctex";
    case UbershaderGap::LightingShadow:
        return "lighting_shadow";
    case UbershaderGap::ShadowRendering:
        return "shadow_rendering";
    case UbershaderGap::MinMaxBlend:
        return "min_max_blend";
    case UbershaderGap::LogicOp:
        return "logic_op";
    case UbershaderGap::CustomNormal:
        return "custom_normal";
    case UbershaderGap::Count:
        break;
    }
    return "unknown";
}

} // namespace Pica::Shader
