// Copyright 2026 Citra Emulator Project / Azahar Emulator Project
// Licensed under GPLv2 or any later version
// Refer to the misc/licenses/gplv2.txt file included.

#include <utility>
#include <catch2/catch_test_macros.hpp>
#include "video_core/pica/regs_internal.h"
#include "video_core/shader/generator/fs_config_uniform.h"
#include "video_core/shader/generator/pica_fs_config.h"
#include "video_core/shader/generator/profile.h"

using namespace Pica::Shader;
using TextureType = Pica::TexturingRegs::TextureConfig::TextureType;

namespace {
FSConfig MakeConfig() {
    static const Pica::RegsInternal regs{};
    Profile profile{};
    profile.has_logic_op = true;
    profile.has_custom_border_color = true;
    profile.is_vulkan = true;
    return FSConfig{regs, profile};
}
} // namespace

TEST_CASE("Ubershader supports a plain config", "[video_core][ubershader]") {
    REQUIRE(UbershaderSupports(MakeConfig(), UserConfig{}));
    REQUIRE(FindUbershaderGap(MakeConfig(), UserConfig{}) == UbershaderGap::None);
}

TEST_CASE("Ubershader rejects features it does not implement", "[video_core][ubershader]") {
    SECTION("procedural textures") {
        auto config = MakeConfig();
        config.proctex.enable.Assign(1);
        REQUIRE(FindUbershaderGap(config, UserConfig{}) == UbershaderGap::ProcTex);
    }
    SECTION("cube and shadow cube texture types") {
        for (const auto [type, gap] :
             {std::pair{TextureType::TextureCube, UbershaderGap::CubeTexture},
              std::pair{TextureType::ShadowCube, UbershaderGap::ShadowCubeTexture}}) {
            auto config = MakeConfig();
            config.texture.texture0_type.Assign(type);
            REQUIRE(FindUbershaderGap(config, UserConfig{}) == gap);
        }
    }
    SECTION("min/max blend emulation") {
        auto config = MakeConfig();
        config.framebuffer.requested_rgb_blend.eq = Pica::FramebufferRegs::BlendEquation::Min;
        REQUIRE(FindUbershaderGap(config, UserConfig{}) == UbershaderGap::MinMaxBlend);
        config = MakeConfig();
        config.framebuffer.requested_alpha_blend.eq = Pica::FramebufferRegs::BlendEquation::Max;
        REQUIRE(FindUbershaderGap(config, UserConfig{}) == UbershaderGap::MinMaxBlend);
    }
    SECTION("logic op emulation") {
        auto config = MakeConfig();
        config.framebuffer.logic_op.Assign(Pica::FramebufferRegs::LogicOp::Clear);
        REQUIRE(FindUbershaderGap(config, UserConfig{}) == UbershaderGap::LogicOp);
    }
    SECTION("custom normal maps") {
        UserConfig user{};
        user.use_custom_normal.Assign(1);
        REQUIRE(FindUbershaderGap(MakeConfig(), user) == UbershaderGap::CustomNormal);
    }
}

TEST_CASE("Ubershader supports shadow maps", "[video_core][ubershader]") {
    SECTION("2D shadow texture") {
        auto config = MakeConfig();
        config.texture.texture0_type.Assign(TextureType::Shadow2D);
        REQUIRE(FindUbershaderGap(config, UserConfig{}) == UbershaderGap::None);
    }
    SECTION("lighting shadows") {
        auto config = MakeConfig();
        config.lighting.enable.Assign(1);
        config.lighting.enable_shadow.Assign(1);
        REQUIRE(FindUbershaderGap(config, UserConfig{}) == UbershaderGap::None);
    }
    SECTION("shadow rendering") {
        auto config = MakeConfig();
        config.framebuffer.shadow_rendering.Assign(1);
        REQUIRE(FindUbershaderGap(config, UserConfig{}) == UbershaderGap::None);
    }
}

TEST_CASE("BuildFSConfigUniform packs the config", "[video_core][ubershader]") {
    static Pica::RegsInternal regs{};
    regs.texturing.texture1.wrap_t.Assign(
        Pica::TexturingRegs::TextureConfig::WrapMode::ClampToBorder);
    Profile profile{}; // No custom border color, so border emulation stays on.
    profile.has_logic_op = true;
    FSConfig config{regs, profile};

    config.texture.fog_mode.Assign(Pica::TexturingRegs::FogMode::Fog);
    config.texture.fog_flip.Assign(1);
    config.texture.tev_stages[2].sources_raw = 0x123;
    config.texture.tev_stages[2].scales_raw = 0x5;
    config.lighting.lights[5].num.Assign(3);
    config.texture.shadow_texture_orthographic.Assign(1);
    config.framebuffer.shadow_rendering.Assign(1);

    const FSConfigUniformData data = BuildFSConfigUniform(config);
    REQUIRE(sizeof(FSConfigUniformData) == 256);
    REQUIRE(data.fog_mode == 5u);
    REQUIRE(data.fog_flip == 1u);
    REQUIRE(data.tex_border == (1u << 3)); // unit 1, enable_t
    // Sources 0x123: Texture0, SecondaryFragmentColor, PrimaryFragmentColor; color scale 1 (x2).
    REQUIRE((data.tev[2].x & 0xfff) == (3u | 2u << 4 | 1u << 8));
    REQUIRE(((data.tev[2].z >> 4) & 3) == 1u);
    REQUIRE(data.light[1][1] == config.lighting.lights[5].raw);
    REQUIRE(data.shadow_flags == 0x3u);
}

namespace {
using TevStageConfig = Pica::TexturingRegs::TevStageConfig;

u32 Bits(u32 value, u32 offset, u32 count) {
    return (value >> offset) & ((1u << count) - 1);
}

/// A packed modifier: 1 keeps the component, -1 takes one minus it, 0 gives 0.
s32 Modifier(u32 value, u32 offset) {
    const u32 bits = Bits(value, offset, 2);
    return bits == 0b11 ? -1 : static_cast<s32>(bits);
}

/// The components a packed color input reads into r, g and b.
std::array<u32, 3> Components(const Common::Vec4u& stage, u32 input) {
    return {Bits(stage.y, 8 * input, 2), Bits(stage.y, 8 * input + 2, 2),
            Bits(stage.y, 8 * input + 4, 2)};
}

TevStageConfigRaw ToRaw(const TevStageConfig& stage) {
    return {stage.sources_raw, stage.modifiers_raw, stage.ops_raw, stage.scales_raw};
}

TevStageConfig PassthroughStage() {
    TevStageConfig stage{};
    stage.color_source1.Assign(TevStageConfig::Source::Previous);
    stage.alpha_source1.Assign(TevStageConfig::Source::Previous);
    return stage;
}
} // namespace

TEST_CASE("BuildFSConfigUniform decodes the TEV stages", "[video_core][ubershader]") {
    using Source = TevStageConfig::Source;
    using ColorModifier = TevStageConfig::ColorModifier;
    using AlphaModifier = TevStageConfig::AlphaModifier;
    using Operation = TevStageConfig::Operation;

    auto config = MakeConfig();
    for (auto& stage : config.texture.tev_stages) {
        stage = ToRaw(PassthroughStage());
    }

    TevStageConfig s0{};
    s0.color_source1.Assign(Source::Previous); // stage 0 reads source 3 instead
    s0.color_source2.Assign(Source::Constant);
    s0.color_source3.Assign(Source::Texture0);
    s0.alpha_source1.Assign(Source::Previous);
    s0.alpha_source2.Assign(Source::PrimaryColor);
    s0.alpha_source3.Assign(Source::Texture1);
    s0.color_modifier1.Assign(ColorModifier::OneMinusSourceAlpha);
    s0.color_modifier2.Assign(static_cast<ColorModifier>(6)); // undefined: reads 0
    s0.color_modifier3.Assign(ColorModifier::SourceGreen);
    s0.alpha_modifier1.Assign(AlphaModifier::OneMinusSourceBlue);
    s0.alpha_modifier2.Assign(AlphaModifier::SourceRed);
    s0.alpha_modifier3.Assign(AlphaModifier::SourceAlpha);
    s0.color_op.Assign(Operation::Modulate);
    s0.alpha_op.Assign(Operation::Lerp);
    s0.color_scale.Assign(2);
    s0.alpha_scale.Assign(3);
    config.texture.tev_stages[0] = ToRaw(s0);

    TevStageConfig s1{};
    s1.color_source1.Assign(Source::Texture2);
    s1.color_source2.Assign(static_cast<Source>(0x7)); // undefined: reads 0
    s1.color_source3.Assign(Source::Previous);
    s1.alpha_source1.Assign(Source::PreviousBuffer);
    s1.color_op.Assign(Operation::Dot3_RGBA);
    config.texture.tev_stages[1] = ToRaw(s1);

    // A passthrough stage reads only Previous, whatever its unused inputs name.
    TevStageConfig s2 = PassthroughStage();
    s2.color_source2.Assign(Source::Texture3);
    config.texture.tev_stages[2] = ToRaw(s2);

    // Stage 1 writes its color to the combiner buffer, stage 0 its alpha.
    config.texture.combiner_buffer_input.Assign(0x12);

    auto data = BuildFSConfigUniform(config);

    const auto& t0 = data.tev[0];
    // Inputs: color Texture0, Constant, Texture0; alpha Texture1, PrimaryColor, Texture1.
    REQUIRE(Bits(t0.x, 0, 12) == (3u | 8u << 4 | 3u << 8));
    REQUIRE(Bits(t0.x, 16, 12) == (4u | 0u << 4 | 4u << 8));
    REQUIRE(Bits(t0.x, 12, 4) == 1u); // Modulate
    REQUIRE(Bits(t0.x, 28, 4) == 4u); // Lerp
    REQUIRE(Components(t0, 0) == std::array<u32, 3>{3, 3, 3});
    REQUIRE(Components(t0, 2) == std::array<u32, 3>{1, 1, 1});
    REQUIRE(Bits(t0.y, 24, 6) == (2u | 0u << 2 | 3u << 4));
    REQUIRE(Modifier(t0.z, 8) == -1);
    REQUIRE(Modifier(t0.z, 10) == 0);
    REQUIRE(Modifier(t0.z, 12) == 1);
    REQUIRE(Modifier(t0.z, 14) == -1);
    REQUIRE(Modifier(t0.z, 16) == 1);
    REQUIRE(Modifier(t0.z, 18) == 1);
    REQUIRE(Bits(t0.z, 0, 4) == (UberTevActive | UberTevBufferAlpha));
    REQUIRE(Bits(t0.z, 4, 2) == 2u); // x4
    REQUIRE(Bits(t0.z, 6, 2) == 0u); // scale 3 multiplies by 1
    REQUIRE(Bits(t0.z, 20, 2) == 2u);
    REQUIRE(Bits(t0.z, 22, 2) == 3u);

    const auto& t1 = data.tev[1];
    REQUIRE(Bits(t1.x, 0, 12) == (5u | 10u << 4 | 9u << 8));
    REQUIRE(Bits(t1.x, 16, 4) == 7u);
    REQUIRE(Bits(t1.z, 0, 4) == (UberTevActive | UberTevDot3Rgba | UberTevBufferColor));
    REQUIRE(Bits(t1.z, 22, 2) == 0u); // Dot3_RGBA reads no alpha inputs

    REQUIRE(Bits(data.tev[2].z, 0, 4) == 0u);
    REQUIRE(data.tex_used == 0x7u);

    config.lighting.enable.Assign(1);
    config.lighting.enable_shadow.Assign(1);
    config.lighting.shadow_selector.Assign(3);
    data = BuildFSConfigUniform(config);
    REQUIRE(data.tex_used == 0xFu);
}
