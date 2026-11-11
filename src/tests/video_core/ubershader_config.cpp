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
    FSConfig config{regs};
    Profile profile{};
    profile.has_logic_op = true;
    profile.has_custom_border_color = true;
    profile.is_vulkan = true;
    config.ApplyProfile(profile);
    return config;
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
    static const Pica::RegsInternal regs{};
    FSConfig config{regs};
    config.texture.requested_wrap[1].t = Pica::TexturingRegs::TextureConfig::WrapMode::ClampToBorder;
    Profile profile{}; // No custom border color, so ApplyProfile turns on border emulation.
    profile.has_logic_op = true;
    config.ApplyProfile(profile);

    config.texture.fog_mode.Assign(Pica::TexturingRegs::FogMode::Fog);
    config.texture.fog_flip.Assign(1);
    config.texture.tev_stages[2].sources_raw = 0x123;
    config.texture.tev_stages[2].scales_raw = 0x5;
    config.lighting.lights[5].num.Assign(3);
    config.texture.shadow_texture_orthographic.Assign(1);
    config.framebuffer.shadow_rendering.Assign(1);

    const FSConfigUniformData data = BuildFSConfigUniform(config);
    REQUIRE(sizeof(FSConfigUniformData) == 240);
    REQUIRE(data.fog_mode == 5u);
    REQUIRE(data.fog_flip == 1u);
    REQUIRE(data.tex_border == (1u << 3)); // unit 1, enable_t
    REQUIRE(data.tev[2][0] == 0x123u);
    REQUIRE(data.tev[2][3] == 0x5u);
    REQUIRE(data.light[1][1] == config.lighting.lights[5].raw);
    REQUIRE(data.shadow_flags == 0x3u);
}
