// Copyright 2026 Citra Emulator Project / Azahar Emulator Project
// Licensed under GPLv2 or any later version
// Refer to the misc/licenses/gplv2.txt file included.

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
}

TEST_CASE("Ubershader rejects features it does not implement", "[video_core][ubershader]") {
    SECTION("procedural textures") {
        auto config = MakeConfig();
        config.proctex.enable.Assign(1);
        REQUIRE_FALSE(UbershaderSupports(config, UserConfig{}));
    }
    SECTION("cube and shadow texture types") {
        for (const auto type :
             {TextureType::TextureCube, TextureType::Shadow2D, TextureType::ShadowCube}) {
            auto config = MakeConfig();
            config.texture.texture0_type.Assign(type);
            REQUIRE_FALSE(UbershaderSupports(config, UserConfig{}));
        }
    }
    SECTION("lighting shadows") {
        auto config = MakeConfig();
        config.lighting.enable.Assign(1);
        config.lighting.enable_shadow.Assign(1);
        REQUIRE_FALSE(UbershaderSupports(config, UserConfig{}));
    }
    SECTION("shadow rendering") {
        auto config = MakeConfig();
        config.framebuffer.shadow_rendering.Assign(1);
        REQUIRE_FALSE(UbershaderSupports(config, UserConfig{}));
    }
    SECTION("min/max blend emulation") {
        auto config = MakeConfig();
        config.framebuffer.requested_rgb_blend.eq = Pica::FramebufferRegs::BlendEquation::Min;
        REQUIRE_FALSE(UbershaderSupports(config, UserConfig{}));
        config = MakeConfig();
        config.framebuffer.requested_alpha_blend.eq = Pica::FramebufferRegs::BlendEquation::Max;
        REQUIRE_FALSE(UbershaderSupports(config, UserConfig{}));
    }
    SECTION("logic op emulation") {
        auto config = MakeConfig();
        config.framebuffer.logic_op.Assign(Pica::FramebufferRegs::LogicOp::Clear);
        REQUIRE_FALSE(UbershaderSupports(config, UserConfig{}));
    }
    SECTION("custom normal maps") {
        UserConfig user{};
        user.use_custom_normal.Assign(1);
        REQUIRE_FALSE(UbershaderSupports(MakeConfig(), user));
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

    const FSConfigUniformData data = BuildFSConfigUniform(config);
    REQUIRE(sizeof(FSConfigUniformData) == 240);
    REQUIRE(data.fog_mode == 5u);
    REQUIRE(data.fog_flip == 1u);
    REQUIRE(data.tex_border == (1u << 3)); // unit 1, enable_t
    REQUIRE(data.tev[2][0] == 0x123u);
    REQUIRE(data.tev[2][3] == 0x5u);
    REQUIRE(data.light[1][1] == config.lighting.lights[5].raw);
}
