// Copyright 2023-2026 Citra Emulator Project / Azahar Emulator Project
// Licensed under GPLv2 or any later version
// Refer to the misc/licenses/gplv2.txt file included.

#pragma once

#include <cstdlib>
#include <span>
#ifdef __linux__
#include <sys/resource.h>
#include <unistd.h>
#endif

#include "video_core/renderer_vulkan/vk_common.h"

namespace Vulkan {

/**
 * @brief Creates a vulkan shader module from GLSL by converting it to SPIR-V using glslang.
 * @param code The string containing GLSL code.
 * @param stage The pipeline stage the shader will be used in.
 */
std::vector<u32> CompileGLSL(std::string_view code, vk::ShaderStageFlagBits stage,
                             std::string_view premable = "");

/**
 * @brief Creates a vulkan shader module from SPIR-V bytecode.
 * @param code The SPIR-V bytecode data.
 * @param device The vulkan device handle
 */
vk::ShaderModule CompileSPV(std::span<const u32> code, vk::Device device);

vk::ShaderModule Compile(std::string_view code, vk::ShaderStageFlagBits stage, vk::Device device,
                         std::string_view premable = "");

/// PoC only: AZAHAR_WORKER_NICE=N sets the calling compile worker thread's nice value, once per
/// thread. Common::SetCurrentThreadPriority() can't do this on Linux: it sets sched_priority under
/// SCHED_OTHER, which the kernel ignores.
inline void ApplyCompileWorkerNice() {
#ifdef __linux__
    thread_local const bool applied = [] {
        if (const char* nice = std::getenv("AZAHAR_WORKER_NICE")) {
            setpriority(PRIO_PROCESS, static_cast<id_t>(gettid()), std::atoi(nice));
        }
        return true;
    }();
    static_cast<void>(applied);
#endif
}

} // namespace Vulkan
