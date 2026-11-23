// Copyright 2022 Citra Emulator Project
// Licensed under GPLv2 or any later version
// Refer to the misc/licenses/gplv2.txt file included.

#include <string>
#include "common/microprofile.h"

namespace VideoCore {

MICROPROFILE_DEFINE(RasterizerCache_CopySurface, "RasterizerCache", "CopySurface",
                    MP_RGB(128, 192, 64));
MICROPROFILE_DEFINE(RasterizerCache_UploadSurface, "RasterizerCache", "UploadSurface",
                    MP_RGB(128, 192, 64));
MICROPROFILE_DEFINE(RasterizerCache_DownloadSurface, "RasterizerCache", "DownloadSurface",
                    MP_RGB(128, 192, 64));
MICROPROFILE_DEFINE(RasterizerCache_Invalidation, "RasterizerCache", "Invalidation",
                    MP_RGB(128, 64, 192));

namespace PocDownloadLog {
// Not inline in rasterizer_cache.h: MinGW GCC emits an inline thread_local's TLS init function in
// every file that includes it, and the link fails on the duplicates.
thread_local std::string reason;
thread_local std::string caller;
thread_local std::string validate_detail;
} // namespace PocDownloadLog

} // namespace VideoCore
