// Copyright 2026 Citra Emulator Project / Azahar Emulator Project
// Licensed under GPLv2 or any later version
// Refer to the license.txt file included.

#pragma once

#include "libretro.h"

// RetroArch's video views API, until the vendored libretro.h has it.
#ifndef RETRO_ENVIRONMENT_SET_VIDEO_VIEWS

#define RETRO_ENVIRONMENT_SET_VIDEO_VIEWS (95 | RETRO_ENVIRONMENT_EXPERIMENTAL)
#define RETRO_ENVIRONMENT_GET_VIDEO_VIEWS_STATUS (96 | RETRO_ENVIRONMENT_EXPERIMENTAL)

#define RETRO_VIDEO_VIEWS_STATUS_PRESENTS (1 << 0)
#define RETRO_VIDEO_VIEWS_STATUS_STEREO (1 << 1)

#define RETRO_VIDEO_VIEW_EYE_NONE 0
#define RETRO_VIDEO_VIEW_EYE_LEFT 1
#define RETRO_VIDEO_VIEW_EYE_RIGHT 2
#define RETRO_VIDEO_VIEWS_MAX 8

struct retro_video_view {
    unsigned x;
    unsigned y;
    unsigned width;
    unsigned height;
    unsigned screen;
    unsigned eye;
    float aspect_ratio;
};

struct retro_video_views {
    const struct retro_video_view* views;
    unsigned num_views;
};

#endif
