// Copyright 2026 Citra Emulator Project / Azahar Emulator Project
// Licensed under GPLv2 or any later version
// Refer to the license.txt file included.

#include "citra_libretro/video_views.h"
#include "core/3ds.h"

namespace LibRetro::VideoViews {

namespace {

Mode current_mode;

retro_video_view MakeView(const Common::Rectangle<u32>& rect, u32 x_offset, unsigned screen,
                          unsigned eye) {
    retro_video_view view{};
    view.x = rect.left + x_offset;
    view.y = rect.top;
    view.width = rect.GetWidth();
    view.height = rect.GetHeight();
    view.screen = screen;
    view.eye = eye;
    view.aspect_ratio = 0.0f;
    return view;
}

} // namespace

Mode SelectMode(bool frontend_layout, unsigned status, bool renderer_stereo) {
    Mode mode;
    mode.active = frontend_layout && (status & RETRO_VIDEO_VIEWS_STATUS_PRESENTS) != 0;
    mode.stereo = mode.active && renderer_stereo && (status & RETRO_VIDEO_VIEWS_STATUS_STEREO) != 0;
    return mode;
}

const Mode& CurrentMode() {
    return current_mode;
}

void SetCurrentMode(const Mode& mode) {
    current_mode = mode;
}

std::pair<u32, u32> PackedSize(bool stereo, u32 scale) {
    const u32 width = static_cast<u32>(Core::kScreenTopWidth) * (stereo ? 2 : 1);
    const u32 height = static_cast<u32>(Core::kScreenTopHeight + Core::kScreenBottomHeight);
    return {width * scale, height * scale};
}

std::vector<retro_video_view> BuildMap(const Layout::FramebufferLayout& layout, bool stereo,
                                       bool swapped) {
    const unsigned top = swapped ? 1 : 0;
    const unsigned bottom = swapped ? 0 : 1;
    std::vector<retro_video_view> views;
    if (stereo) {
        views.push_back(MakeView(layout.top_screen, 0, top, RETRO_VIDEO_VIEW_EYE_LEFT));
        // Where DrawTopScreen puts the right eye in the full side-by-side draw.
        views.push_back(
            MakeView(layout.top_screen, layout.width / 2, top, RETRO_VIDEO_VIEW_EYE_RIGHT));
    } else {
        views.push_back(MakeView(layout.top_screen, 0, top, RETRO_VIDEO_VIEW_EYE_NONE));
    }
    views.push_back(MakeView(layout.bottom_screen, 0, bottom, RETRO_VIDEO_VIEW_EYE_NONE));
    return views;
}

} // namespace LibRetro::VideoViews
