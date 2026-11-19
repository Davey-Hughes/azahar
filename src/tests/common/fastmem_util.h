// Copyright 2026 Azahar Emulator Project
// Licensed under GPLv2 or any later version
// Refer to the misc/licenses/gplv2.txt file included.

#pragma once

#ifdef __linux__

#include <cerrno>
#include <unistd.h>

namespace Tests {

/// Whether one byte at p can be read, checked by the kernel instead of by touching it.
inline bool IsReadable(const void* p) {
    int fds[2];
    if (pipe(fds) != 0) {
        return false;
    }
    const bool readable = write(fds[1], p, 1) == 1;
    close(fds[0]);
    close(fds[1]);
    return readable;
}

} // namespace Tests

#endif
