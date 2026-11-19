// Copyright 2026 Azahar Emulator Project
// Copyright 2025 Panda3DS Emulator Project
// Copyright 2019 yuzu Emulator Project
// Licensed under GPLv2 or any later version
// Refer to the misc/licenses/gplv2.txt file included.

#pragma once

#ifdef __linux__

#include <cstddef>
#include "common/common_types.h"

namespace Common {

/**
 * Memory backed by a shareable fd, so the same pages can also be mapped at other addresses
 * through a VirtualArena. Linux and Android only.
 */
class HostMemory {
public:
    explicit HostMemory(std::size_t backing_size);
    ~HostMemory();

    HostMemory(const HostMemory&) = delete;
    HostMemory& operator=(const HostMemory&) = delete;

    /// Base of the read/write mapping of the whole backing, or nullptr if creating it failed.
    u8* BackingBasePointer() {
        return backing_base;
    }

    bool Contains(const u8* host_ptr) const {
        return backing_base && host_ptr >= backing_base && host_ptr < backing_base + backing_size;
    }

    std::size_t OffsetOf(const u8* host_ptr) const {
        return static_cast<std::size_t>(host_ptr - backing_base);
    }

private:
    friend class VirtualArena;

    std::size_t backing_size;
    int fd = -1;
    u8* backing_base = nullptr;
};

/// A PROT_NONE reservation that pages of a HostMemory can be mapped into.
class VirtualArena {
public:
    VirtualArena(HostMemory& backing, std::size_t size);
    ~VirtualArena();

    VirtualArena(const VirtualArena&) = delete;
    VirtualArena& operator=(const VirtualArena&) = delete;

    /// Base of the reservation, or nullptr if reserving it failed.
    u8* Base() {
        return base;
    }

    /// Maps backing pages read/write at arena_offset. Offsets and length must be page aligned.
    /// On failure the range may be left in any state; Unmap() it before relying on it.
    bool Map(std::size_t arena_offset, std::size_t backing_offset, std::size_t length);

    /// Makes a range inaccessible again.
    bool Unmap(std::size_t arena_offset, std::size_t length);

private:
    HostMemory& backing;
    std::size_t size;
    u8* base = nullptr;
};

} // namespace Common

#endif
