// Copyright 2026 Azahar Emulator Project
// Copyright 2025 Panda3DS Emulator Project
// Copyright 2021 yuzu Emulator Project
// Copyright 2008 Dolphin Emulator Project
// Licensed under GPLv2 or any later version
// Refer to the misc/licenses/gplv2.txt file included.

#ifdef __linux__

#include <cerrno>
#include <cstring>
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/syscall.h>
#include <unistd.h>
#include "common/assert.h"
#include "common/host_memory.h"
#include "common/logging/log.h"

#ifdef __ANDROID__
#include <dlfcn.h>
#endif

#ifndef MAP_NORESERVE
#define MAP_NORESERVE 0
#endif

namespace Common {

namespace {

// memfd_create through syscall(): bionic's wrapper needs API 30 and minSdk is 29.
int CreateBackingFd(std::size_t size) {
#ifdef __ANDROID__
    // ASharedMemory_create is API 26, looked up so libandroid stays an optional dependency.
    static void* libandroid = dlopen("libandroid.so", RTLD_LAZY | RTLD_LOCAL);
    using CreateFn = int (*)(const char*, std::size_t);
    static auto shared_memory_create =
        libandroid ? reinterpret_cast<CreateFn>(dlsym(libandroid, "ASharedMemory_create"))
                   : nullptr;
    if (shared_memory_create) {
        const int fd = shared_memory_create("HostMemory", size);
        if (fd >= 0) {
            return fd;
        }
    }
#endif
    const int fd = static_cast<int>(syscall(__NR_memfd_create, "HostMemory", 1U /*MFD_CLOEXEC*/));
    if (fd < 0) {
        return fd;
    }
    if (ftruncate(fd, static_cast<off_t>(size)) != 0) {
        close(fd);
        return -1;
    }
    return fd;
}

} // Anonymous namespace

HostMemory::HostMemory(std::size_t backing_size_) : backing_size{backing_size_} {
    fd = CreateBackingFd(backing_size);
    if (fd < 0) {
        LOG_ERROR(Common_Memory, "Creating the backing fd failed: {}", std::strerror(errno));
        return;
    }
    void* ptr = mmap(nullptr, backing_size, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
    if (ptr == MAP_FAILED) {
        LOG_ERROR(Common_Memory, "Mapping the backing failed: {}", std::strerror(errno));
        close(fd);
        fd = -1;
        return;
    }
    backing_base = static_cast<u8*>(ptr);
}

HostMemory::~HostMemory() {
    if (backing_base) {
        munmap(backing_base, backing_size);
    }
    if (fd >= 0) {
        close(fd);
    }
}

VirtualArena::VirtualArena(HostMemory& backing_, std::size_t size_)
    : backing{backing_}, size{size_} {
    if (!backing.BackingBasePointer()) {
        return;
    }
    void* ptr =
        mmap(nullptr, size, PROT_NONE, MAP_PRIVATE | MAP_ANONYMOUS | MAP_NORESERVE, -1, 0);
    if (ptr == MAP_FAILED) {
        LOG_ERROR(Common_Memory, "Reserving 0x{:X} bytes failed: {}", size, std::strerror(errno));
        return;
    }
    base = static_cast<u8*>(ptr);
}

VirtualArena::~VirtualArena() {
    if (base) {
        munmap(base, size);
    }
}

bool VirtualArena::Map(std::size_t arena_offset, std::size_t backing_offset, std::size_t length) {
    ASSERT(arena_offset + length <= size && backing_offset + length <= backing.backing_size);
    return mmap(base + arena_offset, length, PROT_READ | PROT_WRITE, MAP_SHARED | MAP_FIXED,
                backing.fd, static_cast<off_t>(backing_offset)) != MAP_FAILED;
}

bool VirtualArena::Unmap(std::size_t arena_offset, std::size_t length) {
    ASSERT(arena_offset + length <= size);
    // Replaced rather than unmapped, so the range stays reserved.
    return mmap(base + arena_offset, length, PROT_NONE,
                MAP_PRIVATE | MAP_ANONYMOUS | MAP_NORESERVE | MAP_FIXED, -1, 0) != MAP_FAILED;
}

} // namespace Common

#endif
