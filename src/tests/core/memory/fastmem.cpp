// Copyright 2026 Azahar Emulator Project
// Licensed under GPLv2 or any later version
// Refer to the misc/licenses/gplv2.txt file included.

#ifdef __linux__

#include <algorithm>
#include <cstdlib>
#include <filesystem>
#include <string>
#include <sys/mman.h>
#include <unistd.h>
#include <catch2/catch_test_macros.hpp>
#include "common/memory_ref.h"
#include "core/core.h"
#include "core/core_timing.h"
#include "core/hle/kernel/process.h"
#include "core/memory.h"
#include "tests/common/fastmem_util.h"

using Tests::IsReadable;

namespace {

constexpr VAddr TEST_VADDR = 0x10000000;

/// A MemorySystem built with AZAHAR_FASTMEM=1; the variable is unset again right after.
struct FastmemMemory {
    explicit FastmemMemory(Core::System& system) : memory{(setenv("AZAHAR_FASTMEM", "1", 1), system)} {
        unsetenv("AZAHAR_FASTMEM");
    }
    Memory::MemorySystem memory;
};

/// Points the newest HostMemory fd at a pipe, so mapping more of the backing fails with ENODEV.
bool BreakNewestHostMemoryFd() {
    int newest = -1;
    for (const auto& entry : std::filesystem::directory_iterator("/proc/self/fd")) {
        std::error_code ec;
        const auto target = std::filesystem::read_symlink(entry.path(), ec).string();
        if (!ec && target.starts_with("/memfd:HostMemory")) {
            newest = std::max(newest, std::stoi(entry.path().filename().string()));
        }
    }
    int fds[2];
    if (newest < 0 || pipe(fds) != 0) {
        return false;
    }
    const bool ok = dup2(fds[0], newest) == newest;
    close(fds[0]);
    close(fds[1]);
    return ok;
}

} // Anonymous namespace

TEST_CASE("memory.Fastmem arena mirrors the page table", "[core][memory][fastmem]") {
    Core::Timing timing(1, 100);
    Core::System system;
    FastmemMemory fm{system};
    auto& memory = fm.memory;
    Kernel::KernelSystem kernel(memory, timing, [] {}, Kernel::MemoryMode::NewProd, 1);
    auto process = kernel.CreateProcess(kernel.CreateCodeSet("", 0));
    auto& pt = *process->vm_manager.page_table;

    REQUIRE(memory.IsFastmemEnabled());
    u8* const base = memory.GetFastmemBase(pt);
    REQUIRE(base != nullptr);
    REQUIRE(!IsReadable(base + TEST_VADDR));

    SECTION("map") {
        memory.MapMemoryRegion(pt, TEST_VADDR, 0x2000, memory.GetFCRAMRef(0x3000));
        REQUIRE(IsReadable(base + TEST_VADDR));
        REQUIRE(IsReadable(base + TEST_VADDR + 0x1000));
        REQUIRE(!IsReadable(base + TEST_VADDR + 0x2000));

        memory.GetFCRAMPointer(0x3000)[0x10] = 0xA5;
        REQUIRE(base[TEST_VADDR + 0x10] == 0xA5);
        base[TEST_VADDR + 0x1004] = 0x5A;
        REQUIRE(*memory.GetFCRAMPointer(0x4004) == 0x5A);
    }

    SECTION("unmap") {
        memory.MapMemoryRegion(pt, TEST_VADDR, 0x2000, memory.GetFCRAMRef(0x3000));
        memory.UnmapRegion(pt, TEST_VADDR, 0x2000);
        REQUIRE(!IsReadable(base + TEST_VADDR));
        REQUIRE(!IsReadable(base + TEST_VADDR + 0x1000));
    }

    SECTION("rasterizer cached") {
        constexpr u32 offset = 0x3000;
        const VAddr vaddr = Memory::LINEAR_HEAP_VADDR + offset;
        memory.MapMemoryRegion(pt, vaddr, 0x2000, memory.GetFCRAMRef(offset));
        REQUIRE(IsReadable(base + vaddr));

        memory.RasterizerMarkRegionCached(Memory::FCRAM_PADDR + offset, 0x1000, true);
        REQUIRE(!IsReadable(base + vaddr));
        REQUIRE(IsReadable(base + vaddr + 0x1000));

        memory.RasterizerMarkRegionCached(Memory::FCRAM_PADDR + offset, 0x1000, false);
        REQUIRE(IsReadable(base + vaddr));
        memory.GetFCRAMPointer(offset)[0x20] = 0x77;
        REQUIRE(base[vaddr + 0x20] == 0x77);
    }

    SECTION("watchpoint") {
        memory.MapMemoryRegion(pt, TEST_VADDR, 0x2000, memory.GetFCRAMRef(0x3000));
        memory.RegisterWatchpoint(*process, TEST_VADDR + 0x10, 4);
        REQUIRE(!IsReadable(base + TEST_VADDR));
        REQUIRE(IsReadable(base + TEST_VADDR + 0x1000));

        memory.UnregisterWatchpoint(*process, TEST_VADDR + 0x10, 4);
        REQUIRE(IsReadable(base + TEST_VADDR));
    }

    SECTION("foreign backing") {
        memory.MapMemoryRegion(pt, TEST_VADDR, 0x1000, memory.GetFCRAMRef(0x3000));
        REQUIRE(IsReadable(base + TEST_VADDR));
        auto buffer = std::make_shared<BufferMem>(0x1000);
        memory.MapMemoryRegion(pt, TEST_VADDR, 0x1000, MemoryRef(buffer));
        REQUIRE(!IsReadable(base + TEST_VADDR));
    }

    SECTION("two page tables") {
        auto process_b = kernel.CreateProcess(kernel.CreateCodeSet("", 0));
        auto& pt_b = *process_b->vm_manager.page_table;
        u8* const base_b = memory.GetFastmemBase(pt_b);
        REQUIRE(base_b != nullptr);
        REQUIRE(base_b != base);

        memory.MapMemoryRegion(pt, TEST_VADDR, 0x1000, memory.GetFCRAMRef(0x3000));
        REQUIRE(IsReadable(base + TEST_VADDR));
        REQUIRE(!IsReadable(base_b + TEST_VADDR));
    }
}

TEST_CASE("memory.Fastmem arenas rebuilt", "[core][memory][fastmem]") {
    Core::Timing timing(1, 100);
    Core::System system;
    FastmemMemory fm{system};
    auto& memory = fm.memory;
    Kernel::KernelSystem kernel(memory, timing, [] {}, Kernel::MemoryMode::NewProd, 1);
    auto process = kernel.CreateProcess(kernel.CreateCodeSet("", 0));
    auto& pt = *process->vm_manager.page_table;
    u8* const base = memory.GetFastmemBase(pt);
    REQUIRE(base != nullptr);

    memory.MapMemoryRegion(pt, TEST_VADDR, 0x2000, memory.GetFCRAMRef(0x3000));
    memory.GetFCRAMPointer(0x3000)[0x10] = 0xA5;

    // A stale arena both ways: a mapped page made inaccessible behind the table's back, and a
    // page the table no longer maps left accessible.
    REQUIRE(mmap(base + TEST_VADDR, 0x1000, PROT_NONE, MAP_PRIVATE | MAP_ANONYMOUS | MAP_FIXED, -1,
                 0) != MAP_FAILED);
    pt.GetPointerArray()[(TEST_VADDR >> Memory::CITRA_PAGE_BITS) + 1] = nullptr;
    REQUIRE(!IsReadable(base + TEST_VADDR));
    REQUIRE(IsReadable(base + TEST_VADDR + 0x1000));

    memory.RebuildFastmemArenas();
    REQUIRE(IsReadable(base + TEST_VADDR));
    REQUIRE(base[TEST_VADDR + 0x10] == 0xA5);
    REQUIRE(!IsReadable(base + TEST_VADDR + 0x1000));
}

TEST_CASE("memory.Fastmem arena update failure falls back to the page table",
          "[core][memory][fastmem]") {
    Core::Timing timing(1, 100);
    Core::System system;
    FastmemMemory fm{system};
    auto& memory = fm.memory;
    Kernel::KernelSystem kernel(memory, timing, [] {}, Kernel::MemoryMode::NewProd, 1);
    auto process = kernel.CreateProcess(kernel.CreateCodeSet("", 0));
    auto& pt = *process->vm_manager.page_table;
    u8* const base = memory.GetFastmemBase(pt);
    REQUIRE(base != nullptr);

    memory.MapMemoryRegion(pt, TEST_VADDR, 0x1000, memory.GetFCRAMRef(0x3000));
    REQUIRE(IsReadable(base + TEST_VADDR));

    REQUIRE(BreakNewestHostMemoryFd());
    constexpr VAddr second = TEST_VADDR + 0x10000;
    memory.MapMemoryRegion(pt, second, 0x1000, memory.GetFCRAMRef(0x5000));

    // The whole arena is inaccessible now, including the page mapped before the failure, so JIT
    // accesses fault into the callbacks; the page table itself still works.
    REQUIRE(!IsReadable(base + TEST_VADDR));
    REQUIRE(!IsReadable(base + second));
    memory.GetFCRAMPointer(0x5000)[0] = 0x5A;
    REQUIRE(memory.Read8(*process, second) == 0x5A);

    // The arena stays off for this table.
    memory.MapMemoryRegion(pt, TEST_VADDR + 0x20000, 0x1000, memory.GetFCRAMRef(0x6000));
    memory.RebuildFastmemArenas();
    REQUIRE(!IsReadable(base + TEST_VADDR));
    REQUIRE(!IsReadable(base + TEST_VADDR + 0x20000));
}

TEST_CASE("memory.Fastmem off keeps today's layout", "[core][memory][fastmem]") {
    Core::Timing timing(1, 100);
    Core::System system;
    unsetenv("AZAHAR_FASTMEM");
    Memory::MemorySystem memory{system};
    Kernel::KernelSystem kernel(memory, timing, [] {}, Kernel::MemoryMode::NewProd, 1);
    auto process = kernel.CreateProcess(kernel.CreateCodeSet("", 0));

    REQUIRE(!memory.IsFastmemEnabled());
    REQUIRE(memory.GetFastmemBase(*process->vm_manager.page_table) == nullptr);
}

#endif
