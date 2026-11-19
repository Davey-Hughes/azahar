// Copyright 2026 Azahar Emulator Project
// Licensed under GPLv2 or any later version
// Refer to the misc/licenses/gplv2.txt file included.

#ifdef __linux__

#include <catch2/catch_test_macros.hpp>
#include "common/host_memory.h"
#include "tests/common/fastmem_util.h"

using Tests::IsReadable;

TEST_CASE("HostMemory backing aliases into an arena", "[common][fastmem]") {
    Common::HostMemory backing(0x10000);
    REQUIRE(backing.BackingBasePointer() != nullptr);
    Common::VirtualArena arena(backing, 0x100000);
    REQUIRE(arena.Base() != nullptr);

    backing.BackingBasePointer()[0x2000] = 0xA5;
    arena.Map(0x10000, 0x2000, 0x1000);
    REQUIRE(arena.Base()[0x10000] == 0xA5);

    arena.Base()[0x10004] = 0x5A;
    REQUIRE(backing.BackingBasePointer()[0x2004] == 0x5A);

    REQUIRE(backing.Contains(backing.BackingBasePointer() + 0xFFFF));
    REQUIRE(!backing.Contains(backing.BackingBasePointer() + 0x10000));
    REQUIRE(!backing.Contains(arena.Base() + 0x10000));
    REQUIRE(backing.OffsetOf(backing.BackingBasePointer() + 0x2004) == 0x2004);
}

TEST_CASE("VirtualArena unmapped pages are inaccessible", "[common][fastmem]") {
    Common::HostMemory backing(0x10000);
    Common::VirtualArena arena(backing, 0x100000);
    REQUIRE(arena.Base() != nullptr);

    REQUIRE(!IsReadable(arena.Base() + 0x10000));
    arena.Map(0x10000, 0x2000, 0x1000);
    REQUIRE(IsReadable(arena.Base() + 0x10000));
    REQUIRE(!IsReadable(arena.Base() + 0x11000));
    arena.Unmap(0x10000, 0x1000);
    REQUIRE(!IsReadable(arena.Base() + 0x10000));
}

#endif
