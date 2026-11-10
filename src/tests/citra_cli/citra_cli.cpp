// Copyright 2026 Citra Emulator Project / Azahar Emulator Project
// Licensed under GPLv3 or any later version
// Refer to the LICENSE.txt file included.

#include <array>
#include <catch2/catch_test_macros.hpp>
#include "citra_cli/citra_cli.h"

namespace {

// getopt keeps a pointer into the last scanned argument between calls, so the argument strings
// must outlive every later call in the process. String literals do.
template <typename... Args>
std::array<char*, sizeof...(Args) + 2> MakeArgv(Args... args) {
    return {const_cast<char*>("azahar"), const_cast<char*>(args)..., nullptr};
}

template <std::size_t N>
bool CheckForCLIOptions(std::array<char*, N>& argv) {
    return CitraCLI::CheckForOptions(CitraCLI::cli_capture_optstring, static_cast<int>(N - 1),
                                     argv.data());
}

} // Anonymous namespace

TEST_CASE("CheckForOptions leaves argv unchanged", "[citra_cli]") {
    auto gdbport = MakeArgv("-g", "24689", "-f", "rom.3ds");
    const auto gdbport_before = gdbport;
    CheckForCLIOptions(gdbport);
    REQUIRE(gdbport == gdbport_before);

    auto compress = MakeArgv("rom.3ds", "-c", "other.3ds", "-o", "out");
    const auto compress_before = compress;
    CheckForCLIOptions(compress);
    REQUIRE(compress == compress_before);
}
