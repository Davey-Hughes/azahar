// Copyright 2026 Citra Emulator Project / Azahar Emulator Project
// Licensed under GPLv3 or any later version
// Refer to the LICENSE.txt file included.

#include <array>
#include <iostream>
#include <sstream>
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

TEST_CASE("CheckForOptions ignores long options", "[citra_cli]") {
    auto fullscreen = MakeArgv("--fullscreen", "rom.3ds");
    REQUIRE_FALSE(CheckForCLIOptions(fullscreen));

    auto version = MakeArgv("--version");
    REQUIRE_FALSE(CheckForCLIOptions(version));

    auto room = MakeArgv("--room");
    REQUIRE_FALSE(CheckForCLIOptions(room));

    auto movie_record = MakeArgv("--movie-record", "movie.ctm", "rom.3ds");
    REQUIRE_FALSE(CheckForCLIOptions(movie_record));

    auto gdbport = MakeArgv("--gdbport", "24689", "rom.3ds");
    REQUIRE_FALSE(CheckForCLIOptions(gdbport));

    auto attached_value = MakeArgv("--movie-record=cool.ctm", "rom.3ds");
    REQUIRE_FALSE(CheckForCLIOptions(attached_value));
}

TEST_CASE("CheckForOptions stops at the end of options", "[citra_cli]") {
    auto argv = MakeArgv("--", "-c", "rom.3ds");
    REQUIRE_FALSE(CheckForCLIOptions(argv));
}

TEST_CASE("CheckForOptions ignores single-dash frontend options", "[citra_cli]") {
    auto qt_platform = MakeArgv("-platform", "xcb", "rom.3ds");
    REQUIRE_FALSE(CheckForCLIOptions(qt_platform));

    auto qt_geometry = MakeArgv("-qwindowgeometry", "1280x720", "rom.3ds");
    REQUIRE_FALSE(CheckForCLIOptions(qt_geometry));

    auto room_name = MakeArgv("--room", "-nCoolroom");
    REQUIRE_FALSE(CheckForCLIOptions(room_name));
}

TEST_CASE("CheckForOptions detects compression options", "[citra_cli]") {
    auto compress = MakeArgv("-c", "rom.3ds");
    REQUIRE(CheckForCLIOptions(compress));

    auto compress_attached = MakeArgv("-crom.3ds");
    REQUIRE(CheckForCLIOptions(compress_attached));

    auto decompress = MakeArgv("-x", "rom.zcci", "-o", "out");
    REQUIRE(CheckForCLIOptions(decompress));

    auto after_non_option = MakeArgv("rom.3ds", "-c", "other.3ds");
    REQUIRE(CheckForCLIOptions(after_non_option));

    auto after_long_option = MakeArgv("--fullscreen", "-c", "rom.3ds");
    REQUIRE(CheckForCLIOptions(after_long_option));

    auto after_attached_value = MakeArgv("--gdbport=24689", "-c", "rom.3ds");
    REQUIRE(CheckForCLIOptions(after_attached_value));
}

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

TEST_CASE("ParseCommand fails without a command", "[citra_cli]") {
    auto argv = MakeArgv("-f", "rom.3ds");

    // Capture std::cout to check that no command (e.g. compression) was run.
    std::ostringstream output;
    auto* const original_cout_buffer = std::cout.rdbuf(output.rdbuf());
    const int result = CitraCLI::ParseCommand(static_cast<int>(argv.size() - 1), argv.data());
    std::cout.rdbuf(original_cout_buffer);

    REQUIRE(result != 0);
    REQUIRE(output.str().empty());
}
