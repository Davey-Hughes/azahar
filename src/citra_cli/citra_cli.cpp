// Copyright 2026 Citra Emulator Project / Azahar Emulator Project
// Licensed under GPLv3 or any later version
// Refer to the LICENSE.txt file included.

#include <vector>
#undef _UNICODE
#include <getopt.h>
#ifndef _MSC_VER
#include <unistd.h>
#endif

#include "citra_cli/citra_cli.h"
#include "citra_cli/compression_cli.h"

namespace CitraCLI {

bool CheckForOptions(const char* optstring, int argc, char* argv[]) {
    // Scan a copy, as getopt can permute argv in place and the Qt frontend parses argv afterwards
    // (see LaunchQtFrontend() in src/citra_qt/citra_qt.cpp).
    std::vector<char*> args(argv, argv + argc);
    args.push_back(nullptr);

    // Use getopt_long with no long options so that e.g. "--fullscreen" is skipped as a single
    // unknown option, rather than being parsed by getopt as the short options "-f -u -l -l -s -c".
    static const struct option no_long_opts[] = {{nullptr, 0, nullptr, 0}};

    const int original_opterr = opterr;
    opterr = 0; // Temporarily suppress invalid option messages

    bool return_value = false;
    int option;
    while ((option = getopt_long(argc, args.data(), optstring, no_long_opts, nullptr)) != -1) {
        for (size_t i = 0; optstring[i] != '\0'; ++i) {
            if (optstring[i] == ':') {
                continue;
            }
            if (option == optstring[i]) {
                return_value = true;
                break;
            }
        }
    }

    opterr = original_opterr;
    optind = 1; // Reset getopt so that it can be used again
    return return_value;
}

int ParseCommand(int argc, char* argv[]) {
    if (CheckForOptions(compression_ops_optstring, argc, argv)) {
        return ParseCompressionCommand(argc, argv);
    }
}

} // namespace CitraCLI
