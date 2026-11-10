// Copyright 2026 Citra Emulator Project / Azahar Emulator Project
// Licensed under GPLv3 or any later version
// Refer to the LICENSE.txt file included.

#pragma once

#include <optional>
#include <string>

namespace CitraCLI {

struct CompressionOptions {
    std::optional<std::string> compress_path;   // The path of a decompressed file to be compressed
    std::optional<std::string> decompress_path; // The path of a compressed file to be decompressed
    std::optional<std::string> output_dir_path; // The directory which will contain processed file
};

CompressionOptions ParseCompressionOptions(int argc, char* argv[]);
int ParseCompressionCommand(int argc, char* argv[]);

} // namespace CitraCLI
