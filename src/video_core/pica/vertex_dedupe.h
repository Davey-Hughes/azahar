// Copyright 2023-2026 Citra Emulator Project / Azahar Emulator Project
// Licensed under GPLv3 or any later version
// Refer to the LICENSE.txt file included.

#pragma once

#include <array>
#include <span>
#include <vector>
#include "common/common_types.h"

namespace Pica {

/// Collapses a draw's vertex IDs into unique IDs in first-use order, plus a per-index slot.
/// The buffers only grow, so a warm instance does not allocate per draw.
class VertexDedupe {
public:
    void BuildIndexed(std::span<const u8> indices);
    void BuildIndexed(std::span<const u16> indices);

    /// Non-indexed draw: IDs are first_vertex + [0, count) and may exceed 65535.
    void BuildRange(u32 first_vertex, u32 count);

    /// Vertex IDs in first-use order.
    std::span<const u32> Unique() const {
        return unique;
    }

    /// For each index, its position in Unique().
    std::span<const u32> Slots() const {
        return slots;
    }

    /// Largest vertex ID of the last build, 0 for an empty draw.
    u32 MaxVertex() const {
        return max_vertex;
    }

private:
    template <typename T>
    void Build(std::span<const T> indices);

    std::array<u32, 0x10000> stamps{};
    std::array<u32, 0x10000> table_slots{};
    u32 generation = 0;
    std::vector<u32> unique;
    std::vector<u32> slots;
    u32 max_vertex = 0;
};

} // namespace Pica
