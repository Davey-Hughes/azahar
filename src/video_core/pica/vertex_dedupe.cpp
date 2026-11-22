// Copyright 2023-2026 Citra Emulator Project / Azahar Emulator Project
// Licensed under GPLv3 or any later version
// Refer to the LICENSE.txt file included.

#include <algorithm>
#include <numeric>
#include "video_core/pica/vertex_dedupe.h"

namespace Pica {

template <typename T>
void VertexDedupe::Build(std::span<const T> indices) {
    if (++generation == 0) {
        stamps.fill(0);
        generation = 1;
    }
    unique.clear();
    slots.clear();
    slots.reserve(indices.size());
    max_vertex = 0;

    for (const T index : indices) {
        if (stamps[index] != generation) {
            stamps[index] = generation;
            table_slots[index] = static_cast<u32>(unique.size());
            unique.push_back(index);
            max_vertex = std::max<u32>(max_vertex, index);
        }
        slots.push_back(table_slots[index]);
    }
}

void VertexDedupe::BuildIndexed(std::span<const u8> indices) {
    Build(indices);
}

void VertexDedupe::BuildIndexed(std::span<const u16> indices) {
    Build(indices);
}

void VertexDedupe::BuildRange(u32 first_vertex, u32 count) {
    unique.resize(count);
    slots.resize(count);
    std::iota(unique.begin(), unique.end(), first_vertex);
    std::iota(slots.begin(), slots.end(), 0u);
    max_vertex = count ? first_vertex + count - 1 : 0;
}

} // namespace Pica
