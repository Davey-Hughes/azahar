// Copyright 2023 Citra Emulator Project
// Licensed under GPLv2 or any later version
// Refer to the misc/licenses/gplv2.txt file included.

#include <cstring>
#include "common/alignment.h"
#include "common/logging/log.h"
#include "video_core/pica/vertex_loader.h"

namespace Pica {

VertexLoader::VertexLoader(Memory::MemorySystem& memory_, const PipelineRegs& regs)
    : memory{memory_} {
    const auto& attribute_config = regs.vertex_attributes;
    num_total_attributes = attribute_config.GetNumTotalAttributes();

    vertex_attribute_sources.fill(0xdeadbeef);

    for (u32 i = 0; i < 16; i++) {
        vertex_attribute_is_default[i] = attribute_config.IsDefaultAttribute(i);
    }

    // Setup attribute data from loaders
    for (u32 loader = 0; loader < 12; ++loader) {
        const auto& loader_config = attribute_config.attribute_loaders[loader];

        u32 offset = 0;

        // TODO: What happens if a loader overwrites a previous one's data?
        for (u32 component = 0; component < loader_config.component_count; ++component) {
            if (component >= 12) {
                LOG_ERROR(HW_GPU,
                          "Overflow in the vertex attribute loader {} trying to load component {}",
                          loader, component);
                continue;
            }

            u32 attribute_index = loader_config.GetComponent(component);
            if (attribute_index < 12) {
                offset = Common::AlignUp(offset,
                                         attribute_config.GetElementSizeInBytes(attribute_index));
                vertex_attribute_sources[attribute_index] = loader_config.data_offset + offset;
                vertex_attribute_strides[attribute_index] =
                    static_cast<u32>(loader_config.byte_count);
                vertex_attribute_formats[attribute_index] =
                    attribute_config.GetFormat(attribute_index);
                vertex_attribute_elements[attribute_index] =
                    attribute_config.GetNumElements(attribute_index);
                offset += attribute_config.GetStride(attribute_index);
            } else if (attribute_index < 16) {
                // Attribute ids 12, 13, 14 and 15 signify 4, 8, 12 and 16-byte paddings,
                // respectively
                offset = Common::AlignUp(offset, 4);
                offset += (attribute_index - 11) * 4;
            } else {
                UNREACHABLE(); // This is truly unreachable due to the number of bits for each
                               // component
            }
        }
    }
}

VertexLoader::~VertexLoader() = default;

void VertexLoader::LoadVertex(PAddr base_address, u32 index, u32 vertex, AttributeBuffer& input,
                              AttributeBuffer& input_default_attributes) const {
    for (s32 i = 0; i < num_total_attributes; ++i) {
        // Load the default attribute if we're configured to do so
        if (vertex_attribute_is_default[i]) {
            input[i] = input_default_attributes[i];
            continue;
        }

        // TODO(yuriks): In this case, no data gets loaded and the vertex
        // remains with the last value it had. This isn't currently maintained
        // as global state, however, and so won't work in Citra yet.
        if (vertex_attribute_elements[i] == 0) {
            LOG_ERROR(HW_GPU, "Vertex retension unimplemented");
            continue;
        }

        // Load per-vertex data from the loader arrays
        const PAddr source_addr =
            base_address + vertex_attribute_sources[i] + vertex_attribute_strides[i] * vertex;

        switch (vertex_attribute_formats[i]) {
        case PipelineRegs::VertexAttributeFormat::BYTE:
            LoadAttribute<s8>(source_addr, i, input);
            break;
        case PipelineRegs::VertexAttributeFormat::UBYTE:
            LoadAttribute<u8>(source_addr, i, input);
            break;
        case PipelineRegs::VertexAttributeFormat::SHORT:
            LoadAttribute<s16>(source_addr, i, input);
            break;
        case PipelineRegs::VertexAttributeFormat::FLOAT:
            LoadAttribute<f32>(source_addr, i, input);
            break;
        }

        // Default attribute values set if array elements have < 4 components. This
        // is *not* carried over from the default attribute settings even if they're
        // enabled for this attribute.
        for (u32 comp = vertex_attribute_elements[i]; comp < 4; comp++) {
            input[i][comp] = comp == 3 ? f24::One() : f24::Zero();
        }
    }
}

bool VertexLoader::ResolveArrays(PAddr base_address, u32 max_vertex, VertexArrays& out) const {
    out = {};
    for (u32 i = 0; i < 16; ++i) {
        if (vertex_attribute_is_default[i] || vertex_attribute_elements[i] == 0) {
            continue;
        }
        u32 element_size = 4;
        switch (vertex_attribute_formats[i]) {
        case PipelineRegs::VertexAttributeFormat::BYTE:
        case PipelineRegs::VertexAttributeFormat::UBYTE:
            element_size = 1;
            break;
        case PipelineRegs::VertexAttributeFormat::SHORT:
            element_size = 2;
            break;
        case PipelineRegs::VertexAttributeFormat::FLOAT:
            break;
        }
        const u64 first = u64{base_address} + vertex_attribute_sources[i];
        const u64 last = first + u64{vertex_attribute_strides[i]} * max_vertex +
                         u64{vertex_attribute_elements[i]} * element_size - 1;
        if (last > 0xFFFFFFFFull) {
            return false;
        }
        const u8* first_ptr = memory.GetPhysicalPointer(static_cast<PAddr>(first));
        const u8* last_ptr = memory.GetPhysicalPointer(static_cast<PAddr>(last));
        if (first_ptr == nullptr || last_ptr == nullptr ||
            static_cast<u64>(last_ptr - first_ptr) != last - first) {
            return false;
        }
        // GetPhysMemRegionInfo() accepts the one-past-end address, so check the last byte
        // is inside the region (exclusive end) as well
        const u64 region_end = memory.GetPhysicalRegionEnd(static_cast<PAddr>(first));
        if (last >= region_end) {
            return false;
        }
        out.base[i] = first_ptr;
    }
    return true;
}

void VertexLoader::LoadVertex(const VertexArrays& arrays, u32 vertex, AttributeBuffer& input,
                              const AttributeBuffer& input_default_attributes) const {
    for (s32 i = 0; i < num_total_attributes; ++i) {
        if (vertex_attribute_is_default[i]) {
            input[i] = input_default_attributes[i];
            continue;
        }

        if (vertex_attribute_elements[i] == 0) {
            LOG_ERROR(HW_GPU, "Vertex retension unimplemented");
            continue;
        }

        // 64-bit, as ResolveArrays() checked it
        const u8* src = arrays.base[i] + u64{vertex_attribute_strides[i]} * vertex;
        const u32 elements = vertex_attribute_elements[i];

        // Read through memcpy: the array is not necessarily aligned for the element type
        switch (vertex_attribute_formats[i]) {
        case PipelineRegs::VertexAttributeFormat::BYTE:
            for (u32 comp = 0; comp < elements; ++comp) {
                s8 v;
                std::memcpy(&v, src + comp, sizeof(v));
                input[i][comp] = f24::FromFloat32(v);
            }
            break;
        case PipelineRegs::VertexAttributeFormat::UBYTE:
            for (u32 comp = 0; comp < elements; ++comp) {
                u8 v;
                std::memcpy(&v, src + comp, sizeof(v));
                input[i][comp] = f24::FromFloat32(v);
            }
            break;
        case PipelineRegs::VertexAttributeFormat::SHORT:
            for (u32 comp = 0; comp < elements; ++comp) {
                s16 v;
                std::memcpy(&v, src + comp * sizeof(v), sizeof(v));
                input[i][comp] = f24::FromFloat32(v);
            }
            break;
        case PipelineRegs::VertexAttributeFormat::FLOAT:
            for (u32 comp = 0; comp < elements; ++comp) {
                f32 v;
                std::memcpy(&v, src + comp * sizeof(v), sizeof(v));
                input[i][comp] = f24::FromFloat32(v);
            }
            break;
        }

        for (u32 comp = elements; comp < 4; comp++) {
            input[i][comp] = comp == 3 ? f24::One() : f24::Zero();
        }
    }
}

} // namespace Pica
