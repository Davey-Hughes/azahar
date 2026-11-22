// Copyright 2023-2026 Citra Emulator Project / Azahar Emulator Project
// Licensed under GPLv2 or any later version
// Refer to the misc/licenses/gplv2.txt file included.

#include <algorithm>
#include <array>
#include <atomic>
#include <bit>
#include <chrono>
#include <cstring>
#include <memory>
#include <thread>
#include <vector>
#include <catch2/catch_test_macros.hpp>
#include <nihstro/inline_assembly.h>
#include "common/arch.h"
#include "core/core.h"
#include "core/memory.h"
#include "video_core/pica/regs_internal.h"
#include "video_core/pica/regs_pipeline.h"
#include "video_core/pica/shader_setup.h"
#include "video_core/pica/shader_unit.h"
#include "video_core/pica/vertex_dedupe.h"
#include "video_core/pica/vertex_loader.h"
#include "video_core/pica/vertex_threads.h"
#include "video_core/pica/vertex_worker_pool.h"
#include "video_core/shader/shader.h"

#if CITRA_ARCH(x86_64)
#include <immintrin.h>
#endif

using Pica::VertexDedupe;
using Pica::VertexWorkerPool;
using namespace std::chrono_literals;
using DestRegister = nihstro::DestRegister;
using OpCode = nihstro::OpCode;
using SourceRegister = nihstro::SourceRegister;

namespace {
std::vector<u32> V(std::span<const u32> s) {
    return {s.begin(), s.end()};
}
using Vec = std::vector<u32>;
} // namespace

TEST_CASE("dedupe u16 keeps first-use order", "[video_core][vertex_threads]") {
    VertexDedupe d;
    const std::vector<u16> idx{5, 3, 5, 7, 3};
    d.BuildIndexed(std::span<const u16>{idx});
    REQUIRE(V(d.Unique()) == Vec{5, 3, 7});
    REQUIRE(V(d.Slots()) == Vec{0, 1, 0, 2, 1});
    REQUIRE(d.MaxVertex() == 7);
}

TEST_CASE("dedupe u8", "[video_core][vertex_threads]") {
    VertexDedupe d;
    const std::vector<u8> idx{2, 2, 9};
    d.BuildIndexed(std::span<const u8>{idx});
    REQUIRE(V(d.Unique()) == Vec{2, 9});
    REQUIRE(V(d.Slots()) == Vec{0, 0, 1});
    REQUIRE(d.MaxVertex() == 9);
}

TEST_CASE("dedupe builds don't leak", "[video_core][vertex_threads]") {
    VertexDedupe d;
    const std::vector<u16> a{5, 3, 5, 7, 3};
    d.BuildIndexed(std::span<const u16>{a});
    const std::vector<u16> b{3, 5};
    d.BuildIndexed(std::span<const u16>{b});
    REQUIRE(V(d.Unique()) == Vec{3, 5});
    REQUIRE(V(d.Slots()) == Vec{0, 1});
    REQUIRE(d.MaxVertex() == 5);
}

TEST_CASE("dedupe range", "[video_core][vertex_threads]") {
    VertexDedupe d;
    d.BuildRange(10, 4);
    REQUIRE(V(d.Unique()) == Vec{10, 11, 12, 13});
    REQUIRE(V(d.Slots()) == Vec{0, 1, 2, 3});
    REQUIRE(d.MaxVertex() == 13);
    d.BuildRange(70000, 3);
    REQUIRE(V(d.Unique()) == Vec{70000, 70001, 70002});
    REQUIRE(V(d.Slots()) == Vec{0, 1, 2});
    REQUIRE(d.MaxVertex() == 70002);
}

TEST_CASE("dedupe empty", "[video_core][vertex_threads]") {
    VertexDedupe d;
    d.BuildIndexed(std::span<const u16>{});
    REQUIRE(d.Unique().empty());
    REQUIRE(d.Slots().empty());
    REQUIRE(d.MaxVertex() == 0);
}

TEST_CASE("pool runs every chunk exactly once", "[video_core][vertex_threads]") {
    VertexWorkerPool pool(3, 0, 20us);
    REQUIRE(pool.NumWorkers() == 3);
    constexpr std::array<u32, 6> sizes{0, 1, 2, 7, 64, 257};
    for (u32 job = 0; job < 3000; ++job) {
        // On this job's stack: a chunk run after Run() returns lands in a later job's counters.
        std::array<std::atomic<u32>, 257> counters{};
        std::atomic<u32> out_of_range{0};
        const u32 num_chunks = sizes[job % sizes.size()];
        pool.Run(num_chunks, [&](u32 chunk) {
            if (chunk < counters.size()) {
                counters[chunk].fetch_add(1, std::memory_order_relaxed);
            } else {
                out_of_range.fetch_add(1, std::memory_order_relaxed);
            }
        });
        if (out_of_range.load() != 0) {
            FAIL("job " << job << ": chunk index out of range");
        }
        for (u32 i = 0; i < counters.size(); ++i) {
            const u32 expected = i < num_chunks ? 1 : 0;
            if (counters[i].load() != expected) {
                FAIL("job " << job << " (" << num_chunks << " chunks): chunk " << i << " ran "
                            << counters[i].load() << " times");
            }
        }
    }
}

TEST_CASE("pool with no workers runs on the caller", "[video_core][vertex_threads]") {
    VertexWorkerPool pool(0, 0, 20us);
    REQUIRE(pool.NumWorkers() == 0);
    const auto caller = std::this_thread::get_id();
    std::atomic<u32> calls{0};
    std::atomic<u32> off_caller{0};
    for (const u32 num_chunks : {0u, 1u, 64u, 257u}) {
        const u64 waited = pool.Run(num_chunks, [&](u32) {
            calls.fetch_add(1);
            if (std::this_thread::get_id() != caller) {
                off_caller.fetch_add(1);
            }
        });
        REQUIRE(waited == 0);
    }
    REQUIRE(calls.load() == 1 + 64 + 257);
    REQUIRE(off_caller.load() == 0);
}

TEST_CASE("parked workers wake for later jobs", "[video_core][vertex_threads]") {
    VertexWorkerPool pool(2, 0, 0us);
    const auto caller = std::this_thread::get_id();
    std::atomic<u32> on_workers{0};
    for (u32 job = 0; job < 200; ++job) {
        std::array<std::atomic<u32>, 64> counters{};
        pool.Run(64, [&](u32 chunk) {
            counters[chunk].fetch_add(1);
            if (std::this_thread::get_id() != caller) {
                on_workers.fetch_add(1);
            }
            std::this_thread::sleep_for(50us);
        });
        for (u32 i = 0; i < counters.size(); ++i) {
            if (counters[i].load() != 1) {
                FAIL("job " << job << ": chunk " << i << " ran " << counters[i].load() << " times");
            }
        }
        std::this_thread::sleep_for(2ms);
    }
    REQUIRE(on_workers.load() > 0);
}

TEST_CASE("pool destructor joins", "[video_core][vertex_threads]") {
    for (u32 i = 0; i < 50; ++i) {
        std::atomic<u32> calls{0};
        VertexWorkerPool pool(3, 0, 20us);
        if (i % 2 == 0) {
            pool.Run(64, [&](u32) { calls.fetch_add(1); });
            REQUIRE(calls.load() == 64);
        }
    }
}

#if CITRA_ARCH(x86_64)
namespace {
constexpr u32 FtzDaz = _MM_FLUSH_ZERO_MASK | _MM_DENORMALS_ZERO_MASK;

/// Restores the caller's MXCSR.
struct MxcsrGuard {
    const u32 saved = _mm_getcsr();
    ~MxcsrGuard() {
        _mm_setcsr(saved);
    }
};

struct DenormalJobs {
    u32 off_caller = 0;
    std::vector<float> results;
};

/// Runs 64-chunk jobs whose chunks compute a denormal product, until a chunk ran off the caller
/// (at most 1000 jobs).
DenormalJobs RunDenormalJobs(VertexWorkerPool& pool) {
    const auto caller = std::this_thread::get_id();
    DenormalJobs out;
    for (u32 job = 0; job < 1000 && out.off_caller == 0; ++job) {
        std::array<float, 64> results{};
        std::atomic<u32> off_caller{0};
        pool.Run(64, [&](u32 chunk) {
            volatile float a = 1e-30f;
            results[chunk] = a * 1e-10f;
            if (std::this_thread::get_id() != caller) {
                off_caller.fetch_add(1, std::memory_order_relaxed);
            }
            // Time for parked or descheduled workers to claim chunks on a busy machine
            std::this_thread::sleep_for(20us);
        });
        out.off_caller += off_caller.load();
        out.results.insert(out.results.end(), results.begin(), results.end());
    }
    return out;
}
} // namespace

TEST_CASE("workers run with the caller's flush-to-zero", "[video_core][vertex_threads]") {
    // Created first: a new thread starts with its creator's MXCSR.
    VertexWorkerPool pool(3, 0, 20us);
    DenormalJobs run;
    {
        const MxcsrGuard guard;
        _mm_setcsr(_mm_getcsr() | FtzDaz);
        run = RunDenormalJobs(pool);
    }
    // Checked with the caller's MXCSR back: DAZ would read a denormal as zero.
    REQUIRE(run.off_caller > 0);
    for (u32 i = 0; i < run.results.size(); ++i) {
        if (std::bit_cast<u32>(run.results[i]) != 0) {
            FAIL("result " << i << " is " << run.results[i] << ", not 0");
        }
    }
}

TEST_CASE("workers drop flush-to-zero when the caller has it off", "[video_core][vertex_threads]") {
    VertexWorkerPool pool(3, 0, 20us);
    const MxcsrGuard guard;
    // So the workers last ran with flush-to-zero on
    _mm_setcsr(_mm_getcsr() | FtzDaz);
    REQUIRE(RunDenormalJobs(pool).off_caller > 0);
    _mm_setcsr(_mm_getcsr() & ~FtzDaz);
    const DenormalJobs run = RunDenormalJobs(pool);
    REQUIRE(run.off_caller > 0);
    for (u32 i = 0; i < run.results.size(); ++i) {
        if (std::bit_cast<u32>(run.results[i]) == 0) {
            FAIL("result " << i << " is 0, not a denormal");
        }
    }
}
#endif

namespace {
// One loader: attr 0 FLOAT x3, attr 1 UBYTE x4, attr 2 SHORT x2 (20 bytes per vertex); attr 3
// default
void ConfigureLoader(Pica::PipelineRegs& regs) {
    auto& va = regs.vertex_attributes;
    va.format0.Assign(Pica::PipelineRegs::VertexAttributeFormat::FLOAT);
    va.size0.Assign(2);
    va.format1.Assign(Pica::PipelineRegs::VertexAttributeFormat::UBYTE);
    va.size1.Assign(3);
    va.format2.Assign(Pica::PipelineRegs::VertexAttributeFormat::SHORT);
    va.size2.Assign(1);
    va.attribute_mask.Assign(1u << 3);
    va.max_attribute_index.Assign(3);
    auto& loader = va.attribute_loaders[0];
    loader.data_offset.Assign(0);
    loader.comp0.Assign(0);
    loader.comp1.Assign(1);
    loader.comp2.Assign(2);
    loader.byte_count.Assign(20);
    loader.component_count.Assign(3);
}
} // namespace

TEST_CASE("resolved loader matches LoadVertex", "[video_core][vertex_threads]") {
    Core::System system;
    Memory::MemorySystem memory{system};
    Pica::PipelineRegs regs{};
    ConfigureLoader(regs);
    const PAddr base = Memory::FCRAM_PADDR + 0x1000;
    u8* data = memory.GetPhysicalPointer(base);
    REQUIRE(data != nullptr);
    for (u32 i = 0; i < 256 * 20; ++i) {
        data[i] = static_cast<u8>(i * 37 + 11);
    }
    // Keep the floats finite
    for (u32 v = 0; v < 256; ++v) {
        const float f[3] = {static_cast<float>(v) * 0.5f, -1.25f * v, 3.0f};
        std::memcpy(data + v * 20, f, sizeof(f));
    }

    const Pica::VertexLoader loader{memory, regs};
    Pica::VertexArrays arrays;
    REQUIRE(loader.ResolveArrays(base, 255, arrays));
    REQUIRE(arrays.base[3] == nullptr);

    Pica::AttributeBuffer defaults{};
    for (u32 a = 0; a < 16; ++a) {
        for (u32 c = 0; c < 4; ++c) {
            defaults[a][c] = Pica::f24::FromFloat32(static_cast<float>(a * 4 + c));
        }
    }
    for (const u32 vertex : {0u, 1u, 17u, 255u}) {
        Pica::AttributeBuffer expected{};
        Pica::AttributeBuffer actual{};
        Pica::AttributeBuffer defaults_copy = defaults;
        loader.LoadVertex(base, 0, vertex, expected, defaults_copy);
        loader.LoadVertex(arrays, vertex, actual, defaults);
        REQUIRE(std::memcmp(&expected, &actual,
                            sizeof(expected[0]) * loader.GetNumTotalAttributes()) == 0);
    }
}

// The brief's "past the region end" and "unmapped" cases are not run: an unmapped lookup makes
// Memory::MemorySystem::GetPhysMemRegionInfo() (src/core/memory.cpp) log the running core's PC,
// which dereferences a null core in a bare Core::System. This case reaches the same false return
// without that log: both ends map, but to different host buffers.
TEST_CASE("ResolveArrays fails when the array spans two regions", "[video_core][vertex_threads]") {
    Core::System system;
    Memory::MemorySystem memory{system};
    Pica::PipelineRegs regs{};
    ConfigureLoader(regs);
    const Pica::VertexLoader loader{memory, regs};
    Pica::VertexArrays arrays;
    // Vertex 0 is in VRAM, the last vertex lands in DSP RAM
    const PAddr base = Memory::VRAM_PADDR + Memory::VRAM_SIZE - 16;
    REQUIRE_FALSE(loader.ResolveArrays(base, 6350000, arrays));
    // Same loader, array inside one region
    REQUIRE(loader.ResolveArrays(Memory::FCRAM_PADDR + 0x1000, 10, arrays));
}

TEST_CASE("ResolveArrays rejects an array ending at the region end",
          "[video_core][vertex_threads]") {
    Core::System system;
    Memory::MemorySystem memory{system};
    Pica::PipelineRegs regs{};
    ConfigureLoader(regs);
    const Pica::VertexLoader loader{memory, regs};
    Pica::VertexArrays arrays;
    // Last array byte (SHORT x2 at offset 16 -> ends at offset 19 of the last vertex)
    const PAddr end = Memory::FCRAM_PADDR + Memory::FCRAM_N3DS_SIZE;
    REQUIRE(loader.ResolveArrays(end - 20, 0, arrays));
    REQUIRE_FALSE(loader.ResolveArrays(end - 19, 0, arrays));
}

namespace {
std::unique_ptr<Pica::ShaderSetup> CompileShaderSetup(
    std::initializer_list<nihstro::InlineAsm> code) {
    const auto shbin = nihstro::InlineAsm::CompileToRawBinary(code);
    auto shader = std::make_unique<Pica::ShaderSetup>();
    Pica::ProgramCode program_code{};
    Pica::SwizzleData swizzle_data{};
    std::transform(shbin.program.begin(), shbin.program.end(), program_code.begin(),
                   [](const auto& x) { return x.hex; });
    std::transform(shbin.swizzle_table.begin(), shbin.swizzle_table.end(), swizzle_data.begin(),
                   [](const auto& x) { return x.hex; });
    shader->UpdateProgramCode(program_code);
    shader->UpdateSwizzleData(swizzle_data);
    return shader;
}

void MapPosition(Pica::RasterizerRegs& rast) {
    rast.vs_output_total.Assign(1);
    auto& map = rast.vs_output_attributes[0];
    map.map_x.Assign(Pica::RasterizerRegs::VSOutputAttributes::POSITION_X);
    map.map_y.Assign(Pica::RasterizerRegs::VSOutputAttributes::POSITION_Y);
    map.map_z.Assign(Pica::RasterizerRegs::VSOutputAttributes::POSITION_Z);
    map.map_w.Assign(Pica::RasterizerRegs::VSOutputAttributes::POSITION_W);
}

void FillInput(u32 v, Pica::AttributeBuffer& input) {
    const float f = static_cast<float>(v);
    input[0] = {Pica::f24::FromFloat32(f), Pica::f24::FromFloat32(2 * f),
                Pica::f24::FromFloat32(-f), Pica::f24::FromFloat32(1.0f)};
}
} // namespace

TEST_CASE("config parse", "[video_core][vertex_threads]") {
    auto c = Pica::ParseVertexThreadsConfig(nullptr, nullptr, nullptr, nullptr, nullptr);
    REQUIRE(c.threads == 0);
    REQUIRE(c.min_unique == 64);
    REQUIRE_FALSE(c.check);
    REQUIRE(c.nice == 0);
    REQUIRE(c.spin_us == 50);
    c = Pica::ParseVertexThreadsConfig("", "", "", "", "");
    REQUIRE(c.threads == 0);
    REQUIRE(c.min_unique == 64);
    c = Pica::ParseVertexThreadsConfig("3", nullptr, nullptr, nullptr, nullptr);
    REQUIRE(c.threads == 3);
    c = Pica::ParseVertexThreadsConfig("1", nullptr, nullptr, nullptr, nullptr);
    REQUIRE(c.threads == 0);
    c = Pica::ParseVertexThreadsConfig("9", nullptr, nullptr, nullptr, nullptr);
    REQUIRE(c.threads == 4);
    c = Pica::ParseVertexThreadsConfig("2", "256", "1", "5", "0");
    REQUIRE(c.threads == 2);
    REQUIRE(c.min_unique == 256);
    REQUIRE(c.check);
    REQUIRE(c.nice == 5);
    REQUIRE(c.spin_us == 0);
    c = Pica::ParseVertexThreadsConfig(nullptr, nullptr, "0", nullptr, nullptr);
    REQUIRE_FALSE(c.check);
    for (const char* bad : {"abc", "-5", "4294967296", "12x"}) {
        c = Pica::ParseVertexThreadsConfig(bad, bad, nullptr, bad, bad);
        REQUIRE(c.threads == 0);
        REQUIRE(c.min_unique == 64);
        REQUIRE(c.spin_us == 50);
    }
    c = Pica::ParseVertexThreadsConfig(nullptr, nullptr, nullptr, "-5", nullptr);
    REQUIRE(c.nice == -5);
    c = Pica::ParseVertexThreadsConfig(nullptr, nullptr, nullptr, "12x", nullptr);
    REQUIRE(c.nice == 0);
    c = Pica::ParseVertexThreadsConfig(nullptr, "4294967295", nullptr, nullptr, nullptr);
    REQUIRE(c.min_unique == 4294967295u);
}

TEST_CASE("used fields", "[video_core][vertex_threads]") {
    using namespace Pica;
    auto regs = std::make_unique<RegsInternal>();
    std::memset(regs.get(), 0, sizeof(RegsInternal));
    regs->lighting.disable.Assign(1);
    REQUIRE(UsedOutputFields(*regs) == (FieldPos | FieldColor));
    regs->lighting.disable.Assign(0);
    REQUIRE(UsedOutputFields(*regs) == (FieldPos | FieldColor | FieldQuat | FieldView));
    regs->lighting.disable.Assign(1);
    regs->texturing.main_config.texture0_enable.Assign(1);
    REQUIRE(UsedOutputFields(*regs) == (FieldPos | FieldColor | FieldTc0 | FieldTc0W));
    regs->texturing.main_config.texture0_enable.Assign(0);
    regs->texturing.main_config.texture2_enable.Assign(1);
    regs->texturing.main_config.texture2_use_coord1.Assign(1);
    REQUIRE(UsedOutputFields(*regs) == (FieldPos | FieldColor | FieldTc1 | FieldTc2));

    // texture1 alone reads tc1 only.
    regs->texturing.main_config.texture2_enable.Assign(0);
    regs->texturing.main_config.texture2_use_coord1.Assign(0);
    regs->texturing.main_config.texture1_enable.Assign(1);
    REQUIRE(UsedOutputFields(*regs) == (FieldPos | FieldColor | FieldTc1));
    regs->texturing.main_config.texture1_enable.Assign(0);

    // texture2 without use_coord1 reads tc2 only.
    regs->texturing.main_config.texture2_enable.Assign(1);
    REQUIRE(UsedOutputFields(*regs) == (FieldPos | FieldColor | FieldTc2));
    regs->texturing.main_config.texture2_enable.Assign(0);

    // Procedural texture coordinates, with every texture unit off.
    regs->texturing.main_config.texture3_enable.Assign(1);
    const u32 proctex_fields[3] = {FieldTc0, FieldTc1, FieldTc2};
    for (u32 coord = 0; coord < 3; ++coord) {
        regs->texturing.main_config.texture3_coordinates.Assign(coord);
        REQUIRE(UsedOutputFields(*regs) == (FieldPos | FieldColor | proctex_fields[coord]));
    }
}

TEST_CASE("first mismatch", "[video_core][vertex_threads]") {
    using namespace Pica;
    OutputVertex a{};
    OutputVertex b{};
    std::memset(&a, 0, sizeof(a));
    std::memset(&b, 0, sizeof(b));
    constexpr u32 used = FieldPos | FieldColor;
    REQUIRE_FALSE(FirstMismatch(a, b, used).has_value());
    b.tc1.x = f24::FromFloat32(1.0f);
    REQUIRE_FALSE(FirstMismatch(a, b, used).has_value());
    REQUIRE(FirstMismatch(a, b, used | FieldTc1) == FieldTc1);
    b = a;
    b.color.y = f24::FromFloat32(0.5f);
    REQUIRE(FirstMismatch(a, b, used) == FieldColor);
    b = a;
    a.pos.x = f24::FromFloat32(0.0f);
    b.pos.x = f24::FromFloat32(-0.0f);
    REQUIRE(FirstMismatch(a, b, used) == FieldPos);
    REQUIRE(std::string(OutputFieldName(FieldTc0W)) == "tc0_w");
}

TEST_CASE("ShadeChunk matches a serial loop", "[video_core][vertex_threads]") {
    using namespace Pica;
    for (const bool jit : {false, true}) {
        const auto engine = CreateEngine(jit);
        auto setup = CompileShaderSetup({
            {OpCode::Id::MUL, DestRegister::MakeOutput(0), SourceRegister::MakeInput(0),
             SourceRegister::MakeFloat(0)},
            {OpCode::Id::END},
        });
        setup->uniforms.f[0] = {f24::FromFloat32(2), f24::FromFloat32(3), f24::FromFloat32(4),
                                f24::FromFloat32(5)};
        engine->SetupBatch(*setup, 0);
        ShaderRegs vs_regs{};
        vs_regs.output_mask.Assign(1);
        vs_regs.max_input_attribute_index.Assign(0);
        RasterizerRegs rast{};
        MapPosition(rast);

        constexpr u32 count = 1000;
        std::vector<u32> unique(count);
        for (u32 i = 0; i < count; ++i) {
            unique[i] = i;
        }
        std::vector<OutputVertex> parallel(count);
        const ShadeDraw draw{engine.get(), setup.get(), &vs_regs, &rast,
                             unique,       parallel,    FillInput};
        VertexWorkerPool pool(3, 0, 20us);
        pool.Run(NumVertexChunks(count), [&](u32 chunk) { ShadeChunk(draw, chunk); });

        ShaderUnit unit;
        AttributeBuffer out{};
        for (u32 k = 0; k < count; ++k) {
            AttributeBuffer input{};
            FillInput(unique[k], input);
            unit.LoadInput(vs_regs, input);
            engine->Run(*setup, unit);
            unit.WriteOutput(vs_regs, out);
            const OutputVertex expected(rast, out);
            if (std::memcmp(&expected.pos, &parallel[k].pos, sizeof(expected.pos)) != 0) {
                FAIL("jit " << jit << ": vertex " << k << " differs");
            }
        }
    }
}

TEST_CASE("ShadeChunk output is the same for any thread count", "[video_core][vertex_threads]") {
    using namespace Pica;
    const auto engine = CreateEngine(false);
    auto setup = CompileShaderSetup({
        {OpCode::Id::ADD, DestRegister::MakeTemporary(0), SourceRegister::MakeTemporary(0),
         SourceRegister::MakeInput(0)},
        {OpCode::Id::MOV, DestRegister::MakeOutput(0), SourceRegister::MakeTemporary(0)},
        {OpCode::Id::END},
    });
    engine->SetupBatch(*setup, 0);
    ShaderRegs vs_regs{};
    vs_regs.output_mask.Assign(1);
    vs_regs.max_input_attribute_index.Assign(0);
    RasterizerRegs rast{};
    MapPosition(rast);

    constexpr u32 count = 200;
    std::vector<u32> unique(count);
    for (u32 i = 0; i < count; ++i) {
        unique[i] = i + 1;
    }
    std::vector<std::vector<OutputVertex>> results;
    for (const u32 workers : {0u, 1u, 3u}) {
        std::vector<OutputVertex> outputs(count);
        const ShadeDraw draw{engine.get(), setup.get(), &vs_regs, &rast,
                             unique,       outputs,     FillInput};
        VertexWorkerPool pool(workers, 0, 20us);
        pool.Run(NumVertexChunks(count), [&](u32 chunk) { ShadeChunk(draw, chunk); });
        results.push_back(std::move(outputs));
    }
    for (u32 r = 1; r < results.size(); ++r) {
        REQUIRE(std::memcmp(results[0].data(), results[r].data(), count * sizeof(OutputVertex)) ==
                0);
    }
    float sum = 0;
    for (u32 k = 0; k < count; ++k) {
        if (k % VertexChunkSize == 0) {
            sum = 0;
        }
        sum += static_cast<float>(unique[k]);
        if (results[0][k].pos.x.ToFloat32() != sum) {
            FAIL("vertex " << k << ": " << results[0][k].pos.x.ToFloat32() << " != " << sum);
        }
    }
}
