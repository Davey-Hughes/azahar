// Copyright 2026 Azahar Emulator Project
// Licensed under GPLv2 or any later version
// Refer to the misc/licenses/gplv2.txt file included.

#include <algorithm>
#include <initializer_list>
#include <string>
#include <string_view>
#include <utility>
#include <catch2/catch_test_macros.hpp>
#include <fmt/format.h>
#include <nihstro/inline_assembly.h>
#include "video_core/shader/generator/glsl_shader_decompiler.h"

using nihstro::DestRegister;
using nihstro::Instruction;
using nihstro::OpCode;
using nihstro::SourceRegister;

namespace {

struct Program {
    Pica::ProgramCode code{};
    Pica::SwizzleData swizzle{};
};

/// Assembles `code`, then puts each flow control instruction at its offset (nihstro can't assemble
/// them).
Program MakeProgram(std::initializer_list<nihstro::InlineAsm> code,
                    std::initializer_list<std::pair<u32, Instruction>> flow_control) {
    const auto shbin = nihstro::InlineAsm::CompileToRawBinary(code);
    Program program;
    std::transform(shbin.program.begin(), shbin.program.end(), program.code.begin(),
                   [](const auto& x) { return x.hex; });
    std::transform(shbin.swizzle_table.begin(), shbin.swizzle_table.end(), program.swizzle.begin(),
                   [](const auto& x) { return x.hex; });
    for (const auto& [offset, instr] : flow_control) {
        program.code[offset] = instr.hex;
    }
    return program;
}

nihstro::InlineAsm MovFromInput(u32 input) {
    return {OpCode::Id::MOV, DestRegister::MakeOutput(0), SourceRegister::MakeInput(input)};
}

/// 0: the given jump to 3, 1: mov o0, v0, 2: end, 3: mov o0, v1, 4: end
Program MakeForwardJumpProgram(const Instruction& jump) {
    return MakeProgram(
        {{OpCode::Id::NOP}, MovFromInput(0), {OpCode::Id::END}, MovFromInput(1), {OpCode::Id::END}},
        {{0, jump}});
}

Instruction MakeJmpu(u32 dest_offset, bool invert_test) {
    Instruction jmpu = {};
    jmpu.opcode = nihstro::OpCode(OpCode::Id::JMPU);
    jmpu.flow_control.bool_uniform_id = 0;
    jmpu.flow_control.dest_offset = dest_offset;
    jmpu.flow_control.num_instructions = invert_test ? 1 : 0;
    return jmpu;
}

Instruction MakeJmpcX(u32 dest_offset) {
    Instruction jmpc = {};
    jmpc.opcode = nihstro::OpCode(OpCode::Id::JMPC);
    jmpc.flow_control.op = Instruction::FlowControlType::Op::JustX;
    jmpc.flow_control.refx = 1;
    jmpc.flow_control.dest_offset = dest_offset;
    return jmpc;
}

std::string Decompile(const Program& program, u32 main_offset, u16 bool_uniforms) {
    return Pica::Shader::Generator::GLSL::DecompileProgram(
        program.code, program.swizzle, main_offset, bool_uniforms,
        [](u32 index) { return fmt::format("vs_in_reg{}", index); },
        [](u32 index) { return fmt::format("vs_out_attr{}", index); }, false);
}

bool Contains(const std::string& text, std::string_view part) {
    return text.find(part) != std::string::npos;
}

} // namespace

TEST_CASE("JMPU with known bools needs no jump dispatcher", "[video_core][shader][glsl]") {
    const Program program = MakeForwardJumpProgram(MakeJmpu(3, false));

    const std::string taken = Decompile(program, 0, 0b1);
    REQUIRE(!taken.empty());
    REQUIRE_FALSE(Contains(taken, "jmp_to"));
    REQUIRE(Contains(taken, "vs_in_reg1"));
    REQUIRE_FALSE(Contains(taken, "vs_in_reg0"));

    const std::string not_taken = Decompile(program, 0, 0b0);
    REQUIRE(!not_taken.empty());
    REQUIRE_FALSE(Contains(not_taken, "jmp_to"));
    REQUIRE(Contains(not_taken, "vs_in_reg0"));
    REQUIRE_FALSE(Contains(not_taken, "vs_in_reg1"));
}

TEST_CASE("JMPU with an inverted test", "[video_core][shader][glsl]") {
    const Program program = MakeForwardJumpProgram(MakeJmpu(3, true));

    const std::string taken = Decompile(program, 0, 0b0);
    REQUIRE_FALSE(Contains(taken, "jmp_to"));
    REQUIRE(Contains(taken, "vs_in_reg1"));
    REQUIRE_FALSE(Contains(taken, "vs_in_reg0"));

    const std::string not_taken = Decompile(program, 0, 0b1);
    REQUIRE_FALSE(Contains(not_taken, "jmp_to"));
    REQUIRE(Contains(not_taken, "vs_in_reg0"));
    REQUIRE_FALSE(Contains(not_taken, "vs_in_reg1"));
}

TEST_CASE("Backward JMPU that doesn't loop needs no jump dispatcher",
          "[video_core][shader][glsl]") {
    // 0: mov o0, v1, 1: end, 2 (main): jmpu b0 -> 0, 3: mov o0, v0, 4: end
    const Program program = MakeProgram(
        {MovFromInput(1), {OpCode::Id::END}, {OpCode::Id::NOP}, MovFromInput(0), {OpCode::Id::END}},
        {{2, MakeJmpu(0, false)}});

    const std::string taken = Decompile(program, 2, 0b1);
    REQUIRE(!taken.empty());
    REQUIRE_FALSE(Contains(taken, "jmp_to"));
    REQUIRE(Contains(taken, "vs_in_reg1"));
    REQUIRE_FALSE(Contains(taken, "vs_in_reg0"));
}

TEST_CASE("JMPU loop keeps the jump dispatcher", "[video_core][shader][glsl]") {
    // 0: jmpc x -> 3, 1: mov o0, v0, 2: jmpu b0 -> 0, 3: mov o0, v1, 4: end
    const Program program = MakeProgram(
        {{OpCode::Id::NOP}, MovFromInput(0), {OpCode::Id::NOP}, MovFromInput(1), {OpCode::Id::END}},
        {{0, MakeJmpcX(3)}, {2, MakeJmpu(0, false)}});

    const std::string glsl = Decompile(program, 0, 0b1);
    REQUIRE(!glsl.empty());
    REQUIRE(Contains(glsl, "jmp_to = 0u"));
    REQUIRE(Contains(glsl, "vs_in_reg0"));
    REQUIRE(Contains(glsl, "vs_in_reg1"));
}

TEST_CASE("JMPU out of a dispatcher case", "[video_core][shader][glsl]") {
    // 0: jmpc x -> 2, 1: jmpu b0 -> 4, 2: mov o0, v0, 3: end, 4: mov o0, v1, 5: end
    // The JMPC makes a dispatcher; the followed JMPU leaves case 0 for a new case 4.
    const Program program = MakeProgram({{OpCode::Id::NOP},
                                         {OpCode::Id::NOP},
                                         MovFromInput(0),
                                         {OpCode::Id::END},
                                         MovFromInput(1),
                                         {OpCode::Id::END}},
                                        {{0, MakeJmpcX(2)}, {1, MakeJmpu(4, false)}});

    const std::string glsl = Decompile(program, 0, 0b1);
    REQUIRE(!glsl.empty());
    REQUIRE(Contains(glsl, "jmp_to = 4u"));
    REQUIRE(Contains(glsl, "case 4u"));
    REQUIRE(Contains(glsl, "vs_in_reg0"));
    REQUIRE(Contains(glsl, "vs_in_reg1"));
}

TEST_CASE("JMPC keeps the jump dispatcher", "[video_core][shader][glsl]") {
    const std::string glsl = Decompile(MakeForwardJumpProgram(MakeJmpcX(3)), 0, 0b0);
    REQUIRE(Contains(glsl, "jmp_to"));
    REQUIRE(Contains(glsl, "vs_in_reg0"));
    REQUIRE(Contains(glsl, "vs_in_reg1"));
}
