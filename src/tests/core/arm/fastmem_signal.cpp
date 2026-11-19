// Copyright 2026 Azahar Emulator Project
// Licensed under GPLv2 or any later version
// Refer to the misc/licenses/gplv2.txt file included.

#ifdef __linux__

#include <chrono>
#include <csignal>
#include <cstring>
#include <optional>
#include <thread>
#include <sys/mman.h>
#include <sys/prctl.h>
#include <sys/wait.h>
#include <unistd.h>
#include <catch2/catch_test_macros.hpp>
#include <dynarmic/interface/A32/a32.h>
#include "common/common_types.h"

namespace {

constexpr std::size_t ARENA_SIZE = (std::size_t{1} << 32) + 0x1000;
constexpr u32 FALLBACK_VALUE = 0xCAFEF00D;

/// Guest code lives in arena page 0; everything else reads FALLBACK_VALUE through the callbacks.
class TestCallbacks final : public Dynarmic::A32::UserCallbacks {
public:
    explicit TestCallbacks(const u8* arena_) : arena{arena_} {}

    u32 Read(u32 vaddr) {
        if (vaddr < 0x1000 - 3) {
            u32 value;
            std::memcpy(&value, arena + vaddr, sizeof(value));
            return value;
        }
        return FALLBACK_VALUE;
    }

    std::optional<u32> MemoryReadCode(u32 vaddr) override {
        return Read(vaddr);
    }
    u8 MemoryRead8(u32 vaddr) override {
        return static_cast<u8>(Read(vaddr));
    }
    u16 MemoryRead16(u32 vaddr) override {
        return static_cast<u16>(Read(vaddr));
    }
    u32 MemoryRead32(u32 vaddr) override {
        return Read(vaddr);
    }
    u64 MemoryRead64(u32 vaddr) override {
        return Read(vaddr);
    }
    void MemoryWrite8(u32, u8) override {}
    void MemoryWrite16(u32, u16) override {}
    void MemoryWrite32(u32, u32) override {}
    void MemoryWrite64(u32, u64) override {}
    void InterpreterFallback(u32, std::size_t) override {
        _exit(10);
    }
    void CallSVC(u32) override {}
    void ExceptionRaised(u32, Dynarmic::A32::Exception) override {
        _exit(11);
    }
    void AddTicks(u64 ticks) override {
        ticks_left = ticks > ticks_left ? 0 : ticks_left - ticks;
    }
    u64 GetTicksRemaining() override {
        return ticks_left;
    }

    u64 ticks_left = 2;

private:
    const u8* arena;
};

/// Runs a fastmem JIT through one faulting load, reports that on progress_fd, then faults from
/// C++. Exits only on failure.
[[noreturn]] void Child(int progress_fd) {
    // No core dump for the expected crash.
    prctl(PR_SET_DUMPABLE, 0);

    void* reservation = mmap(nullptr, ARENA_SIZE, PROT_NONE,
                             MAP_PRIVATE | MAP_ANONYMOUS | MAP_NORESERVE, -1, 0);
    if (reservation == MAP_FAILED) {
        _exit(2);
    }
    u8* arena = static_cast<u8*>(reservation);
    if (mprotect(arena, 0x1000, PROT_READ | PROT_WRITE) != 0) {
        _exit(3);
    }
    const u32 code[] = {
        0xE5910000, // ldr r0, [r1]
        0xEAFFFFFE, // b .
    };
    std::memcpy(arena, code, sizeof(code));

    TestCallbacks callbacks{arena};
    Dynarmic::A32::UserConfig config;
    config.callbacks = &callbacks;
    config.fastmem_pointer = reinterpret_cast<uintptr_t>(arena);
    config.recompile_on_fastmem_failure = true;
    config.fastmem_exclusive_access = false;
    Dynarmic::A32::Jit jit{config};
    const auto run = [&] {
        jit.Regs()[1] = 0x5000;
        jit.Regs()[15] = 0;
        jit.SetCpsr(0x000001D0); // User mode, ARM
        callbacks.ticks_left = 2;
        jit.Run();
        return jit.Regs()[0];
    };

    // Readable: the load goes straight to the arena.
    constexpr u32 arena_value = 0x11111111;
    if (mprotect(arena + 0x5000, 0x1000, PROT_READ | PROT_WRITE) != 0) {
        _exit(7);
    }
    std::memcpy(arena + 0x5000, &arena_value, sizeof(arena_value));
    if (run() != arena_value) {
        _exit(8);
    }

    // Inaccessible: the same block faults into the fallback.
    if (mprotect(arena + 0x5000, 0x1000, PROT_NONE) != 0) {
        _exit(9);
    }
    if (run() != FALLBACK_VALUE) {
        _exit(4);
    }
    const char done = 'J';
    if (write(progress_fd, &done, 1) != 1) {
        _exit(6);
    }

    [[maybe_unused]] volatile u8 value = *static_cast<volatile u8*>(arena + 0x5000);
    _exit(5);
}

} // Anonymous namespace

TEST_CASE("arm_dynarmic.Fastmem non-JIT faults stay fatal", "[core][arm][fastmem]") {
    int progress[2];
    REQUIRE(pipe(progress) == 0);
    const pid_t pid = fork();
    REQUIRE(pid >= 0);
    if (pid == 0) {
        close(progress[0]);
        Child(progress[1]);
    }
    close(progress[1]);

    int status = 0;
    pid_t done = 0;
    for (int i = 0; i < 200 && done == 0; i++) {
        done = waitpid(pid, &status, WNOHANG);
        if (done == 0) {
            std::this_thread::sleep_for(std::chrono::milliseconds(50));
        }
    }
    if (done == 0) {
        kill(pid, SIGKILL);
        waitpid(pid, &status, 0);
        FAIL("the child hung instead of crashing");
    }
    char reached = 0;
    const bool jit_ok = read(progress[0], &reached, 1) == 1 && reached == 'J';
    close(progress[0]);
    INFO("exit status " << (WIFEXITED(status) ? WEXITSTATUS(status) : -1));
    REQUIRE(jit_ok);
    REQUIRE(WIFSIGNALED(status));
    REQUIRE(WTERMSIG(status) == SIGSEGV);
}

#endif
