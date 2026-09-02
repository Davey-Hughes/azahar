add_library(audio_core STATIC EXCLUDE_FROM_ALL
    audio_types.h
    codec.cpp
    codec.h
    dsp_interface.cpp
    dsp_interface.h
    hle/aac_decoder.cpp
    hle/aac_decoder.h
    hle/common.h
    hle/decoder.cpp
    hle/decoder.h
    hle/filter.cpp
    hle/filter.h
    hle/hle.cpp
    hle/hle.h
    hle/mixers.cpp
    hle/mixers.h
    hle/shared_memory.h
    hle/source.cpp
    hle/source.h
    lle/lle.cpp
    lle/lle.h
    input.h
    input_details.cpp
    input_details.h
    interpolate.cpp
    interpolate.h
    null_input.h
    null_sink.h
    precompiled_headers.h
    sink.h
    sink_details.cpp
    sink_details.h
    speedup_lowpass.h
    speedup_params.h
    static_input.cpp
    static_input.h
    time_stretch.cpp
    time_stretch.h
    wsola_stretcher.cpp
    wsola_stretcher.h

    $<$<BOOL:${ENABLE_SDL2}>:sdl2_sink.cpp sdl2_sink.h>
    $<$<BOOL:${ENABLE_CUBEB}>:cubeb_sink.cpp cubeb_sink.h cubeb_input.cpp cubeb_input.h>
    $<$<BOOL:${ENABLE_LIBRETRO}>:libretro_sink.cpp libretro_sink.h libretro_input.cpp libretro_input.h>
    $<$<BOOL:${ENABLE_OPENAL}>:openal_input.cpp openal_input.h openal_sink.cpp openal_sink.h>
)

create_target_directory_groups(audio_core)

target_link_libraries(audio_core PUBLIC citra_common citra_core)
target_link_libraries(audio_core PRIVATE faad2 SoundTouch teakra)

if(ENABLE_SDL2)
    target_link_libraries(audio_core PRIVATE SDL2::SDL2)
    target_compile_definitions(audio_core PRIVATE HAVE_SDL2)
endif()

if(ENABLE_CUBEB)
    target_link_libraries(audio_core PRIVATE cubeb)
    target_compile_definitions(audio_core PUBLIC HAVE_CUBEB)
endif()

if(ENABLE_OPENAL)
    target_link_libraries(audio_core PRIVATE OpenAL)
    target_compile_definitions(audio_core PUBLIC HAVE_OPENAL)
    add_definitions(-DAL_LIBTYPE_STATIC)
endif()

if (CITRA_USE_PRECOMPILED_HEADERS)
    target_precompile_headers(audio_core PRIVATE precompiled_headers.h)
endif()

if (ENABLE_AUDIOBENCH)
    # audiobench needs -fno-access-control, which is GCC/Clang only. Without it the target does
    # not compile at all: RunSearchActivity() (tools/audiobench.cpp) reads WsolaStretcher's
    # private search state directly, so MSVC would fail with "member is private" errors. The
    # whole target is therefore gated on the compiler rather than just the flag, so turning the
    # option on does not break an MSVC build.
    if (NOT MSVC AND (CMAKE_CXX_COMPILER_ID STREQUAL "GNU" OR CMAKE_CXX_COMPILER_ID MATCHES "Clang"))
        add_executable(audiobench tools/audiobench.cpp)
        target_link_libraries(audiobench PRIVATE audio_core citra_common Threads::Threads)
        # Reading that private state is how the search is measured without adding any test-only
        # surface to the shipped class. This relaxes access control for this diagnostic target
        # only - WsolaStretcher itself, and every other target that links it, is unaffected.
        target_compile_options(audiobench PRIVATE -fno-access-control)
        create_target_directory_groups(audiobench)
    else()
        message(WARNING
            "ENABLE_AUDIOBENCH is ON but the audiobench target needs -fno-access-control, which "
            "only GCC and Clang provide. Skipping it for compiler '${CMAKE_CXX_COMPILER_ID}'.")
    endif()
endif()
