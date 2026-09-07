# Portable build for the CUDA rasterizer.
#
# src/Makefile still works and is still the WSL path; this exists so the
# project also builds natively on Windows (MSVC) and Linux, where the frame
# no longer has to cross WSL2's ~1.8 GB/s paravirtualization boundary and
# CUDA/OpenGL interop is actually reachable.
#
#   cmake -B build -DCMAKE_BUILD_TYPE=Release
#   cmake --build build --config Release
#
# Windows needs an SDL2 built for MSVC (the MinGW package will not link).
# With vcpkg:
#   cmake -B build -DCMAKE_TOOLCHAIN_FILE=<vcpkg>/scripts/buildsystems/vcpkg.cmake
# With the SDL2-devel-*-VC release:
#   cmake -B build -DCMAKE_PREFIX_PATH=<path>/SDL2-2.30.12

cmake_minimum_required(VERSION 3.21)

# Must be set BEFORE project(), which otherwise defines it to a conservative
# default (52) that a later `if(NOT DEFINED ...)` can never override. Building
# for the wrong architecture is silent: sm_52 cubin still runs on an sm_86 card
# through PTX JIT, so the only symptom is that the numbers are wrong.
#
# The engine refuses to start without a GPU, so a build machine has one and
# "native" is the right default. Override when producing a binary for other
# hardware: -DCMAKE_CUDA_ARCHITECTURES=86
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  set(CMAKE_CUDA_ARCHITECTURES native CACHE STRING "CUDA architectures to build for")
endif()

project(tinyrenderer
        VERSION 2.0
        DESCRIPTION "Tiled CUDA software rasterizer"
        LANGUAGES CXX CUDA)

# ---------------------------------------------------------------------------
# Standards and defaults
# ---------------------------------------------------------------------------

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# Single-config generators get Release; multi-config (MSVC) picks at build time.
get_property(_multi_config GLOBAL PROPERTY GENERATOR_IS_MULTI_CONFIG)
if(NOT _multi_config AND NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "" FORCE)
endif()

# Everything lands in one place so the SDL2 DLL copy below has one target dir.
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
foreach(_cfg IN ITEMS DEBUG RELEASE RELWITHDEBINFO MINSIZEREL)
  set(CMAKE_RUNTIME_OUTPUT_DIRECTORY_${_cfg} ${CMAKE_BINARY_DIR}/bin)
endforeach()

# ---------------------------------------------------------------------------
# SDL2
# ---------------------------------------------------------------------------
# vcpkg, the SDL2-devel VC release, and Ubuntu's libsdl2-dev all ship a CMake
# config package exporting SDL2::SDL2. Fall back to pkg-config for the distros
# that do not.

find_package(SDL2 CONFIG QUIET)
if(NOT SDL2_FOUND)
  find_package(SDL2 QUIET)
endif()

if(TARGET SDL2::SDL2)
  set(SDL2_LINK_TARGETS SDL2::SDL2)
  if(TARGET SDL2::SDL2main)
    # Windows only: rewrites main() into WinMain. Must precede SDL2::SDL2.
    list(INSERT SDL2_LINK_TARGETS 0 SDL2::SDL2main)
  endif()
elseif(SDL2_FOUND)
  # Older find modules export variables rather than targets.
  add_library(sdl2_compat INTERFACE)
  target_include_directories(sdl2_compat INTERFACE ${SDL2_INCLUDE_DIRS})
  target_link_libraries(sdl2_compat INTERFACE ${SDL2_LIBRARIES})
  set(SDL2_LINK_TARGETS sdl2_compat)
else()
  find_package(PkgConfig QUIET)
  if(PkgConfig_FOUND)
    pkg_check_modules(SDL2 IMPORTED_TARGET sdl2)
  endif()
  if(TARGET PkgConfig::SDL2)
    set(SDL2_LINK_TARGETS PkgConfig::SDL2)
  else()
    message(FATAL_ERROR
      "SDL2 not found.\n"
      "  Linux:   sudo apt install libsdl2-dev\n"
      "  Windows: vcpkg install sdl2:x64-windows, then configure with\n"
      "           -DCMAKE_TOOLCHAIN_FILE=<vcpkg>/scripts/buildsystems/vcpkg.cmake\n"
      "  (an SDL2 built for MinGW will not link against MSVC)")
  endif()
endif()

# ---------------------------------------------------------------------------
# Warning and definition flags, per compiler
# ---------------------------------------------------------------------------
# Carried on an interface target so every target below gets them identically,
# and so nvcc only ever sees them via -Xcompiler on the host pass.

add_library(project_flags INTERFACE)

if(MSVC)
  target_compile_options(project_flags INTERFACE
    $<$<COMPILE_LANGUAGE:CXX>:/W3 /permissive->)
  target_compile_definitions(project_flags INTERFACE
    _USE_MATH_DEFINES        # MSVC does not define M_PI without it (Engine.cpp)
    _CRT_SECURE_NO_WARNINGS
    NOMINMAX                 # windows.h min/max macros break std::min/std::max
    WIN32_LEAN_AND_MEAN)
else()
  target_compile_options(project_flags INTERFACE
    $<$<COMPILE_LANGUAGE:CXX>:-Wall -Wextra>)
endif()

target_include_directories(project_flags INTERFACE ${CMAKE_SOURCE_DIR}/src)

# ---------------------------------------------------------------------------
# render_core: geometry, models, images, scene graph, and the whole GPU path
# ---------------------------------------------------------------------------
# No SDL2 dependency, which is what lets the headless regression tests link
# against it without pulling a window system in.

add_library(render_core STATIC
  src/geometry.cpp
  src/model.cpp
  src/transform.cpp
  src/tgaimage.cpp
  src/hdri.cpp
  src/Scene.cpp
  src/SceneNode.cpp

  # one file per pipeline stage; rasterizer.cu owns the device memory
  src/cuda/setup.cu
  src/cuda/binning.cu
  src/cuda/raster.cu
  src/cuda/shade.cu
  src/cuda/ssao.cu
  src/cuda/ibl.cu
  src/cuda/resolve.cu
  src/cuda/rasterizer.cu

  # not a pipeline stage: presents the finished frame without a host round trip
  src/cuda/present.cu)

# present.cu calls into the GL driver via cuda_gl_interop.h.
find_package(OpenGL REQUIRED)

target_link_libraries(render_core PUBLIC project_flags OpenGL::GL)

# No __device__ symbol crosses a translation unit, so separable compilation
# is not needed and whole-program device linking stays off.
set_target_properties(render_core PROPERTIES
  CUDA_SEPARABLE_COMPILATION OFF
  POSITION_INDEPENDENT_CODE ON)

# ---------------------------------------------------------------------------
# imgui: vendored in third_party/, built as its own target so its warnings
# stay out of ours and project_flags does not apply to it.
# ---------------------------------------------------------------------------

set(IMGUI_DIR ${CMAKE_SOURCE_DIR}/third_party/imgui)

add_library(imgui STATIC
  ${IMGUI_DIR}/imgui.cpp
  ${IMGUI_DIR}/imgui_draw.cpp
  ${IMGUI_DIR}/imgui_tables.cpp
  ${IMGUI_DIR}/imgui_widgets.cpp
  ${IMGUI_DIR}/backends/imgui_impl_sdl2.cpp
  ${IMGUI_DIR}/backends/imgui_impl_opengl3.cpp
  ${IMGUI_DIR}/backends/imgui_impl_sdlrenderer2.cpp)

target_include_directories(imgui PUBLIC ${IMGUI_DIR})
target_link_libraries(imgui PUBLIC ${SDL2_LINK_TARGETS} OpenGL::GL)
set_target_properties(imgui PROPERTIES POSITION_INDEPENDENT_CODE ON)

# ---------------------------------------------------------------------------
# render_engine: the SDL2 window, input, and frame loop
# ---------------------------------------------------------------------------

# UI.cpp is the only file that includes imgui, and nothing in the pipeline
# depends on it: the tests link render_engine and never construct a UI.
add_library(render_engine STATIC src/Engine.cpp src/GLPresenter.cpp src/UI.cpp)
target_link_libraries(render_engine PUBLIC render_core ${SDL2_LINK_TARGETS}
                      PRIVATE imgui)

# ---------------------------------------------------------------------------
# The engine binary
# ---------------------------------------------------------------------------

add_executable(unified_engine src/main.cpp)
target_link_libraries(unified_engine PRIVATE render_engine)

# Model paths in main.cpp are relative to the repo root.
set_target_properties(unified_engine PROPERTIES
  VS_DEBUGGER_WORKING_DIRECTORY ${CMAKE_SOURCE_DIR})

# Windows has no rpath: the loader needs SDL2.dll beside the executable.
if(WIN32)
  add_custom_command(TARGET unified_engine POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
            $<TARGET_RUNTIME_DLLS:unified_engine> $<TARGET_FILE_DIR:unified_engine>
    COMMAND_EXPAND_LISTS)
endif()

# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------
# Two groups, matching src/Makefile. The regression suite links the CPU
# reference rasterizer, which nothing in the engine links: its only job is to
# be an independent implementation the kernels can disagree with.
# The engine tests drive the real Engine headlessly via SDL's dummy driver.
#
# Both resolve models relative to src/, so that is the working directory ctest
# runs them in.
#
# Most take the model as argv[1] and are handed one from assets/, which ships
# with the repository, so a fresh clone gets a suite that actually renders
# something. test_shaded and test_ssao are the exceptions: their thresholds and
# landmarks are calibrated against african_head specifically, so they keep it as
# their default and skip when it is absent rather than being repointed at
# geometry the numbers do not describe. tools/fetch_models.py downloads it.

option(BUILD_TESTING_RENDERER "Build the rasterizer test suite" ON)

if(BUILD_TESTING_RENDERER)
  enable_testing()

  set(REFERENCE_RASTER src/tests/reference_raster.cpp)

  # Differential and unit tests: no SDL2, no window.
  set(CORE_TESTS test_cull test_blit test_mesh test_shaded test_frustum)

  # End-to-end tests that need the real Engine, and therefore SDL2.
  set(ENGINE_TESTS test_engine test_ssao test_ssaa test_meshcache
                   test_environment)

  # Models from assets/, as the tests see them from src/.
  set(ASSET_BUNNY  ../assets/bunny/bunny.obj)
  set(ASSET_GLASS  ../assets/toycar/Glass.obj)

  # Which model each test is run against. Anything not listed here takes its
  # own default, which is what test_shaded and test_ssao rely on.
  set(MODEL_test_cull       ${ASSET_BUNNY})
  set(MODEL_test_mesh       ${ASSET_BUNNY})
  set(MODEL_test_frustum    ${ASSET_BUNNY})
  set(MODEL_test_engine     ${ASSET_BUNNY})
  set(MODEL_test_ssaa       ${ASSET_BUNNY})
  set(MODEL_test_environment ${ASSET_BUNNY})
  # needs two distinct models: the cache it checks is keyed by Model*
  set(MODEL_test_meshcache  ${ASSET_BUNNY} ${ASSET_GLASS})

  # Local benchmarking and inspection tools. These are gitignored, so build
  # whichever happen to be present rather than failing on the ones that are not.
  set(OPTIONAL_TOOLS test_wind bench bench_cull bench_tile profile_frame
                     render_png bench_meshes capture_scene capture_clip
                     bench_overdraw)

  # Needs neither a GPU nor a model: it integrates the BRDF functions on the
  # host and checks the result against their definitions, so it is the one
  # test that actually asserts something on a CI runner with no device.
  add_executable(test_brdf src/tests/test_brdf.cpp)
  target_include_directories(test_brdf PRIVATE ${CMAKE_SOURCE_DIR}/src)
  add_test(NAME test_brdf COMMAND test_brdf)

  foreach(_t IN LISTS CORE_TESTS)
    add_executable(${_t} src/tests/${_t}.cpp ${REFERENCE_RASTER})
    target_include_directories(${_t} PRIVATE ${CMAKE_SOURCE_DIR}/src/tests)
    target_link_libraries(${_t} PRIVATE render_core)
    add_test(NAME ${_t} COMMAND ${_t} ${MODEL_${_t}}
             WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}/src)
    # 77 means the test declined to run -- no CUDA device, or a model it needs
    # is not present. That has to read as skipped, not passed: a test that
    # renders nothing and compares nothing agrees with itself perfectly.
    set_tests_properties(${_t} PROPERTIES SKIP_RETURN_CODE 77)
  endforeach()

  foreach(_t IN LISTS ENGINE_TESTS)
    add_executable(${_t} src/tests/${_t}.cpp)
    target_include_directories(${_t} PRIVATE ${CMAKE_SOURCE_DIR}/src/tests)
    target_link_libraries(${_t} PRIVATE render_engine)
    add_test(NAME ${_t} COMMAND ${_t} ${MODEL_${_t}}
             WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}/src)
    # test_engine opens no window; the others also run without a display.
    set_tests_properties(${_t} PROPERTIES ENVIRONMENT "SDL_VIDEODRIVER=dummy"
                                          SKIP_RETURN_CODE 77)
  endforeach()

  foreach(_t IN LISTS OPTIONAL_TOOLS)
    if(EXISTS ${CMAKE_SOURCE_DIR}/src/tests/${_t}.cpp)
      add_executable(${_t} EXCLUDE_FROM_ALL src/tests/${_t}.cpp ${REFERENCE_RASTER})
      target_include_directories(${_t} PRIVATE ${CMAKE_SOURCE_DIR}/src/tests)
      target_link_libraries(${_t} PRIVATE render_engine)
    endif()
  endforeach()

  if(WIN32)
    # Same DLL problem as the engine binary, for every test that links SDL2.
    foreach(_t IN LISTS ENGINE_TESTS)
      add_custom_command(TARGET ${_t} POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
                $<TARGET_RUNTIME_DLLS:${_t}> $<TARGET_FILE_DIR:${_t}>
        COMMAND_EXPAND_LISTS)
    endforeach()
  endif()
endif()

# ---------------------------------------------------------------------------

message(STATUS "")
message(STATUS "tinyrenderer ${PROJECT_VERSION}")
message(STATUS "  CUDA compiler     ${CMAKE_CUDA_COMPILER_VERSION}")
message(STATUS "  CUDA architectures ${CMAKE_CUDA_ARCHITECTURES}")
message(STATUS "  SDL2 via          ${SDL2_LINK_TARGETS}")
message(STATUS "  tests             ${BUILD_TESTING_RENDERER}")
message(STATUS "")
