cmake_minimum_required(VERSION 3.10)
project(InferenceEngine VERSION 1.0)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

include_directories(include)
file(GLOB SOURCES "src/*.cpp")
add_executable(engine ${SOURCES})

# PLATFORM SPECIFIC FLAGS
if(APPLE AND CMAKE_SYSTEM_PROCESSOR MATCHES "arm64")
    # Apple Silicon: NEON is enabled by default, but we can add O3
    message(STATUS "Detected Apple Silicon (ARM64). Enabling NEON optimizations.")
    target_compile_options(engine PRIVATE -O3)
elseif(MSVC)
    # Windows
    add_compile_options(/arch:AVX2 /O2)
else()
    # Linux / Intel Mac
    message(STATUS "Detected Intel/AMD. Enabling AVX2.")
    add_compile_options(-mavx2 -mfma -O3)
endif()