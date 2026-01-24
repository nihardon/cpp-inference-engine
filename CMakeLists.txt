cmake_minimum_required(VERSION 3.10)
project(InferenceEngine VERSION 1.0)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

if(APPLE AND CMAKE_SYSTEM_PROCESSOR MATCHES "arm64")
    # Homebrew installs libomp here on Apple Silicon
    set(OpenMP_ROOT "/opt/homebrew/opt/libomp")
    # Hinting the prefix path helps find_package locate it
    list(APPEND CMAKE_PREFIX_PATH "/opt/homebrew/opt/libomp")
endif()

find_package(OpenMP REQUIRED)

include_directories(include)
file(GLOB SOURCES "src/*.cpp")
add_executable(engine ${SOURCES})

target_link_libraries(engine PRIVATE OpenMP::OpenMP_CXX)

if(APPLE AND CMAKE_SYSTEM_PROCESSOR MATCHES "arm64")
    message(STATUS "Detected Apple Silicon. Linking Homebrew OpenMP.")
    target_include_directories(engine PRIVATE ${OpenMP_ROOT}/include)
    target_compile_options(engine PRIVATE -Xpreprocessor -fopenmp -O3)
    target_link_libraries(engine PRIVATE ${OpenMP_ROOT}/lib/libomp.dylib)
elseif(MSVC)
    add_compile_options(/arch:AVX2 /O2 /openmp)
else()
    message(STATUS "Detected Intel/AMD. Enabling AVX2 + OpenMP.")
    add_compile_options(-mavx2 -mfma -O3 -fopenmp)
endif()