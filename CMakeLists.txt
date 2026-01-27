cmake_minimum_required(VERSION 3.10)
project(InferenceEngine VERSION 1.0)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

# Dependencies
# Fetch pybind11 automatically from GitHub
include(FetchContent)
FetchContent_Declare(
  pybind11
  GIT_REPOSITORY https://github.com/pybind/pybind11.git
  GIT_TAG        v2.11.1
)
FetchContent_MakeAvailable(pybind11)

# Find OpenMP
if(APPLE AND CMAKE_SYSTEM_PROCESSOR MATCHES "arm64")
    set(OpenMP_ROOT "/opt/homebrew/opt/libomp")
    list(APPEND CMAKE_PREFIX_PATH "/opt/homebrew/opt/libomp")
endif()
find_package(OpenMP REQUIRED)

include_directories(include)

# Build the Python module
file(GLOB SOURCES "src/*.cpp")
pybind11_add_module(engine ${SOURCES})

# Link libraries
target_link_libraries(engine PRIVATE OpenMP::OpenMP_CXX)

# Optimization flags
if(APPLE AND CMAKE_SYSTEM_PROCESSOR MATCHES "arm64")
    target_include_directories(engine PRIVATE ${OpenMP_ROOT}/include)
    target_compile_options(engine PRIVATE -Xpreprocessor -fopenmp -O3)
    target_link_libraries(engine PRIVATE ${OpenMP_ROOT}/lib/libomp.dylib)
elseif(MSVC)
    add_compile_options(/arch:AVX2 /O2 /openmp)
else()
    add_compile_options(-mavx2 -mfma -O3 -fopenmp)
endif()