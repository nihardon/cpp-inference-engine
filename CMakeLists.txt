cmake_minimum_required(VERSION 3.10)

# Project Name
project(InferenceEngine VERSION 1.0)

# Use C++ 17 (Modern standard, good for resumes)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

# Tell CMake where to look for header files (.h)
include_directories(include)

# Gather all .cpp files in src/
file(GLOB SOURCES "src/*.cpp")

# Create an executable named "engine" using those source files
add_executable(engine ${SOURCES})