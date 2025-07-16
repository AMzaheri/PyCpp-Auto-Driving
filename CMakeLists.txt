cmake_minimum_required(VERSION 3.10)
project(SimulatedAVLaneAssist LANGUAGES CXX) # This is the main project name

# Set common output directories for all targets to the main BUILD directory
# ${PROJECT_BINARY_DIR} automatically expands to the directory where cmake was run from (e.g., 'build')
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}") # For executables
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}") # For shared libs (.so, .dylib, .dll)
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}") # For static libs (.a, .lib)

# Add pybind11 as a subdirectory from the project root
add_subdirectory(pybind11)

# Add C++ source directory as a subdirectory
# CMake will now process src/cpp/CMakeLists.txt from here
add_subdirectory(src/cpp)