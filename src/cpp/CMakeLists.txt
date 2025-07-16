# Minimum required CMake version (can stay 3.10 or align with top-level)
cmake_minimum_required(VERSION 3.10) 

# You can keep `project(LaneKeepingInference LANGUAGES CXX)` here if you want
# this `src/cpp` directory to still be buildable as a standalone project,
# but if it's primarily a sub-project, you might omit it. For now, keep it.
project(LaneKeepingInference LANGUAGES CXX) 

# Set C++ standard (e.g., C++17)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# --- OpenCV Configuration ---
find_package(OpenCV REQUIRED)
if(OpenCV_FOUND)
    message(STATUS "OpenCV found. Version: ${OpenCV_VERSION}")
    include_directories(${OpenCV_INCLUDE_DIRS})
else()
    message(FATAL_ERROR "OpenCV not found! Please ensure it's installed and findable by CMake.")
endif()
# --- End OpenCV Configuration ---

# --- ONNX Runtime Configuration ---
set(ONNXRUNTIME_BASE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../../third_party")
set(ONNXRUNTIME_VERSION_SUBDIR "onnxruntime-osx-arm64-1.17.1")

if (EXISTS "${ONNXRUNTIME_BASE_DIR}/${ONNXRUNTIME_VERSION_SUBDIR}/include/onnxruntime_cxx_api.h")
    set(ONNXRUNTIME_INSTALL_PATH "${ONNXRUNTIME_BASE_DIR}/${ONNXRUNTIME_VERSION_SUBDIR}")
    message(STATUS "Found ONNX Runtime in version-specific subfolder: ${ONNXRUNTIME_VERSION_SUBDIR}")
else()
    set(ONNXRUNTIME_INSTALL_PATH "${ONNXRUNTIME_BASE_DIR}")
    message(STATUS "Found ONNX Runtime directly in third_party folder (likely CI setup).")
endif()

find_library(ONNXRUNTIME_LIB
    NAMES onnxruntime
    PATHS "${ONNXRUNTIME_INSTALL_PATH}/lib"
    NO_DEFAULT_PATH
)
if(NOT ONNXRUNTIME_LIB)
    message(FATAL_ERROR "ONNX Runtime library not found at ${ONNXRUNTIME_INSTALL_PATH}/lib")
else()
    message(STATUS "ONNX Runtime library found at: ${ONNXRUNTIME_LIB}")
endif()
# --- End ONNX Runtime Configuration ---


# These lines are from inference syep
# Add executables for source files (e.g., src/cpp/inference_test.cpp)
# Paths here are relative to *this* CMakeLists.txt (src/cpp/)
#add_executable(lane_keeping_test_inference inference_test.cpp)
#add_executable(lane_keeping_real_inference inference_real.cpp)


# Add the new Pybind11 module target
# Paths here are relative to *this* CMakeLists.txt (src/cpp/)
pybind11_add_module(cpp_inference
    bindings.cpp # Simplified path
    LaneKeepingInference.cpp # Simplified path
)

# --- Configure and Link Targets ---

# For lane_keeping_test_inference
#target_include_directories(lane_keeping_test_inference PRIVATE
#    "${ONNXRUNTIME_INSTALL_PATH}/include"
#)
#target_link_libraries(lane_keeping_test_inference PRIVATE
#    ${ONNXRUNTIME_LIB}
#)

# For lane_keeping_real_inference
#target_include_directories(lane_keeping_real_inference PRIVATE
#    "${ONNXRUNTIME_INSTALL_PATH}/include"
#)
#target_link_libraries(lane_keeping_real_inference PRIVATE
#    ${ONNXRUNTIME_LIB}
#    ${OpenCV_LIBS}
#)

# Link the new Python module to its dependencies
target_include_directories(cpp_inference PRIVATE
    "${ONNXRUNTIME_INSTALL_PATH}/include"
    ${OpenCV_INCLUDE_DIRS}
    # This might be needed depending on your setup, but pybind11_add_module often handles it
    # ${pybind11_INCLUDE_DIRS} 
)
target_link_libraries(cpp_inference PRIVATE
    ${ONNXRUNTIME_LIB}
    ${OpenCV_LIBS}
    # pybind11_add_module automatically links to the necessary pybind11 libraries
)

# Optional: Specify output directories. If defined at top-level, these may not be needed here
# set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/bin)
# set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/lib)
# set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/lib)
