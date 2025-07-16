# C++ Python Integration Module for Lane Keeping Model

This module provides a robust C++ implementation for performing inference using the NVIDIA PilotNet model, designed for seamless integration into Python workflows. It leverages **ONNX Runtime** for efficient model execution, **OpenCV** for image preprocessing, and **Pybind11** for creating the Python bindings.

This hybrid architecture enables you to benefit from Python's flexibility for data handling and scripting, combined with C++'s raw speed for computation-heavy inference. It offers an ideal solution for real-time applications by providing:
* Efficient single-image inference on actual image data.
* **Optimized batch inference** on multiple images from a directory.

## Table of Contents

* [Dependencies](#dependencies)
* [Setup Instructions](#setup-instructions)
    * [1. Download ONNX Runtime](#1-download-onnx-runtime)
    * [2. Install OpenCV](#2-install-opencv)
    * [3. Place the ONNX Model](#3-place-the-onnx-model)
    * [4. Prepare Test Images](#4-prepare-test-images)
    * [5. Install Pybind11](#5-install-pybind11)
* [Build Instructions](#build-instructions)
* [Using Python Bindings for Inference](#using-python-bindings-for-inference)

---

## Dependencies

* **CMake:** Version 3.10 or higher.
* **C++ Compiler:** A modern C++ compiler supporting **C++17 standard**.
* **ONNX Runtime:** Specifically, **v1.17.1 for macOS ARM64** (or the version appropriate for your system). This is the ONNX inference engine.
* **OpenCV:** Essential for image loading, manipulation, and preprocessing.
* **Pybind11:** A lightweight header-only library for creating Python bindings of C++ code.

## Setup Instructions

These steps will guide you through setting up the necessary components to build and use the C++ inference module via its Python bindings.

### 1. Download ONNX Runtime

Download the pre-built ONNX Runtime package for the version appropriate for your system from the [ONNX Runtime GitHub Releases page](https://github.com/microsoft/onnxruntime/releases).

Extract the downloaded `.zip` file. Place the extracted folder into a `third_party/` directory at the **root of the `simulated-av-lane-assist` project**.

```bash
mkdir -p third_party
unzip /path/to/downloaded/onnxruntimezip_folder -d third_party/
```

Ensure `third_party/` is listed in your project's `.gitignore` file.

### 2. Install OpenCV

```bash
# Install Homebrew (skip if already installed)
/bin/bash -c "$(curl -fsSL [https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh](https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh))"

# Install OpenCV via Homebrew
brew install opencv
```

For other operating systems, please refer to the [official OpenCV installation](https://docs.opencv.org/4.x/d7/d9f/tutorial_linux_install.html) guide or relevant package managers.

### 3. Place the ONNX Model

Ensure your ONNX model file (e.g., `nvidia_pilotnet.onnx`) is placed inside the `models/` directory at the root of `simulated-av-lane-assist` project.

### 4. Prepare Test Images (for Real Data Inference)

 * The `data/test_images/` directory contains two sample `.png` images, used for demonstrating the real image inference capabilities.
 * Sample images, simulating straight and curved road conditions, can be created using the `src/python/data_generator.py` script.
 
### 5.  Install Pybind11

Pybind11 is included as a git submodule in this project. Ensure it's initialized and updated:
```bash
git submodule update --init --recursive
```

## Build and Run

The project uses a top-level CMakeLists.txt to manage the build process, which includes compiling the C++ Python integration module.

1. Navigate to the top-level `build` directory:

```bash
cd /path/to/simulated-av-lane-assist/build
```

(If build directory doesn't exist at the project root, create it: mkdir build)
2. Run CMake to configure the project:

```bash
cmake ..
```

CMake will configure the build for the `cpp_inference` Python module.

3. Build the project:

```bash
make
```
or
```bash
make -j$(nproc) # Use -j$(nproc) to compile in parallel for faster build
```
 
This will compile the Python C++ extension module: `cpp_inference.so` (on macOS/Linux) or `cpp_inference.pyd` (on Windows), which will be placed directly in the `build/` directory.

## Using Python Bindings for Inference

The `cpp_inference` module exposes the `LaneKeepingInference` class, providing flexible methods to perform model inference directly from your Python applications. It supports two primary modes:

* **Single-Image Inference:** Use the `run_inference(image_path)` method to process an individual image file and obtain its steering angle prediction.
* **Batch Inference from Directory:** Utilise the `run_inference_on_directory_batched(directory_path, batch_size)` method to efficiently process all images within a specified directory in configurable batches. This significantly boosts throughput for large datasets by leveraging the C++ module's optimised batching logic.

For practical examples demonstrating how to use both single-image and batch inference, please refer to the [`inference_demo.py`](../../inference_demo.py) script located in the project root.

Note: After building, the `cpp_inference.so` (or `.pyd`) file will be located in your `build/` directory. For Python to find it, you need to add this directory to your `PYTHONPATH`.

