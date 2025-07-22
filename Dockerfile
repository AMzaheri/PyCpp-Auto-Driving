# Dockerfile

# Use a base Ubuntu image. It's robust for C++ builds and allows easy installation of Python.
FROM ubuntu:22.04

# Set environment variables for non-interactive apt-get installs and unbuffered Python output
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

# Set the working directory inside the container. All subsequent commands will run from here.
WORKDIR /app

# Install system dependencies:
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    build-essential \
    cmake \
    libopencv-dev \
    python3 \
    python3-pip \
    # Add python3-dev for Python development headers and static libraries
    python3-dev \
    wget \
    ca-certificates && \
    rm -rf /var/lib/apt/lists/*

# --- ONNX Runtime Setup ---
# This step downloads and extracts ONNX Runtime directly into the container.
# This ensures the correct Linux x64 version is always used, independent of the local host.
ARG ONNX_RUNTIME_VERSION="1.17.1"
#ARG ONNX_RUNTIME_ARCH="x64" # We are on an x64 Linux base image (ubuntu:22.04)
ARG ONNX_RUNTIME_ARCH="aarch64"

RUN echo "Downloading ONNX Runtime v${ONNX_RUNTIME_VERSION} for Linux-${ONNX_RUNTIME_ARCH}..." && \
    wget -q https://github.com/microsoft/onnxruntime/releases/download/v${ONNX_RUNTIME_VERSION}/onnxruntime-linux-${ONNX_RUNTIME_ARCH}-${ONNX_RUNTIME_VERSION}.tgz -O /tmp/onnxruntime.tgz && \
    mkdir -p third_party && \
    tar -xzf /tmp/onnxruntime.tgz -C third_party/ --strip-components=1 && \
    rm /tmp/onnxruntime.tgz && \
    echo "ONNX Runtime setup complete."

# Set ONNXRUNTIME_HOME to the directory where ONNX Runtime was extracted.
# CMake's find_package(ONNXRuntime) often uses this hint.
ENV ONNXRUNTIME_HOME="/app/third_party"

# Copy the entire project source code into the container's /app directory.
# This includes src/, inference_demo.py, CMakeLists.txt, models/, etc.
# IMPORTANT: This should be done *after* ONNX Runtime is setup, to avoid copying
# host-specific ONNX Runtime binaries if they were in the local third_party/.
COPY . /app

# --- Build the C++ Python module (more verbose for debugging) ---
RUN mkdir -p build

# Configure CMake. We separate this step to see its specific output.
# Adding -DCMAKE_VERBOSE_MAKEFILE=ON for more detailed build commands.
RUN cmake -B build -S . -DCMAKE_VERBOSE_MAKEFILE=ON

# Build the project. Separating this step also helps.
# Adding --verbose for even more detailed compiler output.
RUN cmake --build build -j$(nproc) -v

# Set the PYTHONPATH environment variable.
# This tells Python where to find the compiled `cpp_inference.so` module.
ENV PYTHONPATH=/app/build:${PYTHONPATH}


# Install Python packages from requirements.txt
# Copy requirements.txt first to leverage Docker's build cache
COPY requirements.txt .
RUN pip3 install --no-cache-dir -r requirements.txt

# Command to run when the container starts.
# This will execute the Python inference demo script.
# You can change this to start a server or other application later.
CMD ["python3", "inference_demo.py"]
