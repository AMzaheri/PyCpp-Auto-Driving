# Stage 1: Build C++ components and download ONNX Runtime for AMD64
FROM ubuntu:22.04 AS builder

# Set environment variables for non-interactive installation
ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=Europe/London

# Update and install build dependencies
RUN sed -i 's|http://archive.ubuntu.com/ubuntu/|http://uk.archive.ubuntu.com/ubuntu/|g' /etc/apt/sources.list && \
    sed -i 's|http://security.ubuntu.com/ubuntu/|http://uk.archive.ubuntu.com/ubuntu/|g' /etc/apt/sources.list && \
    apt-get update && \
    apt-get install -y --no-install-recommends \
        build-essential \
        cmake \
        git \
        wget \
        unzip \
        python3 \
        python3-pip \
        libopencv-dev \
        libboost-all-dev \
        ninja-build \     
    && rm -rf /var/lib/apt/lists/*

# Install pybind11
WORKDIR /app
RUN git clone https://github.com/pybind/pybind11.git && \
    cd pybind11 && \
    pip install .

# Create the application directory and copy source files for C++ build
WORKDIR /app
COPY src /app/src
COPY CMakeLists.txt /app/

# --- Download and extract ONNX Runtime for linux-x64 (AMD64) ---
# Use a specific version. Ensure this matches what C++ code expects if tightly coupled.
ENV ONNX_RUNTIME_VERSION="1.17.1" 
ENV ONNX_RUNTIME_TARBALL="onnxruntime-linux-x64-${ONNX_RUNTIME_VERSION}.tgz"
ENV ONNX_RUNTIME_DOWNLOAD_URL="https://github.com/microsoft/onnxruntime/releases/download/v${ONNX_RUNTIME_VERSION}/${ONNX_RUNTIME_TARBALL}"

# Create the directory for third-party libraries within the build stage
RUN mkdir -p /app/third_party/lib && \
    wget -O /tmp/${ONNX_RUNTIME_TARBALL} ${ONNX_RUNTIME_DOWNLOAD_URL} && \
    tar -xzf /tmp/${ONNX_RUNTIME_TARBALL} -C /tmp/ && \
    cp /tmp/onnxruntime-linux-x64-${ONNX_RUNTIME_VERSION}/lib/libonnxruntime.so /app/third_party/lib/ && \
    cp -r /tmp/onnxruntime-linux-x64-${ONNX_RUNTIME_VERSION}/include /app/third_party/include/ && \
    rm -rf /tmp/${ONNX_RUNTIME_TARBALL} /tmp/onnxruntime-linux-x64-${ONNX_RUNTIME_VERSION}

# Configure and build the C++ project
RUN mkdir -p build && \
    cmake -S . -B build -DPYBIND11_PATH=/app/pybind11 \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_INTERPROCEDURAL_OPTIMIZATION=OFF && \
    cmake --build build -j4 -v

# Stage 2: Final image for deployment
FROM ubuntu:22.04 AS final

# --- START OF CHANGES FOR FINAL STAGE ---

# Set environment variables for non-interactive installation and timezone
ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=Europe/London 

# Set Python environment variables (uncommented, good practice for Python apps in Docker)
ENV PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app/src/python:/app

# Install Python runtime dependencies (essential for the Flask app and OpenCV)
# Included mirror changes and additional runtime libraries
RUN sed -i 's|http://archive.ubuntu.com/ubuntu/|http://uk.archive.ubuntu.com/ubuntu/|g' /etc/apt/sources.list && \
    sed -i 's|http://security.ubuntu.com/ubuntu/|http://uk.archive.ubuntu.com/ubuntu/|g' /etc/apt/sources.list && \
    apt-get update -y && \
    apt-get install -y --no-install-recommends \
        python3 \
        python3-pip \
        libopencv-dev \
        libgomp1 \
        libstdc++6 \
    && apt-get autoremove -y && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

# --- END OF CHANGES FOR FINAL STAGE ---

# Set working directory inside the container
WORKDIR /app

# Copy compiled C++ artifacts and Python code from the builder stage
COPY --from=builder /app/build/cpp_inference.cpython-310-x86_64-linux-gnu.so /app/
COPY --from=builder /app/third_party/lib/libonnxruntime.so /app/third_party/lib/

# Copy the Flask application and HTML templates
COPY app/ /app/app/

# Copy your ONNX model
COPY models/nvidia_pilotnet.onnx /app/ 

# Install Python dependencies from requirements.txt
COPY requirements.txt /app/
RUN pip3 install --no-cache-dir -r requirements.txt

# Expose the port the Flask/web server listens on
# Cloud Run expects the app to listen on the PORT env var (defaults to 8080)
EXPOSE 8080

# Command to run Flask application using Gunicorn
# 'app.main:app' refers to 'app' directory, 'main.py' module, and 'app' Flask instance
CMD ["gunicorn", "--bind", "0.0.0.0:$PORT", "app.main:app"]
