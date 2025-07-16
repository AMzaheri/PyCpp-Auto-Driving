# simulated-av-lane-assist
A simulated autonomous driving project implementing a lane-keeping assistant using a hybrid Python-C++ architecture with MLOps principles.

## Project Structure

```
├── .gitignore
├── README.md
├── data/
│      └──test_images/
├── models/
└── src/
│     ├── cpp/
│     └── python/
├── third_party/        <-- Contains external dependencies like ONNX Runtime
├── CMakeLists.txt      <-- Top-level CMake configuration for building the project
├── inference_demo.py       # Python script demonstrating C++ inference (single & batch)
├── pybind11/               # Git submodule for Pybind11 library (used for C++ to Python bindings)

```

## Dataset

This project utilises a custom-generated synthetic driving dataset, specifically designed for training machine learning models for autonomous vehicle lane-keeping.

The dataset comprises **34,379 grayscale images** captured from a simulated front-facing camera, covering driving scenarios on both **straight road segments** and a **curved (left-turn) arc**. Each image is paired with a corresponding steering angle label.

**Access the complete dataset on Kaggle:**
[Link to the Kaggle Dataset Here] 
(`https://www.kaggle.com/datasets/afsanehm/simulated-driving-data`)

For a detailed explanation of the data generation process, including the specific parameters used for diversification and the dataset's internal structure, please refer to the dedicated [`Data_README.md`](data/README.md).

### ML Model Training & Accessibility

The deep learning model was trained and evaluated on a dedicated Kaggle notebook: [Deep Learning for Simulated Driving](https://www.kaggle.com/code/afsanehm/deep-learning-for-simulated-driving). The trained model checkpoints (including the best-performing version) are accessible within this repository in the `models/` directory.


## C++ Inference Module (Hybrid Python-C++ Integration)

This project's core objective is to demonstrate a hybrid Python-C++ architecture for high-performance AI model deployment. The C++ inference module is specifically designed for integration into Python-based workflows, leveraging **ONNX Runtime (v1.17.1)** for efficient model execution and **Pybind11** for creating the necessary Python bindings.

This approach combines Python's flexibility for data handling and scripting with C++'s raw speed for computation-heavy inference, making it ideal for real-time autonomous driving applications.

**For detailed setup, build, and usage instructions for the C++ Python Integration Module, please refer to:**
[**`src/cpp/README_integration.md`**](src/cpp/README_integration.md)

**Note:** If you are looking for the standalone C++ executable inference examples (e.g., `lane_keeping_test_inference`, `lane_keeping_real_inference`) without Python integration, please refer to the `feature/cpp_inference` branch of [this repository](https://github.com/AMzaheri/simulated-av-lane-assist/blob/feature/cpp-inference/src/cpp/README_inference.md)

### Key Inference Capabilities (via Python)

The C++ module, fully accessible from Python, provides advanced functionalities for real-world image inference:

* **Flexible Image Loading & Preprocessing:**
    * Loads images from specified file paths.
    * Applies essential preprocessing steps (resizing, grayscale conversion, normalization) tailored to the model's input requirements.
* **Efficient Model Execution:**
    * Feeds processed image data directly into the ONNX model for high-speed inference.
* **Single-Image Inference:**
    * Provides a direct method to get a steering angle prediction for one image at a time.
* **Batch Inference from Directory (Optimized):**
    * Efficiently processes multiple images from a specified directory in configurable batches. This leverages the model's dynamic batching capability, running a single ONNX Runtime session for an entire group of images, significantly boosting throughput for large datasets.

This design facilitates the direct application of trained models on actual visual inputs, enabling a performant pathway for integrating the AI model into real-time or embedded systems.

### How to Use

The core C++ inference logic is exposed to Python via `pybind11`. For detailed Python usage examples, including single-image inference and optimised batch inference, please refer to [`inference_demo.py`](inference_demo.py).


