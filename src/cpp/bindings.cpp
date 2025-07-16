/* 
This code creates a Python-callable "wrapper" for C++ LaneKeepingInference class, 
It allows  to create objects of this class and call its methods directly from Python.
*/

#include <pybind11/pybind11.h>
#include <pybind11/stl.h> // For automatic conversion of STL containers (like std::vector)
#include "LaneKeepingInference.h"

namespace py = pybind11;

PYBIND11_MODULE(cpp_inference, m) {
    m.doc() = "A Python wrapper for the C++ ONNX inference engine.";

    // Bind the C++ class `LaneKeepingInference`
    py::class_<LaneKeepingInference>(m, "LaneKeepingInference")
        // Expose the constructor, which takes a model path string
        .def(py::init<const std::string&>())
        // Expose the `run_inference` method
        .def("run_inference", &LaneKeepingInference::run_inference, 
             "Runs inference on an image and returns the steering angle.")
        // NEW: Expose the batch inference method
        .def("run_inference_on_directory_batched", &LaneKeepingInference::run_inference_on_directory_batched,
             py::arg("directory_path"), py::arg("batch_size") = 1,
             "Run inference on all images in a directory, processed in batches, and return a list of steering angles.");
}
