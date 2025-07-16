// -----------------------------------------------------
// A header file to declare a class that encapsulates the ONNX Runtime session and the inference logic
// -----------------------------------------------------
#ifndef LANE_KEEPING_INFERENCE_H
#define LANE_KEEPING_INFERENCE_H

#include <onnxruntime_cxx_api.h>
#include <string>
#include <opencv2/opencv.hpp>
#include <vector>

class LaneKeepingInference {
public:
    // Constructor to initialize the ONNX Runtime session
    LaneKeepingInference(const std::string& model_path);

    // Method to run inference on a single image file path
    float run_inference(const std::string& image_path);

    // NEW: Batch inference method for a directory of images
    std::vector<float> run_inference_on_directory_batched(
        const std::string& directory_path,
        int batch_size = 1 // Default batch size is 1, can be overridden
    );

private:
    Ort::Env env;
    Ort::SessionOptions session_options; 
    Ort::Session session;
    Ort::AllocatorWithDefaultOptions allocator;
    std::vector<const char*> input_node_names;
    std::vector<const char*> output_node_names;
    std::vector<int64_t> input_shape;
    std::vector<int64_t> output_shape;
    
    // Helper function for image preprocessing
    std::vector<float> preprocess_image(const std::string& image_path);

    static Ort::SessionOptions create_configured_session_options(); // <-- Static helper
};

#endif // LANE_KEEPING_INFERENCE_H
