#include "LaneKeepingInference.h"
#include <iostream>
#include <numeric>
#include <filesystem>
#include <array>
#include <onnxruntime_cxx_api.h>
#include <opencv2/opencv.hpp>
#include <stdexcept> // For std::runtime_error
#include <vector>    // Include vector if not already there for std::vector

// --- New static helper function to configure session options ---
Ort::SessionOptions LaneKeepingInference::create_configured_session_options() {
    Ort::SessionOptions options;
    options.SetIntraOpNumThreads(1);
    options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_EXTENDED);
    return options;
}
// ---------------------------------------------------------------
// Constructor implementation
LaneKeepingInference::LaneKeepingInference(const std::string& model_path)
    : env(ORT_LOGGING_LEVEL_WARNING, "LaneKeepingInference"),
      session_options(create_configured_session_options()),
      session(env, model_path.c_str(), session_options)
{
    // 1. Get Input Node Info
    size_t num_input_nodes = session.GetInputCount();
    input_node_names.reserve(num_input_nodes);

    for (size_t i = 0; i < num_input_nodes; i++) {
        auto name = session.GetInputNameAllocated(i, allocator);
        input_node_names.push_back(name.release());

        Ort::TypeInfo type_info = session.GetInputTypeInfo(i);
        auto tensor_info = type_info.GetTensorTypeAndShapeInfo();

        if (i == 0) { // Assuming only one input for this model
            input_shape = tensor_info.GetShape();

            // --- CRITICAL FIX: Handle dynamic batch dimension (-1) ---
            if (input_shape[0] == -1) {
                std::cout << "DEBUG (Constructor): Detected dynamic batch size (-1) from model. Setting to 1 for inference." << std::endl;
                input_shape[0] = 1; // Set batch size to 1 for single-image inference
            }
            // --- END FIX ---
        }
        
        // --- DEBUG PRINT: Input Node Details (keep these, they are helpful) ---
        //std::cout << "DEBUG (Constructor): Input Node " << i << " Name: " << input_node_names[i] << std::endl;
        //std::cout << "DEBUG (Constructor): Input Node " << i << " Data Type: " << tensor_info.GetElementType() << std::endl;
        //std::cout << "DEBUG (Constructor): Input Node " << i << " Shape from Model (after potential fix): ["; // Updated message
        for (size_t j = 0; j < input_shape.size(); ++j) { // Print 'input_shape' which is now modified
            std::cout << input_shape[j] << (j == input_shape.size() - 1 ? "" : ", ");
        }
        std::cout << "]" << std::endl;

        // --- REMOVE THIS THROW: We are now handling the -1 ---
        // You can remove or comment out this block completely, as the -1 is now handled.
        /*
        for (int64_t dim : tensor_info.GetShape()) {
            if (dim < 0) {
                throw std::runtime_error("ERROR (Constructor): Model defined input shape with a negative dimension!");
            }
        }
        */
        // Or, a general check for other unexpected negative dims (not batch):
        for (size_t k = 1; k < input_shape.size(); ++k) { // Check dims other than batch
            if (input_shape[k] < 0) {
                throw std::runtime_error("ERROR (Constructor): Non-batch model input shape dimension is negative after processing!");
            }
        }
    }

    // 2. Get Output Node Info
    size_t num_output_nodes = session.GetOutputCount();
    output_node_names.reserve(num_output_nodes);

    for (size_t i = 0; i < num_output_nodes; i++) {
        auto name = session.GetOutputNameAllocated(i, allocator);
        output_node_names.push_back(name.release()); // .release() transfers ownership
        Ort::TypeInfo type_info = session.GetOutputTypeInfo(i);
        output_shape = type_info.GetTensorTypeAndShapeInfo().GetShape(); // Assuming only one output
    }
}

// Preprocessing method with debugging and checks
std::vector<float> LaneKeepingInference::preprocess_image(const std::string& image_path) {
    // 1. Load the image
    cv::Mat image = cv::imread(image_path, cv::IMREAD_COLOR);
    if (image.empty()) {
        throw std::runtime_error("Could not load image from " + image_path);
    }
    //std::cout << "DEBUG (Preprocess): Image loaded. Original dimensions (HxWxC): " 
    //          << image.rows << "x" << image.cols << "x" << image.channels() << std::endl;
    if (image.rows <= 0 || image.cols <= 0 || image.channels() <= 0) {
        throw std::runtime_error("ERROR (Preprocess): Loaded image has zero or negative dimensions/channels.");
    }

    // 2. Convert to grayscale
    cv::Mat gray_image;
    cv::cvtColor(image, gray_image, cv::COLOR_BGR2GRAY);
    //std::cout << "DEBUG (Preprocess): Converted to grayscale. Dimensions (HxW): " 
    //          << gray_image.rows << "x" << gray_image.cols << std::endl;


    // 3. Resize the image (assuming our model expects 200x66)
    cv::Mat resized_image;
    cv::resize(gray_image, resized_image, cv::Size(200, 66), 0, 0, cv::INTER_AREA);
    //std::cout << "DEBUG (Preprocess): Image resized. New dimensions (HxW): " 
    //          << resized_image.rows << "x" << resized_image.cols << std::endl;
    if (resized_image.rows <= 0 || resized_image.cols <= 0) {
        throw std::runtime_error("ERROR (Preprocess): Resized image has zero or negative dimensions.");
    }
    
    // 4. Convert to float and normalize (e.g., to [-1, 1] range based on 1.0/127.5, -1.0)
    cv::Mat float_image;
    resized_image.convertTo(float_image, CV_32FC1, 1.0 / 127.5, -1.0);
    //std::cout << "DEBUG (Preprocess): Converted to float and normalized. Type: " << float_image.type() << std::endl;


    if (!float_image.isContinuous()) {
        //std::cout << "DEBUG (Preprocess): float_image not continuous, cloning..." << std::endl;
        float_image = float_image.clone();
    }
    
    // 5. Copy to a 1D vector
    // Your model expects 1 batch, 1 channel, 66 height, 200 width.
    // So the total size is 1*1*66*200 = 13200
    // Ensure the size matches the expected input_shape from the model
    size_t expected_vector_size = 1 * input_shape[1] * input_shape[2] * input_shape[3]; // Assuming N=1, C,H,W from input_shape
    // OR directly use the known size for this specific model if input_shape is unreliable:
    // size_t expected_vector_size = 1 * 1 * 66 * 200; // For 1-channel, 66x200 model

    std::vector<float> input_tensor_values(expected_vector_size);
    
    // Check if the source data size matches the destination vector size
    if (float_image.total() * float_image.elemSize() != input_tensor_values.size() * sizeof(float)) {
        throw std::runtime_error("ERROR (Preprocess): Mismatch between source image data size and target vector size.");
    }

    std::memcpy(input_tensor_values.data(), float_image.data, input_tensor_values.size() * sizeof(float));
    //std::cout << "DEBUG (Preprocess): Final input tensor vector size: " << input_tensor_values.size() << std::endl;

    if (input_tensor_values.empty()) {
        throw std::runtime_error("ERROR (Preprocess): Input tensor values vector is empty after preprocessing.");
    }

    return input_tensor_values;
}
//------------------------------------------------------

// Main inference method with debugging and checks
float LaneKeepingInference::run_inference(const std::string& image_path) {
    // 1. Preprocess the image
    std::vector<float> input_tensor_values = preprocess_image(image_path);
    
    //std::cout << "DEBUG (Inference): Image path: " << image_path << std::endl; // Added std::endl

    // 2. Create the input tensor
    Ort::MemoryInfo memory_info = Ort::MemoryInfo::CreateCpu(
        OrtAllocatorType::OrtArenaAllocator,
        OrtMemType::OrtMemTypeDefault);

    // --- DEBUG PRINT: Input shape used for tensor creation ---
    //std::cout << "DEBUG (Inference): Shape being used for Ort::Value::CreateTensor: [";
    for (size_t i = 0; i < input_shape.size(); ++i) {
        std::cout << input_shape[i] << (i == input_shape.size() - 1 ? "" : ", ");
    }
    std::cout << "]" << std::endl;

    // --- CRITICAL CHECK: For negative dimensions right before tensor creation ---
    for (int64_t dim : input_shape) {
        if (dim < 0) {
            throw std::runtime_error("ERROR (Inference): Negative dimension found in input_shape array before tensor creation!");
        }
    }


    Ort::Value input_tensor = Ort::Value::CreateTensor<float>(
        memory_info,
        input_tensor_values.data(),
        input_tensor_values.size(),
        input_shape.data(),
        input_shape.size()
    );
    if (!input_tensor.IsTensor()) { 
    throw std::runtime_error("ERROR (Inference): Input tensor creation failed or has wrong type.");
}

    //std::cout << "DEBUG (Inference): Input tensor created successfully." << std::endl;

    // 3. Run inference
    auto output_tensors = session.Run(
        Ort::RunOptions{nullptr},
        input_node_names.data(),
        &input_tensor,
        1,
        output_node_names.data(),
        1
    );
    //std::cout << "DEBUG (Inference): Session.Run completed." << std::endl;

    // 4. Process output
    Ort::Value& output_tensor = output_tensors[0];
    float* output_data = output_tensor.GetTensorMutableData<float>();
    Ort::TensorTypeAndShapeInfo output_tensor_info = output_tensor.GetTensorTypeAndShapeInfo();
    std::vector<int64_t> actual_output_shape = output_tensor_info.GetShape();

    // Validate output shape and throw an error if unexpected
    bool is_empty = actual_output_shape.empty();
    bool is_single_element_shape = (actual_output_shape.size() == 2) &&
                                   (actual_output_shape[0] == 1) &&
                                   (actual_output_shape[1] == 1);

    std::string shape_str_output = "[";
    for (size_t i = 0; i < actual_output_shape.size(); ++i) {
        shape_str_output += std::to_string(actual_output_shape[i]);
        if (i < actual_output_shape.size() - 1) {
            shape_str_output += ", ";
        }
    }
    shape_str_output += "]";
    //std::cout << "DEBUG (Inference): Actual model output shape: " << shape_str_output << std::endl;


    if (!is_empty && !is_single_element_shape) {    
        throw std::runtime_error("ERROR (Inference): Unexpected model output shape: " + shape_str_output);
    }
    
    return output_data[0];
}
//------------------------------------------------------
// NEW: Batch inference method for a directory of images
std::vector<float> LaneKeepingInference::run_inference_on_directory_batched(
    const std::string& directory_path,
    int batch_size
) {
    if (batch_size <= 0) {
        throw std::invalid_argument("Batch size must be a positive integer.");
    }
    if (!std::filesystem::is_directory(directory_path)) {
        throw std::runtime_error("Directory not found or is not a directory: " + directory_path);
    }

    std::vector<std::string> image_files;
    for (const auto& entry : std::filesystem::directory_iterator(directory_path)) {
        if (entry.is_regular_file()) {
            std::string ext = entry.path().extension().string();
            // Convert extension to lowercase for case-insensitive comparison
            std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower);
            if (ext == ".jpg" || ext == ".jpeg" || ext == ".png") {
                image_files.push_back(entry.path().string());
            }
        }
    }
    std::sort(image_files.begin(), image_files.end()); // Sort for consistent order

    if (image_files.empty()) {
        std::cout << "DEBUG (Batch Inference): No image files found in " << directory_path << std::endl;
        return {}; // Return empty vector if no images
    }

    //std::cout << "DEBUG (Batch Inference): Found " << image_files.size() 
    //          << " images. Processing in batches of " << batch_size << std::endl;

    std::vector<float> all_predicted_angles;
    
    // Calculate expected size of a single preprocessed image data
    // Assuming input_shape is (1, C, H, W) initially for single image preprocessing
    // input_shape[1] = C (1 for grayscale)
    // input_shape[2] = H (66)
    // input_shape[3] = W (200)
    size_t single_image_data_size = input_shape[1] * input_shape[2] * input_shape[3];

    // Store the original input_shape's batch dimension.
    // This is important because input_shape is a member variable,
    // and we'll modify its batch dimension for each batch.
    int64_t original_input_batch_dim = input_shape[0]; 

    for (size_t i = 0; i < image_files.size(); i += batch_size) {
        std::vector<std::string> current_batch_paths;
        // Determine the actual batch size for the current loop iteration (last batch might be smaller)
        int current_actual_batch_size = std::min((int)batch_size, (int)(image_files.size() - i));

        std::vector<float> batched_input_tensor_values(current_actual_batch_size * single_image_data_size);
        
        //std::cout << "DEBUG (Batch Inference): Processing batch starting with " 
        //          << image_files[i] << " (size: " << current_actual_batch_size << ")" << std::endl;

        for (int j = 0; j < current_actual_batch_size; ++j) {
            const std::string& image_path = image_files[i + j];
            try {
                std::vector<float> single_image_data = preprocess_image(image_path);
                if (single_image_data.size() != single_image_data_size) {
                    throw std::runtime_error("Preprocessing returned unexpected size for " + image_path);
                }
                // Copy data into the batched tensor at the correct offset
                std::memcpy(batched_input_tensor_values.data() + (j * single_image_data_size),
                            single_image_data.data(),
                            single_image_data_size * sizeof(float));
            } catch (const std::exception& e) {
                std::cerr << "ERROR (Batch Preprocessing): Failed to preprocess image " << image_path << ": " << e.what() << std::endl;
                // Decide how to handle this: skip, fill with zeros, etc.
                // For now, we'll fill with zeros and continue. This might impact other inferences in the batch.
                // A more robust solution might require skipping the entire batch or custom error flags.
                std::fill(batched_input_tensor_values.data() + (j * single_image_data_size),
                          batched_input_tensor_values.data() + ((j + 1) * single_image_data_size),
                          0.0f);
            }
        }

        // Adjust the batch dimension of input_shape for the current inference
        // This is crucial for dynamic batching
        input_shape[0] = current_actual_batch_size; 

        // Create the input tensor for the batch
        Ort::MemoryInfo memory_info = Ort::MemoryInfo::CreateCpu(
            OrtAllocatorType::OrtArenaAllocator,
            OrtMemType::OrtMemTypeDefault);

        Ort::Value input_tensor = Ort::Value::CreateTensor<float>(
            memory_info,
            batched_input_tensor_values.data(),
            batched_input_tensor_values.size(),
            input_shape.data(),
            input_shape.size()
        );

        if (!input_tensor.IsTensor()) {
            throw std::runtime_error("ERROR (Batch Inference): Batch input tensor creation failed.");
        }
        //std::cout << "DEBUG (Batch Inference): Input tensor created successfully for batch. Shape: [";
        for (size_t j = 0; j < input_shape.size(); ++j) {
            std::cout << input_shape[j] << (j == input_shape.size() - 1 ? "" : ", ");
        }
        std::cout << "]" << std::endl;


        // Run inference for the entire batch
        auto output_tensors = session.Run(
            Ort::RunOptions{nullptr},
            input_node_names.data(),
            &input_tensor,
            1,
            output_node_names.data(),
            1
        );
        //std::cout << "DEBUG (Batch Inference): Session.Run completed for batch." << std::endl;

        // Process output for the batch
        Ort::Value& output_tensor = output_tensors[0];
        float* output_data = output_tensor.GetTensorMutableData<float>();
        Ort::TensorTypeAndShapeInfo output_tensor_info = output_tensor.GetTensorTypeAndShapeInfo();
        std::vector<int64_t> actual_output_shape = output_tensor_info.GetShape();

        // The output shape should now be [current_actual_batch_size, 1] if output is (N,1)
        if (actual_output_shape.size() != 2 || actual_output_shape[0] != current_actual_batch_size || actual_output_shape[1] != 1) {
            std::string shape_str_output = "[";
            for (size_t k = 0; k < actual_output_shape.size(); ++k) {
                shape_str_output += std::to_string(actual_output_shape[k]);
                if (k < actual_output_shape.size() - 1) {
                    shape_str_output += ", ";
                }
            }
            shape_str_output += "]";
            throw std::runtime_error("ERROR (Batch Inference): Unexpected batch model output shape: " + shape_str_output);
        }

        // Collect individual predictions from the batch output
        for (int k = 0; k < current_actual_batch_size; ++k) {
            all_predicted_angles.push_back(output_data[k]);
        }
    }

    // After all batches are processed, we might want to reset input_shape[0] to its original value if needed
    // However, it's generally safer to just set it dynamically per batch as done above.
    // If we have other methods that expect a fixed input_shape[0] (e.g., 1), we might reset it here:
    // input_shape[0] = original_input_batch_dim;

    //std::cout << "DEBUG (Batch Inference): Completed all batches. Total predictions: " 
    //          << all_predicted_angles.size() << std::endl;
    return all_predicted_angles;
}
