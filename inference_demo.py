import sys
import os
import time 


# --- Configuration ---
# Get the absolute path to the project root (where this script is located)
project_root = os.path.dirname(os.path.abspath(__file__)) 


# Add the build directory to the Python path
build_dir = os.path.join(project_root, 'build')
if build_dir not in sys.path:
    sys.path.insert(0, build_dir)

try:
    import cpp_inference
    print("Successfully imported cpp_inference module!")
except ImportError as e:
    print(f"Error importing cpp_inference: {e}")
    print("Please ensure the 'build' directory (containing your .so/.pyd file) is correctly in your PYTHONPATH.")
    sys.exit(1)
#--------------------------
# Construct absolute paths to model and test images directory
model_path = os.path.join(project_root, 'models', 'nvidia_pilotnet.onnx')
test_images_dir = os.path.join(project_root, 'data', 'test_images')

# --- Initialise C++ Inference Engine ---
print("--- Initialising C++ Inference Engine ---")
try:
    inference_engine = cpp_inference.LaneKeepingInference(model_path)
    print("C++ Inference Engine initialised successfully.")
except Exception as e:
    print(f"Error initialising C++ Inference Engine: {e}")
    exit()

# --- Section 1: Single Image Inference Demo ---
print("\n--- DEMO: Single Image Inference ---")
single_image_path = os.path.join(test_images_dir, 'frame_00029.png') # Example image path

if os.path.exists(single_image_path):
    print(f"Processing single image: {os.path.basename(single_image_path)}")
    try:
        start_time = time.time()
        steering_angle = inference_engine.run_inference(single_image_path)
        end_time = time.time() 
        
        duration = end_time - start_time
        print(f"Predicted Steering Angle: {steering_angle:.4f}")
        print(f"Inference Time (Single Image): {duration:.4f} seconds")
    except Exception as e:
        print(f"Error during single image inference: {e}")
else:
    print(f"Single image not found at: {single_image_path}")
    print("Please ensure 'data/test_images/frame_00029.png' exists.")

# --- Section 2: Batch Inference from Directory Demo ---
print("\n--- DEMO: Batch Inference from Directory ---")
batch_size = 1 # You can change this to test different batch sizes

if os.path.isdir(test_images_dir):
    print(f"Processing images in '{os.path.basename(test_images_dir)}' in batches of {batch_size}")
    try:
        start_time = time.time()
        all_angles = inference_engine.run_inference_on_directory_batched(test_images_dir, batch_size)
        end_time = time.time()
        
        duration = end_time - start_time
        total_images_processed = len(all_angles)

        print(f"Batch Inference complete! Total images processed: {total_images_processed}")
        print(f"Total Inference Time (Batch): {duration:.4f} seconds")
        
        if duration > 0 and total_images_processed > 0:
            images_per_second = total_images_processed / duration
            print(f"Images Per Second (IPS): {images_per_second:.2f}")
        
        if all_angles: # Only print results if there are any
            print("Predicted Steering Angles (first 5 results shown for brevity):")
            for i, angle in enumerate(all_angles[:5]): # Print only first 5 for brevity
                print(f"  Result {i+1}: {angle:.4f}")
            if total_images_processed > 5:
                print(f"  ... and {total_images_processed - 5} more results.")
        else:
            print("No steering angles returned. Check if directory contains valid images.")

    except Exception as e:
        print(f"Error during batch inference: {e}")
else:
    print(f"Test images directory not found at: {test_images_dir}")
    print("Please ensure 'data/test_images/' exists and contains images.")

print("\n--- Python inference demo finished successfully ---")
