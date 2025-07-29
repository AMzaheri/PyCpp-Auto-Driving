# app/main.py

import os
from flask import Flask, request, jsonify, render_template
import base64
import cv2
import numpy as np
from PIL import Image
import io

# --- Import C++ inference module ---
# Because cpp_inference.cpython-310-x86_64-linux-gnu.so is copied to /app/
# and /app/ is the WORKDIR, Python can import it directly.
try:
    import cpp_inference
    print("Successfully imported cpp_inference module in Flask app!")
except ImportError as e:
    print(f"FATAL ERROR: Could not import cpp_inference: {e}")
    exit(1) # Exit if the core inference module can't be loaded

app = Flask(__name__)

# --- Global Inference Engine Initialization ---
# Initialize the C++ inference engine once when the Flask app starts.
# This avoids re-loading the model for every request, which is much more efficient.
inference_engine = None # Declare globally

@app.before_first_request
def initialize_inference_engine():
    """
    Initializes the C++ inference engine when the first request comes in.
    This ensures it's ready to serve.
    """
    global inference_engine
    
    # Construct model path relative to the current working directory (/app/)
    # The ONNX model is also copied to /app/
    model_name = 'nvidia_pilotnet.onnx' # Use the actual filename
    model_path = os.path.join(os.getcwd(), model_name) # os.getcwd() will be /app/

    print(f"Attempting to load model from: {model_path}")

    try:
        inference_engine = cpp_inference.LaneKeepingInference(model_path)
        print("C++ Inference Engine initialised successfully for Flask app.")
    except Exception as e:
        print(f"FATAL ERROR: Could not initialise C++ Inference Engine: {e}")
        exit(1) # Exit if the inference engine cannot be initialised


# --- Inference function ---
def run_inference_on_image(image_data_np):
    """
    Runs inference using the globally initialised LaneKeepingInference engine.
    """
    if inference_engine is None:
        raise RuntimeError("Inference engine not initialised. This should not happen.")
    
    # Assuming predict method takes a numpy array and returns an angle
    angle = inference_engine.predict(image_data_np)
    return angle


# --- Flask Routes ---
@app.route('/')
def index():
    """Renders the HTML upload form."""
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    """Handles image upload, runs inference, and returns prediction."""
    if 'image' not in request.files:
        return jsonify({'error': 'No image file provided'}), 400

    file = request.files['image']
    if file.filename == '':
        return jsonify({'error': 'No selected image file'}), 400

    if file:
        try:
            image_bytes = file.read()
            nparr = np.frombuffer(image_bytes, np.uint8)
            img_np = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

            if img_np is None:
                raise ValueError("Could not decode image.")

            predicted_angle = run_inference_on_image(img_np)

            return jsonify({'angle': predicted_angle, 'message': 'Prediction successful!'})

        except Exception as e:
            # Log the full traceback for debugging in Cloud Run logs
            import traceback
            traceback.print_exc() 
            return jsonify({'error': f'Processing failed: {str(e)}. Check server logs for details.'}), 500

    return jsonify({'error': 'Unknown error occurred'}), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 8080))
    app.run(host='0.0.0.0', port=port)
