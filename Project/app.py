import os
from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
from utils.predict import load_prediction_model, predict_digit
from utils.preprocess import base64_to_tensor
from utils.captcha_generator import generate_captcha_digit

app = Flask(__name__)
CORS(app)

# Global dictionary to hold loaded models
MODELS = {}
DEVICES = {}

# Paths to models
MODEL_PATHS = {
    'EfficientNetV2B0': os.path.join('models', 'EfficientNetV2B0_digit.pth'),
    'MobileNetV2': os.path.join('models', 'MobileNetV2_digit.pth')
}

def load_models_on_startup():
    print("Loading models...")
    for model_name, path in MODEL_PATHS.items():
        if os.path.exists(path):
            print(f"Found {model_name} at {path}. Loading...")
            model, device = load_prediction_model(path, model_name)
            if model:
                MODELS[model_name] = model
                DEVICES[model_name] = device
                print(f"Successfully loaded {model_name} on {device}")
        else:
            print(f"Warning: Model file not found for {model_name} at {path}")

# Load models when the app starts
load_models_on_startup()

@app.route('/')
def home():
    """Render the homepage."""
    return render_template('index.html')

@app.route('/api/captcha', methods=['GET'])
def get_captcha():
    """Returns a new random CAPTCHA digit."""
    digit = generate_captcha_digit()
    return jsonify({
        'status': 'success',
        'captcha_digit': digit
    })

@app.route('/api/predict', methods=['POST'])
def predict():
    """
    Receives base64 image data, preprocesses it, runs inference using
    the selected model, and compares the prediction to the CAPTCHA target.
    """
    data = request.json
    
    if not data or 'image' not in data or 'target_digit' not in data or 'model_name' not in data:
        return jsonify({'status': 'error', 'message': 'Invalid input data'}), 400
        
    base64_image = data['image']
    target_digit = int(data['target_digit'])
    model_name = data['model_name']
    
    # Check if image is blank/empty based on length
    if len(base64_image) < 100:
        return jsonify({'status': 'error', 'message': 'Blank canvas submitted'}), 400

    if model_name not in MODELS:
        return jsonify({'status': 'error', 'message': f'Model {model_name} is not loaded on the server.'}), 400

    model = MODELS[model_name]
    device = DEVICES[model_name]

    try:
        # 1. Preprocess the image
        # Target size for EfficientNet/MobileNetV2 training was 64x64
        tensor = base64_to_tensor(base64_image, target_size=(64, 64))
        
        # 2. Predict
        predicted_digit, confidence = predict_digit(model, device, tensor)
        
        # 3. Verification logic
        is_verified = (predicted_digit == target_digit)
        
        # 4. Reject low confidence
        if confidence < 0.4:
            is_verified = False
            message = "Confidence too low, please draw more clearly."
        else:
            message = "Verification Successful" if is_verified else "Verification Failed"

        return jsonify({
            'status': 'success',
            'predicted_digit': predicted_digit,
            'confidence': confidence,
            'is_verified': is_verified,
            'message': message
        })

    except Exception as e:
        print(f"Prediction Error: {str(e)}")
        return jsonify({'status': 'error', 'message': 'Error processing image'}), 500

if __name__ == '__main__':
    # Start Flask server
    app.run(debug=True, host='0.0.0.0', port=5000)
