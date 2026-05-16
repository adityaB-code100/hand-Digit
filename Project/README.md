# AI Handwritten CAPTCHA Verification

This project is a full-stack handwritten digit CAPTCHA verification system using PyTorch models (EfficientNetV2B0 and MobileNetV2), Flask, and a custom HTML5 canvas frontend.

## Features
- **Frontend**: A sleek, dark-mode UI with a drawing canvas to capture user input.
- **Backend**: Flask API that serves the frontend, processes the image, and orchestrates model inference.
- **AI Models**: Integrates directly with pre-trained PyTorch models without retraining.
- **Preprocessing**: Exact replication of training validation transforms for accurate inference.

## Prerequisites
- Python 3.8+
- Pre-trained models: `mobilenetv2.pth` and `efficientnetv2b0.pth` should be placed in `project/models/`.

## Local Setup
1. Create a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Run the Flask application:
   ```bash
   python app.py
   ```
4. Access the web app at `http://127.0.0.1:5000`.

## Deployment
This project is configured to run on Render or Railway out of the box using `requirements.txt` and Gunicorn (to be added for production).
