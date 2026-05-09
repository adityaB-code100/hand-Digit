import torch
from torchvision import transforms
from PIL import Image
from models import build_model
from preprocess import MODEL_SIZES

def load_prediction_model(model_path, model_name='EfficientNetV2B0', num_classes=10):
    """
    Loads a saved PyTorch model for inference.
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Initialize the model structure
    model = build_model(model_name=model_name, num_classes=num_classes)
    
    # Load the trained weights
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    
    return model, device

def predict_digit(model, device, image_path, model_name='EfficientNetV2B0'):
    """
    Runs inference on a single image and returns the predicted digit (0-9).
    This function is designed to be easily imported into a Flask app.
    """
    target_size = MODEL_SIZES.get(model_name, (64, 64))
    
    # Same validation transforms used during training
    transform = transforms.Compose([
        transforms.Resize(target_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                             std=[0.229, 0.224, 0.225])
    ])
    
    try:
        image = Image.open(image_path).convert('RGB')
    except Exception as e:
        print(f"Error loading image: {e}")
        return None
        
    # Preprocess image
    image_tensor = transform(image).unsqueeze(0).to(device) # Add batch dimension
    
    # Inference
    with torch.no_grad():
        output = model(image_tensor)
        probabilities = torch.softmax(output, dim=1)
        predicted_class = torch.argmax(probabilities, dim=1).item()
        confidence = probabilities[0][predicted_class].item()
        
    return predicted_class, confidence

if __name__ == "__main__":
    # Example usage / Testing
    import os
    
    # Assuming standard test folder structure
    test_image_path = os.path.join("..", "handDigitDataset", "digit_0", os.listdir(os.path.join("..", "handDigitDataset", "digit_0"))[0])
    model_path = os.path.join("saved_models", "EfficientNetV2B0_digit.pth")
    
    if os.path.exists(model_path) and os.path.exists(test_image_path):
        print(f"Loading model from {model_path}...")
        model, device = load_prediction_model(model_path, 'EfficientNetV2B0')
        
        print(f"Running prediction on {test_image_path}...")
        digit, conf = predict_digit(model, device, test_image_path, 'EfficientNetV2B0')
        print(f"Predicted Digit: {digit} (Confidence: {conf:.2%})")
    else:
        print("Model or test image not found. Ensure you have trained the model first.")
