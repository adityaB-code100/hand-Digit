import torch
import torch.nn as nn
import timm
from torchvision import models

def build_model(model_name='EfficientNetV2B0', num_classes=10):
    """
    Builds the requested PyTorch model with a custom classification head
    matching the training phase exactly.
    """
    if model_name == 'EfficientNetV2B0':
        model = timm.create_model('tf_efficientnetv2_b0', pretrained=False, num_classes=0)
        in_features = model.num_features
        model.classifier = nn.Sequential(
            nn.Linear(in_features, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(512, num_classes)
        )
    elif model_name == 'MobileNetV2':
        model = models.mobilenet_v2(weights=None)
        in_features = model.classifier[1].in_features
        model.classifier = nn.Sequential(
            nn.Dropout(p=0.4, inplace=False),
            nn.Linear(in_features, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(p=0.4, inplace=False),
            nn.Linear(512, num_classes)
        )
    else:
        raise ValueError(f"Model {model_name} not supported.")
        
    return model

def load_prediction_model(model_path, model_name='EfficientNetV2B0', num_classes=10):
    """
    Loads a saved PyTorch model for inference.
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    model = build_model(model_name=model_name, num_classes=num_classes)
    
    try:
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.to(device)
        model.eval()
        return model, device
    except Exception as e:
        print(f"Error loading {model_name} from {model_path}: {e}")
        return None, device

def predict_digit(model, device, image_tensor):
    """
    Runs inference on a single image tensor and returns the predicted digit (0-9) and confidence.
    """
    image_tensor = image_tensor.to(device)
    
    with torch.no_grad():
        output = model(image_tensor)
        probabilities = torch.softmax(output, dim=1)
        predicted_class = torch.argmax(probabilities, dim=1).item()
        confidence = probabilities[0][predicted_class].item()
        
    return predicted_class, confidence
