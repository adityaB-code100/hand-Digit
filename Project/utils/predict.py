import torch
import torch.nn as nn
import timm
from torchvision import models

def build_model(model_name='EfficientNetV2B0', num_classes=10):
    """
    Builds the requested PyTorch model with a custom classification head
    matching the training phase exactly.
    """
    mapping = {
        'EfficientNetV2B0': 'tf_efficientnetv2_b0',
        'EfficientNetV2B1': 'tf_efficientnetv2_b1',
        'EfficientNetV2B2': 'tf_efficientnetv2_b2',
        'EfficientNetV2B3': 'tf_efficientnetv2_b3',
    }
    
    timm_model_name = mapping.get(model_name, model_name)
    
    if timm_model_name.lower() in ('mobilenetv2', 'mobilenet_v2'):
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
        # Standard timm model
        model = timm.create_model(timm_model_name, pretrained=False, num_classes=0)
        in_features = model.num_features
        model.classifier = nn.Sequential(
            nn.Linear(in_features, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(512, num_classes)
        )
        
    return model

def index_to_char(index):
    """
    Maps class index (0-61) back to the corresponding character:
    - 0-9: '0'-'9'
    - 10-35: 'A'-'Z'
    - 36-61: 'a'-'z'
    """
    if 0 <= index <= 9:
        return str(index)
    elif 10 <= index <= 35:
        return chr(ord('A') + (index - 10))
    elif 36 <= index <= 61:
        return chr(ord('a') + (index - 36))
    return str(index)

def load_prediction_model(model_path, model_name='EfficientNetV2B0', num_classes=10):
    """
    Loads a saved PyTorch model for inference.
    Automatically detects the number of classes from the state dict keys.
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    try:
        state_dict = torch.load(model_path, map_location='cpu')
        
        # Auto-detect num_classes from state dict
        detected_num_classes = num_classes
        for key in reversed(state_dict.keys()):
            if 'classifier' in key and ('bias' in key or 'weight' in key):
                detected_num_classes = state_dict[key].shape[0]
                break
                
        print(f"Loaded weights from {model_path}. Auto-detected classes: {detected_num_classes}")
        
        model = build_model(model_name=model_name, num_classes=detected_num_classes)
        model.load_state_dict(state_dict)
        model.to(device)
        model.eval()
        
        # Attach attribute to the model for dynamic query at runtime
        model.num_classes = detected_num_classes
        
        return model, device
    except Exception as e:
        print(f"Error loading {model_name} from {model_path}: {e}")
        return None, device

def predict_digit(model, device, image_tensor):
    """
    Runs inference on a single image tensor and returns the predicted digit index and confidence.
    Retained for backward compatibility with 10-class layouts.
    """
    image_tensor = image_tensor.to(device)
    with torch.no_grad():
        output = model(image_tensor)
        probabilities = torch.softmax(output, dim=1)
        predicted_class = torch.argmax(probabilities, dim=1).item()
        confidence = probabilities[0][predicted_class].item()
        
    return predicted_class, confidence

def predict_character(model, device, image_tensor):
    """
    Runs inference on a single image tensor and returns (predicted_class_index, char_label, confidence).
    Supports all 62 alphanumeric classes.
    """
    image_tensor = image_tensor.to(device)
    with torch.no_grad():
        output = model(image_tensor)
        probabilities = torch.softmax(output, dim=1)
        predicted_class = torch.argmax(probabilities, dim=1).item()
        confidence = probabilities[0][predicted_class].item()
        
    char_label = index_to_char(predicted_class)
    return predicted_class, char_label, confidence
