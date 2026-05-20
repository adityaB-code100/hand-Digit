import torch
import torch.nn as nn
import timm
from torchvision import models

def build_model(model_name='EfficientNetV2B0', num_classes=10):
    """
    Builds the requested PyTorch model (timm or torchvision) with a custom classification head
    dynamically sized to num_classes.
    """
    # Normalize model names to support both custom and standard timm/torchvision names
    mapping = {
        'EfficientNetV2B0': 'tf_efficientnetv2_b0',
        'EfficientNetV2B1': 'tf_efficientnetv2_b1',
        'EfficientNetV2B2': 'tf_efficientnetv2_b2',
        'EfficientNetV2B3': 'tf_efficientnetv2_b3',
    }
    
    timm_model_name = mapping.get(model_name, model_name)
    
    if timm_model_name.lower() in ('mobilenetv2', 'mobilenet_v2'):
        print(f"Building MobileNetV2 with num_classes={num_classes}")
        # Using torchvision for MobileNetV2
        model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.DEFAULT)
        in_features = model.classifier[1].in_features
        
        # Replace the classifier
        model.classifier = nn.Sequential(
            nn.Dropout(p=0.4, inplace=False),
            nn.Linear(in_features, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(p=0.4, inplace=False),
            nn.Linear(512, num_classes)
        )
        
    else:
        print(f"Building timm model '{timm_model_name}' with num_classes={num_classes}")
        try:
            # Using timm for the exact architecture
            model = timm.create_model(timm_model_name, pretrained=True, num_classes=0)
            in_features = model.num_features
            
            # Create a custom classification head
            model.classifier = nn.Sequential(
                nn.Linear(in_features, 512),
                nn.BatchNorm1d(512),
                nn.ReLU(),
                nn.Dropout(0.4),
                nn.Linear(512, num_classes)
            )
        except Exception as e:
            # Fallback to tf_efficientnetv2_b0 if build fails
            print(f"Error building '{timm_model_name}' ({e}). Falling back to 'tf_efficientnetv2_b0'.")
            model = timm.create_model('tf_efficientnetv2_b0', pretrained=True, num_classes=0)
            in_features = model.num_features
            model.classifier = nn.Sequential(
                nn.Linear(in_features, 512),
                nn.BatchNorm1d(512),
                nn.ReLU(),
                nn.Dropout(0.4),
                nn.Linear(512, num_classes)
            )
            
    return model

def setup_device():
    """Configures the training device (GPU if available)."""
    if torch.cuda.is_available():
        device = torch.device('cuda')
        print(f"CUDA GPU detected: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device('cpu')
        print("No GPU detected. Using CPU.")
    return device
