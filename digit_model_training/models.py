import torch
import torch.nn as nn
import timm
from torchvision import models

def build_model(model_name='EfficientNetV2B0', num_classes=10):
    """
    Builds the requested PyTorch model with a custom classification head.
    """
    if model_name == 'EfficientNetV2B0':
        # Using timm for the exact tf_efficientnetv2_b0 architecture
        # pretrained=True downloads the ImageNet weights
        model = timm.create_model('tf_efficientnetv2_b0', pretrained=True, num_classes=0)
        
        # Determine the number of features in the final layer
        in_features = model.num_features
        
        # Create a custom classification head
        model.classifier = nn.Sequential(
            nn.Linear(in_features, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(512, num_classes)
        )
        
    elif model_name == 'MobileNetV2':
        # Using torchvision for MobileNetV2
        # Weights default to ImageNet (IMAGENET1K_V1/V2)
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
        raise ValueError(f"Model {model_name} not supported. Choose from EfficientNetV2B0, MobileNetV2.")
        
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
