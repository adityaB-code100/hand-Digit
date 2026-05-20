import os
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image, ImageOps

# Model input sizes (for backward compatibility)
MODEL_SIZES = {
    'EfficientNetV2B0': (64, 64),
    'MobileNetV2': (64, 64)
}

class DigitDataset(Dataset):
    def __init__(self, image_paths, labels, transform=None, preprocessing_approach='standard', fallback_size=(64, 64)):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform
        self.preprocessing_approach = preprocessing_approach.lower()
        self.fallback_size = fallback_size

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        label = self.labels[idx]
        
        try:
            if self.preprocessing_approach == 'bitmask':
                # Apply bitmask preprocessing (white digit on black background)
                # 1. Read image in grayscale
                img_gray = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                if img_gray is None:
                    raise ValueError(f"Failed to read image: {img_path}")
                
                # 2. Otsu's binary thresholding with inversion (assuming dark digit on light background)
                _, thresh = cv2.threshold(img_gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
                
                # 3. Crop to the bounding box of the non-zero pixels
                pts = np.argwhere(thresh > 0)
                if len(pts) > 0:
                    y_min, x_min = pts.min(axis=0)
                    y_max, x_max = pts.max(axis=0)
                    cropped = thresh[y_min:y_max+1, x_min:x_max+1]
                    
                    # 4. Convert to PIL and add 10% padding (keeps aspect ratio and mimics Flask preprocessor)
                    pil_img = Image.fromarray(cropped)
                    padding = int(max(pil_img.size) * 0.1)
                    if padding < 2:
                        padding = 2
                    pil_img = ImageOps.expand(pil_img, border=padding, fill=0)
                else:
                    pil_img = Image.fromarray(thresh)
                
                # 5. Convert to RGB as the pre-trained EfficientNet/MobileNet expect 3 channels
                image = pil_img.convert('RGB')
            else:
                # Standard loading
                image = Image.open(img_path).convert('RGB')
                
        except Exception as e:
            # Fallback for corrupted images
            image = Image.new('RGB', self.fallback_size, color='black')
            
        if self.transform:
            image = self.transform(image)
            
        # Convert label to tensor
        label = torch.tensor(label, dtype=torch.long)
        
        return image, label

def get_transforms(target_size, is_training=True):
    """Returns torchvision transforms for training or validation."""
    if is_training:
        return transforms.Compose([
            transforms.Resize(target_size),
            transforms.RandomRotation(10),
            transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1)),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.ToTensor(),
            # EfficientNet and MobileNet typically use ImageNet normalization
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                 std=[0.229, 0.224, 0.225])
        ])
    else:
        return transforms.Compose([
            transforms.Resize(target_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                 std=[0.229, 0.224, 0.225])
        ])

def create_dataloaders(X_train, y_train, X_test, y_test, target_size=None, preprocessing_approach='standard', model_name=None, batch_size=64, num_workers=4):
    """
    Creates PyTorch DataLoaders for training and testing.
    Optimized for GPU with pin_memory=True.
    Supports either target_size or model_name for sizing, and standard/bitmask preprocessing.
    """
    if target_size is None:
        if model_name is not None:
            target_size = MODEL_SIZES.get(model_name, (64, 64))
        else:
            target_size = (64, 64)
            
    train_transform = get_transforms(target_size, is_training=True)
    test_transform = get_transforms(target_size, is_training=False)
    
    train_dataset = DigitDataset(X_train, y_train, transform=train_transform, preprocessing_approach=preprocessing_approach, fallback_size=target_size)
    test_dataset = DigitDataset(X_test, y_test, transform=test_transform, preprocessing_approach=preprocessing_approach, fallback_size=target_size)
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=num_workers,
        pin_memory=True,  # Optimizes transfer to GPU
        drop_last=True    # Helps with batch norm stability
    )
    
    test_loader = DataLoader(
        test_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=num_workers,
        pin_memory=True
    )
    
    return train_loader, test_loader
