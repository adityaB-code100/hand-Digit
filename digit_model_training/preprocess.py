import os
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image

# Model input sizes
MODEL_SIZES = {
    'EfficientNetV2B0': (64, 64),
    'MobileNetV2': (64, 64)
}

class DigitDataset(Dataset):
    def __init__(self, image_paths, labels, transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        label = self.labels[idx]
        
        # Load image as RGB (EfficientNet/MobileNet expect 3 channels)
        try:
            image = Image.open(img_path).convert('RGB')
        except Exception as e:
            # Fallback for corrupted images if any slipped through
            image = Image.new('RGB', (64, 64), color='black')
            
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

def create_dataloaders(X_train, y_train, X_test, y_test, model_name='EfficientNetV2B0', batch_size=64, num_workers=4):
    """
    Creates PyTorch DataLoaders for training and testing.
    Optimized for GPU with pin_memory=True.
    """
    target_size = MODEL_SIZES.get(model_name, (64, 64))
    
    train_transform = get_transforms(target_size, is_training=True)
    test_transform = get_transforms(target_size, is_training=False)
    
    train_dataset = DigitDataset(X_train, y_train, transform=train_transform)
    test_dataset = DigitDataset(X_test, y_test, transform=test_transform)
    
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
