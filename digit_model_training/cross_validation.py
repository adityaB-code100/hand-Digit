import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import StratifiedKFold
from preprocess import create_dataloaders
from models import build_model, setup_device
from tqdm import tqdm

def run_cross_validation(X, y, model_name, epochs, batch_size, n_splits=5):
    """
    Performs Stratified K-Fold Cross Validation using PyTorch.
    """
    device = setup_device()
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    
    fold_accuracies = []
    fold_losses = []
    
    print(f"\n--- Starting {n_splits}-Fold Cross Validation for {model_name} ---")
    
    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
        print(f"\nFold {fold + 1}/{n_splits}")
        
        X_train_fold = [X[i] for i in train_idx]
        y_train_fold = [y[i] for i in train_idx]
        X_val_fold = [X[i] for i in val_idx]
        y_val_fold = [y[i] for i in val_idx]
        
        # Create dataloaders
        train_loader, val_loader = create_dataloaders(
            X_train_fold, y_train_fold, X_val_fold, y_val_fold, 
            model_name=model_name, batch_size=batch_size
        )
        
        # Build model
        model = build_model(model_name=model_name).to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=1e-4)
        scaler = torch.amp.GradScaler('cuda') if torch.cuda.is_available() else None
        
        best_val_loss = float('inf')
        patience_counter = 0
        patience = 3
        best_val_acc = 0
        
        # Train
        for epoch in range(epochs):
            model.train()
            for images, labels in train_loader:
                images, labels = images.to(device), labels.to(device)
                optimizer.zero_grad()
                
                if scaler:
                    with torch.amp.autocast('cuda'):
                        outputs = model(images)
                        loss = criterion(outputs, labels)
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    outputs = model(images)
                    loss = criterion(outputs, labels)
                    loss.backward()
                    optimizer.step()
            
            # Evaluate
            model.eval()
            val_loss = 0
            correct = 0
            total = 0
            with torch.no_grad():
                for images, labels in val_loader:
                    images, labels = images.to(device), labels.to(device)
                    outputs = model(images)
                    loss = criterion(outputs, labels)
                    val_loss += loss.item() * images.size(0)
                    _, predicted = outputs.max(1)
                    total += labels.size(0)
                    correct += predicted.eq(labels).sum().item()
            
            val_loss /= total
            val_acc = correct / total
            
            # Early stopping check
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_val_acc = val_acc
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    break
        
        print(f"Fold {fold + 1} - Validation Accuracy: {best_val_acc:.4f}, Validation Loss: {best_val_loss:.4f}")
        fold_accuracies.append(best_val_acc)
        fold_losses.append(best_val_loss)
        
    mean_acc = np.mean(fold_accuracies)
    std_acc = np.std(fold_accuracies)
    mean_loss = np.mean(fold_losses)
    
    print(f"\n--- Cross Validation Results for {model_name} ---")
    print(f"Fold-wise Accuracies: {['{:.4f}'.format(a) for a in fold_accuracies]}")
    print(f"Mean Accuracy: {mean_acc:.4f} (+/- {std_acc:.4f})")
    print(f"Mean Loss: {mean_loss:.4f}")
    
    return mean_acc, std_acc
