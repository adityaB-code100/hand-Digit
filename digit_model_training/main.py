import os
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from sklearn.model_selection import train_test_split

from dataset_loader import load_dataset, prepare_data_splits
from preprocess import create_dataloaders
from cross_validation import run_cross_validation
from models import build_model, setup_device
from evaluate import evaluate_model, plot_training_history
from utils import setup_directories, TimeTracker, get_model_size, format_time, log_gpu_memory

def train_model(model, train_loader, val_loader, device, epochs, patience=3):
    """
    Trains the PyTorch model using mixed precision (AMP) and Early Stopping.
    """
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    scaler = torch.amp.GradScaler('cuda') if torch.cuda.is_available() else None
    
    best_val_loss = float('inf')
    patience_counter = 0
    best_model_state = None
    
    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}
    
    for epoch in range(epochs):
        model.train()
        train_loss, train_correct, train_total = 0, 0, 0
        
        # Training loop
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            
            # Forward pass with AMP
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
                
            train_loss += loss.item() * images.size(0)
            _, predicted = outputs.max(1)
            train_total += labels.size(0)
            train_correct += predicted.eq(labels).sum().item()
            
        epoch_train_loss = train_loss / train_total
        epoch_train_acc = train_correct / train_total
        
        # Validation loop
        model.eval()
        val_loss, val_correct, val_total = 0, 0, 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                
                if scaler:
                    with torch.amp.autocast('cuda'):
                        outputs = model(images)
                        loss = criterion(outputs, labels)
                else:
                    outputs = model(images)
                    loss = criterion(outputs, labels)
                    
                val_loss += loss.item() * images.size(0)
                _, predicted = outputs.max(1)
                val_total += labels.size(0)
                val_correct += predicted.eq(labels).sum().item()
                
        epoch_val_loss = val_loss / val_total
        epoch_val_acc = val_correct / val_total
        
        history['train_loss'].append(epoch_train_loss)
        history['train_acc'].append(epoch_train_acc)
        history['val_loss'].append(epoch_val_loss)
        history['val_acc'].append(epoch_val_acc)
        
        # Early Stopping & Checkpointing
        if epoch_val_loss < best_val_loss:
            best_val_loss = epoch_val_loss
            patience_counter = 0
            best_model_state = model.state_dict().copy()
        else:
            patience_counter += 1
            if patience_counter >= patience:
                break
                
    # Restore best weights
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
        
    return history

def run_training_experiment(
    model_name,
    dataset_path="../Dataset",
    target_size=(64, 64),
    preprocessing_approach='standard',
    load_full_characters=False,
    epochs=5,
    batch_size=64,
    num_workers=4
):
    """
    Runs a training experiment using a single dataset split into Train (70%), Val (10%), and Test (20%).
    Returns a dictionary of key metrics and training history.
    """
    setup_directories()
    device = setup_device()
    tracker = TimeTracker()
    
    print(f"\n--- RUNNING EXPERIMENT: Model={model_name} | Size={target_size} | Preprocess={preprocessing_approach} | Alphanumeric={load_full_characters} ---")
    
    # 1. Load Dataset
    image_paths, labels = load_dataset(dataset_path, load_full_characters=load_full_characters)
    num_classes = len(set(labels))
    
    # Split into 80% train_val, 20% hold-out test
    X_train_full, X_test, y_train_full, y_test = prepare_data_splits(image_paths, labels, test_size=0.2)
    
    # Split the 80% train_val into 70% train, 10% val
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_full, y_train_full, test_size=0.125, stratify=y_train_full, random_state=42
    )
    
    print(f"Dataset split results:")
    print(f"  Training samples:   {len(X_train)}")
    print(f"  Validation samples: {len(X_val)}")
    print(f"  Testing samples:    {len(X_test)}")
    
    # 2. Create Dataloaders
    train_loader, val_loader = create_dataloaders(
        X_train, y_train, X_val, y_val, 
        target_size=target_size, preprocessing_approach=preprocessing_approach,
        batch_size=batch_size, num_workers=num_workers
    )
    
    _, test_loader = create_dataloaders(
        X_test, y_test, X_test, y_test,
        target_size=target_size, preprocessing_approach=preprocessing_approach,
        batch_size=batch_size, num_workers=num_workers
    )
        
    # 3. Build Model
    model = build_model(model_name=model_name, num_classes=num_classes).to(device)
    
    # 4. Train
    tracker.start()
    history = train_model(model, train_loader, val_loader, device, epochs=epochs)
    train_time = tracker.stop()
    
    # Save model weights
    mode_suffix = "full" if load_full_characters else "digit"
    model_filename = f"{model_name}_{target_size[0]}x{target_size[1]}_{preprocessing_approach}_{mode_suffix}.pth"
    model_save_path = os.path.join("saved_models", model_filename)
    torch.save(model.state_dict(), model_save_path)
    
    model_size_mb = get_model_size(model)
    
    # 5. Evaluate Validation Accuracy
    print("\nEvaluating on Validation Split...")
    val_metrics = evaluate_model(model, val_loader, device, model_name)
    val_acc = val_metrics['Accuracy']
    
    # 6. Evaluate Test Accuracy
    print("\nEvaluating on Hold-out Test Split...")
    test_metrics = evaluate_model(model, test_loader, device, model_name)
    test_acc = test_metrics['Accuracy']
        
    print(f"\nExperiment Complete!")
    print(f"Validation Accuracy: {val_acc:.4f}")
    print(f"Test Accuracy:       {test_acc:.4f}")
    print(f"Training Time: {format_time(train_time)} | Model Size: {model_size_mb:.2f} MB")
    
    return {
        'Model': model_name,
        'Size': f"{target_size[0]}x{target_size[1]}",
        'Preprocessing': preprocessing_approach,
        'Alphanumeric': load_full_characters,
        'Validation Accuracy': val_acc,
        'Test Accuracy': test_acc,
        'Model Size (MB)': model_size_mb,
        'Training Time (s)': train_time,
        'Training Time Formatted': format_time(train_time),
        'History': history,
        'Weights Saved': model_save_path
    }

def main():
    # Change working directory to the script's directory so relative paths work properly
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    
    # Run a simple default training run for backward compatibility
    result = run_training_experiment(
        model_name='EfficientNetV2B0',
        dataset_path="../Dataset",
        target_size=(64, 64),
        preprocessing_approach='standard',
        load_full_characters=False,
        epochs=3
    )
    
    print("\nMain script execution finished.")

if __name__ == "__main__":
    main()
