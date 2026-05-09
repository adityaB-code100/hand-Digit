import os
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

from dataset_loader import load_dataset, prepare_data_splits
from preprocess import create_dataloaders
from cross_validation import run_cross_validation
from models import build_model, setup_device
from evaluate import evaluate_model, plot_training_history
from utils import setup_directories, TimeTracker, get_model_size, format_time, log_gpu_memory

def train_model(model, train_loader, test_loader, device, model_name, epochs):
    """
    Trains the PyTorch model using mixed precision (AMP) and Early Stopping.
    """
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    scaler = torch.amp.GradScaler('cuda') if torch.cuda.is_available() else None
    
    best_val_loss = float('inf')
    patience = 3
    patience_counter = 0
    best_model_state = None
    
    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}
    
    for epoch in range(epochs):
        print(f"\nEpoch {epoch+1}/{epochs}")
        model.train()
        train_loss, train_correct, train_total = 0, 0, 0
        
        # Training loop with tqdm
        for images, labels in tqdm(train_loader, desc="Training"):
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            
            # Forward pass with AMP
            if scaler:
                with torch.amp.autocast('cuda'):
                    outputs = model(images)
                    loss = criterion(outputs, labels)
                # Backward pass
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
            for images, labels in test_loader:
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
        
        print(f"Train Loss: {epoch_train_loss:.4f} | Train Acc: {epoch_train_acc:.4f} | Val Loss: {epoch_val_loss:.4f} | Val Acc: {epoch_val_acc:.4f}")
        log_gpu_memory()
        
        # Early Stopping & Checkpointing
        if epoch_val_loss < best_val_loss:
            best_val_loss = epoch_val_loss
            patience_counter = 0
            best_model_state = model.state_dict().copy()
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping triggered at epoch {epoch+1}.")
                break
                
    # Restore best weights
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
        
    return history

def main():
    # Change working directory to the script's directory so relative paths work properly
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    
    # Setup directories for outputs
    setup_directories()
    
    # Setup Device (GPU)
    device = setup_device()
    
    # Configuration
    dataset_path = "../handDigitDataset"
    epochs = 15  # Slightly higher epochs allowed due to Early Stopping and faster GPU
    batch_size = 64
    num_workers = 4 # Optimized for data loading on Windows
    models_to_train = ['EfficientNetV2B0', 'MobileNetV2']
    
    print(f"Starting PyTorch training pipeline...")
    
    # 1. Load and split dataset
    print("\n--- Data Loading ---")
    image_paths, labels = load_dataset(dataset_path)
    X_train_cv, X_test, y_train_cv, y_test = prepare_data_splits(image_paths, labels, test_size=0.2)
    
    results = []
    
    # 2. Iterate over chosen models
    for model_name in models_to_train:
        print(f"\n{'='*50}")
        print(f"Processing Model: {model_name}")
        print(f"{'='*50}")
        
        tracker = TimeTracker()
        
        # --- Step A: Cross Validation ---
        # Note: If CV takes too long, you can disable it by commenting this section.
        # cv_mean_acc, cv_std_acc = run_cross_validation(X_train_cv, y_train_cv, model_name, epochs, batch_size)
        cv_mean_acc = "N/A" # Skipping CV for full run speed, uncomment above if needed
        cv_std_acc = "N/A"
        
        # --- Step B: Final Model Training ---
        print(f"\n--- Training Final {model_name} Model on 80% split ---")
        train_loader, test_loader = create_dataloaders(
            X_train_cv, y_train_cv, X_test, y_test, 
            model_name=model_name, batch_size=batch_size, num_workers=num_workers
        )
        
        model = build_model(model_name=model_name).to(device)
        
        tracker.start()
        history = train_model(model, train_loader, test_loader, device, model_name, epochs)
        train_time = tracker.stop()
        
        # Plot training history
        plot_training_history(history, model_name)
        
        # Save model (.pth format)
        model_save_path = os.path.join("saved_models", f"{model_name}_digit.pth")
        torch.save(model.state_dict(), model_save_path)
        
        # Get model size
        model_size_mb = get_model_size(model)
        
        # --- Step C: Evaluation ---
        eval_metrics = evaluate_model(model, test_loader, device, model_name)
        
        # Collect results
        results.append({
            'Model': model_name,
            'CV Accuracy': f"{cv_mean_acc}" if isinstance(cv_mean_acc, str) else f"{cv_mean_acc:.4f} (+/- {cv_std_acc:.4f})",
            'Test Accuracy': eval_metrics['Accuracy'],
            'Precision': eval_metrics['Precision'],
            'Recall': eval_metrics['Recall'],
            'F1 Score': eval_metrics['F1 Score'],
            'Model Size (MB)': f"{model_size_mb:.2f}",
            'Training Time': format_time(train_time)
        })
        
    # --- Step D: Summary Report ---
    print("\n" + "="*50)
    print("FINAL MODEL COMPARISON (PyTorch)")
    print("="*50)
    
    results_df = pd.DataFrame(results)
    print(results_df.to_string(index=False))
    
    # Save comparison to CSV
    results_df.to_csv(os.path.join("reports", "final_model_comparison.csv"), index=False)
    print("\nTraining pipeline complete. Reports, graphs, and models have been saved to respective folders.")

if __name__ == "__main__":
    main()
