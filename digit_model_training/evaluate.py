import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc, precision_recall_fscore_support
from sklearn.preprocessing import label_binarize
import torch
from tqdm import tqdm

def evaluate_model(model, test_loader, device, model_name, graphs_dir="graphs", reports_dir="reports"):
    """
    Evaluates the model and generates all requested metrics and graphs.
    Returns a dictionary of metrics.
    """
    print(f"\nEvaluating {model_name}...")
    model.eval()
    
    y_true = []
    y_pred_probs = []
    y_pred = []
    
    with torch.no_grad():
        for images, labels in tqdm(test_loader, desc="Evaluating"):
            images = images.to(device)
            
            # Predict probabilities and classes
            outputs = model(images)
            probs = torch.softmax(outputs, dim=1)
            preds = torch.argmax(probs, dim=1)
            
            y_true.extend(labels.cpu().numpy())
            y_pred_probs.extend(probs.cpu().numpy())
            y_pred.extend(preds.cpu().numpy())
            
    y_true = np.array(y_true)
    y_pred_probs = np.array(y_pred_probs)
    y_pred = np.array(y_pred)
    
    # 1. Classification Report & Precision/Recall/F1
    report = classification_report(y_true, y_pred, output_dict=True)
    precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='weighted')
    
    # Save Classification Report as CSV
    report_df = pd.DataFrame(report).transpose()
    report_df.to_csv(os.path.join(reports_dir, f"{model_name}_classification_report.csv"))
    
    # 2. Confusion Matrix
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=range(10), yticklabels=range(10))
    plt.title(f'{model_name} Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.savefig(os.path.join(graphs_dir, f"{model_name}_confusion_matrix.png"))
    plt.close()
    
    # 3. ROC Curve
    try:
        # Binarize labels for multi-class ROC
        y_true_bin = label_binarize(y_true, classes=range(10))
        n_classes = y_true_bin.shape[1]
        
        plt.figure(figsize=(10, 8))
        for i in range(n_classes):
            fpr, tpr, _ = roc_curve(y_true_bin[:, i], y_pred_probs[:, i])
            roc_auc = auc(fpr, tpr)
            plt.plot(fpr, tpr, lw=2, label=f'Class {i} (AUC = {roc_auc:.2f})')
            
        plt.plot([0, 1], [0, 1], 'k--', lw=2)
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f'{model_name} Multi-class ROC Curve')
        plt.legend(loc="lower right")
        plt.savefig(os.path.join(graphs_dir, f"{model_name}_roc_curve.png"))
        plt.close()
    except Exception as e:
        print(f"Error plotting ROC curve: {e}")

    # Return metrics for the final comparison table
    return {
        'Precision': precision,
        'Recall': recall,
        'F1 Score': f1,
        'Accuracy': report['accuracy']
    }

def plot_training_history(history, model_name, graphs_dir="graphs"):
    """Plots training & validation accuracy and loss."""
    # Accuracy plot
    plt.figure(figsize=(10, 6))
    plt.plot(history['train_acc'], label='Training Accuracy')
    if 'val_acc' in history:
        plt.plot(history['val_acc'], label='Validation Accuracy')
    plt.title(f'{model_name} Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.savefig(os.path.join(graphs_dir, f"{model_name}_accuracy.png"))
    plt.close()
    
    # Loss plot
    plt.figure(figsize=(10, 6))
    plt.plot(history['train_loss'], label='Training Loss')
    if 'val_loss' in history:
        plt.plot(history['val_loss'], label='Validation Loss')
    plt.title(f'{model_name} Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(os.path.join(graphs_dir, f"{model_name}_loss.png"))
    plt.close()
