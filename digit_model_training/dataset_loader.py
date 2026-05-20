import os
import glob
from collections import Counter
import cv2
import pandas as pd
from sklearn.model_selection import train_test_split

def load_dataset(dataset_path, load_full_characters=False):
    """
    Scans the dataset folder, verifies images, and extracts paths and labels.
    Supports either digit-only (10 classes) or full alphanumeric (62 classes) mode.
    """
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"Dataset path {dataset_path} does not exist.")

    image_paths = []
    labels = []
    
    # Auto-detect folder style (character_0 vs digit_0)
    character_style_exists = os.path.exists(os.path.join(dataset_path, "character_0"))
    
    if character_style_exists:
        if load_full_characters:
            # Full 62 classes: digits (0-9), uppercase (10-35), lowercase (36-61)
            class_to_idx = {}
            for i in range(10):
                class_to_idx[f"character_{i}"] = i
            for c in range(ord('A'), ord('Z') + 1):
                class_to_idx[f"{chr(c)}_caps"] = 10 + (c - ord('A'))
            for c in range(ord('a'), ord('z') + 1):
                class_to_idx[f"character_{chr(c)}"] = 36 + (c - ord('a'))
        else:
            # Digits only (10 classes)
            class_to_idx = {f"character_{i}": i for i in range(10)}
    else:
        # Hand digit style: digit_0 to digit_9
        class_to_idx = {f"digit_{i}": i for i in range(10)}
        
    print(f"Scanning dataset in: {dataset_path} (Alphanumeric: {load_full_characters})")
    print(f"Class-to-Index mapping holds {len(class_to_idx)} classes.")
    
    corrupted_files = 0
    valid_files = 0

    for class_folder, idx in class_to_idx.items():
        folder_path = os.path.join(dataset_path, class_folder)
        if not os.path.exists(folder_path):
            # If full characters are requested, some folders might be missing, so warning is fine
            if not load_full_characters:
                print(f"Warning: Folder {folder_path} not found.")
            continue
            
        extensions = ('*.jpg', '*.jpeg', '*.png')
        files = []
        for ext in extensions:
            files.extend(glob.glob(os.path.join(folder_path, ext)))
            files.extend(glob.glob(os.path.join(folder_path, ext.upper())))
            
        for file in files:
            # Check if file is valid image
            try:
                img = cv2.imread(file)
                if img is None:
                    corrupted_files += 1
                else:
                    image_paths.append(file)
                    labels.append(idx)
                    valid_files += 1
            except Exception as e:
                corrupted_files += 1

    print(f"Found {valid_files} valid images.")
    print(f"Found {corrupted_files} corrupted images (skipped).")
    
    return image_paths, labels

def get_dataset_stats(labels, split_name="Dataset"):
    """Prints statistics about the dataset."""
    counts = Counter(labels)
    print(f"\n--- {split_name} Statistics ---")
    print(f"Total images: {len(labels)}")
    print("Images per class:")
    for cls in range(10):
        print(f"  Digit {cls}: {counts.get(cls, 0)}")

def prepare_data_splits(image_paths, labels, test_size=0.2, random_state=42):
    """
    Splits the data into an initial train (used for CV) and hold-out test set.
    """
    X_train_cv, X_test, y_train_cv, y_test = train_test_split(
        image_paths, labels, test_size=test_size, stratify=labels, random_state=random_state
    )
    
    get_dataset_stats(labels, "Full Dataset")
    get_dataset_stats(y_train_cv, "Train/CV Split (80%)")
    get_dataset_stats(y_test, "Hold-out Test Split (20%)")
    
    return X_train_cv, X_test, y_train_cv, y_test
