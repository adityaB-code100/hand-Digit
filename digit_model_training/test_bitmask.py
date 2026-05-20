import os
import cv2
import numpy as np
from PIL import Image, ImageOps

def apply_bitmask(img_path):
    # Load image in grayscale
    img_gray = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if img_gray is None:
        print(f"Failed to load {img_path}")
        return None
    
    # Apply Otsu's threshold with binary inversion
    # This assumes dark digit on light background, which turns it into white digit on black background
    _, thresh = cv2.threshold(img_gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    
    # Find bounding box of the white digit
    pts = np.argwhere(thresh > 0)
    if len(pts) > 0:
        y_min, x_min = pts.min(axis=0)
        y_max, x_max = pts.max(axis=0)
        # Crop the thresholded image
        cropped = thresh[y_min:y_max+1, x_min:x_max+1]
        
        # Pad cropped image to keep aspect ratio and avoid touching borders
        # Convert to PIL for easy padding or do it in numpy
        pil_img = Image.fromarray(cropped)
        padding = int(max(pil_img.size) * 0.1) # 10% padding
        pil_img = ImageOps.expand(pil_img, border=padding, fill=0)
    else:
        pil_img = Image.fromarray(thresh)
        
    return pil_img

import glob
# Test on handDigitDataset
h_files = glob.glob("../handDigitDataset/digit_0/*.png") + glob.glob("../handDigitDataset/digit_0/*.jpg")
if h_files:
    pil_h = apply_bitmask(h_files[0])
    if pil_h:
        print("Hand digit bitmask shape:", pil_h.size)
        arr = np.array(pil_h)
        print(f"Hand digit bitmask unique values: {np.unique(arr)}")

# Test on Dataset
d_files = glob.glob("../Dataset/character_0/*.png") + glob.glob("../Dataset/character_0/*.jpg")
if d_files:
    pil_d = apply_bitmask(d_files[0])
    if pil_d:
        print("Dataset digit bitmask shape:", pil_d.size)
        arr = np.array(pil_d)
        print(f"Dataset digit bitmask unique values: {np.unique(arr)}")
