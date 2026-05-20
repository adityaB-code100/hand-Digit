import os
import glob
from PIL import Image
import numpy as np

def inspect_image_details(img_path):
    print(f"Inspecting: {img_path}")
    img = Image.open(img_path)
    arr = np.array(img)
    print(f"Shape: {arr.shape}")
    unique_vals = np.unique(arr)
    print(f"Unique values: {unique_vals[:20]} ... (total {len(unique_vals)})")
    
    # Check background: let's look at the corner pixel
    print(f"Corner pixel (0,0): {arr[0, 0]}")
    # Check center pixel
    h, w = arr.shape[:2]
    print(f"Center pixel: {arr[h//2, w//2]}")
    print("-" * 40)

# Find first image in handDigitDataset
h_files = glob.glob("../handDigitDataset/digit_0/*.png") + glob.glob("../handDigitDataset/digit_0/*.jpg")
if h_files:
    inspect_image_details(h_files[0])

# Find first image in Dataset
d_files = glob.glob("../Dataset/character_0/*.png") + glob.glob("../Dataset/character_0/*.jpg")
if d_files:
    inspect_image_details(d_files[0])
