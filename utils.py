"""
Utility functions for Pokemon Card Recognition System
"""

import tensorflow as tf
import numpy as np
from PIL import Image, ImageOps
import cv2
from tqdm import tqdm
import os

def smart_resize(image_path, target_size=(448, 448)):
    """Preserve aspect ratio with smart padding"""
    img = Image.open(image_path).convert('RGB')
    
    # Handle size difference between database and query images
    # Database images are typically 200x280, while query images can be much larger
    
    ratio = min(target_size[0]/img.width, target_size[1]/img.height)
    new_size = (int(img.width*ratio), int(img.height*ratio))
    img = img.resize(new_size, Image.LANCZOS)
    
    delta_w = target_size[0] - img.width
    delta_h = target_size[1] - img.height
    padding = (delta_w//2, delta_h//2, delta_w-(delta_w//2), delta_h-(delta_h//2))
    return ImageOps.expand(img, padding, fill='white')

def extract_features(image_path, model):
    """Enhanced feature extraction with TTA"""
    img = smart_resize(image_path)
    img = tf.keras.preprocessing.image.img_to_array(img)
    
    # Test-Time Augmentation
    augmentations = [
        img,
        np.rot90(img, k=1),
        np.rot90(img, k=3),
        img[:, ::-1]  # Horizontal flip
    ]
    
    # Process all augmentations in a single batch for better GPU utilization
    batch = np.array(augmentations)
    processed = tf.keras.applications.efficientnet.preprocess_input(batch)
    feats = model.predict(processed, verbose=0)
    
    # Average the features
    avg_feat = np.mean(feats, axis=0)
    # Normalize
    return avg_feat / np.linalg.norm(avg_feat)

def verify_with_orb(query_path, db_path):
    """Feature matching verification"""
    img1 = cv2.imread(query_path, 0)
    img2 = cv2.imread(db_path, 0)
    
    if img1 is None or img2 is None:
        return 0
    
    # Resize query image to match database image size
    # This is crucial for handling size differences
    h2, w2 = img2.shape
    img1 = cv2.resize(img1, (w2, h2))
    
    orb = cv2.ORB_create(5000)
    kp1, des1 = orb.detectAndCompute(img1, None)
    kp2, des2 = orb.detectAndCompute(img2, None)
    
    if des1 is None or des2 is None:
        return 0
        
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(des1, des2)
    
    if len(matches) > 20:
        return len(matches)/min(len(des1), len(des2))
    return 0

def detect_and_crop_card(image_path):
    """Detect and crop card from image"""
    image = cv2.imread(image_path)
    if image is None:
        return None
    
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Apply Gaussian blur
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # Apply Canny edge detection
    edges = cv2.Canny(blurred, 50, 150)
    
    # Find contours
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        return None
    
    # Find the largest contour by area
    largest_contour = max(contours, key=cv2.contourArea)
    
    # Get bounding rectangle
    x, y, w, h = cv2.boundingRect(largest_contour)
    
    # Check if the contour is large enough to be a card
    if w < 100 or h < 100:
        return None
    
    # Check aspect ratio (Pokemon cards have ~2.5:3.5 ratio)
    aspect_ratio = h / w
    if not (0.6 < aspect_ratio < 0.85):  # Allow some tolerance
        return None
    
    # Crop the image
    cropped = image[y:y+h, x:x+w]
    
    return cropped

class ProgressLogger:
    def __init__(self, total, desc="Processing", leave=True):
        self.pbar = tqdm(
            total=total,
            desc=desc,
            unit="img",
            ncols=100,
            ascii=True,
            leave=leave,
            bar_format="{desc}: {percentage:3.0f}% |{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]"
        )
        
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_value, traceback):
        self.close()
        # Return False to propagate exceptions if any
        return False

    def update(self):
        self.pbar.update(1)
        
    def close(self):
        self.pbar.close()
        
def print_colored(message, color="blue"):
    colors = {
        "red": "\033[91m",
        "green": "\033[92m",
        "yellow": "\033[93m",
        "blue": "\033[94m",
        "end": "\033[0m"
    }
    print(f"{colors.get(color, colors['blue'])}{message}{colors['end']}")
