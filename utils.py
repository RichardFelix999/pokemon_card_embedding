import tensorflow as tf
import numpy as np
from PIL import Image, ImageOps
import cv2
import sys
from tqdm import tqdm
def smart_resize(image_path, target_size=(448, 448)):
    """Preserve aspect ratio with smart padding"""
    img = Image.open(image_path).convert('RGB')
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
    
    features = []
    for aug in augmentations:
        processed = tf.keras.applications.efficientnet.preprocess_input(aug)
        processed = np.expand_dims(processed, axis=0)
        feat = model.predict(processed, verbose=0)[0]
        features.append(feat)
    
    return np.mean(features, axis=0) / np.linalg.norm(np.mean(features, axis=0))

def verify_with_orb(query_path, db_path):
    """Feature matching verification"""
    img1 = cv2.imread(query_path, 0)
    img2 = cv2.imread(db_path, 0)
    
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

class ProgressLogger:
    def __init__(self, total, desc="Processing"):
        self.pbar = tqdm(
            total=total,
            desc=desc,
            unit="img",
            ncols=100,
            ascii=True,
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