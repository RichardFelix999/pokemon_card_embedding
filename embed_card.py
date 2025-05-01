import tensorflow as tf
import numpy as np
import os
import json
from annoy import AnnoyIndex
from PIL import UnidentifiedImageError

# Configuration
DATABASE_DIR = "card_images"
CARD_DATA_PATH = "card_data.json"
MODEL_NAME = "card_search_model"

# Load card data
with open(CARD_DATA_PATH, "r") as f:
    card_data = json.load(f)["results"]

# Build feature extraction model
efficientnet_model = tf.keras.applications.EfficientNetB4(
    include_top=False, 
    weights='imagenet', 
    input_shape=(224, 224, 3)
)
model = tf.keras.Sequential([
    efficientnet_model,
    tf.keras.layers.GlobalAveragePooling2D(),
])

def extract_features(image_path):
    try:
        img = tf.keras.preprocessing.image.load_img(image_path, target_size=(224, 224))
        img = tf.keras.preprocessing.image.img_to_array(img)
        img = tf.keras.applications.efficientnet.preprocess_input(img)
        img = np.expand_dims(img, axis=0)
        features = model.predict(img)
        features /= np.linalg.norm(features, axis=-1, keepdims=True)
        return features
    except UnidentifiedImageError:
        print(f"Skipping corrupt image: {image_path}")
        return None

def build_database():
    database_features = []
    database_meta = []

    for i, card in enumerate(card_data):
        file_name = f"{card['productId']}.{card['imageUrl'].split('.')[-1]}"
        image_path = os.path.join(DATABASE_DIR, file_name)
        
        if os.path.exists(image_path):
            features = extract_features(image_path)
            if features is not None:
                database_features.append(features.flatten())
                database_meta.append({
                    "json_id": i,
                    "product_id": card["productId"],
                    "file_name": file_name
                })

    return np.array(database_features), database_meta

def save_artifacts(features, meta):
    # Save features
    np.save(f"{MODEL_NAME}_features.npy", features)
    
    # Save metadata
    with open(f"{MODEL_NAME}_meta.json", "w") as f:
        json.dump(meta, f)
    
    # Build and save Annoy index
    index = AnnoyIndex(features.shape[1], 'angular')
    for i, vec in enumerate(features):
        index.add_item(i, vec)
    index.build(400)  # Using 400 trees for better accuracy
    index.save(f"{MODEL_NAME}.ann")

if __name__ == "__main__":
    features, meta = build_database()
    save_artifacts(features, meta)
    print(f"Database built with {len(features)} valid images")
