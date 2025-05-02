import os
import json
import numpy as np
import hnswlib
from utils import *
import tensorflow as tf

# Configuration
DATABASE_DIR = "card_images"
CARD_DATA_PATH = "test_data.json"
MODEL_NAME = "card_search_model_v2"

# Enhanced Model Architecture
def build_model():
    base_model = tf.keras.applications.EfficientNetV2L(
        include_top=False,
        weights='imagenet',
        input_shape=(448, 448, 3),
        include_preprocessing=True
    )
    
    model = tf.keras.Sequential([
        base_model,
        tf.keras.layers.GlobalAveragePooling2D(),
        tf.keras.layers.Dense(2048, activation='swish'),
        tf.keras.layers.LayerNormalization(),
        tf.keras.layers.Dropout(0.1),
        tf.keras.layers.Dense(1024, activation=None)
    ])
    return model

def build_database():
    model = build_model()
    database_features = []
    database_meta = []

    with open(CARD_DATA_PATH) as f:
        card_data = json.load(f)["results"]

    print_colored("\n⚙️ Starting database creation...", "blue")
    print_colored(f"📊 Total cards: {len(card_data)}", "yellow")
    
    # Initialize progress logger
    valid_count = 0
    total = len(card_data)
    progress = ProgressLogger(total, "Embedding Cards")
    
    for i, card in enumerate(card_data):
        file_name = f"{card['productId']}.{card['imageUrl'].split('.')[-1]}"
        image_path = os.path.join(DATABASE_DIR, file_name)
        
        if os.path.exists(image_path):
            try:
                features = extract_features(image_path, model)
                database_features.append(features)
                database_meta.append({
                    "json_id": i,
                    "product_id": card["productId"],
                    "file_name": file_name
                })
                valid_count += 1
            except Exception as e:
                print_colored(f"⚠️ Error {image_path}: {str(e)}", "red")
        else:
            print_colored(f"❌ Missing {image_path}", "yellow")
            
        progress.update()
    
    progress.close()
    
    print_colored(f"\n✅ Successfully processed {valid_count}/{total} cards", "green")
    return np.array(database_features), database_meta

def save_artifacts(features, meta):
    print_colored("\n💾 Saving database artifacts...", "blue")
    
    # Save features
    with ProgressLogger(1, "Saving Features") as pbar:
        np.save(f"{MODEL_NAME}_features.npy", features)
        pbar.update()
    
    # Save metadata
    with ProgressLogger(1, "Saving Metadata") as pbar:
        with open(f"{MODEL_NAME}_meta.json", "w") as f:
            json.dump(meta, f, indent=2)
        pbar.update()
    
    # Build HNSW index
    print_colored("🔨 Building search index...", "blue")
    dim = features.shape[1]
    index = hnswlib.Index(space='l2', dim=dim)
    
    with ProgressLogger(2, "Index Construction") as pbar:
        index.init_index(max_elements=len(features), ef_construction=400, M=64)
        pbar.update()
        index.add_items(features, np.arange(len(features)))
        pbar.update()
    
    with ProgressLogger(1, "Saving Index") as pbar:
        index.save_index(f"{MODEL_NAME}.hnsw")
        pbar.update()
    
    print_colored("🎉 Database creation complete!", "green")

if __name__ == "__main__":
    features, meta = build_database()
    save_artifacts(features, meta)
    print(f"Database built with {len(features)} valid images")
