import numpy as np
import json
import os
import cv2
from annoy import AnnoyIndex
import tensorflow as tf

# Configuration
MODEL_NAME = "card_search_model"
DATABASE_DIR = "card_images"
QUERY_DIR = "input"

# Load feature extraction model
efficientnet_model = tf.keras.applications.EfficientNetB4(
    include_top=False, 
    weights='imagenet', 
    input_shape=(224, 224, 3)
)
model = tf.keras.Sequential([
    efficientnet_model,
    tf.keras.layers.GlobalAveragePooling2D(),
])

def load_artifacts():
    features = np.load(f"{MODEL_NAME}_features.npy")
    with open(f"{MODEL_NAME}_meta.json", "r") as f:
        meta = json.load(f)
    
    index = AnnoyIndex(features.shape[1], 'angular')
    index.load(f"{MODEL_NAME}.ann")
    
    return index, features, meta

def extract_features(image_path):
    img = tf.keras.preprocessing.image.load_img(image_path, target_size=(224, 224))
    img = tf.keras.preprocessing.image.img_to_array(img)
    img = tf.keras.applications.efficientnet.preprocess_input(img)
    img = np.expand_dims(img, axis=0)
    features = model.predict(img)
    features /= np.linalg.norm(features, axis=-1, keepdims=True)
    return features.flatten()

def search_image(query_path, index, features, meta, card_data, top_n=5):
    query_feature = extract_features(query_path)
    indices = index.get_nns_by_vector(query_feature, top_n, search_k=40000)
    
    results = []
    for idx in indices:
        card_info = card_data[meta[idx]["json_id"]]
        distance = np.linalg.norm(query_feature - features[idx])
        results.append({
            "product_id": meta[idx]["product_id"],
            "distance": float(distance),
            "image_path": os.path.join(DATABASE_DIR, meta[idx]["file_name"]),
            "card_data": card_info
        })
    
    return results

def display_results(results):
    for i, result in enumerate(results):
        print(f"\nResult {i+1}:")
        print(f"Product ID: {result['product_id']}")
        print(f"Distance: {result['distance']:.4f}")
        print(f"Card Name: {result['card_data'].get('name', 'N/A')}")
        
        img = cv2.imread(result["image_path"])
        if img is not None:
            cv2.imshow(f"Result {i+1}", img)
    
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    # Load pre-built artifacts
    index, features, meta = load_artifacts()
    
    # Load card data
    with open("card_data.json", "r") as f:
        card_data = json.load(f)["results"]

    # Example search
    query_path = os.path.join(QUERY_DIR, "pok1.png")
    if os.path.exists(query_path):
        results = search_image(query_path, index, features, meta, card_data)
        display_results(results)
    else:
        print(f"Query image not found: {query_path}")
