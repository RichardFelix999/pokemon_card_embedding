import os
import json
import numpy as np
import hnswlib
from utils import extract_features, verify_with_orb
import tensorflow as tf
import cv2
from embed import build_model

# Configuration
MODEL_NAME = "card_search_model_v2"
DATABASE_DIR = "card_images"
QUERY_DIR = "input"

def load_artifacts():
    features = np.load(f"{MODEL_NAME}_features.npy")
    with open(f"{MODEL_NAME}_meta.json") as f:
        meta = json.load(f)
    
    index = hnswlib.Index(space='l2', dim=features.shape[1])
    index.load_index(f"{MODEL_NAME}.hnsw")
    index.set_ef(300)  # Query-time accuracy parameter
    
    return index, features, meta

def search(query_path, index, features, meta, card_data):
    model = build_model()  # Same as embed.py
    query_feature = extract_features(query_path, model)
    
    # First-stage retrieval
    indices, distances = index.knn_query(query_feature, k=50)
    
    # Second-stage verification
    results = []
    for i, d in zip(indices[0], distances[0]):
        if d > 0.2:  # L2 distance threshold
            continue
            
        # ORB verification
        db_path = os.path.join(DATABASE_DIR, meta[i]["file_name"])
        orb_score = verify_with_orb(query_path, db_path)
        
        if orb_score < 0.85:
            continue
            
        results.append({
            "product_id": meta[i]["product_id"],
            "distance": float(d),
            "orb_score": orb_score,
            "image_path": db_path,
            "card_data": card_data[meta[i]["json_id"]]
        })
    
    return sorted(results, key=lambda x: x['distance'])[:2]

def display_results(results):
    for i, r in enumerate(results):
        print(f"\nMatch {i+1}:")
        print(f"Product ID: {r['product_id']}")
        print(f"L2 Distance: {r['distance']:.4f}")
        print(f"ORB Score: {r['orb_score']:.2%}")
        print(f"Card Name: {r['card_data'].get('name', 'N/A')}")
        
        img = cv2.imread(r['image_path'])
        cv2.imwrite(f"match_{i+1}_{r['product_id']}.jpg", img)  # Save instead of show
        print(f"Saved match image to: match_{i+1}_{r['product_id']}.jpg")

if __name__ == "__main__":
    index, features, meta = load_artifacts()
    with open("card_data.json") as f:
        card_data = json.load(f)["results"]
    
    for query_name in os.listdir(QUERY_DIR):
        query_path = os.path.join(QUERY_DIR, query_name)
        if os.path.isfile(query_path):
            print(f"\nSearching: {query_name}")
            results = search(query_path, index, features, meta, card_data)
            display_results(results)
