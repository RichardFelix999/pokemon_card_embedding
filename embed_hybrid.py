import os
import json
import numpy as np
import hnswlib
from config import *
from utils import FeatureExtractor

def build_database():
    extractor = FeatureExtractor()

    with open("card_data.json") as f:
        cards = json.load(f)["results"]

    image_paths = []
    texts = []
    meta = []

    for card in cards:
        img_path = os.path.join("card_images", f"{card['productId']}.jpg")
        if not os.path.exists(img_path):
            continue
        card_text = f"{card.get('name','')} {card.get('type','')} {card.get('effect_text','')}"
        image_paths.append(img_path)
        texts.append(card_text)
        meta.append({
            "card_id": card["productId"],
            "text": card_text[:500],
            "img_path": img_path
        })

    features = extractor.hybrid_embedding(image_paths, texts).astype(np.float32)

    # Save features and metadata
    np.save("features.npy", features)
    with open("meta.json", "w") as f:
        json.dump(meta, f)

    # Build HNSW index
    index = hnswlib.Index(space=INDEX_CONFIG["space"], dim=INDEX_CONFIG["dim"])
    index.init_index(max_elements=len(features)*2, M=INDEX_CONFIG["M"], ef_construction=INDEX_CONFIG["ef_construction"])
    index.add_items(features)
    index.save_index("tcg_index.bin")

if __name__ == "__main__":
    build_database()
