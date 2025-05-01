import os
import json
import numpy as np
import hnswlib
from utils import FeatureExtractor
from config import INDEX_CONFIG

class TCGSearch:
    def __init__(self):
        self.extractor = FeatureExtractor()
        self.index = hnswlib.Index(space=INDEX_CONFIG["space"], dim=INDEX_CONFIG["dim"])
        self.index.load_index("tcg_index.bin")
        self.index.set_ef(INDEX_CONFIG["ef_search"])
        self.features = np.load("features.npy")
        with open("meta.json") as f:
            self.meta = json.load(f)

    def search(self, query_img_path, query_text="", top_k=5):
        query_emb = self.extractor.hybrid_embedding([query_img_path], [query_text])[0]
        indices, distances = self.index.knn_query(query_emb, k=top_k*10)

        results = []
        for idx, dist in zip(indices[0], distances[0]):
            exact_dist = np.linalg.norm(query_emb - self.features[idx])
            results.append({
                "card_id": self.meta[idx]["card_id"],
                "distance": float(exact_dist),
                "metadata": self.meta[idx]
            })

        return sorted(results, key=lambda x: x["distance"])[:top_k]

if __name__ == "__main__":
    searcher = TCGSearch()

    # Example query
    query_image = "input/query.jpg"
    query_text = "rare holographic monster card"

    results = searcher.search(query_image, query_text, top_k=5)

    for i, res in enumerate(results, 1):
        print(f"Result {i}: Card ID {res['card_id']}, Distance: {res['distance']:.4f}")
        print(f"Text snippet: {res['metadata']['text']}\n")
