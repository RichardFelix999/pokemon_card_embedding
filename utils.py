import torch
import numpy as np
from sentence_transformers import SentenceTransformer
from transformers import CLIPModel, CLIPProcessor
from PIL import Image
from config import MODEL_CONFIG, GPU_CONFIG

class FeatureExtractor:
    def __init__(self):
        self.device = GPU_CONFIG["device"]
        self.mixed_precision = GPU_CONFIG["mixed_precision"]

        self.text_model = SentenceTransformer(
            MODEL_CONFIG["text_model"],
            device=self.device,
            compute_dtype=torch.float16 if self.mixed_precision else torch.float32
        )
        self.clip_model = CLIPModel.from_pretrained(
            MODEL_CONFIG["image_model"]
        ).to(self.device)
        self.clip_processor = CLIPProcessor.from_pretrained(
            MODEL_CONFIG["image_model"]
        )

    def get_text_embedding(self, texts):
        # texts: list of strings or single string
        with torch.no_grad():
            embeddings = self.text_model.encode(
                texts,
                batch_size=GPU_CONFIG["batch_size"],
                convert_to_tensor=True,
                device=self.device,
                show_progress_bar=True
            )
        return embeddings

    def get_image_embedding(self, image_paths):
        all_embs = []
        with torch.no_grad():
            for i in range(0, len(image_paths), GPU_CONFIG["batch_size"]):
                batch_paths = image_paths[i:i + GPU_CONFIG["batch_size"]]
                images = [Image.open(p).convert("RGB") for p in batch_paths]
                inputs = self.clip_processor(images=images, return_tensors="pt", padding=True)
                inputs = {k: v.to(self.device) for k, v in inputs.items()}

                if self.mixed_precision:
                    with torch.cuda.amp.autocast():
                        outputs = self.clip_model.get_image_features(**inputs)
                else:
                    outputs = self.clip_model.get_image_features(**inputs)

                all_embs.append(outputs)
            all_embs = torch.cat(all_embs, dim=0)
        return all_embs

    def hybrid_embedding(self, image_paths, texts):
        # image_paths: list of image paths
        # texts: list of corresponding text strings
        text_embs = self.get_text_embedding(texts)
        image_embs = self.get_image_embedding(image_paths)

        # Weighted sum and normalize
        combined = text_embs * MODEL_CONFIG["hybrid_ratio"] + image_embs * (1 - MODEL_CONFIG["hybrid_ratio"])
        combined = combined / combined.norm(dim=1, keepdim=True)
        return combined.cpu().numpy()
