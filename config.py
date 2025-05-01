MODEL_CONFIG = {
    "text_model": "NovaSearch/stella_en_1.5B_v5",
    "image_model": "openai/clip-vit-base-patch32",
    "hybrid_ratio": 0.6  # Weight for text vs image features
}

INDEX_CONFIG = {
    "space": "l2",
    "dim": 1024,  # Stella base dimension (CLIP ViT-B/32 image dim is 512, combined weighted sum fits 1024)
    "M": 64,
    "ef_construction": 400,
    "ef_search": 512
}

GPU_CONFIG = {
    "device": "cuda:0",
    "batch_size": 128,
    "mixed_precision": True,  # Enable FP16 mixed precision
    "num_workers": 4
}
