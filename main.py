"""
Pokemon Card Recognition System - Main Script
"""

import os
import json
import numpy as np
import tensorflow as tf
import cv2
import time
import argparse
from tqdm import tqdm
import hnswlib
from utils import extract_features, verify_with_orb, print_colored, ProgressLogger, smart_resize, detect_and_crop_card

# Configuration
MODEL_NAME = "card_search_model"
DATABASE_DIR = "card_images"
QUERY_DIR = "input"
BATCH_SIZE = 16

# Configure GPU memory growth to avoid OOM errors
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print_colored(f"🚀 Found {len(gpus)} GPU(s). Memory growth enabled.", "green")
    except RuntimeError as e:
        print_colored(f"❌ GPU error: {e}", "red")
else:
    print_colored("⚠️ No GPU found. Running on CPU.", "yellow")

# Enable mixed precision for faster computation
try:
    tf.keras.mixed_precision.set_global_policy('mixed_float16')
    print_colored("⚡ Mixed precision enabled (float16)", "green")
except Exception as e:
    print_colored(f"⚠️ Mixed precision not available: {e}", "yellow")

def build_model():
    """Build and return the model for feature extraction"""
    # Use EfficientNetV2S for better performance
    base_model = tf.keras.applications.EfficientNetV2S(
        include_top=False,
        weights='imagenet',
        input_shape=(448, 448, 3),
        include_preprocessing=True
    )
    
    model = tf.keras.Sequential([
        base_model,
        tf.keras.layers.GlobalAveragePooling2D(),
        tf.keras.layers.Dense(1024, activation='swish'),
        tf.keras.layers.LayerNormalization(),
        tf.keras.layers.Dropout(0.1),
        tf.keras.layers.Dense(512, activation=None)
    ])
    
    # Compile the model for better performance
    model.compile(optimizer='adam', loss='mse')
    return model

def batch_extract_features(image_paths, model):
    """Extract features in batches for better GPU utilization"""
    all_features = []
    all_valid_paths = []
    
    # Preprocess images
    batch_images = []
    valid_paths = []
    
    for path in image_paths:
        try:
            img = smart_resize(path)
            img_array = tf.keras.preprocessing.image.img_to_array(img)
            batch_images.append(img_array)
            valid_paths.append(path)
        except Exception as e:
            print_colored(f"⚠️ Error preprocessing {path}: {str(e)}", "red")
    
    if not batch_images:
        return [], []
    
    # Process batch
    batch_images = np.array(batch_images)
    processed = tf.keras.applications.efficientnet.preprocess_input(batch_images)
    features = model.predict(processed, verbose=0)
    
    # Normalize features
    for i in range(features.shape[0]):
        features[i] = features[i] / np.linalg.norm(features[i])
    
    return features, valid_paths

def build_database():
    """Build the card database from images"""
    start_time = time.time()
    model = build_model()
    
    # Warm up the model
    dummy_input = np.zeros((1, 448, 448, 3), dtype=np.float32)
    _ = model.predict(dummy_input, verbose=0)
    
    database_features = []
    database_meta = []

    with open("test_data.json") as f:
        card_data = json.load(f)["results"]

    print_colored("\n⚙️ Starting database creation...", "blue")
    print_colored(f"📊 Total cards: {len(card_data)}", "yellow")
    
    # Collect all valid image paths first
    all_image_paths = []
    image_to_card_map = {}
    
    for i, card in enumerate(card_data):
        file_name = f"{card['productId']}.{card['imageUrl'].split('.')[-1]}"
        image_path = os.path.join(DATABASE_DIR, file_name)
        
        if os.path.exists(image_path):
            all_image_paths.append(image_path)
            image_to_card_map[image_path] = {
                "json_id": i,
                "product_id": card["productId"],
                "file_name": file_name
            }
        else:
            print_colored(f"❌ Missing {image_path}", "yellow")
    
    # Process in batches
    total_batches = (len(all_image_paths) + BATCH_SIZE - 1) // BATCH_SIZE
    progress = ProgressLogger(total_batches, "Processing Batches")
    
    for i in range(0, len(all_image_paths), BATCH_SIZE):
        batch_paths = all_image_paths[i:i+BATCH_SIZE]
        features, valid_paths = batch_extract_features(batch_paths, model)
        
        for j, (feature, path) in enumerate(zip(features, valid_paths)):
            database_features.append(feature)
            database_meta.append(image_to_card_map[path])
        
        progress.update()
    
    progress.close()
    
    elapsed = time.time() - start_time
    print_colored(f"\n✅ Successfully processed {len(database_features)} cards in {elapsed:.2f} seconds", "green")
    print_colored(f"⚡ Average time per card: {elapsed/len(database_features):.4f} seconds", "green")
    
    return np.array(database_features), database_meta

def save_artifacts(features, meta):
    """Save database artifacts to disk"""
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
        # Optimize parameters for better accuracy/speed
        index.init_index(max_elements=len(features), ef_construction=400, M=64)
        pbar.update()
        index.add_items(features, np.arange(len(features)))
        pbar.update()
    
    with ProgressLogger(1, "Saving Index") as pbar:
        index.save_index(f"{MODEL_NAME}.hnsw")
        pbar.update()
    
    print_colored("🎉 Database creation complete!", "green")

def load_artifacts():
    """Load database artifacts from disk"""
    features = np.load(f"{MODEL_NAME}_features.npy")
    with open(f"{MODEL_NAME}_meta.json") as f:
        meta = json.load(f)
    
    index = hnswlib.Index(space='l2', dim=features.shape[1])
    index.load_index(f"{MODEL_NAME}.hnsw")
    index.set_ef(300)  # Query-time accuracy parameter
    
    return index, features, meta

def search(query_path, index, features, meta, card_data):
    """Search for a card in the database"""
    model = build_model()
    
    print_colored(f"\n🔍 Processing query: {os.path.basename(query_path)}", "blue")
    
    # Detect and crop card
    cropped = detect_and_crop_card(query_path)
    if cropped is None:
        print_colored("❌ No card detected in the image", "red")
        return []
    
    # Save cropped image temporarily
    temp_path = os.path.join(QUERY_DIR, "temp_cropped.jpg")
    cv2.imwrite(temp_path, cropped)
    query_path_processed = temp_path
    
    # Feature extraction
    with ProgressLogger(1, "Extracting Features") as pbar:
        query_feature = extract_features(query_path_processed, model)
        pbar.update()
    
    # First-stage retrieval
    with ProgressLogger(1, "ANN Search") as pbar:
        indices, distances = index.knn_query(query_feature, k=50)
        pbar.update()
    
    # Second-stage verification
    results = []
    print_colored("🔬 Verifying matches...", "blue")
    
    for i, d in tqdm(zip(indices[0], distances[0]), total=len(indices[0]), desc="Verification"):
        if d > 0.2:
            continue
            
        db_path = os.path.join(DATABASE_DIR, meta[i]["file_name"])
        
        with ProgressLogger(1, "ORB Matching", leave=False) as pbar:
            orb_score = verify_with_orb(query_path_processed, db_path)
            pbar.update()
        
        if orb_score < 0.85:
            continue
            
        results.append({
            "product_id": meta[i]["product_id"],
            "distance": float(d),
            "orb_score": orb_score,
            "image_path": db_path,
            "card_data": card_data[meta[i]["json_id"]]
        })
    
    # Clean up temporary files
    if os.path.exists(temp_path):
        os.remove(temp_path)
    
    return sorted(results, key=lambda x: x['distance'])[:2]

def display_results(results):
    """Display search results"""
    for i, r in enumerate(results):
        print(f"\nMatch {i+1}:")
        print(f"Product ID: {r['product_id']}")
        print(f"L2 Distance: {r['distance']:.4f}")
        print(f"ORB Score: {r['orb_score']:.2%}")
        print(f"Card Name: {r['card_data'].get('name', 'N/A')}")
        
        img = cv2.imread(r['image_path'])
        if img is not None:
            cv2.imshow(f"Match {i+1}", cv2.resize(img, (600, 600)))
    
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def batch_search(query_dir, index, features, meta, card_data):
    """Search for multiple cards in batch"""
    model = build_model()
    
    # Get all query images
    query_files = [f for f in os.listdir(query_dir) if os.path.isfile(os.path.join(query_dir, f))]
    
    all_results = {}
    
    for query_file in query_files:
        query_path = os.path.join(query_dir, query_file)
        results = search(query_path, index, features, meta, card_data)
        
        if results:
            all_results[query_file] = results
            print_colored(f"✅ Found matches for {query_file}", "green")
        else:
            print_colored(f"❌ No matches found for {query_file}", "red")
    
    return all_results

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description="Pokemon Card Recognition System")
    subparsers = parser.add_subparsers(dest="command", help="Command to run")
    
    # Build database command
    build_parser = subparsers.add_parser("build", help="Build card database")
    build_parser.add_argument("--force", action="store_true", help="Force rebuild database")
    
    # Search command
    search_parser = subparsers.add_parser("search", help="Search for a card")
    search_parser.add_argument("--image", required=True, help="Path to query image")
    
    # Batch search command
    batch_parser = subparsers.add_parser("batch", help="Search for multiple cards")
    batch_parser.add_argument("--dir", default=QUERY_DIR, help="Directory with query images")
    
    args = parser.parse_args()
    
    if args.command == "build":
        # Build database
        print_colored("🚀 Building card database...", "blue")
        features, meta = build_database()
        save_artifacts(features, meta)
        print_colored("✅ Database built successfully!", "green")
    
    elif args.command == "search":
        # Load phase
        print_colored("🚀 Initializing search system...", "blue")
        
        with ProgressLogger(3, "Loading Artifacts") as pbar:
            index, features, meta = load_artifacts()
            pbar.update()
            with open("test_data.json") as f:
                card_data = json.load(f)["results"]
            pbar.update()
            pbar.update()
        
        # Search
        results = search(args.image, index, features, meta, card_data)
        
        # Display results
        if results:
            display_results(results)
        else:
            print_colored("❌ No matches found", "red")
    
    elif args.command == "batch":
        # Load phase
        print_colored("🚀 Initializing batch search system...", "blue")
        
        with ProgressLogger(3, "Loading Artifacts") as pbar:
            index, features, meta = load_artifacts()
            pbar.update()
            with open("test_data.json") as f:
                card_data = json.load(f)["results"]
            pbar.update()
            pbar.update()
        
        # Batch search
        all_results = batch_search(args.dir, index, features, meta, card_data)
        
        # Summary
        print_colored(f"\n📊 Batch Search Summary:", "blue")
        print_colored(f"Total images: {len(os.listdir(args.dir))}", "yellow")
        print_colored(f"Images with matches: {len(all_results)}", "green")
        print_colored(f"Images without matches: {len(os.listdir(args.dir)) - len(all_results)}", "red")
    
    else:
        parser.print_help()

if __name__ == "__main__":
    main()
