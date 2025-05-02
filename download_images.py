import os
import json
import requests
from tqdm import tqdm

os.makedirs("card_images", exist_ok=True)

with open("test_data.json", "r", encoding="utf-8") as f:
    data = json.load(f)["results"]

for card in tqdm(data, desc="Downloading Images"):
    image_url = card.get("imageUrl")
    if not image_url:
        continue

    product_id = str(card["productId"])
    filename = f"card_images/{product_id}.jpg"

    if os.path.exists(filename):
        continue

    try:
        response = requests.get(image_url, timeout=10)
        if response.status_code == 200:
            with open(filename, "wb") as f:
                f.write(response.content)
    except Exception as e:
        print(f"Failed downloading {product_id}: {e}")
