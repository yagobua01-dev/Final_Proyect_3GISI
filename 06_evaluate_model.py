import torch
import numpy as np
import faiss
import json
import os
import random
from transformers import CLIPProcessor, CLIPModel
from PIL import Image
from tqdm import tqdm
import warnings

# Suppress warnings
warnings.filterwarnings("ignore")

# 1. Configuration and Paths
EMBEDDINGS_DIR = "embeddings/"
EMBEDDINGS_FILE = os.path.join(EMBEDDINGS_DIR, "image_embeddings.npy")
PATHS_FILE = os.path.join(EMBEDDINGS_DIR, "image_paths.json")
CAPTIONS_FILE = "data/captions.txt" # Make sure this file exists!

SAMPLE_SIZE = 100  # Number of random images to test
TOP_K = 5          # We will calculate Recall@5

print("Loading evaluation system...")

# 2. Load FAISS Database
image_embeddings = np.load(EMBEDDINGS_FILE)
with open(PATHS_FILE, "r") as f:
    image_paths = json.load(f)

dimension = image_embeddings.shape[1]
index = faiss.IndexFlatIP(dimension)
index.add(image_embeddings)

# 3. Load CLIP Model
model_id = "openai/clip-vit-base-patch32"
model = CLIPModel.from_pretrained(model_id)
processor = CLIPProcessor.from_pretrained(model_id)

def get_text_embedding(query_text):
    dummy_image = Image.new('RGB', (224, 224), (0, 0, 0))
    inputs = processor(text=[query_text], images=dummy_image, return_tensors="pt", padding=True)
    with torch.no_grad():
        text_features = model(**inputs).text_embeds
        text_features = text_features / text_features.norm(p=2, dim=-1, keepdim=True)
    return text_features.cpu().numpy().astype(np.float32)

# 4. Load and Parse Captions
print(f"Reading ground truth from {CAPTIONS_FILE}...")
image_to_captions = {}

try:
    with open(CAPTIONS_FILE, "r", encoding="utf-8") as f:
        lines = f.readlines()
        # Skip header if it exists (usually "image,caption")
        if "image,caption" in lines[0].lower() or "image" in lines[0].lower():
            lines = lines[1:]
            
        for line in lines:
            parts = line.strip().split(",", 1)
            if len(parts) == 2:
                img_name, caption = parts
                # Ensure the dictionary has a list for each image
                if img_name not in image_to_captions:
                    image_to_captions[img_name] = []
                image_to_captions[img_name].append(caption)
except FileNotFoundError:
    print(f"ERROR: Could not find {CAPTIONS_FILE}. Please make sure you downloaded the captions text file from Kaggle.")
    exit()

# 5. Prepare Test Set
# We only want to test images that we actually successfully processed in Phase 5
processed_filenames = [os.path.basename(path) for path in image_paths]
valid_test_images = [img for img in image_to_captions.keys() if img in processed_filenames]

print(f"Found {len(valid_test_images)} valid images with captions.")

# Pick a random sample to evaluate
random.seed(42) # Seed for reproducibility
test_sample = random.sample(valid_test_images, min(SAMPLE_SIZE, len(valid_test_images)))

# 6. Evaluation Loop
print(f"\n--- Starting Evaluation: Recall@{TOP_K} on {len(test_sample)} random samples ---")
hits = 0

for target_image in tqdm(test_sample, desc="Evaluating"):
    # Pick one random human caption for this image
    caption = random.choice(image_to_captions[target_image])
    
    # Process the text query
    query_vector = get_text_embedding(caption)
    
    # Search in FAISS
    distances, indices = index.search(query_vector, TOP_K)
    
    # Check if the target_image is in the Top K results
    target_found = False
    for idx in indices[0]:
        retrieved_path = image_paths[idx]
        retrieved_filename = os.path.basename(retrieved_path)
        
        if retrieved_filename == target_image:
            target_found = True
            break
            
    if target_found:
        hits += 1

# 7. Final Results
recall_score = (hits / len(test_sample)) * 100
print("\n" + "="*50)
print("FINAL EVALUATION RESULTS")
print("="*50)
print(f"Total Queries Evaluated: {len(test_sample)}")
print(f"Successful Hits (Target image in Top {TOP_K}): {hits}")
print(f"Recall@{TOP_K}: {recall_score:.2f}%")
print("="*50)