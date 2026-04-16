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

