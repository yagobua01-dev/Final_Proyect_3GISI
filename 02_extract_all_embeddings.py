import torch
from transformers import CLIPProcessor, CLIPModel
from PIL import Image
import os
import numpy as np
import json
from tqdm import tqdm

# 1. Configuration and Paths
IMAGE_DIR = "data/Images/"
EMBEDDINGS_DIR = "embeddings/"
os.makedirs(EMBEDDINGS_DIR, exist_ok=True)

print("Loading CLIP model and processor...")
model_id = "openai/clip-vit-base-patch32"
model = CLIPModel.from_pretrained(model_id)
processor = CLIPProcessor.from_pretrained(model_id)
print("Model loaded successfully!")

# 2. Gather all image files
# We scan the directory to find all .jpg files
image_files = [f for f in os.listdir(IMAGE_DIR) if f.endswith('.jpg')]
image_paths = [os.path.join(IMAGE_DIR, f) for f in image_files]

print(f"Found {len(image_paths)} images. Starting extraction process...")

all_embeddings = []
valid_image_paths = []

# 3. Iterate over the dataset with a progress bar
for img_path in tqdm(image_paths, desc="Processing Images"):
    try:
        # Open and ensure image is in RGB format to avoid channel errors
        image = Image.open(img_path).convert("RGB")
        
        # Prepare inputs
        inputs = processor(text=[""], images=image, return_tensors="pt", padding=True)
        
        with torch.no_grad():
            outputs = model(**inputs)
            # Extract the 512-dimensional vector
            embedding = outputs.image_embeds
            
            # L2 Normalization
            # We normalize the vector so its length is 1. This allows us to use 
            # simple dot product (Inner Product) to calculate Cosine Similarity later.
            embedding = embedding / embedding.norm(p=2, dim=-1, keepdim=True)
            
        # Convert the PyTorch tensor to a standard NumPy array
        # We use .squeeze() to turn [1, 512] into just [512]
        numpy_embedding = embedding.squeeze().cpu().numpy()
        
        all_embeddings.append(numpy_embedding)
        valid_image_paths.append(img_path)
        
    except Exception as e:
        # If any image is corrupted, we catch the error and continue
        print(f"\nError processing {img_path}: {e}")

# 4. Save the results to disk
print("\nExtraction complete. Saving embeddings to disk...")

# Stack all individual [512] arrays into a massive [N, 512] matrix
embedding_matrix = np.array(all_embeddings, dtype=np.float32)

# Save the numerical vectors
np.save(os.path.join(EMBEDDINGS_DIR, "image_embeddings.npy"), embedding_matrix)

# Save the exact file paths matching each vector index using JSON
with open(os.path.join(EMBEDDINGS_DIR, "image_paths.json"), "w") as f:
    json.dump(valid_image_paths, f)

print(f"Success! Saved {embedding_matrix.shape[0]} embeddings.")
print(f"Matrix shape: {embedding_matrix.shape}")
print("Data saved in the 'embeddings/' folder.")