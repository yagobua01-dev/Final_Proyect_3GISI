import torch
import numpy as np
import faiss
import json
import os
from transformers import CLIPProcessor, CLIPModel
from PIL import Image

# 1. Configuration and Paths
EMBEDDINGS_DIR = "embeddings/"
EMBEDDINGS_FILE = os.path.join(EMBEDDINGS_DIR, "image_embeddings.npy")
PATHS_FILE = os.path.join(EMBEDDINGS_DIR, "image_paths.json")

# 2. Load the pre-computed database
print("Loading image embeddings from disk...")
image_embeddings = np.load(EMBEDDINGS_FILE)

with open(PATHS_FILE, "r") as f:
    image_paths = json.load(f)

print(f"Loaded {image_embeddings.shape[0]} embeddings.")

# 3. Initialize FAISS Index (Advanced Machine Learning: Vector Search)
dimension = image_embeddings.shape[1]  # 512
index = faiss.IndexFlatIP(dimension)
index.add(image_embeddings)
print("FAISS index built successfully!")

# 4. Load CLIP Model
print("Loading CLIP multimodal processor...")
model_id = "openai/clip-vit-base-patch32"
model = CLIPModel.from_pretrained(model_id)
processor = CLIPProcessor.from_pretrained(model_id)

def search_images(query_text, top_k=5):
    """
    Converts text to an embedding and searches the FAISS index.
    """
    # FIX: We create a blank dummy image to satisfy the multimodal model's requirement
    dummy_image = Image.new('RGB', (224, 224), (0, 0, 0))
    
    # Process both text and dummy image
    inputs = processor(text=[query_text], images=dummy_image, return_tensors="pt", padding=True)
    
    with torch.no_grad():
        # Pass everything through the main model
        outputs = model(**inputs)
        
        # Explicitly extract the text embeddings
        text_features = outputs.text_embeds
        
        # Normalize the vector (crucial for cosine similarity)
        text_features = text_features / text_features.norm(p=2, dim=-1, keepdim=True)
        
    # Convert to numpy array for FAISS
    text_vector = text_features.cpu().numpy().astype(np.float32)
    
    # Perform the search. 'distances' contains the similarity scores, 
    # 'indices' contains the positions of the matching images in our array.
    distances, indices = index.search(text_vector, top_k)
    
    print(f"\n--- Top {top_k} Results for: '{query_text}' ---")
    results = []
    for i in range(top_k):
        idx = indices[0][i]
        score = distances[0][i]
        path = image_paths[idx]
        results.append((path, score))
        print(f"{i+1}. Score: {score:.4f} | Image: {path}")
        
    return results

# 5. Interactive Terminal Loop
if __name__ == "__main__":
    print("\n" + "="*50)
    print("MULTIMODAL SEARCH ENGINE READY")
    print("Type 'exit' or 'quit' to stop.")
    print("="*50)
    
    while True:
        user_query = input("\nEnter your search query (English): ")
        if user_query.lower() in ['exit', 'quit']:
            break
            
        if user_query.strip() == "":
            continue
            
        search_images(user_query, top_k=5)