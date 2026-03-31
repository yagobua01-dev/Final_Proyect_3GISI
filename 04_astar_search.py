import torch
import numpy as np
import faiss
import json
import os
import heapq
from transformers import CLIPProcessor, CLIPModel
from PIL import Image

# 1. Configuration and Paths
EMBEDDINGS_DIR = "embeddings/"
EMBEDDINGS_FILE = os.path.join(EMBEDDINGS_DIR, "image_embeddings.npy")
PATHS_FILE = os.path.join(EMBEDDINGS_DIR, "image_paths.json")

print("Loading database and models...")
image_embeddings = np.load(EMBEDDINGS_FILE)
with open(PATHS_FILE, "r") as f:
    image_paths = json.load(f)

dimension = image_embeddings.shape[1]
index = faiss.IndexFlatIP(dimension)
index.add(image_embeddings)

model_id = "openai/clip-vit-base-patch32"
model = CLIPModel.from_pretrained(model_id)
processor = CLIPProcessor.from_pretrained(model_id)
print("System ready!")

def get_text_embedding(query_text):
    """Generates the text embedding using CLIP, handling the multimodal requirement."""
    dummy_image = Image.new('RGB', (224, 224), (0, 0, 0))
    inputs = processor(text=[query_text], images=dummy_image, return_tensors="pt", padding=True)
    with torch.no_grad():
        text_features = model(**inputs).text_embeds
        text_features = text_features / text_features.norm(p=2, dim=-1, keepdim=True)
    return text_features.cpu().numpy().astype(np.float32)

# A* SEARCH ALGORITHM

def calculate_diversity_penalty(new_img_idx, current_state_indices, candidate_embeddings):
    """Calculates how similar the new image is to the already selected ones."""
    if not current_state_indices:
        return 0.0
    
    new_emb = candidate_embeddings[new_img_idx]
    penalties = []
    
    for idx in current_state_indices:
        existing_emb = candidate_embeddings[idx]
        sim = np.dot(new_emb, existing_emb)
        penalties.append(sim)
        
    return np.mean(penalties) 

def a_star_selection(query_embedding, top_n_indices, top_n_scores, candidate_embeddings, k=5, w1=0.7, w2=0.3):
    """
    Selects k images using A* search.
    Optimized to generate combinations instead of permutations to prevent combinatorial explosion.
    """
    open_list = []
    best_possible_relevance_cost = 1.0 - np.max(top_n_scores)
    initial_state = ()
    heapq.heappush(open_list, (0.0, 0.0, initial_state))
    
    while open_list:
        f_cost, g_cost, current_state = heapq.heappop(open_list)
        
        # Goal check
        if len(current_state) == k:
            return current_state
            
        # Successor generation constraint: strictly increasing indices to form combinations
        start_idx = current_state[-1] + 1 if current_state else 0
        
        for i in range(start_idx, len(top_n_indices)):
            new_state = current_state + (i,)
            
            relevance_cost = 1.0 - top_n_scores[i]
            diversity_cost = calculate_diversity_penalty(i, current_state, candidate_embeddings)
            
            step_cost = (w1 * relevance_cost) + (w2 * diversity_cost)
            new_g_cost = g_cost + step_cost
            
            remaining_steps = k - len(new_state)
            h_cost = remaining_steps * (w1 * best_possible_relevance_cost) 
            
            new_f_cost = new_g_cost + h_cost
            
            heapq.heappush(open_list, (new_f_cost, new_g_cost, new_state))
            
    return None

def search_with_astar(query_text, pool_size=20, final_k=5):
    """Full pipeline: Text embedding -> FAISS initial retrieval -> A* final selection."""
    query_vector = get_text_embedding(query_text)
    
    # 1. FAISS retrieval
    distances, indices = index.search(query_vector, pool_size)
    top_n_indices = indices[0]
    top_n_scores = distances[0]
    
    candidate_embeddings = image_embeddings[top_n_indices]
    
    # 2. A* Search
    best_subset_local_indices = a_star_selection(
        query_vector, top_n_indices, top_n_scores, candidate_embeddings, k=final_k
    )
    
    print(f"\n--- A* Optimized Results for: '{query_text}' ---")
    if best_subset_local_indices:
        for i, local_idx in enumerate(best_subset_local_indices):
            global_idx = top_n_indices[local_idx]
            original_score = top_n_scores[local_idx]
            path = image_paths[global_idx]
            print(f"{i+1}. Original FAISS Score: {original_score:.4f} | Image: {path}")
    else:
        print("A* could not find a valid combination.")

if __name__ == "__main__":
    print("\n" + "="*50)
    print("A* INTELLIGENT SEARCH ENGINE READY")
    print("Type 'exit' or 'quit' to stop.")
    print("="*50)
    
    while True:
        user_query = input("\nEnter search query to test A*: ")
        if user_query.lower() in ['exit', 'quit']:
            break
        if user_query.strip() == "":
            continue
            
        search_with_astar(user_query)