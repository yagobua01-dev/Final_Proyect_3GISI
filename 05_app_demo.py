import torch
import numpy as np
import faiss
import json
import os
import heapq
import gradio as gr
import whisper
from transformers import CLIPProcessor, CLIPModel
from PIL import Image
import warnings
import nltk
from nltk.corpus import wordnet

# Suppress warnings for a cleaner terminal output
warnings.filterwarnings("ignore")

# Download necessary resources for NLP (WordNet for Query Expansion)
nltk.download('wordnet')
nltk.download('omw-1.4')

# 1. Configuration and Paths
EMBEDDINGS_DIR = "embeddings/"
EMBEDDINGS_FILE = os.path.join(EMBEDDINGS_DIR, "image_embeddings.npy")
PATHS_FILE = os.path.join(EMBEDDINGS_DIR, "image_paths.json")

print("Initializing Multimodal System...")

# 2. Load FAISS Database
print("Loading vector database...")
image_embeddings = np.load(EMBEDDINGS_FILE)
with open(PATHS_FILE, "r") as f:
    image_paths = json.load(f)

dimension = image_embeddings.shape[1]
index = faiss.IndexFlatIP(dimension)
index.add(image_embeddings)

# 3. Load Multimodal Models (Vision & Text)
print("Loading CLIP model...")
model_id = "openai/clip-vit-base-patch32"
model = CLIPModel.from_pretrained(model_id)
processor = CLIPProcessor.from_pretrained(model_id)

# 4. Load Speech-to-Text Model
print("Loading Whisper model (Speech recognition)...")
whisper_model = whisper.load_model("base")
print("All systems go! Launching UI...")

# QUERY EXPANSION

def expand_query(text):
    """
    NLP Technique: Query Expansion using WordNet synonyms.
    This improves the semantic reach of the search by adding related terms.
    """
    words = text.lower().split()
    expanded_words = set(words)
    
    # We expand the most important words (first 4) to keep the query relevant
    for word in words[:4]:
        # Skip very short words
        if len(word) < 3:
            continue
        for syn in wordnet.synsets(word):
            for lemma in syn.lemmas():
                # Add synonyms, replacing underscores with spaces
                synonym = lemma.name().replace('_', ' ')
                expanded_words.add(synonym)
                
    return " ".join(list(expanded_words))

# CORE LOGIC

def get_text_embedding(query_text):
    dummy_image = Image.new('RGB', (224, 224), (0, 0, 0))
    inputs = processor(text=[query_text], images=dummy_image, return_tensors="pt", padding=True)
    with torch.no_grad():
        text_features = model(**inputs).text_embeds
        text_features = text_features / text_features.norm(p=2, dim=-1, keepdim=True)
    return text_features.cpu().numpy().astype(np.float32)

def calculate_diversity_penalty(new_img_idx, current_state_indices, candidate_embeddings):
    if not current_state_indices:
        return 0.0
    new_emb = candidate_embeddings[new_img_idx]
    penalties = [np.dot(new_emb, candidate_embeddings[idx]) for idx in current_state_indices]
    return np.mean(penalties) 

def a_star_selection(query_embedding, top_n_indices, top_n_scores, candidate_embeddings, k=5, w1=0.7, w2=0.3):
    open_list = []
    best_possible_relevance_cost = 1.0 - np.max(top_n_scores)
    heapq.heappush(open_list, (0.0, 0.0, ()))
    
    while open_list:
        f_cost, g_cost, current_state = heapq.heappop(open_list)
        if len(current_state) == k:
            return current_state
            
        start_idx = current_state[-1] + 1 if current_state else 0
        for i in range(start_idx, len(top_n_indices)):
            new_state = current_state + (i,)
            relevance_cost = 1.0 - top_n_scores[i]
            diversity_cost = calculate_diversity_penalty(i, current_state, candidate_embeddings)
            
            step_cost = (w1 * relevance_cost) + (w2 * diversity_cost)
            new_g_cost = g_cost + step_cost
            h_cost = (k - len(new_state)) * (w1 * best_possible_relevance_cost) 
            
            heapq.heappush(open_list, (new_g_cost + h_cost, new_g_cost, new_state))
    return None

# UI INTEGRATION LOGIC

def process_query(text_input, audio_input):
    raw_query = ""
    status_message = ""
    
    # 1. Handle Input (Speech-to-Text or Direct Text)
    if audio_input is not None:
        result = whisper_model.transcribe(audio_input)
        raw_query = result["text"].strip()
        input_type = "Speech"
    elif text_input:
        raw_query = text_input.strip()
        input_type = "Text"
    else:
        return [], "Please provide either text or a voice recording."
    
    # 2. NLP Enhancement: Query Expansion
    expanded_query = expand_query(raw_query)
    status_message = f"[{input_type}] Raw: '{raw_query}' | Expanded: '{expanded_query}'"
        
    # 3. Perform Multimodal Search
    query_vector = get_text_embedding(expanded_query)
    distances, indices = index.search(query_vector, 20)
    
    top_n_indices = indices[0]
    top_n_scores = distances[0]
    candidate_embeddings = image_embeddings[top_n_indices]
    
    # 4. Apply Intelligent Systems (A* Filter)
    best_subset_local_indices = a_star_selection(query_vector, top_n_indices, top_n_scores, candidate_embeddings, k=5)
    
    # 5. Gather results for the UI Gallery
    result_images = []
    if best_subset_local_indices:
        for local_idx in best_subset_local_indices:
            global_idx = top_n_indices[local_idx]
            path = image_paths[global_idx]
            result_images.append(path)
            
    return result_images, status_message

# GRADIO WEB INTERFACE

with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown("Multimodal Semantic Search Engine")
    
    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("### 1. Input")
            text_box = gr.Textbox(label="Text Query", placeholder="e.g. A dog running on the beach...")
            gr.Markdown("**OR**")
            audio_box = gr.Audio(sources=["microphone"], type="filepath", label="Voice Query (English)")
            
            search_button = gr.Button("Search Images", variant="primary")
            status_text = gr.Textbox(label="NLP & System Status", interactive=False)
            
        with gr.Column(scale=2):
            gr.Markdown("### 2. A* Optimized Results")
            gallery = gr.Gallery(label="Results", show_label=False, columns=3, rows=2, height="auto")

    search_button.click(
        fn=process_query, 
        inputs=[text_box, audio_box], 
        outputs=[gallery, status_text]
    )

if __name__ == "__main__":
    demo.launch()