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

# Suppress warnings for a cleaner terminal output
warnings.filterwarnings("ignore")

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
# We use the "base" model. It is small enough to run quickly on a CPU but very accurate for English.
whisper_model = whisper.load_model("base")
print("All systems go! Launching UI...")

# CORE LOGIC

def get_text_embedding(query_text):
    if isinstance(input_image, np.ndarray):
        input_image = Image.fromarray(input_image)
    if input_image.mode != "RGB":
        input_image = input_image.convert("RGB")
    inputs = processor(images=input_image, return_tensors="pt", padding=True)
    with torch.no_grad():
        text_features = model(**inputs).text_embeds
        text_features = text_features / text_features.norm(p=2, dim=-1, keepdim=True)
    return text_features.cpu().numpy().astype(np.float32)

def get_image_embedding(input_image):
    """Genera el embedding visual de la imagen de consulta."""
    if isinstance(input_image, np.ndarray):
        input_image = Image.fromarray(input_image)
    if input_image.mode != "RGB":
        input_image = input_image.convert("RGB")
    inputs = processor(text=[""], images=input_image, return_tensors="pt", padding=True)
    with torch.no_grad():
        outputs = model(**inputs)
        image_features = outputs.image_embeds
        # L2 Normalization
        image_features = image_features / image_features.norm(p=2, dim=-1, keepdim=True)
    return image_features.cpu().numpy().astype(np.float32)

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

def process_query(text_input, audio_input, image_input):
    """
    Handles user input (Voice, Text, or Image).
    Generates the corresponding vector and performs the A* search.
    """
    try:
        status_message = ""
        query_vector = None
        
        # 1. Identificar el tipo de entrada y generar el vector
        if audio_input is not None:
            result = whisper_model.transcribe(audio_input)
            raw_query = result["text"].strip()
            status_message = f"[Audio] Buscando: '{raw_query}'"
            query_vector = get_text_embedding(raw_query)
            
        elif text_input:
            raw_query = text_input.strip()
            status_message = f"[Texto] Buscando: '{raw_query}'"
            query_vector = get_text_embedding(raw_query)
            
        elif image_input is not None:
            status_message = f"[Imagen] Ejecutando búsqueda visual inversa..."
            query_vector = get_image_embedding(image_input)
            
        else:
            return [], "Por favor, proporciona texto, un audio o sube una imagen."
            
        # 2. Perform Multimodal Search
        distances, indices = index.search(query_vector, 20)
        
        top_n_indices = indices[0]
        top_n_scores = distances[0]
        candidate_embeddings = image_embeddings[top_n_indices]
        
        # 3. Apply Intelligent Systems (A* Filter)
        best_subset_local_indices = a_star_selection(query_vector, top_n_indices, top_n_scores, candidate_embeddings, k=5)
        
        # 4. Gather results for the UI Gallery
        result_images = []
        if best_subset_local_indices:
            for local_idx in best_subset_local_indices:
                global_idx = top_n_indices[local_idx]
                path = image_paths[global_idx]
                result_images.append(path)
                
        return result_images, status_message

    except Exception as e:
        import traceback
        print(traceback.format_exc()) # Lo imprime detallado en la terminal
        return [], f"ERROR: {str(e)}"
# GRADIO WEB INTERFACE

with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown("# Multimodal Semantic Search Engine")
    
    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("### 1. Enter your query")
            text_box = gr.Textbox(label="Text Query", placeholder="e.g. A dog playing with a red ball...")
            gr.Markdown("**OR**")
            audio_box = gr.Audio(sources=["microphone"], type="filepath", label="Voice Query (English)")
            image_query_box = gr.Image(sources=["upload", "webcam"], type="pil", label="3. Image Search")
            
            search_button = gr.Button("Search Images", variant="primary")
            status_text = gr.Textbox(label="System Status", interactive=False)
            
        with gr.Column(scale=2):
            gr.Markdown("### 2. A* Optimized Results")
            # Gallery component to display multiple images beautifully
            gallery = gr.Gallery(label="Selected Images", show_label=False, elem_id="gallery", columns=3, rows=2, height="auto")

    # Connect the UI elements to the python function
    search_button.click(
        fn=process_query, 
        inputs=[text_box, audio_box, image_query_box], 
        outputs=[gallery, status_text]
    )

# Launch the local web server
if __name__ == "__main__":
    demo.launch()