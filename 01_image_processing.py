import torch
from transformers import CLIPProcessor, CLIPModel
from PIL import Image
import os

# 1. Load the pre-trained CLIP model and processor
print("Loading CLIP model...")
model_id = "openai/clip-vit-base-patch32"
model = CLIPModel.from_pretrained(model_id)
processor = CLIPProcessor.from_pretrained(model_id)
print("Model loaded successfully!")

# 2. Test with a single image from your dataset
sample_image_path = "data/Images/1000268201_693b08cb0e.jpg" 

if os.path.exists(sample_image_path):
    print(f"Processing image: {sample_image_path}")
    image = Image.open(sample_image_path)
    
    # We provide a dummy text string so the processor generates 'input_ids'
    # This satisfies the multimodal forward pass requirements of CLIPModel
    inputs = processor(text=[""], images=image, return_tensors="pt", padding=True)
    
    with torch.no_grad():
        # The model receives both modalities and won't crash
        outputs = model(**inputs)
        # We exclusively extract the visual representation
        image_features = outputs.image_embeds
        
    print("Success! Embedding generated.")
    print(f"Tensor Shape: {image_features.shape}")
    
else:
    print(f"Error: Could not find the image at {sample_image_path}")
    print("Please check your folder structure.")