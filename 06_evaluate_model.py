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

