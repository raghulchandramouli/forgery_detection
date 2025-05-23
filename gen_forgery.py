import os
import random
import shutil
import argparse
from PIL import Image
import cv2
import torch
from diffusers import StableDiffusionInpaintPipeline
from transformers import pipeline

# set device
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# Load models
# Stable diffusion Inapainting Model.

try:
    inpaint_pipe = StableDiffusionInpaintPipeline.from_pretrained(
        "stabilityai/stable-diffusion-2-inpainting",
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
    ).to(device)
    
    print("Inpainting model loaded successfully.")
    
except Exception as e:
    print(f"Error loading inpainting model: {e}")
    inpaint_pipe = None # setting it to None if there's an error, and loading fails
    
    
print("Using OpenCV for copy-move forgery simulation")

# helper function to create-inpainting-forgery
def create_inpainting_forgery(
    image_path, output_path
):
    
    """
    Creates an inpainting forgery of the given image using stable diffusion.
    Selects a random area and inpaints it.
    """
    
    if inpaint_pipe is None:
        print("Inpainting model not loaded. Skipping inpainting forgery creation.")
        return False
    
    try:
        img = Image.open(image_path)
        img = img.convert("RGB")
        w, h = img.size
        
        if h < 128 or w < 128:
            # skips small images is does not meet the size requirement.
            print(f"Image {image_path} is too small. Skipping inpainting forgery creation.")
            return False
        
        mask = Image.new("RGB", (w, h), 0) # setting color for black background
        mask_draw = ImageDraw.Draw(mask)