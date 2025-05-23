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
        
        # define mask size (e.g., 1/5th of the image) so if Image size is 512, then mask size will be 102.4
        mask_w = random.randint(w // 10, w // 4) # random mask size between 1/5th and 1/2th of the image width
        mask_h = random.randint(h // 10, h // 4) # random mask size between 1/5th and 1/2th of the image height
        
        # Randomly select a region to inpaint
        mask_x1 = random.randint(0, w - mask_w) 
        mask_y1 = random.randint(0, h - mask_h)   
        mask_x2 = mask_x1 + mask_w
        mask_y2 = mask_y1 + mask_h

        mask_draw.rectangle([mask_x1, mask_y1, mask_x2, mask_y2], fill=255) # setting color for white mask
        
        # prompt:
        prompt = "A photo realistic image that blends in with the background"
        
        # Resizing the image to fit the inpainting model guidelines
        input_img_resized = img.resize((512, 512))
        mask_resized = mask.resize((512, 512))
        
        # Inpainting the masked region
        inpainted_image = inpaint_pipe(
            prompt=prompt,
            image=input_img_resized,
            mask_image=mask_resized,
            num_inference_steps=50,
            
        ).images[0]
        
        inpainted_image = inpainted_image.resize((w, h))

        inpainted_image.save(output_path)
        return True
    
    except Exception as e:
        print(f"Error creating inpainting forgery: {e}")
        return False
    
# helper function to create-copy-move forgery
