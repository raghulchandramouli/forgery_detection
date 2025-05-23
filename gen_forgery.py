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
def create_copy_move_forgery(
    image_path, output_path  
):
    
    """
    Creates a copy-move forgery of the given image using OpenCV.
    Copies a random area and moves it to a random position.
    """
    
    try:
        img = cv2.imread(image_path)
        if img is None:
            print(f"Error reading image {image_path}. Skipping copy-move forgery creation.")
            return False
        
        h, w, _ = img.shape
        if h < 100 or w < 100:
            print(f"Image {image_path} is too small. Skipping copy-move forgery creation.")
            return False
        
        patch_size = min(h, w) // 10
        if patch_size < 20: patch_size = 20 # Minimum patch size
        
        # Randomly select a region to copy
        src_x1 = random.randint(0, w - patch_size)
        src_y1 = random.randint(0, h - patch_size)
        src_x2 = src_x1 + patch_size
        src_y2 = src_y1 + patch_size
        
        # Randomly select a position to move the copied region
        while True:
            dst_x1 = random.randint(0, w - patch_size)
            dst_y1 = random.randint(0, h - patch_size)
            dst_x2 = dst_x1 + patch_size
            dst_y2 = dst_y1 + patch_size
            
            # check for overlap or proximity
            if (
                abs(dst_x1 - src_x1) > patch_size
                or abs(dst_y1 - src_y1) > patch_size
            ):
                break # found a valid position
            
        patch = img[src_y1:src_y2, src_x1:src_x2].copy()
        img[dst_y1:dst_y2, dst_x1:dst_x2] = patch
        
        cv2.imwrite(output_path, img)
        return True

    except Exception as e:
        print(f"Error creating copy-move forgery: {e}")
        return False
    
def generate_forged_dataset(
  input_dir,
  output_dir,
  num_forgeries_per_image=1  
):
    
    """
    Generates forgeries for the given dataset.
    
    Args:
        input_dir (str): Path to the input directory containing real images.
        output_dir (str): Path to the output directory where forgeries will be saved.
        num_forgeries_per_image (int): Number of forgeries to generate per real image.
    """
    
    os.makedirs(output_dir, exist_ok=True)
    
    clean_output_dir = os.path.join(output_dir, 'clean')
    forget_output_dir = os.path.join(output_dir, 'forged')

    os.makedirs(clean_output_dir, exist_ok=True)
    os.makedirs(forget_output_dir, exist_ok=True)
    
    image_files = [
        f for f in os.listdir(input_dir) if f.endswith(('.jpg', '.jpeg', '.png'))
    ]
    
    random.shuffle(image_files)
    
    list_file_path = os.path.join(output_dir, 'list.txt')
    
    with open(list_file_path, 'w') as f:
        for i, image_files in ennumerate(image_files):
            input_image_path = os.path.join(input_dir, image_files)
            base_name, ext = os.path.splitext(clean_output_dir, image_files)
            
            # 1. save clean image
            clean_output_name = f"{base_name}_clean{ext}"
            clean_image_path = os.path.join(clean_output_dir, clean_output_name)
            
            if not os.path.exists(clean_image_path):
                shutil.copy(input_image_path, clean_image_path)
                
            f.write(f"{os.path.relpath(clean_image_path, output_dir)}0\n") # 0 for clean image

            # 2. generate forgeries and save them as well:
            
            