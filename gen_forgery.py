import os
import random
import shutil
import argparse
from PIL import Image, ImageDraw # Import ImageDraw
import cv2
import numpy as np
import torch
from diffusers import StableDiffusionInpaintPipeline
from transformers import pipeline

# Set device
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# Load models
# Stable diffusion Inpainting Model.

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

        mask = Image.new("L", (w, h), 0) # setting color for black background, use "L" for grayscale mask
        mask_draw = ImageDraw.Draw(mask)

        # define mask size (e.g., 1/5th of the image) so if Image size is 512, then mask size will be 102.4
        mask_w = random.randint(w // 10, w // 4) # random mask size between 1/10th and 1/4th of the image width
        mask_h = random.randint(h // 10, h // 4) # random mask size between 1/10th and 1/4th of the image height

        # Randomly select a region to inpaint
        mask_x1 = random.randint(0, w - mask_w)
        mask_y1 = random.randint(0, h - mask_h)
        mask_x2 = mask_x1 + mask_w
        mask_y2 = mask_y1 + mask_h

        mask_draw.rectangle([mask_x1, mask_y1, mask_x2, mask_y2], fill=255) # setting color for white mask

        # prompt:
        prompt = "A photo realistic image that blends in with the background"

        # Resizing the image to fit the inpainting model guidelines
        # Stable Diffusion 2 Inpainting model expects 768x768
        input_img_resized = img.resize((768, 768))
        mask_resized = mask.resize((768, 768))

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

# helper function to create-copy-move-forgery
def create_copy_move_forgery(image_path, output_path):
    """
    Creates a simple copy-move forgery using OpenCV.
    Copies a random patch from the image and pastes it elsewhere.
    """
    try:
        img = cv2.imread(image_path)
        if img is None:
            print(f"Warning: Could not read image {image_path}. Skipping copy-move.")
            return False

        h, w, _ = img.shape
        if h < 100 or w < 100: # Skip very small images
             print(f"Warning: Image {image_path} is too small for copy-move. Skipping.")
             return False

        # Define patch size (e.g., 1/10th of the smaller dimension)
        patch_size = min(h, w) // 10
        if patch_size < 20: patch_size = 20 # Minimum patch size

        # Randomly select source patch coordinates
        src_x1 = random.randint(0, w - patch_size)
        src_y1 = random.randint(0, h - patch_size)
        src_x2 = src_x1 + patch_size
        src_y2 = src_y1 + patch_size

        # Randomly select destination patch coordinates (ensure not too close to source)
        # Simple check: ensure destination is at least patch_size away from source
        while True:
            dst_x1 = random.randint(0, w - patch_size)
            dst_y1 = random.randint(0, h - patch_size)
            dst_x2 = dst_x1 + patch_size
            dst_y2 = dst_y1 + patch_size

            # Check for overlap or proximity
            if (abs(dst_x1 - src_x1) > patch_size or abs(dst_y1 - src_y1) > patch_size):
                 break # Found a suitable destination

        # Copy the patch
        patch = img[src_y1:src_y2, src_x1:src_x2].copy()

        # Paste the patch
        img[dst_y1:dst_y2, dst_x1:dst_x2] = patch

        # Simple blending at the edges of the pasted patch for basic harmonization
        # This is a very basic approach; a real GAN would do this much better
        # You could add more sophisticated blending here if needed
        # For now, we just paste directly.

        cv2.imwrite(output_path, img)
        return True

    except Exception as e:
        print(f"Error creating copy-move for {image_path}: {e}")
        return False


def generate_forged_dataset(input_dir, output_dir, num_forgeries_per_image=1):
    """
    Generates clean and forged images from the input directory.

    Args:
        input_dir (str): Directory containing the original clean images.
        output_dir (str): Directory to save the generated data.
        num_forgeries_per_image (int): Number of forgery types to apply per image (currently supports 2: copy-move, inpainting).
    """
    os.makedirs(output_dir, exist_ok=True)
    clean_output_dir = os.path.join(output_dir, 'clean')
    forged_output_dir = os.path.join(output_dir, 'forged')
    os.makedirs(clean_output_dir, exist_ok=True)
    os.makedirs(forged_output_dir, exist_ok=True)

    image_files = [f for f in os.listdir(input_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    random.shuffle(image_files) # Shuffle to process images in a random order

    list_file_path = os.path.join(output_dir, 'list.txt')

    with open(list_file_path, 'w') as f:
        for i, image_file in enumerate(image_files):
            input_image_path = os.path.join(input_dir, image_file)
            base_name, ext = os.path.splitext(image_file)

            # 1. Save Clean Image
            clean_image_name = f"{base_name}_clean{ext}"
            clean_image_path = os.path.join(clean_output_dir, clean_image_name)
            if not os.path.exists(clean_image_path):
                 shutil.copy(input_image_path, clean_image_path)
            f.write(f"{os.path.relpath(clean_image_path, output_dir)} 0\n") # 0 for clean

            # 2. Generate and Save Forged Images
            if num_forgeries_per_image > 0:
                # Copy-Move Forgery
                copy_move_image_name = f"{base_name}_copymove{ext}"
                copy_move_image_path = os.path.join(forged_output_dir, copy_move_image_name)
                if not os.path.exists(copy_move_image_path):
                    if create_copy_move_forgery(input_image_path, copy_move_image_path):
                        f.write(f"{os.path.relpath(copy_move_image_path, output_dir)} 1\n") # 1 for forged
                    else:
                        print(f"Skipping copy-move entry for {image_file} due to failure.")
                else:
                    print(f"Copy-move forgery for {image_file} already exists. Skipping generation.")
                    f.write(f"{os.path.relpath(copy_move_image_path, output_dir)} 1\n") # Add to list if exists

            if num_forgeries_per_image > 1:
                 # Inpainting Forgery
                 inpainting_image_name = f"{base_name}_inpainting{ext}"
                 inpainting_image_path = os.path.join(forged_output_dir, inpainting_image_name)
                 if not os.path.exists(inpainting_image_path):
                     if create_inpainting_forgery(input_image_path, inpainting_image_path):
                         f.write(f"{os.path.relpath(inpainting_image_path, output_dir)} 1\n") # 1 for forged
                     else:
                         print(f"Skipping inpainting entry for {image_file} due to failure.")
                 else:
                     print(f"Inpainting forgery for {image_file} already exists. Skipping generation.")
                     f.write(f"{os.path.relpath(inpainting_image_path, output_dir)} 1\n") # Add to list if exists


            if (i + 1) % 100 == 0:
                print(f"Processed {i + 1}/{len(image_files)} images.")

    print(f"Forgery generation complete. Data saved to {output_dir}")
    print(f"List file created at {list_file_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generate clean and forged images from a dataset.')
    # Updated default input_dir to point to the correct subdirectory
    parser.add_argument('--input_dir', default='coco_dataset/val2017/val2017', help='Directory containing the original clean images.')
    parser.add_argument('--output_dir', default='forged_data', help='Directory to save the generated data.')
    parser.add_argument('--num_forgeries_per_image', type=int, default=2, help='Number of forgery types to apply per image (0, 1, or 2).')
    args = parser.parse_args()

    generate_forged_dataset(args.input_dir, args.output_dir, args.num_forgeries_per_image)