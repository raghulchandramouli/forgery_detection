import os
import random
import shutil
import argparse
from PIL import Image, ImageDraw
import cv2
import numpy as np
import torch
from diffusers import StableDiffusionInpaintPipeline
import gc

# ============================== #
#     CONFIG / DEVICE SETUP     #
# ============================== #
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"[INFO] Using device: {device}")

# ============================== #
#   LOAD INPAINTING MODEL       #
# ============================== #
def load_inpainting_model():
    try:
        # Clear CUDA cache
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            gc.collect()
        
        pipe = StableDiffusionInpaintPipeline.from_pretrained(
            "runwayml/stable-diffusion-v1-5",
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            safety_checker=None,
        ).to(device)
        
        # Enable memory efficient attention
        pipe.enable_attention_slicing()
        if torch.cuda.is_available():
            pipe.enable_model_cpu_offload()
        
        print("[INFO] Inpainting model loaded successfully.")
        return pipe
    except Exception as e:
        print(f"[WARNING] Failed to load inpainting model: {e}")
        return None


def create_inpainting_forgery(image_path, output_path, pipe):
    if pipe is None:
        print("[SKIP] Inpainting model not loaded.")
        return False

    try:
        img = Image.open(image_path).convert("RGB")
        w, h = img.size
        if w < 128 or h < 128:
            print(f"[SKIP] Image too small for inpainting: {image_path}")
            return False

        # Create random mask
        mask = Image.new("L", (w, h), 0)
        draw = ImageDraw.Draw(mask)
        mask_w = random.randint(w // 8, w // 4)
        mask_h = random.randint(h // 8, h // 4)
        x1 = random.randint(0, w - mask_w)
        y1 = random.randint(0, h - mask_h)
        draw.rectangle([x1, y1, x1 + mask_w, y1 + mask_h], fill=255)

        # Resize for model (512x512 is optimal for SD 1.5)
        target_size = 512
        img_resized = img.resize((target_size, target_size), Image.Resampling.LANCZOS)
        mask_resized = mask.resize((target_size, target_size), Image.Resampling.LANCZOS)

        # Generate inpainting with prompts optimized for SD 1.5
        prompts = [
            "a detailed photograph, highly realistic, professional",
            "a clear photo with natural lighting and perfect details",
            "photorealistic image with high fidelity",
        ]
        
        with torch.inference_mode():
            result = pipe(
                prompt=random.choice(prompts),
                image=img_resized,
                mask_image=mask_resized,
                num_inference_steps=50,  # Reduced steps for faster generation
                guidance_scale=7.5,
            ).images[0]

        # Resize back to original size
        result = result.resize((w, h), Image.Resampling.LANCZOS)
        result.save(output_path, quality=95)
        print(f"[SUCCESS] Created inpainting forgery: {output_path}")
        return True

    except Exception as e:
        print(f"[ERROR] Inpainting failed for {image_path}: {e}")
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            gc.collect()
        return False



def create_copy_move_forgery(image_path, output_path):
    try:
        img = cv2.imread(image_path)
        if img is None:
            print(f"[SKIP] Can't read image: {image_path}")
            return False

        h, w, _ = img.shape
        if min(h, w) < 100:
            print(f"[SKIP] Image too small for copy-move: {image_path}")
            return False

        patch_size = max(20, min(h, w) // 10)
        src_x, src_y = random.randint(0, w - patch_size), random.randint(0, h - patch_size)
        dst_x, dst_y = src_x, src_y

        while abs(dst_x - src_x) < patch_size and abs(dst_y - src_y) < patch_size:
            dst_x = random.randint(0, w - patch_size)
            dst_y = random.randint(0, h - patch_size)

        patch = img[src_y:src_y + patch_size, src_x:src_x + patch_size].copy()
        img[dst_y:dst_y + patch_size, dst_x:dst_x + patch_size] = patch

        cv2.imwrite(output_path, img)
        return True

    except Exception as e:
        print(f"[ERROR] Copy-move failed for {image_path}: {e}")
        return False



def generate_forged_dataset(input_dir, output_dir, num_forgeries_per_image=2, use_inpainting=True, max_images=2500):
    os.makedirs(output_dir, exist_ok=True)
    clean_dir = os.path.join(output_dir, "clean")
    forged_dir = os.path.join(output_dir, "forged")
    os.makedirs(clean_dir, exist_ok=True)
    os.makedirs(forged_dir, exist_ok=True)

    # Load inpainting model (optional)
    inpaint_pipe = load_inpainting_model() if use_inpainting else None

    image_files = [f for f in os.listdir(input_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    random.shuffle(image_files)
    
    # Limit the number of images
    if max_images and max_images < len(image_files):
        print(f"[INFO] Limiting dataset to {max_images} images")
        image_files = image_files[:max_images]

    list_file = os.path.join(output_dir, "list.txt")
    with open(list_file, "w") as f:
        for i, filename in enumerate(image_files):
            base, ext = os.path.splitext(filename)
            src = os.path.join(input_dir, filename)

            # Save clean
            clean_path = os.path.join(clean_dir, f"{base}_clean{ext}")
            shutil.copy(src, clean_path)
            f.write(f"{os.path.relpath(clean_path, output_dir)} 0\n")

            # Copy-Move Forgery
            if num_forgeries_per_image >= 1:
                cm_path = os.path.join(forged_dir, f"{base}_copymove{ext}")
                if create_copy_move_forgery(src, cm_path):
                    f.write(f"{os.path.relpath(cm_path, output_dir)} 1\n")

            # Inpainting Forgery
            if num_forgeries_per_image >= 2 and inpaint_pipe is not None:
                inpaint_path = os.path.join(forged_dir, f"{base}_inpainting{ext}")
                if create_inpainting_forgery(src, inpaint_path, inpaint_pipe):
                    f.write(f"{os.path.relpath(inpaint_path, output_dir)} 1\n")

            if (i + 1) % 50 == 0:
                print(f"[INFO] Processed {i + 1}/{len(image_files)} images.")

    print(f"[DONE] Forgery generation complete. Output: {output_dir}")



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate clean and forged image dataset.")
    parser.add_argument('--input_dir', default='coco_dataset/coco_dataset/val2017', help='Input directory with clean images.')
    parser.add_argument('--output_dir', default='forged_data', help='Directory to store clean and forged images.')
    parser.add_argument('--num_forgeries_per_image', type=int, default=2, help='Number of forgery types (0, 1, or 2).')
    parser.add_argument('--no_inpainting', action='store_true', help='Disable inpainting forgeries.')
    parser.add_argument('--max_images', type=int, default=2500, help='Maximum number of images to process')

    args = parser.parse_args()

    generate_forged_dataset(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        num_forgeries_per_image=args.num_forgeries_per_image,
        use_inpainting=not args.no_inpainting,
        max_images=args.max_images
    )
