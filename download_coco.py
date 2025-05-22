from pycocotools.coco import COCO
import requests
import os
import argparse
import zipfile # Import zipfile for extraction
import json

def download_coco_dataset(output_dir='coco_dataset', limit=5000):
    """
    Downloads a subset of the COCO 2017 validation dataset
    
    Args:
        output_dir (str): The directory to save the downloaded images
        limit (int): The maximum number of images to download
    """
    
    # make the o/p dir:
    os.makedirs(output_dir, exist_ok=True)

    # Define annotation zip file URL and local path
    annotation_zip_url = 'http://images.cocodataset.org/annotations/annotations_trainval2017.zip'
    local_zip_path = os.path.join(output_dir, 'annotations_trainval2017.zip')
    local_annotation_path = os.path.join(output_dir, 'annotations', 'instances_val2017.json') # Correct path within the zip

    # Download the annotation zip file if it doesn't exist
    if not os.path.exists(local_zip_path):
        print(f"Downloading annotation zip file from {annotation_zip_url}...")
        try:
            response = requests.get(annotation_zip_url, stream=True)
            response.raise_for_status() # Raise an HTTPError for bad responses (4xx or 5xx)
            with open(local_zip_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            print("Annotation zip file downloaded successfully.")
        except requests.exceptions.RequestException as e:
            print(f"Error downloading annotation zip file: {e}")
            return # Exit if annotation zip file download fails
    else:
        print("Annotation zip file already exists. Skipping download.")

    # Extract the annotation file from the zip
    annotation_dir = os.path.join(output_dir, 'annotations')
    os.makedirs(annotation_dir, exist_ok=True)

    if not os.path.exists(local_annotation_path):
        print(f"Extracting {os.path.basename(local_annotation_path)} from {os.path.basename(local_zip_path)}...")
        try:
            with zipfile.ZipFile(local_zip_path, 'r') as zip_ref:
                # Find the specific annotation file within the zip
                # The structure is usually annotations/instances_val2017.json
                target_file_in_zip = 'annotations/instances_val2017.json'
                if target_file_in_zip in zip_ref.namelist():
                     zip_ref.extract(target_file_in_zip, output_dir)
                     print("Annotation file extracted successfully.")
                else:
                     print(f"Error: Could not find {target_file_in_zip} in the zip file.")
                     return
        except zipfile.BadZipFile:
             print(f"Error: The downloaded file {local_zip_path} is not a valid zip file.")
             return
        except Exception as e:
            print(f"Error extracting annotation file: {e}")
            return # Exit if extraction fails
    else:
        print("Annotation file already extracted. Skipping extraction.")


    # init COCO API for 2017 validation set using the local file
    # Use the path to the extracted JSON file
    try:
        coco = COCO(local_annotation_path)
    except Exception as e:
        print(f"Error initializing COCO API with {local_annotation_path}: {e}")
        print("Please ensure the extracted annotation file is valid JSON.")
        return # Exit if COCO initialization fails


    # Get all the images IDs:
    img_ids = coco.getImgIds()

    # Limit the number of images:
    # import random
    # random.shuffle(img_ids) # Optional: shuffle if you want a random subset
    img_ids = img_ids[:limit]

    images = coco.loadImgs(img_ids)

    print(f"Attempting to download {len(images)} images...")

    for idx, img in enumerate(images):
        img_url = img['coco_url']
        file_name = img['file_name']
        file_path = os.path.join(output_dir, file_name)

        try:
            if not os.path.exists(file_path):
                img_data = requests.get(
                    img_url,
                    stream=True
                )

                img_data.raise_for_status() # this throws an HTTPError if the HTTP request returned an unsuccessful status code

                with open(file_path, 'wb') as f:
                    for chunk in img_data.iter_content(chunk_size=8192):
                        f.write(chunk)

            else:
                print(f"File {file_name} already exists. Skipping...")

            if (idx+ 1) % 100 == 0:
                print(f"Downloaded {idx+1}/{len(images)} images")

        except requests.exceptions.RequestException as e:
            print(f"Error downloading image {img_url} : {e}")

        except Exception as e:
            print(f"An unexpected error occurred while downloading image {img_url} : {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download a subset of COCO 2017 validation dataset")
    parser.add_argument("--output_dir", default='coco_dataset', help="The directory to save the downloaded images")
    parser.add_argument("--limit", type=int, default=5000, help="The maximum number of images to download")

    args = parser.parse_args()

    download_coco_dataset(output_dir=args.output_dir, limit=args.limit)