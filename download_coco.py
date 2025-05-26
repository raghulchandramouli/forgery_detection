import requests
import os
import argparse
import zipfile
import shutil
from pathlib import Path

def download_and_extract_zip(url, output_path, extract_dir):
    """Downloads a zip file and extracts its contents."""
    # Convert to Path objects to handle cross-platform paths
    output_path = Path(output_path)
    extract_dir = Path(extract_dir)
    
    # Create the output directory if it doesn't exist
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    if not output_path.exists():
        print(f"Downloading zip file from {url}...")
        try:
            response = requests.get(url, stream=True)
            response.raise_for_status()
            with open(output_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            print("Zip file downloaded successfully.")
        except requests.exceptions.RequestException as e:
            print(f"Error downloading zip file from {url}: {e}")
            return False
    else:
        print(f"Zip file already exists at {output_path}. Skipping download.")

    # Check if extract_dir exists and is not empty
    if extract_dir.exists() and any(extract_dir.iterdir()):
        print(f"Extraction directory {extract_dir} is not empty. Skipping extraction.")
        return True

    # If directory doesn't exist or is empty, proceed with extraction
    print(f"Extracting zip file {output_path.name} to {extract_dir}...")
    extract_dir.mkdir(parents=True, exist_ok=True)
    try:
        with zipfile.ZipFile(output_path, 'r') as zip_ref:
            zip_ref.extractall(extract_dir)
        print("Zip file extracted successfully.")
        return True
    except zipfile.BadZipFile:
        print(f"Error: The downloaded file {output_path} is not a valid zip file.")
        return False
    except Exception as e:
        print(f"Error extracting zip file {output_path}: {e}")
        return False

def download_coco_images(output_dir='./coco_dataset'):
    """
    Downloads and extracts the COCO 2017 validation images zip file.

    Args:
        output_dir (str): Directory to save the downloaded and extracted data.
    """
    # Convert to Path object to handle cross-platform paths
    output_dir = Path(output_dir)
    
    # make the o/p dir:
    output_dir.mkdir(parents=True, exist_ok=True)

    # --- Download and Extract Images ---
    images_zip_url = 'http://images.cocodataset.org/zips/val2017.zip'
    local_images_zip_path = output_dir / 'val2017.zip'
    images_extract_dir = output_dir / 'coco_dataset'  # Changed to extract to coco_dataset subfolder

    print("Starting COCO 2017 Validation Images download and extraction...")
    success = download_and_extract_zip(images_zip_url, local_images_zip_path, images_extract_dir)  # Changed to use images_extract_dir

    if success:
        print(f"COCO 2017 Validation images are located in the '{images_extract_dir}' directory.")
    else:
        print("Failed to download or extract COCO images.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download and extract COCO 2017 validation dataset images zip file.")
    parser.add_argument("--output_dir", default='./coco_dataset', 
                       help="The directory to save the downloaded and extracted data.")

    args = parser.parse_args()
    download_coco_images(output_dir=args.output_dir)