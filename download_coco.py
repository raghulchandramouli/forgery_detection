import requests
import os
import argparse
import zipfile # Import zipfile for extraction
import shutil # Import shutil for moving files

def download_and_extract_zip(url, output_path, extract_dir):
    """Downloads a zip file and extracts its contents."""
    if not os.path.exists(output_path):
        print(f"Downloading zip file from {url}...")
        try:
            response = requests.get(url, stream=True)
            response.raise_for_status() # Raise an HTTPError for bad responses (4xx or 5xx)
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
    if os.path.exists(extract_dir) and os.listdir(extract_dir):
        print(f"Extraction directory {extract_dir} is not empty. Skipping extraction.")
        return True # Assume extraction was done previously

    # If directory doesn't exist or is empty, proceed with extraction
    print(f"Extracting zip file {os.path.basename(output_path)} to {extract_dir}...")
    os.makedirs(extract_dir, exist_ok=True) # Ensure the directory exists before extracting
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


def download_coco_images(output_dir='coco_dataset'):
    """
    Downloads and extracts the COCO 2017 validation images zip file.

    Args:
        output_dir (str): Directory to save the downloaded and extracted data.
    """

    # make the o/p dir:
    os.makedirs(output_dir, exist_ok=True)

    # --- Download and Extract Images ---
    images_zip_url = 'http://images.cocodataset.org/zips/val2017.zip'
    local_images_zip_path = os.path.join(output_dir, 'val2017.zip')
    images_extract_dir = os.path.join(output_dir, 'val2017') # Images will be extracted here

    print("Starting COCO 2017 Validation Images download and extraction...")
    success = download_and_extract_zip(images_zip_url, local_images_zip_path, output_dir) # Extract images to output_dir/val2017

    if success:
        print(f"COCO 2017 Validation images are located in the '{images_extract_dir}' directory.")
    else:
        print("Failed to download or extract COCO images.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download and extract COCO 2017 validation dataset images zip file.")
    parser.add_argument("--output_dir", default='coco_dataset', help="The directory to save the downloaded and extracted data.")
    # Removed --limit as we are downloading the whole zip

    args = parser.parse_args()

    download_coco_images(output_dir=args.output_dir)