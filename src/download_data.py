"""
download_data.py
Downloads the UCF-Crime dataset from HuggingFace and extracts it to the data folder.
"""

import os
import zipfile
from pathlib import Path
from huggingface_hub import hf_hub_download

def download_and_extract_dataset():
    repo_id = "hibana2077/UCF-Crime-Dataset"
    filename = "ucf-crime-dataset.zip"
    
    # Target directories
    base_dir = Path(__file__).resolve().parent.parent
    data_dir = base_dir / "data" / "ucf_crime"
    videos_dir = data_dir / "videos"
    annotations_dir = data_dir / "annotations"
    
    videos_dir.mkdir(parents=True, exist_ok=True)
    annotations_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Downloading {filename} from HuggingFace ({repo_id})...")
    print("This might take a while depending on your internet connection.")
    
    try:
        zip_path = hf_hub_download(repo_id=repo_id, filename=filename, repo_type="dataset")
        print(f"Download complete: {zip_path}")
        
        print("Extracting dataset...")
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            # Extract to the videos directory
            zip_ref.extractall(videos_dir)
            
        print(f"Extraction complete! Videos are located in: {videos_dir}")
        
        # Note: The PRD mentions Temporal_Anomaly_Annotation.txt. If it's inside the zip,
        # we can move it to the annotations directory. Let's do a quick scan.
        annotation_file = None
        for root, _, files in os.walk(videos_dir):
            for file in files:
                if file.endswith(".txt") and "annotation" in file.lower():
                    annotation_file = Path(root) / file
                    break
        
        if annotation_file:
            target_path = annotations_dir / "Temporal_Anomaly_Annotation.txt"
            os.rename(annotation_file, target_path)
            print(f"Found and moved annotation file to: {target_path}")
        else:
            print("Note: Temporal_Anomaly_Annotation.txt was not found in the zip archive.")
            print("You may need to download it separately from the official UCF-Crime website.")
            
    except Exception as e:
        print(f"Error downloading dataset: {e}")

if __name__ == "__main__":
    download_and_extract_dataset()
