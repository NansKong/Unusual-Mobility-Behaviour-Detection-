import os
from huggingface_hub import snapshot_download

def download_model():
    model_id = "OPear/videomae-large-finetuned-UCF-Crime"
    local_dir = os.path.join(os.path.dirname(__file__), "..", "models", "videomae_ucf")
    
    print(f"Downloading {model_id} to {local_dir}...")
    
    # Download the model and cache it locally in the models folder
    snapshot_download(
        repo_id=model_id,
        local_dir=local_dir,
        local_dir_use_symlinks=False, # Ensure actual files are downloaded, not symlinks
        ignore_patterns=["*.msgpack", "*.h5", "*.tflite", "*.ot"] # Ignore non-pytorch weights to save bandwidth
    )
    
    print("Download completed successfully!")

if __name__ == "__main__":
    download_model()
