import os   
import torch
from transformers import VideoMAEImageProcessor, VideoMAEModel
import numpy as np

def verify_videomae():
    print("Loading VideoMAE model from HuggingFace...")
    local_dir = os.path.join(os.path.dirname(__file__), "..", "models", "videomae_ucf")
    
    # Load model and processor from local cache
    processor = VideoMAEImageProcessor.from_pretrained(local_dir)
    model = VideoMAEModel.from_pretrained(local_dir)
    
    print("Model loaded successfully. Verifying CLS token shape...")
    
    # Create dummy input (batch_size, num_frames, num_channels, height, width)
    # The PRD specifies 16 frames @ 224x224 RGB
    num_frames = 16
    batch_size = 1
    dummy_video = [np.random.randint(0, 256, (224, 224, 3)).astype(np.uint8) for _ in range(num_frames)]
    
    # Process inputs
    inputs = processor(dummy_video, return_tensors="pt")
    
    # Run model forward pass
    with torch.no_grad():
        outputs = model(**inputs)
    
    # Extract CLS token from outputs.last_hidden_state[:,0,:]
    cls_token = outputs.last_hidden_state[:, 0, :]
    
    print(f"Original hidden state shape: {outputs.last_hidden_state.shape}")
    print(f"Extracted CLS token shape: {cls_token.shape}")
    
    if cls_token.shape == (1, 1024):
        print("Success! CLS token shape is exactly (1024,) per track as expected in the PRD.")
    else:
        print("Error: Unexpected CLS token shape.")

if __name__ == "__main__":
    verify_videomae()
