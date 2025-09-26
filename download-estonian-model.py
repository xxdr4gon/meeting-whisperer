#!/usr/bin/env python3
"""
Download Estonian Whisper model manually
"""
import os
from huggingface_hub import snapshot_download

def download_estonian_model():
    model_name = "TalTechNLP/whisper-large-v3-turbo-et-verbatim"
    local_dir = "./models/hf_et_model"
    
    print(f"Downloading {model_name} to {local_dir}...")
    
    # Create directory if it doesn't exist
    os.makedirs(local_dir, exist_ok=True)
    
    try:
        # Download the model
        snapshot_download(
            repo_id=model_name,
            local_dir=local_dir,
            local_dir_use_symlinks=False  # Use actual files, not symlinks
        )
        print(f"✅ Model downloaded successfully to {local_dir}")
        print(f"📁 Contents: {os.listdir(local_dir)}")
        
    except Exception as e:
        print(f"❌ Error downloading model: {e}")
        return False
    
    return True

if __name__ == "__main__":
    download_estonian_model()
