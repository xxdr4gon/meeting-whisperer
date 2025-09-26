#!/usr/bin/env python3
"""
Download Estonian model manually before building Docker image.
This way you can download it overnight or when you have good internet.
"""

import os
from huggingface_hub import snapshot_download

def main():
    print("Downloading Estonian model...")
    print("This will take 10-20+ minutes with 3-15 Mbps internet...")
    
    # Create models directory
    os.makedirs("./models/hf_et_model", exist_ok=True)
    
    # Download the model
    snapshot_download(
        repo_id="TalTechNLP/whisper-large-v3-turbo-et-verbatim",
        local_dir="./models/hf_et_model"
    )
    
    print("Model downloaded successfully!")
    print("You can now run: docker compose up --build -d")

if __name__ == "__main__":
    main()
