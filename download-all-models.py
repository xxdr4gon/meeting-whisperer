#!/usr/bin/env python3
"""
Comprehensive model download script for Meeting Whisperer
Downloads all required models and datasets into the correct directory structure
"""

import os
import sys
import time
from pathlib import Path
from huggingface_hub import snapshot_download, hf_hub_download
from datasets import load_dataset
import argparse
from tqdm import tqdm
import json

def print_status(message, level="INFO"):
    """Print status message with timestamp and level"""
    timestamp = time.strftime("%H:%M:%S")
    print(f"[{timestamp}] {level}: {message}")

def download_with_progress(repo_id, local_dir, description, use_symlinks=False):
    """Download a model with progress indication"""
    print_status(f"Starting download: {description}")
    print_status(f"Repository: {repo_id}")
    print_status(f"Destination: {local_dir}")
    
    try:
        # Create directory if it doesn't exist
        Path(local_dir).mkdir(parents=True, exist_ok=True)
        
        # Download with progress
        snapshot_download(
            repo_id=repo_id,
            local_dir=local_dir,
            local_dir_use_symlinks=use_symlinks,
            resume_download=True,
            local_files_only=False,
            max_workers=4
        )
        
        print_status(f"Successfully downloaded: {description}")
        return True
        
    except Exception as e:
        print_status(f"Failed to download {description}: {e}", "ERROR")
        return False

def download_dataset_with_progress(repo_id, local_dir, description):
    """Download a dataset with progress indication"""
    print_status(f"Starting dataset download: {description}")
    print_status(f"Repository: {repo_id}")
    print_status(f"Destination: {local_dir}")
    
    try:
        # Create directory if it doesn't exist
        Path(local_dir).mkdir(parents=True, exist_ok=True)
        
        # Load dataset
        print_status("Loading dataset...")
        dataset = load_dataset(repo_id)
        
        # Save dataset locally
        dataset_path = Path(local_dir) / "dataset.json"
        print_status("Saving dataset to local file...")
        
        # Convert to JSON and save
        with open(dataset_path, 'w', encoding='utf-8') as f:
            json.dump(dataset['train'].to_dict(), f, ensure_ascii=False, indent=2)
        
        print_status(f"Successfully downloaded dataset: {description}")
        print_status(f"Dataset saved to: {dataset_path}")
        return True
        
    except Exception as e:
        print_status(f"Failed to download dataset {description}: {e}", "ERROR")
        return False

def verify_download(local_dir, required_files=None):
    """Verify that download was successful"""
    if not Path(local_dir).exists():
        return False
    
    if required_files:
        for file in required_files:
            if not (Path(local_dir) / file).exists():
                return False
    
    return True

def main():
    parser = argparse.ArgumentParser(description="Download all Meeting Whisperer models and datasets")
    parser.add_argument("--base-dir", default="./models", help="Base directory for models")
    parser.add_argument("--symlinks", action="store_true", help="Use symlinks instead of copying files")
    parser.add_argument("--skip-grammar", action="store_true", help="Skip grammar dataset download")
    parser.add_argument("--skip-llama", action="store_true", help="Skip Llama model download")
    parser.add_argument("--skip-qwen", action="store_true", help="Skip Qwen model download")
    
    args = parser.parse_args()
    
    base_dir = Path(args.base_dir)
    
    print_status("Starting comprehensive model download for Meeting Whisperer")
    print_status("=" * 60)
    
    # Define all downloads
    downloads = []
    
    # Estonian ASR Models
    downloads.append({
        "type": "model",
        "repo_id": "TalTechNLP/whisper-large-v3-turbo-et-verbatim",
        "local_dir": str(base_dir / "estonian-asr"),
        "description": "Estonian ASR Model (TalTechNLP Whisper Large v3)",
        "required_files": ["config.json", "tokenizer.json"]
    })
    
    # AI Summarization Models
    if not args.skip_qwen:
        downloads.append({
            "type": "model",
            "repo_id": "Qwen/Qwen2.5-3B-Instruct",
            "local_dir": str(base_dir / "summarization" / "qwen"),
            "description": "Qwen AI Summarization Model (3B parameters)",
            "required_files": ["config.json", "tokenizer.json"]
        })
    
    if not args.skip_llama:
        downloads.append({
            "type": "model",
            "repo_id": "tartuNLP/llama-estllm-protype-0825",
            "local_dir": str(base_dir / "summarization" / "llama"),
            "description": "Llama AI Summarization Model (8B parameters)",
            "required_files": ["config.json", "tokenizer.json"]
        })
    
    # Grammar Dataset
    if not args.skip_grammar:
        downloads.append({
            "type": "dataset",
            "repo_id": "TalTechNLP/grammar_et",
            "local_dir": str(base_dir / "grammar-correction"),
            "description": "Estonian Grammar Correction Dataset",
            "required_files": ["dataset.json"]
        })
    
    # Track success/failure
    results = []
    
    # Download each item
    for i, download in enumerate(downloads, 1):
        print_status(f"Download {i}/{len(downloads)}: {download['description']}")
        print_status("-" * 40)
        
        if download["type"] == "model":
            success = download_with_progress(
                download["repo_id"],
                download["local_dir"],
                download["description"],
                args.symlinks
            )
        else:  # dataset
            success = download_dataset_with_progress(
                download["repo_id"],
                download["local_dir"],
                download["description"]
            )
        
        # Verify download
        if success:
            verified = verify_download(download["local_dir"], download.get("required_files"))
            if verified:
                print_status(f"Verification passed: {download['description']}")
                results.append(True)
            else:
                print_status(f"Verification failed: {download['description']}", "WARNING")
                results.append(False)
        else:
            results.append(False)
        
        print_status("")  # Empty line for readability
    
    # Summary
    print_status("DOWNLOAD SUMMARY")
    print_status("=" * 60)
    
    successful = sum(results)
    total = len(results)
    
    for i, (download, success) in enumerate(zip(downloads, results), 1):
        status = "SUCCESS" if success else "FAILED"
        print_status(f"{i}. {download['description']}: {status}")
    
    print_status("")
    print_status(f"Total: {successful}/{total} downloads successful")
    
    if successful == total:
        print_status("All downloads completed successfully!")
        print_status("Your Meeting Whisperer is ready to use!")
    else:
        print_status("Some downloads failed. Check the logs above for details.", "WARNING")
        print_status("You can retry failed downloads by running the script again.")
    
    print_status("")
    print_status("Directory structure created:")
    print_status(f"  {base_dir}/estonian-asr/          # Estonian ASR models")
    print_status(f"  {base_dir}/summarization/qwen/    # Qwen summarization model")
    print_status(f"  {base_dir}/summarization/llama/   # Llama summarization model")
    print_status(f"  {base_dir}/grammar-correction/    # Grammar correction dataset")
    print_status(f"  {base_dir}/english-asr/           # English ASR (auto-downloaded)")

if __name__ == "__main__":
    main()
