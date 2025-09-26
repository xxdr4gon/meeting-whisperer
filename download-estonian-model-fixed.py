#!/usr/bin/env python3
"""
Download Estonian Whisper model properly using the correct classes
"""
import os
from transformers import WhisperForConditionalGeneration, WhisperProcessor

def download_estonian_model():
    model_name = "TalTechNLP/whisper-large-v3-turbo-et-verbatim"
    local_dir = "./models/hf_et_model"
    
    print(f"Downloading {model_name} to {local_dir}...")
    
    # Create directory if it doesn't exist
    os.makedirs(local_dir, exist_ok=True)
    
    try:
        print("Downloading model...")
        model = WhisperForConditionalGeneration.from_pretrained(model_name)
        
        print("Downloading processor...")
        processor = WhisperProcessor.from_pretrained(model_name)
        
        print("Saving model to local directory...")
        model.save_pretrained(local_dir)
        
        print("Saving processor to local directory...")
        processor.save_pretrained(local_dir)
        
        print(f"✅ Model downloaded successfully to {local_dir}")
        print(f"📁 Contents: {os.listdir(local_dir)}")
        
        # Verify required files exist (Estonian model uses BPE tokenizer)
        required_files = ["config.json", "pytorch_model.bin", "preprocessor_config.json"]
        tokenizer_files = ["vocab.json", "merges.txt", "tokenizer_config.json"]
        all_required = required_files + tokenizer_files
        
        missing_files = [f for f in all_required if not os.path.exists(os.path.join(local_dir, f))]
        
        if missing_files:
            print(f"⚠️  Missing files: {missing_files}")
        else:
            print("✅ All required files present")
            print(f"📄 Model files: {[f for f in os.listdir(local_dir) if f.endswith('.json') or f.endswith('.bin') or f.endswith('.txt')]}")
        
    except Exception as e:
        print(f"❌ Error downloading model: {e}")
        return False
    
    return True

if __name__ == "__main__":
    download_estonian_model()
