"""
AI-powered summarization using Estonian language models from TartuNLP
"""

import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from typing import Dict, List, Optional
import logging

logger = logging.getLogger(__name__)

# Global variables for model caching
_ESTONIAN_SUMMARIZER = None
_ESTONIAN_TOKENIZER = None
_ESTONIAN_LOCK = None

def _get_estonian_summarizer():
    """Get or create Estonian summarization model with caching"""
    global _ESTONIAN_SUMMARIZER, _ESTONIAN_TOKENIZER, _ESTONIAN_LOCK
    
    if _ESTONIAN_LOCK is None:
        import threading
        _ESTONIAN_LOCK = threading.Lock()
    
    if _ESTONIAN_SUMMARIZER is not None:
        return _ESTONIAN_SUMMARIZER, _ESTONIAN_TOKENIZER
    
    with _ESTONIAN_LOCK:
        if _ESTONIAN_SUMMARIZER is not None:
            return _ESTONIAN_SUMMARIZER, _ESTONIAN_TOKENIZER
        
        # Import settings here to avoid circular imports
        from ..config import settings
        
        print("Loading Estonian AI summarization model...")
        
        # Determine model path based on configuration
        summarization_dir = settings.summarization_dir
        model_type = settings.ai_summarization_model
        
        print(f"AI Summarization model type: {model_type}")
        print(f"Summarization directory: {summarization_dir}")
        print(f"Settings object: {settings}")
        print(f"AI_SUMMARIZATION_MODEL env var: {os.environ.get('AI_SUMMARIZATION_MODEL', 'Not set')}")
        
        if model_type == "llama":
            model_name = "tartuNLP/llama-estllm-protype-0825"
            local_model_path = f"{summarization_dir}/llama"
        else:  # qwen
            model_name = "tartuNLP/Qwen2.5-3B-Instruct-hsb-dsb"
            local_model_path = f"{summarization_dir}/qwen"
        
        # Check if the selected model exists, if not try the other one
        if not os.path.exists(local_model_path) or not os.listdir(local_model_path):
            print(f"Selected model not found at {local_model_path}, trying alternative...")
            if model_type == "llama":
                # Try qwen instead
                model_name = "tartuNLP/Qwen2.5-3B-Instruct-hsb-dsb"
                local_model_path = f"{summarization_dir}/qwen"
                model_type = "qwen"
            else:
                # Try llama instead
                model_name = "tartuNLP/llama-estllm-protype-0825"
                local_model_path = f"{summarization_dir}/llama"
                model_type = "llama"
            print(f"Switched to {model_type} model: {local_model_path}")
        
        print(f"Model name: {model_name}")
        print(f"Local model path: {local_model_path}")
        print(f"Path exists: {os.path.exists(local_model_path)}")
        if os.path.exists(local_model_path):
            print(f"Directory contents: {os.listdir(local_model_path)}")
        
        try:
            # Try to load from local directory first
            if os.path.exists(local_model_path) and os.listdir(local_model_path):
                print(f"Loading from local directory: {local_model_path}")
                
                print("Loading tokenizer...")
                _ESTONIAN_TOKENIZER = AutoTokenizer.from_pretrained(
                    local_model_path,
                    trust_remote_code=True,
                    local_files_only=True,  # Airgapped mode - only local files
                    use_fast=True  # Use fast tokenizer if available
                )
                print("Tokenizer loaded successfully")
                
                print("Loading model...")
                _ESTONIAN_SUMMARIZER = AutoModelForCausalLM.from_pretrained(
                    local_model_path,
                    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                    device_map="auto" if torch.cuda.is_available() else "cpu",
                    trust_remote_code=True,
                    low_cpu_mem_usage=True,
                    local_files_only=True,  # Airgapped mode - only local files
                    use_safetensors=True  # Prefer safetensors format
                )
                print("Model loaded successfully")
                
                print("Estonian AI summarization model loaded from local directory")
                
            else:
                print(f"Local model not found at {local_model_path}")
                print("AI summarization requires pre-downloaded models for airgapped operation")
                print(f"Please run: python download-all-models.py --skip-qwen" if model_type == "llama" else f"Please run: python download-all-models.py --skip-llama")
                _ESTONIAN_SUMMARIZER = None
                _ESTONIAN_TOKENIZER = None
                return None, None
            
        except Exception as e:
            print(f"Failed to load Estonian model: {e}")
            print("AI summarization requires pre-downloaded models for airgapped operation")
            print(f"Please run: python download-all-models.py --skip-qwen" if model_type == "llama" else f"Please run: python download-all-models.py --skip-llama")
            _ESTONIAN_SUMMARIZER = None
            _ESTONIAN_TOKENIZER = None
        
        return _ESTONIAN_SUMMARIZER, _ESTONIAN_TOKENIZER


def ai_summarize_text(text: str, language: str = "et", max_length: int = 500) -> str:
    """
    AI-powered summarization using Estonian language models
    
    Args:
        text: Text to summarize
        language: Language code ('et' for Estonian, 'en' for English)
        max_length: Maximum length of summary
    
    Returns:
        Generated summary
    """
    if not text or not text.strip():
        return "No content to summarize."
    
    # For Estonian, use the AI model
    if language == "et":
        return _summarize_with_estonian_ai(text, max_length)
    else:
        # For English, use a simpler approach or fallback to rule-based
        return _summarize_english_simple(text, max_length)


def _summarize_with_estonian_ai(text: str, max_length: int) -> str:
    """Summarize using Estonian AI model"""
    try:
        model, tokenizer = _get_estonian_summarizer()
        
        if model is None or tokenizer is None:
            print("Estonian AI model not available, using fallback")
            return _summarize_estonian_fallback(text, max_length)
        
        # Prepare the prompt for summarization
        prompt = f"""Kokkuvõte järgmisest tekstist:

{text}

Kokkuvõte:"""
        
        # Tokenize input
        inputs = tokenizer.encode(prompt, return_tensors="pt", max_length=2048, truncation=True)
        
        # Generate summary
        with torch.no_grad():
            outputs = model.generate(
                inputs,
                max_new_tokens=max_length,
                temperature=0.7,
                do_sample=True,
                top_p=0.9,
                repetition_penalty=1.1,
                pad_token_id=tokenizer.eos_token_id
            )
        
        # Decode the generated text
        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Extract only the summary part (after "Kokkuvõte:")
        if "Kokkuvõte:" in generated_text:
            summary = generated_text.split("Kokkuvõte:")[-1].strip()
        else:
            summary = generated_text[len(prompt):].strip()
        
        # Clean up the summary
        summary = _clean_summary(summary)
        
        if not summary or len(summary) < 50:
            print("AI summary too short, using fallback")
            return _summarize_estonian_fallback(text, max_length)
        
        return summary
        
    except Exception as e:
        print(f"Error in AI summarization: {e}")
        return _summarize_estonian_fallback(text, max_length)


def _summarize_estonian_fallback(text: str, max_length: int) -> str:
    """Fallback summarization for Estonian using rule-based approach"""
    sentences = text.split('.')
    sentences = [s.strip() for s in sentences if s.strip()]
    
    if len(sentences) <= 3:
        return text
    
    # Take the most informative sentences
    key_sentences = sentences[:min(5, len(sentences))]
    
    # Look for sentences with important keywords
    important_sentences = []
    for sentence in key_sentences:
        if any(word in sentence.lower() for word in ['tähtis', 'peamine', 'oluline', 'kokkuvõte', 'järeldus']):
            important_sentences.append(sentence)
    
    if important_sentences:
        return " • ".join(important_sentences[:3])
    else:
        return " • ".join(key_sentences[:3])


def _summarize_english_simple(text: str, max_length: int) -> str:
    """Simple English summarization"""
    sentences = text.split('.')
    sentences = [s.strip() for s in sentences if s.strip()]
    
    if len(sentences) <= 3:
        return text
    
    # Take the most informative sentences
    key_sentences = sentences[:min(5, len(sentences))]
    
    # Look for sentences with important keywords
    important_sentences = []
    for sentence in key_sentences:
        if any(word in sentence.lower() for word in ['important', 'key', 'main', 'summary', 'conclusion']):
            important_sentences.append(sentence)
    
    if important_sentences:
        return " • ".join(important_sentences[:3])
    else:
        return " • ".join(key_sentences[:3])


def _clean_summary(summary: str) -> str:
    """Clean up the generated summary"""
    # Remove extra whitespace
    summary = " ".join(summary.split())
    
    # Remove common artifacts
    artifacts = [
        "Kokkuvõte:",
        "Summary:",
        "Kokkuvõte",
        "Summary",
        "Järeldus:",
        "Conclusion:",
        "Järeldus",
        "Conclusion"
    ]
    
    for artifact in artifacts:
        if summary.startswith(artifact):
            summary = summary[len(artifact):].strip()
    
    # Ensure it ends with proper punctuation
    if summary and not summary.endswith(('.', '!', '?')):
        summary += "."
    
    return summary


def create_ai_meeting_summary(transcript: str, language: str = "et") -> str:
    """
    Create a comprehensive meeting summary using AI
    
    Args:
        transcript: Full transcript text
        language: Language code ('et' for Estonian, 'en' for English)
    
    Returns:
        Formatted meeting summary
    """
    if not transcript or not transcript.strip():
        return "No content to summarize."
    
    # Generate AI summary
    ai_summary = ai_summarize_text(transcript, language, max_length=300)
    
    # Count statements
    sentences = transcript.split('.')
    statement_count = len([s for s in sentences if s.strip()])
    
    # Create formatted summary
    if language == "et":
        header = "📋 KOKKUVÕTE"
        overview = f"Kokku arutelupunkte: {statement_count} väidet"
        ai_section = "🤖 AI KOKKUVÕTE"
    else:
        header = "📋 MEETING SUMMARY"
        overview = f"Total discussion points: {statement_count} statements"
        ai_section = "🤖 AI SUMMARY"
    
    summary_parts = [
        header,
        "=" * 50,
        f"\n📊 OVERVIEW",
        overview,
        f"\n{ai_section}",
        ai_summary,
        f"\n" + "=" * 50,
        "Generated by Meeting Whisperer AI"
    ]
    
    return "\n".join(summary_parts)
