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
            model_name = "Qwen/Qwen2.5-3B-Instruct"  # Use general multilingual model, not Sorbian-specific
            local_model_path = f"{summarization_dir}/qwen"
        
        # Verify model exists (no fallback since model will always be available)
        if not os.path.exists(local_model_path) or not os.listdir(local_model_path):
            print(f"Model not found at {local_model_path}")
            print("Please run: python download-all-models.py")
            _ESTONIAN_SUMMARIZER = None
            _ESTONIAN_TOKENIZER = None
            return None, None
        
        print(f"Model name: {model_name}")
        print(f"Local model path: {local_model_path}")
        print(f"Path exists: {os.path.exists(local_model_path)}")
        if os.path.exists(local_model_path):
            print(f"Directory contents: {os.listdir(local_model_path)}")
        
        try:
            # Try to load from local directory first
            if os.path.exists(local_model_path) and os.listdir(local_model_path):
                print(f"Loading from local directory: {local_model_path}")
                
                # Check if essential model files exist and are not corrupted
                essential_files = ["config.json", "tokenizer.json", "model.safetensors.index.json"]
                missing_files = [f for f in essential_files if not os.path.exists(os.path.join(local_model_path, f))]
                if missing_files:
                    print(f"Missing essential files: {missing_files}")
                    print("AI summarization requires complete model files for airgapped operation")
                    _ESTONIAN_SUMMARIZER = None
                    _ESTONIAN_TOKENIZER = None
                    return None, None
                
                # Check file sizes to detect corruption
                safetensors_files = [f for f in os.listdir(local_model_path) if f.startswith("model-") and f.endswith(".safetensors")]
                if not safetensors_files:
                    print("No safetensors model files found")
                    _ESTONIAN_SUMMARIZER = None
                    _ESTONIAN_TOKENIZER = None
                    return None, None
                
                # Check if model files are reasonably sized (not empty)
                for safetensors_file in safetensors_files:
                    file_path = os.path.join(local_model_path, safetensors_file)
                    file_size = os.path.getsize(file_path)
                    if file_size < 1024 * 1024:  # Less than 1MB is suspicious
                        print(f"Model file {safetensors_file} is too small ({file_size} bytes), may be corrupted")
                        _ESTONIAN_SUMMARIZER = None
                        _ESTONIAN_TOKENIZER = None
                        return None, None
                    print(f"Model file {safetensors_file}: {file_size / (1024*1024):.1f}MB")
                
                # Try to validate safetensors files by attempting to read them
                print("Validating safetensors files...")
                try:
                    import safetensors
                    for safetensors_file in safetensors_files:
                        file_path = os.path.join(local_model_path, safetensors_file)
                        with open(file_path, "rb") as f:
                            # Try to read the header to validate file integrity
                            header = f.read(8)
                            if len(header) < 8:
                                print(f"Model file {safetensors_file} appears corrupted (header too short)")
                                _ESTONIAN_SUMMARIZER = None
                                _ESTONIAN_TOKENIZER = None
                                return None, None
                    print("Safetensors files validation passed")
                except Exception as validation_error:
                    print(f"Safetensors validation failed: {validation_error}")
                    print("Model files may be corrupted, falling back to rule-based summarization")
                    _ESTONIAN_SUMMARIZER = None
                    _ESTONIAN_TOKENIZER = None
                    return None, None
                
                print("Loading tokenizer...")
                _ESTONIAN_TOKENIZER = AutoTokenizer.from_pretrained(
                    local_model_path,
                    trust_remote_code=True,
                    local_files_only=True,  # Airgapped mode - only local files
                    use_fast=True  # Use fast tokenizer if available
                )
                
                # Set pad token if needed to avoid attention mask warnings
                if _ESTONIAN_TOKENIZER.pad_token is None:
                    _ESTONIAN_TOKENIZER.pad_token = _ESTONIAN_TOKENIZER.eos_token
                    print("Set pad_token to eos_token to avoid attention mask warnings")
                
                print("Tokenizer loaded successfully")
                
                print("Loading model...")
                # Aggressive GPU memory cleanup
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    torch.cuda.synchronize()
                    # Force garbage collection
                    import gc
                    gc.collect()
                    # Additional cleanup - reset peak memory stats
                    torch.cuda.reset_peak_memory_stats()
                    print(f"GPU memory before loading: {torch.cuda.memory_allocated() / 1024**3:.2f}GB / {torch.cuda.memory_reserved() / 1024**3:.2f}GB")
                
                # Try GPU first, fallback to CPU if needed
                print("Loading model on GPU with memory optimization...")
                try:
                    # Add a timeout mechanism to prevent hanging
                    import signal
                    import time
                    
                    def timeout_handler(signum, frame):
                        raise TimeoutError("Model loading timed out")
                    
                    # Set a 5-minute timeout for model loading
                    signal.signal(signal.SIGALRM, timeout_handler)
                    signal.alarm(300)  # 5 minutes timeout
                    # Check if GPU has enough memory (need ~6GB for Qwen 3B)
                    if torch.cuda.is_available():
                        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
                        print(f"GPU memory available: {gpu_memory:.1f}GB")
                        
                        if gpu_memory >= 7.0:  # Need at least 7GB for 3B model
                            print("Loading model on GPU with conservative settings...")
                            try:
                                _ESTONIAN_SUMMARIZER = AutoModelForCausalLM.from_pretrained(
                                    local_model_path,
                                    torch_dtype=torch.float16,  # Use float16 for GPU efficiency
                                    device_map="auto",  # Let transformers choose the best device mapping
                                    trust_remote_code=True,
                                    low_cpu_mem_usage=True,
                                    local_files_only=True,  # Airgapped mode - only local files
                                    use_safetensors=True,  # Prefer safetensors format
                                    max_memory={0: "5GB"}  # More conservative memory limit
                                )
                                print("Model loaded successfully on GPU")
                                print(f"GPU memory after AI model loading: {torch.cuda.memory_allocated() / 1024**3:.2f}GB / {torch.cuda.memory_reserved() / 1024**3:.2f}GB")
                            except Exception as gpu_load_error:
                                print(f"GPU loading failed: {gpu_load_error}")
                                print("Trying CPU loading as fallback...")
                                raise Exception("GPU loading failed")
                        else:
                            print(f"GPU memory insufficient ({gpu_memory:.1f}GB < 7GB), using CPU")
                            raise Exception("Insufficient GPU memory")
                    else:
                        print("CUDA not available, using CPU")
                        raise Exception("CUDA not available")
                        
                except Exception as gpu_error:
                    print(f"GPU loading failed: {gpu_error}")
                    print("Falling back to CPU loading...")
                    try:
                        _ESTONIAN_SUMMARIZER = AutoModelForCausalLM.from_pretrained(
                            local_model_path,
                            torch_dtype=torch.float32,  # Use float32 for CPU stability
                            device_map="cpu",
                            trust_remote_code=True,
                            low_cpu_mem_usage=True,
                            local_files_only=True,  # Airgapped mode - only local files
                            use_safetensors=True,  # Prefer safetensors format
                            max_memory=None  # No memory limits on CPU
                        )
                        print("Model loaded successfully on CPU")
                    except Exception as cpu_error:
                        print(f"CPU loading also failed: {cpu_error}")
                        print("Model files may be corrupted or incompatible")
                        _ESTONIAN_SUMMARIZER = None
                        _ESTONIAN_TOKENIZER = None
                        return None, None
                finally:
                    # Cancel the timeout
                    signal.alarm(0)
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
        print(f"Starting AI summarization for text length: {len(text)}")
        model, tokenizer = _get_estonian_summarizer()
        
        if model is None or tokenizer is None:
            print("Estonian AI model not available, using fallback")
            return _summarize_estonian_fallback(text, max_length)
        
        # If text is very long, truncate it to avoid token limit issues
        if len(text) > 8000:  # Roughly 2000 tokens
            print(f"Text too long ({len(text)} chars), truncating to 8000 chars")
            text = text[:8000] + "..."
        
        # Prepare the prompt for summarization
        prompt = f"""Kokkuvõte järgmisest tekstist:

{text}

Kokkuvõte:"""
        
        # Tokenize input with proper attention mask handling
        # Truncate input if it's too long to avoid max_length issues
        inputs = tokenizer(prompt, return_tensors="pt", max_length=1024, truncation=True, 
                          padding=True, return_attention_mask=True)
        
        input_length = len(inputs["input_ids"][0])
        print(f"Input token length: {input_length}")
        
        # Move all input tensors to the same device as the model
        model_device = next(model.parameters()).device
        inputs = {k: v.to(model_device) for k, v in inputs.items()}
        
        # Generate summary
        print(f"Generating summary with max_new_tokens={max_length}")
        with torch.no_grad():
            try:
                # Add timeout to prevent hanging
                import signal
                import time
                
                def timeout_handler(signum, frame):
                    raise TimeoutError("Generation timed out")
                
                # Set a 2-minute timeout for generation
                signal.signal(signal.SIGALRM, timeout_handler)
                signal.alarm(120)  # 2 minutes timeout
                
                outputs = model.generate(
                    inputs["input_ids"],
                    attention_mask=inputs["attention_mask"],
                    max_new_tokens=min(max_length, 400),  # Increased from 200 to 400
                    min_new_tokens=50,  # Increased minimum tokens
                    num_beams=4,  # Beam search for better quality
                    early_stopping=True,  # Stop when EOS is generated
                    do_sample=False,  # Disable sampling for beam search
                    repetition_penalty=1.1,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                    max_time=60  # Additional timeout parameter
                )
                signal.alarm(0)  # Cancel timeout
                print("AI generation completed successfully")
            except TimeoutError:
                print("Generation timed out, using fallback")
                signal.alarm(0)  # Cancel timeout
                return _summarize_estonian_fallback(text, max_length)
            except Exception as e:
                print(f"Error during AI generation: {e}")
                signal.alarm(0)  # Cancel timeout
                # Fallback to simpler generation
                print("Trying fallback generation...")
                try:
                    outputs = model.generate(
                        inputs["input_ids"],
                        attention_mask=inputs["attention_mask"],
                        max_new_tokens=min(max_length, 200),  # Increased from 100 to 200
                        min_new_tokens=30,  # Increased minimum tokens
                        num_beams=2,  # Fewer beams for speed
                        early_stopping=True,  # Stop when EOS is generated
                        do_sample=False,  # Disable sampling for stability
                        pad_token_id=tokenizer.pad_token_id,
                        eos_token_id=tokenizer.eos_token_id
                    )
                    print("Fallback generation completed")
                except Exception as fallback_error:
                    print(f"Fallback generation also failed: {fallback_error}")
                    return _summarize_estonian_fallback(text, max_length)
        
        # Decode the generated text
        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        print(f"Generated text length: {len(generated_text)}")
        print(f"Generated text preview: {generated_text[:200]}...")
        
        # Extract only the summary part (after "Kokkuvõte:")
        if "Kokkuvõte:" in generated_text:
            # Find all occurrences of "Kokkuvõte:" and take the last one (most likely the AI's response)
            parts = generated_text.split("Kokkuvõte:")
            if len(parts) > 1:
                summary = parts[-1].strip()  # Take the last part after the last "Kokkuvõte:"
                print("Found 'Kokkuvõte:' in generated text")
            else:
                summary = generated_text.strip()
                print("Found 'Kokkuvõte:' but extraction failed, using full text")
        elif "Kokkuvõte" in generated_text:
            parts = generated_text.split("Kokkuvõte")
            if len(parts) > 1:
                summary = parts[-1].strip()
                print("Found 'Kokkuvõte' in generated text")
            else:
                summary = generated_text.strip()
                print("Found 'Kokkuvõte' but extraction failed, using full text")
        else:
            # If no "Kokkuvõte" found, try to extract after the prompt
            prompt_end = prompt.rfind("Kokkuvõte:")
            if prompt_end != -1:
                summary = generated_text[prompt_end + len("Kokkuvõte:"):].strip()
                print("Extracted after prompt 'Kokkuvõte:'")
            else:
                # Last resort: use the whole generated text
                summary = generated_text.strip()
                print("Using entire generated text as summary")
        
        print(f"Extracted summary length: {len(summary)}")
        print(f"Extracted summary preview: {summary[:200]}...")
        
        # Clean up the summary
        summary = _clean_summary(summary)
        print(f"Cleaned summary length: {len(summary)}")
        
        if not summary or len(summary) < 20:  # Lowered from 50 to 20
            print(f"AI summary too short ({len(summary)} chars), using fallback")
            return _summarize_estonian_fallback(text, max_length)
        
        print(f"AI summarization completed successfully, returning summary of length: {len(summary)}")
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
    
    # Remove any remaining prompt text that might have been generated
    if "Kokkuvõte järgmisest tekstist:" in summary:
        summary = summary.split("Kokkuvõte järgmisest tekstist:")[-1].strip()
    
    # Remove any text before the actual summary content
    lines = summary.split('\n')
    # Find the first line that looks like actual content (not empty or just punctuation)
    content_lines = []
    for line in lines:
        line = line.strip()
        if line and len(line) > 3 and not line.startswith(('Kokkuvõte', 'Summary', 'Järeldus', 'Conclusion')):
            content_lines.append(line)
    
    if content_lines:
        summary = ' '.join(content_lines)
    
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


def unload_ai_summarizer():
    """Unload AI summarizer to free GPU memory"""
    global _ESTONIAN_SUMMARIZER, _ESTONIAN_TOKENIZER
    if _ESTONIAN_SUMMARIZER is not None:
        del _ESTONIAN_SUMMARIZER
        _ESTONIAN_SUMMARIZER = None
    if _ESTONIAN_TOKENIZER is not None:
        del _ESTONIAN_TOKENIZER
        _ESTONIAN_TOKENIZER = None
    import torch
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    print("AI summarizer unloaded from GPU memory")
