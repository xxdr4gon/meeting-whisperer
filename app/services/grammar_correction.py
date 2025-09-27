"""
Estonian Grammar Correction Service
Uses the TalTechNLP grammar_et dataset for post-processing transcription corrections
"""

import os
import json
import re
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from difflib import SequenceMatcher
import threading

# Global variables for caching
_GRAMMAR_DATASET = None
_GRAMMAR_LOCK = None

def _load_grammar_dataset() -> Optional[Dict]:
    """Load the Estonian grammar correction dataset with caching"""
    global _GRAMMAR_DATASET, _GRAMMAR_LOCK
    
    if _GRAMMAR_LOCK is None:
        _GRAMMAR_LOCK = threading.Lock()
    
    if _GRAMMAR_DATASET is not None:
        return _GRAMMAR_DATASET
    
    with _GRAMMAR_LOCK:
        if _GRAMMAR_DATASET is not None:
            return _GRAMMAR_DATASET
        
        # Import settings here to avoid circular imports
        from ..config import settings
        
        print("Loading Estonian grammar correction dataset...")
        
        # Try to load from local directory
        grammar_dir = Path(settings.grammar_correction_dir)
        
        # Try different file formats
        dataset_files = [
            grammar_dir / "dataset.json",
            grammar_dir / "grammar_l2_train.jsonl",
            grammar_dir / "grammar_l2_test.jsonl"
        ]
        
        try:
            for dataset_file in dataset_files:
                if dataset_file.exists():
                    if dataset_file.suffix == '.json':
                        # JSON format
                        with open(dataset_file, 'r', encoding='utf-8') as f:
                            data = json.load(f)
                            _GRAMMAR_DATASET = data
                            print(f"Loaded grammar dataset from JSON: {len(data.get('original_string', []))} correction pairs")
                            return _GRAMMAR_DATASET
                    elif dataset_file.suffix == '.jsonl':
                        # JSONL format - convert to expected format
                        original_strings = []
                        correct_strings = []
                        
                        with open(dataset_file, 'r', encoding='utf-8') as f:
                            for line in f:
                                if line.strip():
                                    item = json.loads(line.strip())
                                    original_strings.append(item.get('original', ''))
                                    correct_strings.append(item.get('correct', ''))
                        
                        _GRAMMAR_DATASET = {
                            'original_string': original_strings,
                            'correct_string': correct_strings
                        }
                        print(f"Loaded grammar dataset from JSONL: {len(original_strings)} correction pairs")
                        return _GRAMMAR_DATASET
            
            print(f"Grammar dataset not found in {grammar_dir}")
            print("Please run: python download-all-models.py")
            return None
                
        except Exception as e:
            print(f"Failed to load grammar dataset: {e}")
            return None

def _find_similar_errors(text: str, dataset: Dict, threshold: float = 0.7) -> List[Tuple[str, str, float]]:
    """Find similar error patterns in the dataset - optimized for speed"""
    similar_corrections = []
    
    original_strings = dataset.get('original_string', [])
    correct_strings = dataset.get('correct_string', [])
    
    # Split text into words for faster matching
    text_words = set(text.lower().split())
    
    # Limit search to first 1000 entries for speed
    max_entries = min(1000, len(original_strings))
    
    for i in range(max_entries):
        orig = original_strings[i]
        corr = correct_strings[i]
        
        # Quick word overlap check first
        orig_words = set(orig.lower().split())
        word_overlap = len(text_words.intersection(orig_words)) / max(len(text_words), len(orig_words))
        
        if word_overlap > 0.3:  # Only do expensive similarity if there's word overlap
            # Calculate similarity only for promising candidates
            similarity = SequenceMatcher(None, text.lower(), orig.lower()).ratio()
            
            if similarity >= threshold:
                similar_corrections.append((orig, corr, similarity))
                
                # Early termination if we have enough good matches
                if len(similar_corrections) >= 10:
                    break
    
    # Sort by similarity (highest first)
    similar_corrections.sort(key=lambda x: x[2], reverse=True)
    
    return similar_corrections[:5]  # Return top 5 matches

def _apply_rule_based_corrections(text: str) -> str:
    """Apply rule-based grammar corrections based on common patterns"""
    corrected = text
    
    # Common Estonian grammar rules
    corrections = [
        # Compound words
        (r'\bkuurort linn\b', 'kuurortlinn'),
        (r'\bsuve pealinn\b', 'suvepealinn'),
        (r'\bkahekorraline\b', 'kahekorruseline'),
        (r'\braudne katus\b', 'raudkatus'),
        (r'\blisa tööd\b', 'lisatööd'),
        (r'\bvanemate kasvatamine\b', 'vanematepoolsest kasvatusest'),
        
        # Word order
        (r'\b(.*?) on (.*?) loomulikult\b', r'\1 sõltub loomulikult \2'),
        (r'\b(.*?) on (.*?) väga kerge\b', r'\1 on \2 väga kerge'),
        
        # Case corrections
        (r'\bPärnus\b', 'Pärnust'),
        (r'\bseene korjama\b', 'seeni korjama'),
        (r'\bmarju ja seene\b', 'marju ja seeni'),
        
        # Verb forms
        (r'\brentima\b', 'rentida'),
        (r'\btoimus\b', 'möödus'),
        (r'\blõpetas\b', 'on lõpetanud'),
        
        # Punctuation
        (r'\b(.*?) , (.*?)\b', r'\1, \2'),  # Fix spacing around commas
        (r'\b(.*?) \. (.*?)\b', r'\1. \2'),  # Fix spacing around periods
    ]
    
    for pattern, replacement in corrections:
        corrected = re.sub(pattern, replacement, corrected, flags=re.IGNORECASE)
    
    return corrected

def correct_estonian_grammar(text: str, use_ai_correction: bool = True) -> Dict[str, any]:
    """
    Correct Estonian grammar in transcribed text
    
    Args:
        text: The text to correct
        use_ai_correction: Whether to use AI-based correction (requires dataset)
    
    Returns:
        Dict with corrected text and correction details
    """
    if not text or not text.strip():
        return {
            "original_text": text,
            "corrected_text": text,
            "corrections_applied": 0,
            "correction_method": "none",
            "confidence": 0.0
        }
    
    print(f"Applying Estonian grammar correction to {len(text)} characters...")
    
    # Start with rule-based corrections
    corrected_text = _apply_rule_based_corrections(text)
    corrections_applied = 0
    correction_method = "rule_based"
    confidence = 0.6  # Rule-based corrections have moderate confidence
    
    # Apply AI-based corrections if enabled and dataset is available
    # Skip AI correction for very long texts to maintain speed
    if use_ai_correction and len(text) < 1000:
        dataset = _load_grammar_dataset()
        if dataset:
            print("Applying AI-based grammar corrections...")
            
            # Find similar error patterns
            similar_errors = _find_similar_errors(corrected_text, dataset, threshold=0.6)
            
            if similar_errors:
                print(f"Found {len(similar_errors)} similar error patterns")
                
                # Apply the most similar correction
                best_match = similar_errors[0]
                original_pattern, correct_pattern, similarity = best_match
                
                if similarity > 0.8:  # High confidence match
                    # Apply the correction
                    corrected_text = corrected_text.replace(original_pattern, correct_pattern)
                    corrections_applied += 1
                    correction_method = "ai_based"
                    confidence = similarity
                    print(f"Applied AI correction: '{original_pattern}' → '{correct_pattern}'")
                else:
                    print(f"Similarity too low ({similarity:.2f}) for AI correction")
            else:
                print("No similar error patterns found in dataset")
        else:
            print("Grammar dataset not available, using rule-based corrections only")
    
    # Count actual changes
    if corrected_text != text:
        corrections_applied = len([i for i, (a, b) in enumerate(zip(text, corrected_text)) if a != b])
    
    result = {
        "original_text": text,
        "corrected_text": corrected_text,
        "corrections_applied": corrections_applied,
        "correction_method": correction_method,
        "confidence": confidence
    }
    
    print(f"Grammar correction complete: {corrections_applied} corrections applied")
    print(f"Method: {correction_method}, Confidence: {confidence:.2f}")
    
    return result

def get_grammar_correction_stats() -> Dict[str, any]:
    """Get statistics about the grammar correction dataset"""
    dataset = _load_grammar_dataset()
    
    if not dataset:
        return {
            "dataset_loaded": False,
            "total_corrections": 0,
            "dataset_size": "0 MB"
        }
    
    original_strings = dataset.get('original_string', [])
    correct_strings = dataset.get('correct_string', [])
    
    return {
        "dataset_loaded": True,
        "total_corrections": len(original_strings),
        "dataset_size": f"{len(str(dataset)) / 1024 / 1024:.1f} MB",
        "sample_corrections": list(zip(original_strings[:3], correct_strings[:3]))
    }
