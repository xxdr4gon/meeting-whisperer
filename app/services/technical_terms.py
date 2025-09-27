"""
Technical terms support for better transcription accuracy
"""

import re
from typing import Dict, List

def _get_technical_terms_prompt() -> str:
    """Get technical terms prompt for better transcription accuracy"""
    return """This is a technical meeting with common terms including:
    - Software: API, database, server, client, frontend, backend, framework, library
    - Business: revenue, profit, budget, strategy, marketing, sales, customer
    - Technology: cloud, AI, machine learning, data, analytics, security, encryption
    - General: meeting, discussion, presentation, project, team, deadline, milestone
    - Estonian: tarkvara, andmebaas, server, klient, raamistik, teek, pilv, andmeteadus"""

def _apply_technical_corrections(text: str) -> str:
    """Apply technical terms corrections to transcribed text"""
    if not text:
        return text
    
    # Common technical term corrections
    corrections = {
        # Software terms
        r'\bapi\b': 'API',
        r'\bdatabase\b': 'database',
        r'\bserver\b': 'server',
        r'\bclient\b': 'client',
        r'\bfrontend\b': 'frontend',
        r'\bbackend\b': 'backend',
        r'\bframework\b': 'framework',
        r'\blibrary\b': 'library',
        
        # Business terms
        r'\brevenue\b': 'revenue',
        r'\bprofit\b': 'profit',
        r'\bbudget\b': 'budget',
        r'\bstrategy\b': 'strategy',
        r'\bmarketing\b': 'marketing',
        r'\bsales\b': 'sales',
        r'\bcustomer\b': 'customer',
        
        # Technology terms
        r'\bcloud\b': 'cloud',
        r'\bai\b': 'AI',
        r'\bmachine learning\b': 'machine learning',
        r'\bdata\b': 'data',
        r'\banalytics\b': 'analytics',
        r'\bsecurity\b': 'security',
        r'\bencryption\b': 'encryption',
        
        # Common meeting terms
        r'\bmeeting\b': 'meeting',
        r'\bdiscussion\b': 'discussion',
        r'\bpresentation\b': 'presentation',
        r'\bproject\b': 'project',
        r'\bteam\b': 'team',
        r'\bdeadline\b': 'deadline',
        r'\bmilestone\b': 'milestone',
        
        # Estonian technical terms
        r'\btarkvara\b': 'tarkvara',
        r'\bandmebaas\b': 'andmebaas',
        r'\bserver\b': 'server',
        r'\bklient\b': 'klient',
        r'\braamistik\b': 'raamistik',
        r'\bteek\b': 'teek',
        r'\bpilv\b': 'pilv',
        r'\bandmeteadus\b': 'andmeteadus',
    }
    
    corrected_text = text
    for pattern, replacement in corrections.items():
        corrected_text = re.sub(pattern, replacement, corrected_text, flags=re.IGNORECASE)
    
    return corrected_text

def get_technical_terms_stats() -> Dict[str, int]:
    """Get statistics about technical terms usage"""
    return {
        "total_corrections": len(_get_technical_terms_prompt().split()),
        "categories": ["software", "business", "technology", "meeting", "estonian"]
    }
