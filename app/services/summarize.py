import re
from typing import List, Dict, Any


def summarize_text(text: str, max_sentences: int = 7) -> str:
    """Enhanced memo-style summarization with structured analysis."""
    if not text or not text.strip():
        return "No content to summarize."
    
    # Split into sentences
    sentences = re.split(r'[.!?]+', text)
    sentences = [s.strip() for s in sentences if s.strip()]
    
    if len(sentences) <= 3:
        return text
    
    # Analyze content for memo structure
    analysis = analyze_content(sentences)
    
    # Generate intelligent summary instead of just listing sentences
    summary = generate_intelligent_summary(analysis, sentences)
    
    return summary


def analyze_content(sentences: List[str]) -> Dict[str, Any]:
    """Analyze content to identify key themes, problems, and solutions."""
    problems = []
    solutions = []
    decisions = []
    key_points = []
    action_items = []
    events = []
    people = []
    
    # Enhanced keywords for different categories
    problem_keywords = ['problem', 'issue', 'challenge', 'difficulty', 'concern', 'trouble', 'error', 'bug', 'fault', 'vägivald', 'konflikt', 'probleem']
    solution_keywords = ['solution', 'fix', 'resolve', 'address', 'implement', 'create', 'develop', 'build', 'solve', 'lahendus', 'parandus']
    decision_keywords = ['decide', 'decision', 'choose', 'select', 'agree', 'approve', 'reject', 'conclude', 'determine', 'otsustada', 'kokkulepe']
    action_keywords = ['action', 'task', 'todo', 'next', 'follow', 'implement', 'execute', 'do', 'complete', 'finish', 'tegevus', 'ülesanne']
    event_keywords = ['happened', 'occurred', 'took place', 'event', 'incident', 'situation', 'leidsid aset', 'toimus', 'sündmus']
    people_keywords = ['person', 'people', 'man', 'woman', 'individual', 'inimene', 'isik', 'mees', 'naine']
    
    for sentence in sentences:
        sentence_lower = sentence.lower()
        
        # Check for problems
        if any(keyword in sentence_lower for keyword in problem_keywords):
            problems.append(sentence)
        # Check for solutions
        elif any(keyword in sentence_lower for keyword in solution_keywords):
            solutions.append(sentence)
        # Check for decisions
        elif any(keyword in sentence_lower for keyword in decision_keywords):
            decisions.append(sentence)
        # Check for action items
        elif any(keyword in sentence_lower for keyword in action_keywords):
            action_items.append(sentence)
        # Check for events
        elif any(keyword in sentence_lower for keyword in event_keywords):
            events.append(sentence)
        # Check for people
        elif any(keyword in sentence_lower for keyword in people_keywords):
            people.append(sentence)
        # Other important sentences (longer, with specific patterns)
        elif len(sentence.split()) > 8 and any(word in sentence_lower for word in ['important', 'key', 'main', 'primary', 'critical', 'essential', 'tähtis', 'peamine']):
            key_points.append(sentence)
    
    return {
        'problems': problems[:3],  # Limit to top 3
        'solutions': solutions[:3],
        'decisions': decisions[:2],
        'key_points': key_points[:3],
        'action_items': action_items[:3],
        'events': events[:3],
        'people': people[:3],
        'total_sentences': len(sentences)
    }


def generate_memo(analysis: Dict[str, Any], sentences: List[str]) -> str:
    """Generate a structured memo from the analysis."""
    memo_parts = []
    
    # Header
    memo_parts.append("MEETING SUMMARY")
    memo_parts.append("=" * 50)
    
    # Overview
    memo_parts.append(f"\nOVERVIEW")
    memo_parts.append(f"Total discussion points: {analysis['total_sentences']} statements")
    
    # Problems identified
    if analysis['problems']:
        memo_parts.append(f"\nPROBLEMS IDENTIFIED")
        for i, problem in enumerate(analysis['problems'], 1):
            memo_parts.append(f"{i}. {problem}")
    
    # Solutions discussed
    if analysis['solutions']:
        memo_parts.append(f"\nSOLUTIONS DISCUSSED")
        for i, solution in enumerate(analysis['solutions'], 1):
            memo_parts.append(f"{i}. {solution}")
    
    # Decisions made
    if analysis['decisions']:
        memo_parts.append(f"\nDECISIONS MADE")
        for i, decision in enumerate(analysis['decisions'], 1):
            memo_parts.append(f"{i}. {decision}")
    
    # Key points
    if analysis['key_points']:
        memo_parts.append(f"\nKEY POINTS")
        for i, point in enumerate(analysis['key_points'], 1):
            memo_parts.append(f"{i}. {point}")
    
    # Action items
    if analysis['action_items']:
        memo_parts.append(f"\nACTION ITEMS")
        for i, action in enumerate(analysis['action_items'], 1):
            memo_parts.append(f"{i}. {action}")
    
    # Events mentioned
    if analysis['events']:
        memo_parts.append(f"\nEVENTS MENTIONED")
        for i, event in enumerate(analysis['events'], 1):
            memo_parts.append(f"{i}. {event}")
    
    # People involved
    if analysis['people']:
        memo_parts.append(f"\nPEOPLE INVOLVED")
        for i, person in enumerate(analysis['people'], 1):
            memo_parts.append(f"{i}. {person}")
    
    # If no structured content found, provide a general summary
    if not any([analysis['problems'], analysis['solutions'], analysis['decisions'], analysis['key_points'], analysis['action_items'], analysis['events'], analysis['people']]):
        memo_parts.append(f"\nGENERAL SUMMARY")
        # Take first few sentences as general summary
        general_summary = sentences[:min(5, len(sentences))]
        for i, sentence in enumerate(general_summary, 1):
            memo_parts.append(f"{i}. {sentence}")
    
    memo_parts.append(f"\n" + "=" * 50)
    memo_parts.append("Generated by Meeting Whisperer")
    
    return "\n".join(memo_parts)


def generate_intelligent_summary(analysis: Dict[str, Any], sentences: List[str]) -> str:
    """Generate an intelligent summary that synthesizes information rather than just listing sentences."""
    summary_parts = []
    
    # Header
    summary_parts.append("MEETING SUMMARY")
    summary_parts.append("=" * 50)
    
    # Overview
    summary_parts.append(f"\nOVERVIEW")
    summary_parts.append(f"Total discussion points: {analysis['total_sentences']} statements")
    
    # Create intelligent summaries for each category
    if analysis['events']:
        summary_parts.append(f"\nKEY EVENTS")
        event_summary = synthesize_events(analysis['events'])
        summary_parts.append(event_summary)
    
    if analysis['people']:
        summary_parts.append(f"\nPEOPLE INVOLVED")
        people_summary = synthesize_people(analysis['people'])
        summary_parts.append(people_summary)
    
    if analysis['problems']:
        summary_parts.append(f"\nISSUES IDENTIFIED")
        problem_summary = synthesize_problems(analysis['problems'])
        summary_parts.append(problem_summary)
    
    if analysis['solutions']:
        summary_parts.append(f"\nSOLUTIONS DISCUSSED")
        solution_summary = synthesize_solutions(analysis['solutions'])
        summary_parts.append(solution_summary)
    
    if analysis['decisions']:
        summary_parts.append(f"\nDECISIONS MADE")
        decision_summary = synthesize_decisions(analysis['decisions'])
        summary_parts.append(decision_summary)
    
    if analysis['action_items']:
        summary_parts.append(f"\nACTION ITEMS")
        action_summary = synthesize_actions(analysis['action_items'])
        summary_parts.append(action_summary)
    
    # If no structured content found, create a general summary
    if not any([analysis['events'], analysis['people'], analysis['problems'], analysis['solutions'], analysis['decisions'], analysis['action_items']]):
        summary_parts.append(f"\nGENERAL SUMMARY")
        general_summary = create_general_summary(sentences)
        summary_parts.append(general_summary)
    
    summary_parts.append(f"\n" + "=" * 50)
    summary_parts.append("Generated by Meeting Whisperer")
    
    return "\n".join(summary_parts)


def synthesize_events(events: List[str]) -> str:
    """Synthesize event information into a coherent summary."""
    if not events:
        return "No specific events mentioned."
    
    # Look for common themes in events
    themes = []
    for event in events:
        if any(word in event.lower() for word in ['incident', 'accident', 'sündmus', 'juhtum']):
            themes.append("Incident occurred")
        elif any(word in event.lower() for word in ['meeting', 'discussion', 'kohtumine', 'arutelu']):
            themes.append("Meeting or discussion took place")
        elif any(word in event.lower() for word in ['arrest', 'detention', 'vahistamine', 'kinni']):
            themes.append("Arrest or detention involved")
    
    if themes:
        return f"• {', '.join(set(themes))}\n• Key details: {events[0][:100]}..."
    else:
        return f"• {events[0][:150]}..."


def synthesize_people(people: List[str]) -> str:
    """Synthesize people information into a coherent summary."""
    if not people:
        return "No specific people mentioned."
    
    # Extract key information about people
    key_info = []
    details = []
    
    for person in people:
        person_lower = person.lower()
        
        # Look for specific roles and information
        if any(word in person_lower for word in ['arrest', 'arrested', 'suspect', 'detained', 'vahistamine', 'kahtlustatav']):
            key_info.append("Arrest/suspect information")
            details.append(person)
        elif any(word in person_lower for word in ['age', 'aastane', 'vanus', 'year', 'old']):
            key_info.append("Age information mentioned")
            details.append(person)
        elif any(word in person_lower for word in ['victim', 'kannatanu', 'surnud', 'injured', 'hurt']):
            key_info.append("Victim information")
            details.append(person)
        elif any(word in person_lower for word in ['officer', 'police', 'agent', 'politsei', 'ametnik']):
            key_info.append("Law enforcement involved")
            details.append(person)
        elif any(word in person_lower for word in ['witness', 'tunnistaja', 'eyewitness']):
            key_info.append("Witness information")
            details.append(person)
        else:
            # Generic person information
            details.append(person)
    
    if key_info:
        summary = f"• {', '.join(set(key_info))}"
        if details:
            # Use the most informative detail
            best_detail = max(details, key=len)
            summary += f"\n• Key details: {best_detail[:120]}..."
        return summary
    else:
        # If no specific roles found, provide general information
        best_detail = max(people, key=len)
        return f"• {best_detail[:150]}..."


def synthesize_problems(problems: List[str]) -> str:
    """Synthesize problem information into a coherent summary."""
    if not problems:
        return "No specific problems identified."
    
    # Look for common problem themes
    themes = []
    details = []
    
    for problem in problems:
        problem_lower = problem.lower()
        
        if any(word in problem_lower for word in ['violence', 'vägivald', 'attack', 'rünnak', 'assault']):
            themes.append("Violence/attack involved")
            details.append(problem)
        elif any(word in problem_lower for word in ['conflict', 'konflikt', 'dispute', 'vaidlus', 'argument']):
            themes.append("Conflict or dispute")
            details.append(problem)
        elif any(word in problem_lower for word in ['injury', 'vigastus', 'hurt', 'haav', 'wounded']):
            themes.append("Injuries sustained")
            details.append(problem)
        elif any(word in problem_lower for word in ['drug', 'narkootikum', 'trafficking', 'smuggling']):
            themes.append("Drug-related issues")
            details.append(problem)
        elif any(word in problem_lower for word in ['crime', 'kuritegu', 'criminal', 'illegal']):
            themes.append("Criminal activity")
            details.append(problem)
        else:
            details.append(problem)
    
    if themes:
        summary = f"• {', '.join(set(themes))}"
        if details:
            # Use the most informative detail
            best_detail = max(details, key=len)
            summary += f"\n• Key details: {best_detail[:120]}..."
        return summary
    else:
        # If no specific themes found, provide general information
        best_detail = max(problems, key=len)
        return f"• {best_detail[:150]}..."


def synthesize_solutions(solutions: List[str]) -> str:
    """Synthesize solution information into a coherent summary."""
    if not solutions:
        return "No specific solutions discussed."
    
    # Look for common solution themes
    themes = []
    for solution in solutions:
        if any(word in solution.lower() for word in ['arrest', 'vahistamine', 'detention']):
            themes.append("Arrest/detention as solution")
        elif any(word in solution.lower() for word in ['investigation', 'menetlus', 'inquiry']):
            themes.append("Investigation initiated")
        elif any(word in solution.lower() for word in ['medical', 'haigla', 'treatment']):
            themes.append("Medical treatment provided")
    
    if themes:
        return f"• {', '.join(set(themes))}\n• Details: {solutions[0][:100]}..."
    else:
        return f"• {solutions[0][:150]}..."


def synthesize_decisions(decisions: List[str]) -> str:
    """Synthesize decision information into a coherent summary."""
    if not decisions:
        return "No specific decisions made."
    
    return f"• {decisions[0][:150]}..."


def synthesize_actions(actions: List[str]) -> str:
    """Synthesize action item information into a coherent summary."""
    if not actions:
        return "No specific action items identified."
    
    return f"• {actions[0][:150]}..."


def create_general_summary(sentences: List[str]) -> str:
    """Create a general summary when no specific categories are found."""
    if not sentences:
        return "No content available for summary."
    
    # Look for the most informative sentences
    informative_sentences = []
    
    for sentence in sentences[:5]:  # Check first 5 sentences
        sentence_lower = sentence.lower()
        # Score sentences based on content richness
        score = 0
        
        # Longer sentences are often more informative
        score += len(sentence.split()) * 2
        
        # Look for key information indicators
        if any(word in sentence_lower for word in ['important', 'key', 'main', 'primary', 'critical', 'essential']):
            score += 10
        if any(word in sentence_lower for word in ['discussed', 'mentioned', 'talked', 'said', 'stated']):
            score += 5
        if any(word in sentence_lower for word in ['conclusion', 'result', 'outcome', 'decision']):
            score += 8
        if any(word in sentence_lower for word in ['number', 'amount', 'quantity', 'statistics']):
            score += 3
        
        informative_sentences.append((score, sentence))
    
    # Sort by score and take the best ones
    informative_sentences.sort(key=lambda x: x[0], reverse=True)
    
    if informative_sentences:
        # Use the best sentence
        best_sentence = informative_sentences[0][1]
        return f"• {best_sentence[:200]}..."
    else:
        # Fallback to first sentence
        return f"• {sentences[0][:200]}..."