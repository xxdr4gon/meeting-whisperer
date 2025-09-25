from collections import Counter
import re
from typing import List

SENT_SPLIT = re.compile(r"(?<=[.!?])\s+")
WORD_RE = re.compile(r"[\w']+")


def split_sentences(text: str) -> List[str]:
	return [s.strip() for s in SENT_SPLIT.split(text) if s.strip()]


def summarize_text(text: str, max_sentences: int = 5) -> str:
	if not text:
		return ""
	sentences = split_sentences(text)
	if len(sentences) <= max_sentences:
		return text
	words = WORD_RE.findall(text.lower())
	freq = Counter(words)
	# score sentences by sum of word frequencies
	scored = []
	for idx, s in enumerate(sentences):
		tokens = WORD_RE.findall(s.lower())
		score = sum(freq.get(t, 0) for t in tokens)
		scored.append((score, idx, s))
	scored.sort(reverse=True)
	top = sorted(scored[:max_sentences], key=lambda x: x[1])
	return " ".join(s for _, _, s in top)
