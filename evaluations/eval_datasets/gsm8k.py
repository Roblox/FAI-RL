"""
GSM8K (Grade School Math 8K) dataset evaluation utilities.
"""

import re
from typing import Optional, List

FINAL_ANSWER_DELIMITERS = [
    r"####",                # Standard GSM8K delimiter
    r"ANSWER IS",           # Variations like: The answer is 42
    r"FINAL ANSWER",        # Explicit final answer marker
]

_NUMBER_REGEX = re.compile(r"(-?\$?[0-9][0-9,]*(?:\.[0-9]+)?)")

# Regexes to clean artifacts
_CLEAN_PATTERNS = [
    (re.compile(r","), ""),          # remove thousands separators
    (re.compile(r"\$"), ""),         # remove dollar sign
    (re.compile(r"^=+"), ""),        # leading equals from calc
    (re.compile(r"\.$"), ""),       # trailing period
    (re.compile(r"^\s+|\s+$"), ""), # trim
]


def _normalize_number(s: str) -> str:
    """Normalize a numeric string by removing separators, dollar signs, etc."""
    s = s.strip()
    for pat, repl in _CLEAN_PATTERNS:
        s = pat.sub(repl, s)
    # Remove leading zeros unless decimal like 0.x
    if re.fullmatch(r"0+[0-9]+", s):
        s = str(int(s))
    return s


def _extract_number_candidates(text: str) -> List[str]:
    """Extract all numeric candidates from text."""
    return [m[0] if isinstance(m, tuple) else m for m in _NUMBER_REGEX.findall(text)]


def extract_numeric_answer(text: str) -> Optional[str]:
    """
    Extract numeric answer from GSM8K model response.
    
    Extraction heuristics (in priority order):
      1. JSON pattern: "answer": "<value>" (case-insensitive key)
      2. Delimiter based (#### <answer>) taking the portion after the last delimiter
      3. Phrases like 'The answer is 24' capturing final number
      4. Last numeric token in the text
    
    Args:
        text: Model response text
    
    Returns:
        Extracted and normalized numeric answer or None if not found
    """
    if not text:
        return None

    # 1. JSON pattern
    json_match = re.search(r'"answer"\s*:\s*"?([^"\n]+)"?', text, flags=re.IGNORECASE)
    if json_match:
        candidate = json_match.group(1).strip()
        nums = _extract_number_candidates(candidate)
        if nums:
            return _normalize_number(nums[-1])
        return _normalize_number(candidate)

    upper = text.upper()

    # 2. Delimiter based (use the last occurrence)
    last_segment = None
    for delim in FINAL_ANSWER_DELIMITERS:
        if re.search(delim, upper):
            parts = re.split(delim, text, flags=re.IGNORECASE)
            if parts:
                last_segment = parts[-1]
    if last_segment:
        nums = _extract_number_candidates(last_segment)
        if nums:
            return _normalize_number(nums[-1])
        return _normalize_number(last_segment.split("\n")[0])

    # 3. Phrase: 'answer is <number>'
    ans_is = re.search(r"ANSWER IS\s+(-?\$?[0-9][0-9,]*(?:\.[0-9]+)?)", upper)
    if ans_is:
        return _normalize_number(ans_is.group(1))

    # 4. Fallback to last number anywhere in text
    nums = _extract_number_candidates(text)
    if nums:
        return _normalize_number(nums[-1])

    return None


def extract_predicted_answer(text: str, choice_labels: Optional[List[str]] = None) -> Optional[str]:
    """
    Extract predicted answer from model response for GSM8K dataset.
    
    Args:
        text: Model response text
        choice_labels: Not used for GSM8K (numeric answers), included for interface consistency
    
    Returns:
        Extracted answer or None if not found
    """
    return extract_numeric_answer(text)


def extract_ground_truth(text, choice_labels: Optional[List[str]] = None) -> Optional[str]:
    """
    Extract ground truth answer for GSM8K dataset.
    
    Ground truth entries typically follow the format:
      '... solution reasoning ... #### 42'
    We extract text after the last '####' delimiter. If absent, fallback to last number.
    
    Args:
        text: Ground truth text from dataset
        choice_labels: Not used for GSM8K (numeric answers), included for interface consistency
    
    Returns:
        Processed ground truth answer
    """
    if not text:
        return None
    if '####' in text:
        segment = text.split('####')[-1].strip()
        nums = _extract_number_candidates(segment)
        if nums:
            return _normalize_number(nums[-1])
        return _normalize_number(segment)
    # Fallback last number
    nums = _extract_number_candidates(text)
    if nums:
        return _normalize_number(nums[-1])
    return None

__all__ = ["extract_predicted_answer", "extract_ground_truth"]
