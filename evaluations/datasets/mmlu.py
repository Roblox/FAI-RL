"""
MMLU (Massive Multitask Language Understanding) dataset evaluation utilities.
"""

import re
from typing import Optional, List


def extract_multiple_choice_answer(text: str, choice_labels: Optional[List[str]] = None) -> Optional[str]:
    """
    Extract the first character from text if it matches a choice label.
    
    Args:
        text: Model response text
        choice_labels: List of valid choice labels
    
    Returns:
        Extracted choice label or None if not found
    """
    if not text:
        return None
    
    if choice_labels is None:
        choice_labels = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J"]

    text_upper = text.upper()

    # Pattern 0: JSON format, e.g., "answer": "B" or "answer": "B"
    pattern0 = r'[""]ANSWER[""]\s*:\s*[""]([' + ''.join(choice_labels) + r'])[""]'

    match = re.search(pattern0, text_upper)
    if match:
        return match.group(1)
    
    # Pattern 1: "The best/final/correct answer/option/choice is A" or "The answer/option is A"
    pattern1 = r'(?:THE\s+)?(?:BEST|FINAL|CORRECT)?\s*(?:ANSWER|OPTION|CHOICE)\s+IS\s+([' + ''.join(choice_labels) + r'])'
    match = re.search(pattern1, text_upper)
    if match:
        return match.group(1)
    
    # Pattern 2: "Answer: A" (with optional additional text)
    pattern2 = r'ANSWER:\s*([' + ''.join(choice_labels) + r'])'
    match = re.search(pattern2, text_upper)
    if match:
        return match.group(1)

    # Pattern 3 for "option/answer/choice C is the best/final answer/option/choice"
    pattern3 = r'(?:OPTION|ANSWER|CHOICE)\s+([' + ''.join(choice_labels) + r'])\s+IS\s+(?:THE\s+)?(?:BEST|FINAL)?\s*(?:ANSWER|OPTION|CHOICE)'
    match = re.search(pattern3, text_upper)
    if match:
        return match.group(1)

    # Pattern 4: "the answer/option/choice is option/answer C" (e.g., "the answer is option C")
    pattern4 = r'(?:THE\s+)?(?:ANSWER|OPTION|CHOICE)\s+IS\s+(?:OPTION|ANSWER|CHOICE)\s+([' + ''.join(choice_labels) + r'])'
    match = re.search(pattern4, text_upper)
    if match:
        return match.group(1)

    # Pattern 5: "the best/final/correct answer/option/choice should/would/must be C" (with optional adjectives like "correct")
    pattern5 = r'(?:THE\s+)?(?:BEST|FINAL|CORRECT)?\s*(?:ANSWER|OPTION|CHOICE)\s+(?:SHOULD|WOULD|MUST|COULD)\s+BE\s+([' + ''.join(choice_labels) + r'])'
    match = re.search(pattern5, text_upper)
    if match:
        return match.group(1)

    # Pattern 6: "option/answer/choice D is correct" (e.g., "option D is correct")
    pattern6 = r'(?:OPTION|ANSWER|CHOICE)\s+([' + ''.join(choice_labels) + r'])\s+IS\s+(?:THE\s+)?CORRECT'
    match = re.search(pattern6, text_upper)
    if match:
        return match.group(1)

    # Pattern 7: Fallback to first character
    first_char = text.strip().upper()[0]
    
    return first_char if first_char in choice_labels else None


def convert_answer_index_to_letter(answer_index, choice_labels: Optional[List[str]] = None) -> Optional[str]:
    """
    Convert numerical answer index to choice letter.
    
    Args:
        answer_index: Numerical index (0, 1, 2, 3...)
        choice_labels: List of choice labels
    
    Returns:
        Choice letter (A, B, C, D...)
    """
    if choice_labels is None:
        choice_labels = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J"]
    
    try:
        index = int(answer_index)
        if 0 <= index < len(choice_labels):
            return choice_labels[index]
    except (ValueError, TypeError):
        pass
    
    return None


def extract_predicted_answer(text: str, choice_labels: Optional[List[str]] = None) -> Optional[str]:
    """
    Extract predicted answer from model response for MMLU dataset.
    
    Args:
        text: Model response text
        choice_labels: List of choice labels for multiple choice
    
    Returns:
        Extracted answer or None if not found
    """
    return extract_multiple_choice_answer(text, choice_labels)


def extract_ground_truth(text, choice_labels: Optional[List[str]] = None) -> Optional[str]:
    """
    Extract ground truth answer for MMLU dataset.
    
    Args:
        text: Ground truth text or index
        choice_labels: List of choice labels for multiple choice
    
    Returns:
        Processed ground truth answer
    """
    # For MMLU, ground truth is typically a numerical index
    return convert_answer_index_to_letter(text, choice_labels)

