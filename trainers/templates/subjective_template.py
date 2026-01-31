"""Template for subjective evaluation datasets."""
from typing import Dict, Any, List, Optional


class SubjectiveTemplate:
    """Template for formatting subjective evaluation data for training.
    
    This template is used for datasets where responses are evaluated
    based on subjective quality criteria rather than verifiable answers.
    """
    
    SYSTEM_PROMPT = """You are a helpful AI assistant. Provide thoughtful, accurate, and well-structured responses to user queries."""
    
    @classmethod
    def format_for_training(
        cls, 
        example: Dict[str, Any],
        prompt_column: str = "prompt",
        answer_column: str = "answer"
    ) -> Dict[str, Any]:
        """
        Format an example for training with subjective rewards.
        
        Args:
            example: Raw dataset example
            prompt_column: Column name containing the prompt
            answer_column: Column name containing the expected answer (if any)
            
        Returns:
            Formatted example with 'prompt' and 'answer' fields
        """
        prompt = example.get(prompt_column, "")
        answer = example.get(answer_column, "")
        
        # Format as chat messages
        messages = [
            {"role": "system", "content": cls.SYSTEM_PROMPT},
            {"role": "user", "content": prompt}
        ]
        
        return {
            "prompt": messages,
            "answer": answer if answer else "",
        }
    
    @classmethod
    def format_prompt(cls, prompt: str) -> List[Dict[str, str]]:
        """
        Format a single prompt for inference.
        
        Args:
            prompt: The user's prompt
            
        Returns:
            List of message dictionaries
        """
        return [
            {"role": "system", "content": cls.SYSTEM_PROMPT},
            {"role": "user", "content": prompt}
        ]
    
    @classmethod
    def format_for_inference(
        cls,
        example: Dict[str, Any],
        prompt_column: str = "prompt"
    ) -> Dict[str, Any]:
        """
        Format an example for inference.
        
        Args:
            example: Raw dataset example
            prompt_column: Column name containing the prompt
            
        Returns:
            Formatted example ready for inference
        """
        prompt = example.get(prompt_column, "")
        messages = cls.format_prompt(prompt)
        
        return {
            "prompt": messages,
            **{k: v for k, v in example.items() if k != prompt_column}
        }

