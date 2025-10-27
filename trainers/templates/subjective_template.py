"""Subjective dataset formatting template for human preference learning."""


class UnverifiableDomainTemplate:
    """Template for formatting subjective/preference-based dataset examples."""
    
    @staticmethod
    def format_for_training(example, prompt_col="prompt", answer_col=None):
        """
        Format a subjective example for training.
        
        This template is designed for subjective training where the quality
        of responses is determined by an external evaluator (API) rather than
        ground truth answers.
        
        Args:
            example: Dataset example containing the prompt
            prompt_col: Column name for the question/prompt
            answer_col: Column name for the answer (optional, not used for subjective training)
            
        Returns:
            dict: Formatted example with 'prompt' key
        """
        prompt = example[prompt_col]
        
        # For subjective training, we don't need a ground truth answer
        # The model will generate multiple responses and the reward function
        # will evaluate them via an API call
        
        # If prompt is already a list of messages, use it directly
        if isinstance(prompt, list):
            training_prompt = prompt
        else:
            # Otherwise, create a simple user prompt
            training_prompt = [
                {'role': 'user', 'content': prompt}
            ]
        
        return {'prompt': training_prompt}

