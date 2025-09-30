"""OpenMathInstruct dataset formatting template."""

SYSTEM_PROMPT = """
Respond in the following format:

<reasoning>
...
</reasoning>
<answer>
...
</answer>
"""

XML_COT_FORMAT = """\
<reasoning>
{reasoning}
</reasoning>
<answer>
{answer}
</answer>
"""


class OpenMathInstructTemplate:
    """Template for formatting OpenMathInstruct math dataset examples."""
    
    @staticmethod
    def format_for_training(example, prompt_col="question", answer_col="answer"):
        """
        Format a OpenMathInstruct example for training.
        
        Args:
            example: Dataset example containing question and answer
            prompt_col: Column name for the question/prompt
            answer_col: Column name for the answer
            
        Returns:
            dict: Formatted example with 'prompt' and 'answer' keys
        """
        prompt = example[prompt_col]
        answer = example[answer_col]
        
        training_prompt = [
            {'role': 'system', 'content': SYSTEM_PROMPT}, 
            {'role': 'user', 'content': 'What is the largest single-digit prime number?'},
            {'role': 'assistant', 'content': XML_COT_FORMAT.format(
                reasoning="9 is divisible by 3 and 8 is divisible by 2, but 7 is prime.",
                answer="7"
            )},
            {'role': 'user', 'content': prompt}
        ]
        return {'prompt': training_prompt, 'answer': answer}
    