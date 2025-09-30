"""GSM8K dataset formatting template."""

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


class GSM8KTemplate:
    """Template for formatting GSM8K math dataset examples."""
    
    @staticmethod
    def format_for_training(example, prompt_col="question", answer_col="answer"):
        """
        Format a GSM8K example for training.
        
        Args:
            example: Dataset example containing question and answer
            prompt_col: Column name for the question/prompt
            answer_col: Column name for the answer
            
        Returns:
            dict: Formatted example with 'prompt' and 'answer' keys
        """
        prompt = example[prompt_col]
        answer = example[answer_col]
        
        # Extract final answer for this specific example (assuming GSM8K format with ####)
        final_answer = answer.split('####')[-1].strip() if '####' in answer else answer.strip()
        
        training_prompt = [
            {'role': 'system', 'content': SYSTEM_PROMPT}, 
            {'role': 'user', 'content': 'What is the largest single-digit prime number?'},
            {'role': 'assistant', 'content': XML_COT_FORMAT.format(
                reasoning="9 is divisible by 3 and 8 is divisible by 2, but 7 is prime.",
                answer="7"
            )},
            {'role': 'user', 'content': prompt}
        ]
        return {'prompt': training_prompt, 'answer': final_answer}
    
    @staticmethod
    def get_system_prompt():
        """Get the system prompt for GSM8K."""
        return SYSTEM_PROMPT
    
    @staticmethod
    def get_cot_format():
        """Get the XML COT format template."""
        return XML_COT_FORMAT
