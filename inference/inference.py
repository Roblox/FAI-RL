"""
Main inference script for model inference.
"""

import argparse
import torch
import os
import json, csv
import sys
import pandas as pd
import requests
import re
import warnings
from typing import Dict, Any, Union
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset

# Suppress Pydantic warnings from dependencies (TRL/transformers)
warnings.filterwarnings("ignore", category=UserWarning, module="pydantic._internal._generate_schema")
from pathlib import Path
from peft import PeftModel, PeftConfig

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from core.config import ExperimentConfig


def format_multiple_choice_for_inference(choices, choice_labels=None):
    """
    Format a list of choices into A/B/C/D format for inference.
    
    Args:
        choices: List of choice strings or string representation of list
        choice_labels: List of labels to use (default: ["A", "B", "C", "D", ...])
    
    Returns:
        Formatted string with labeled choices
    """
    if choice_labels is None:
        choice_labels = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J"]
    
    # Handle string representation of list
    if isinstance(choices, str):
        try:
            # Try to evaluate as list if it looks like one
            if choices.startswith('[') and choices.endswith(']'):
                choices = eval(choices)
            else:
                # Split by comma if it's comma-separated
                choices = [choice.strip() for choice in choices.split(',')]
        except:
            # If parsing fails, treat as single choice
            choices = [choices]
    
    formatted_choices = []
    for i, choice in enumerate(choices):
        if i < len(choice_labels):
            formatted_choices.append(f"{choice_labels[i]}. {choice}")
        else:
            # Fallback if we have more choices than labels
            formatted_choices.append(f"{i+1}. {choice}")
    
    return "\n".join(formatted_choices)


def has_template_placeholders(template):
    """Check if a template string contains placeholders like {variable}."""
    return '{' in template and '}' in template


def format_template_prompt(template, example, config):
    """
    Format prompt template with example data, handling special cases like multiple choice.
    
    Args:
        template: Template string with placeholders
        example: Dataset example dictionary
        config: Configuration object
    
    Returns:
        Formatted prompt string
    """
    # Create a copy of the example for formatting
    format_dict = example.copy()
    
    # Handle multiple choice formatting if needed
    if hasattr(config, 'output_type') and config.output_type == "multiple_choice":
        if 'choices' in format_dict:
            choice_labels = getattr(config, 'choice_labels', None)
            formatted_choices = format_multiple_choice_for_inference(
                format_dict['choices'], 
                choice_labels
            )
            format_dict['choices'] = formatted_choices
    
    # Format the template
    try:
        return template.format(**format_dict)
    except KeyError as e:
        print(f"Warning: Missing key in template formatting: {e}")
        return template
    except Exception as e:
        print(f"Warning: Error in template formatting: {e}")
        return template


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Run model inference")
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to inference configuration YAML file"
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug mode with verbose logging"
    )
    
    return parser.parse_args()


def load_model_and_tokenizer(config):
    """Load model and tokenizer based on config."""
    model_path = config.model_path
    
    # Handle relative paths by prepending current working directory
    if not os.path.isabs(model_path):
        model_path = os.path.join(os.getcwd(), model_path)
    
    print(f"Loading model from: {model_path}")
    
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model path does not exist: {model_path}")
    
    # Check if this is a PEFT checkpoint
    adapter_config_path = os.path.join(model_path, "adapter_config.json")
    is_peft_checkpoint = os.path.exists(adapter_config_path)
    
    if is_peft_checkpoint:
        print("Detected PEFT/LoRA checkpoint, loading base model first...")
        
        # Load the PEFT config to get the base model name
        peft_config = PeftConfig.from_pretrained(model_path)
        base_model_name = peft_config.base_model_name_or_path
        
        print(f"Base model: {base_model_name}")
        
        # Load tokenizer from checkpoint
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        
        # Set the pad token if it's not already set
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        # Add the special pad token to match training setup (PPO adds "[PAD]" token)
        if "[PAD]" not in tokenizer.get_vocab():
            tokenizer.add_special_tokens({"pad_token": "[PAD]"})
            print(f"Added [PAD] token to tokenizer. New vocab size: {len(tokenizer)}")
        
        # Load base model first WITHOUT adapter
        print("Loading base model...")
        model = AutoModelForCausalLM.from_pretrained(
            base_model_name,
            torch_dtype=torch.bfloat16,
            device_map="auto"
        )
        
        # Resize embeddings to match tokenizer BEFORE loading adapter
        if model.get_input_embeddings().weight.shape[0] != len(tokenizer):
            print(f"Resizing model embeddings from {model.get_input_embeddings().weight.shape[0]} to {len(tokenizer)}")
            model.resize_token_embeddings(len(tokenizer))
        
        # Now load the PEFT adapter
        print("Loading PEFT adapter...")
        model = PeftModel.from_pretrained(model, model_path)
        
        # Merge adapter weights for faster inference
        print("Merging adapter weights...")
        model = model.merge_and_unload()
        
    else:
        # Regular model loading (non-PEFT)
        print("Loading regular (non-PEFT) model...")
        
        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        
        # Set the pad token if it's not already set
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        # Load the model
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            device_map="auto"
        )
    
    model.eval()  # Set the model to inference mode
    
    return model, tokenizer


def generate_response(model, tokenizer, prompt: str, config):
    """
    Generates a response from the model given a prompt.
    """
    
    # Tokenize the input prompt
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    
    # Get the length of the input tokens
    input_token_length = inputs.input_ids.shape[1]
    
    # Generate output
    with torch.no_grad():  # Disable gradient calculations for inference
        outputs = model.generate(
            **inputs,
            max_new_tokens=config.max_new_tokens,
            do_sample=config.do_sample,
            temperature=config.temperature,
            top_p=config.top_p,
            pad_token_id=tokenizer.pad_token_id
        )
    
    # Slice off the prompt tokens
    generated_tokens = outputs[0][input_token_length:]
    
    # Decode only the new tokens
    response_text = tokenizer.decode(generated_tokens, skip_special_tokens=True)
    
    return response_text

def _get_api_endpoint(model: str) -> str:
    """Determine the appropriate API endpoint based on model type and name."""
    # ST3 models that require the sitetest3 endpoint
    st3_models = [
        'Qwen/Qwen3-235B-A22B-Instruct-2507', 
        'Qwen/Qwen3-8B',
        'openai/gpt-oss-120b', 
        'openai/o3', 
        'openai/gpt-4o', 
        'anthropic/claude-opus-4'
    ]
    
    if model.startswith("google/"):
        model_name = model.split("/")[-1]
        return f"https://apis.sitetest3.simulpong.com/ml-gateway-service/gemini/v1beta/models/{model_name}:generateContent"
    elif model in st3_models:
        return "https://apis.sitetest3.simulpong.com/ml-gateway-service/v1/chat/completions"
    else:
        # Default ST1 endpoint
        return 'https://snc2-apis.sitetest1.simulpong.com/ml-gateway-service/v1/chat/completions'


def _build_google_request_data(prompt: str, config) -> dict:
    """Build request data for Google/Gemini models."""
    return {
        "contents": {
            "role": "user",
            "parts": [{"text": prompt}]
        },
        "generationConfig": {
            "maxOutputTokens": config.max_new_tokens
        }
    }


def _build_openai_request_data(prompt: str, config) -> dict:
    """Build request data for OpenAI models."""
    return {
        "model": config.model,
        "messages": [{"content": prompt, "role": "user"}]
    }


def _build_default_request_data(prompt: str, config) -> dict:
    """Build request data for other models (Anthropic, etc.)."""
    return {
        "model": config.model,
        "max_tokens": config.max_new_tokens,
        "temperature": config.temperature,
        "messages": [{"content": prompt, "role": "user"}]
    }


def _make_api_request(url: str, headers: dict, data: dict, model: str) -> requests.Response:
    """Make the HTTP request to the API endpoint."""
    if model.startswith("google/"):
        # Google API requires data to be JSON string in body
        return requests.post(url, headers=headers, data=json.dumps(data))
    else:
        # Other APIs can use json parameter
        return requests.post(url, headers=headers, json=data)


def _parse_api_response(response_json: dict, model: str) -> str:
    """Extract the response text from the API response JSON."""
    try:
        if model.startswith("google/"):
            return response_json['candidates'][0]['content']['parts'][0]['text']
        else:
            return response_json['choices'][0]['message']['content']
    except (KeyError, IndexError, TypeError):
        return ""


def _validate_api_key(api_key: str) -> None:
    """Validate that the API key is not a placeholder."""
    if api_key == "YOUR_API_KEY":
        raise ValueError(
            "Error: api_key is still set to the placeholder 'YOUR_API_KEY'. "
            "Please replace it with your actual API key in the configuration file."
        )


def generate_response_by_api(
    prompt: str,
    config
) -> Union[Dict[str, Any], str]:
    """Generate response using API-based inference."""
    _validate_api_key(config.api_key)
    
    try:
        # Get the appropriate API endpoint
        url = _get_api_endpoint(config.model)
        
        # Set up headers
        headers = {
            "Content-Type": "application/json",
            "Authorization": config.api_key
        }
        
        # Build request data based on model type
        if config.model.startswith("google/"):
            data = _build_google_request_data(prompt, config)
        elif config.model.startswith("openai/"):
            data = _build_openai_request_data(prompt, config)
        else:
            data = _build_default_request_data(prompt, config)
        
        # Make the API request
        response = _make_api_request(url, headers, data, config.model)
        response.raise_for_status()
        
        # Parse and return the response
        response_json = response.json()
        return _parse_api_response(response_json, config.model)
        
    except requests.exceptions.RequestException as e:
        return ""


def run_inference(config, debug=False):
    """Run inference on the specified dataset."""
    # Determine if we should use API or local model
    use_api = (config.model is not None) and (config.api_key is not None)
    
    if use_api:
        print(f"Using API inference with model: {config.model}")
        model, tokenizer = None, None
    else:
        print("Using local model inference")
        model, tokenizer = load_model_and_tokenizer(config)

    # Load dataset
    print(f"Loading dataset: {config.dataset_name}")
    if hasattr(config, 'dataset_subset') and config.dataset_subset:
        dataset = load_dataset(config.dataset_name, config.dataset_subset)
    else:
        dataset = load_dataset(config.dataset_name)
    
    # Get the appropriate split
    data_split = dataset[config.dataset_split] if config.dataset_split in dataset else dataset[list(dataset.keys())[0]]
    
    print(f"Processing {len(data_split)} examples from the dataset...")
    
    # Process the dataset
    results = []
    
    for i, example in enumerate(data_split):
        # Check if system_prompt contains template placeholders
        if has_template_placeholders(config.system_prompt):
            # Use template formatting
            full_prompt = format_template_prompt(config.system_prompt, example, config)
        else:
            raise ValueError(
                "system_prompt configuration is missing in the template, "
                "or the required placeholder is not present in system_prompt."
            )
        
        # Generate response
        try:
            if debug:
                print(f"\n{'='*50}")
                print(f"DEBUG - Example {i+1}")
                print(f"{'='*50}")
                print("FULL PROMPT:")
                print(f"{full_prompt}")
                print(f"\n{'-'*30}")
            
            # Choose the appropriate response generation method based on config
            if use_api:
                response = generate_response_by_api(
                    prompt=full_prompt,
                    config=config
                )
            else:
                response = generate_response(model, tokenizer, full_prompt, config)
            
            if debug:
                print("Response:")
                print(f"{response}")
                print(f"{'='*50}\n")
            
            # Store the result with flattened used_columns
            result = {'response': response}
            
            # Flatten used_columns into separate columns
            for col in config.dataset_columns:
                result[col] = example.get(col, "")
            
            results.append(result)
            
            print(f"Processed example {i+1}/{len(data_split)}")
            
        except Exception as e:
            print(f"Error processing example {i}: {e}")
            continue
    
    # Create output directory if it doesn't exist
    output_dir = os.path.dirname(config.output_file)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    
    # Save results to CSV file
    df = pd.DataFrame(results)
    df.to_csv(config.output_file, index=False, encoding='utf-8', quoting=csv.QUOTE_ALL)
    
    print(f"\nResults saved to: {config.output_file}")
    print(f"Processed {len(results)} examples successfully")
    
    # Create summary
    summary = {
        'total_examples': len(data_split),
        'successful_examples': len(results),
        'failed_examples': len(data_split) - len(results),
        'config': config.to_dict(),
        'inference_type': 'api' if use_api else 'local',
        'model_info': config.model if use_api else config.model_path,
        'dataset_name': config.dataset_name,
        'dataset_columns_used': config.dataset_columns,
        'system_prompt': config.system_prompt
    }
    
    # Save summary (keep as JSON, base filename on CSV output)
    summary_file = config.output_file.replace('.csv', '_summary.json')
    with open(summary_file, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    
    print(f"Summary saved to: {summary_file}")


def main():
    """Main inference function."""
    global args
    args = parse_args()
    
    # Load configuration
    config = ExperimentConfig.load_inference_config(args.config)
    
    # Check if we should use API-based inference
    use_api = hasattr(config, 'api_key') and config.api_key
    
    print("Starting inference with the following configuration:")
    if use_api:
        print(f"  Model (API): {config.model}")
        print(f"  Inference type: API-based")
    else:
        print(f"  Model path: {config.model_path}")
        print(f"  Inference type: Local model")
    
    print(f"  Dataset: {config.dataset_name}")
    print(f"  Dataset columns: {config.dataset_columns}")
    print(f"  Output file: {config.output_file}")
    print(f"  Temperature: {config.temperature}")
    print(f"  Top-p: {config.top_p}")
    print(f"  Max new tokens: {config.max_new_tokens}")
    print(f"  Do sample: {config.do_sample}")
    print()
    
    try:
        # Run inference
        run_inference(config, debug=args.debug)
        print("Inference completed successfully!")
        
    except Exception as e:
        print(f"Inference failed with error: {str(e)}")
        if args.debug:
            import traceback
            traceback.print_exc()
        raise


if __name__ == "__main__":
    main()