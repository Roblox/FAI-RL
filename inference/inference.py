"""
Main inference script for model inference.
"""

import argparse
import torch
import os
import json, csv
import sys

# Disable fast tokenizer conversion to avoid tiktoken issues
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Add project root to path BEFORE importing local modules
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import yaml
import pandas as pd
import requests
import re
import warnings
import subprocess
import datetime
from typing import Dict, Any, Union
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset
from utils.api_utils import generate_response_by_api

# Suppress Pydantic warnings from dependencies (TRL/transformers)
warnings.filterwarnings("ignore", category=UserWarning, module="pydantic._internal._generate_schema")
warnings.filterwarnings("ignore", message=".*'repr' attribute.*has no effect.*")
warnings.filterwarnings("ignore", message=".*'frozen' attribute.*has no effect.*")
from pathlib import Path
from peft import PeftModel, PeftConfig

from core.config import ExperimentConfig
from utils.config_validation import validate_api_config
from utils.recipe_overrides import apply_overrides_to_recipe, load_recipe_from_yaml
from utils.logging_utils import setup_logging, SafeLogger

# Setup module-level logger with SafeLogger for robustness
# This prevents logging errors from crashing long-running inference jobs
_base_logger = setup_logging("Inference")
logger = SafeLogger(_base_logger)


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
    
    # Handle multiple choice formatting if needed (for MMLU dataset)
    if hasattr(config, 'dataset_name') and config.dataset_name == "cais/mmlu":
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
    parser = argparse.ArgumentParser(
        description="Run model inference",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Using recipe file:
  fai-rl-inference --recipe recipes/inference/llama3_3B.yaml
  
  # Mix recipe file with overrides:
  fai-rl-inference --recipe recipe.yaml model_path='./my_model' temperature=0.7
  
  # Override inference parameters:
  fai-rl-inference --recipe recipe.yaml max_new_tokens=512 do_sample=True
"""
    )
    parser.add_argument(
        "--recipe",
        type=str,
        default=None,
        help="Path to inference recipe YAML file"
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug mode with verbose logging"
    )
    parser.add_argument(
        "--nohup",
        action="store_true",
        help="Run inference in background with nohup (output redirected to logs/Inference_<timestamp>.log)"
    )
    parser.add_argument(
        "overrides",
        nargs="*",
        help="Recipe overrides in key=value format (e.g., model_path='./output' temperature=0.7)"
    )
    
    args = parser.parse_args()
    
    # Add this check: if no arguments provided at all, show help and exit
    if len(sys.argv) == 1:
        parser.print_help(sys.stderr)
        sys.exit(1)
    
    return args


def load_model_and_tokenizer(config):
    """Load model and tokenizer based on config."""
    # Support model_paths (local) and model (HuggingFace hub)
    if hasattr(config, 'model_path') and config.model_path:
        model_identifier = config.model_path
        is_local = True
    elif hasattr(config, 'model') and config.model:
        model_identifier = config.model
        is_local = False
    else:
        raise ValueError("Either model_paths or model must be specified in config")
    
    # Handle relative paths for local models
    if is_local and not os.path.isabs(model_identifier):
        model_identifier = os.path.join(os.getcwd(), model_identifier)
    
    print(f"Loading model from: {model_identifier}")
    
    # Check if path exists for local models
    if is_local and not os.path.exists(model_identifier):
        raise FileNotFoundError(f"Model path does not exist: {model_identifier}")
    
    # Check if this is a PEFT checkpoint (only for local models)
    is_peft_checkpoint = False
    if is_local:
        adapter_config_path = os.path.join(model_identifier, "adapter_config.json")
        is_peft_checkpoint = os.path.exists(adapter_config_path)
    
    if is_peft_checkpoint:
        print("Detected PEFT/LoRA checkpoint, loading base model first...")
        
        # Load the PEFT config to get the base model name
        peft_config = PeftConfig.from_pretrained(model_identifier)
        base_model_name = peft_config.base_model_name_or_path
        
        print(f"Base model: {base_model_name}")
        
        # Load tokenizer from base model instead of checkpoint to avoid conversion issues
        print("Loading tokenizer from base model...")
        try:
            tokenizer = AutoTokenizer.from_pretrained(base_model_name)
        except Exception as e:
            print(f"Warning: Failed to load tokenizer, trying with trust_remote_code=True: {e}")
            tokenizer = AutoTokenizer.from_pretrained(
                base_model_name,
                trust_remote_code=True
            )
        
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
        model = PeftModel.from_pretrained(model, model_identifier)
        
        # Merge adapter weights for faster inference
        print("Merging adapter weights...")
        model = model.merge_and_unload()
        
    else:
        # Regular model loading (non-PEFT) - can be local or from HuggingFace hub
        print(f"Loading regular model from {'local path' if is_local else 'HuggingFace hub'}...")
        
        # Load tokenizer
        try:
            tokenizer = AutoTokenizer.from_pretrained(model_identifier)
        except Exception as e:
            print(f"Warning: Failed to load tokenizer, trying with trust_remote_code=True: {e}")
            tokenizer = AutoTokenizer.from_pretrained(
                model_identifier,
                trust_remote_code=True
            )
        
        # Set the pad token if it's not already set
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        # Load the model
        model = AutoModelForCausalLM.from_pretrained(
            model_identifier,
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


def run_inference(config, debug=False):
    """Run inference on the specified dataset."""
    # Determine if we should use API or local model
    # API requires both model and api_key
    use_api = (hasattr(config, 'model') and config.model is not None) and \
              (hasattr(config, 'api_key') and config.api_key is not None)
    
    # Determine checkpoint paths to process
    checkpoint_paths = []
    if hasattr(config, 'model_paths') and config.model_paths:
        # Checkpoint paths provided
        checkpoint_paths = config.model_paths
        print(f"Running inference on {len(checkpoint_paths)} checkpoint(s)")
    elif use_api:
        # API mode
        checkpoint_paths = [None]  # Placeholder for API
    elif hasattr(config, 'model') and config.model:
        # HuggingFace model
        checkpoint_paths = [None]  # Placeholder for HF model
    else:
        raise ValueError("Either model_paths or model must be specified in config")
    
    # Track if we're running multi-checkpoint inference
    is_multi_checkpoint = len(checkpoint_paths) > 1

    # Load dataset once (shared across all checkpoints)
    print(f"Loading dataset: {config.dataset_name}")
    
    # Check if dataset_name has CSV extension - if so, load from local file
    if config.dataset_name.endswith('.csv'):
        print(f"Detected CSV file, loading from local path: {config.dataset_name}")
        
        # Handle relative and absolute paths
        dataset_path = config.dataset_name
        if not os.path.isabs(dataset_path):
            dataset_path = os.path.join(os.getcwd(), dataset_path)
        
        # Check if file exists
        if not os.path.exists(dataset_path):
            raise FileNotFoundError(f"CSV file not found: {dataset_path}")
        
        # Load CSV using pandas
        df = pd.read_csv(dataset_path)
        print(f"Loaded {len(df)} rows from CSV file")
        
        # Convert DataFrame to list of dicts (similar to HuggingFace dataset format)
        data_split = df.to_dict('records')
    else:
        # Load from HuggingFace
        if hasattr(config, 'dataset_subset') and config.dataset_subset:
            dataset = load_dataset(config.dataset_name, config.dataset_subset)
        else:
            dataset = load_dataset(config.dataset_name)
        
        # Get the appropriate split
        data_split = dataset[config.dataset_split] if config.dataset_split in dataset else dataset[list(dataset.keys())[0]]
    
    print(f"Processing {len(data_split)} examples from the dataset...")
    
    # Process all checkpoints
    all_results = []
    
    for checkpoint_idx, checkpoint_path in enumerate(checkpoint_paths):
        # Load model for this checkpoint (if not using API)
        if use_api:
            print(f"Using API inference with model: {config.model}")
            model, tokenizer = None, None
            checkpoint_name = config.model
        else:
            # Update config with current checkpoint path
            if checkpoint_path is not None:
                config.model_path = checkpoint_path
                print(f"\n{'='*60}")
                print(f"Processing checkpoint {checkpoint_idx + 1}/{len(checkpoint_paths)}: {checkpoint_path}")
                print(f"{'='*60}")
                checkpoint_name = checkpoint_path
            else:
                # Using HuggingFace model directly
                print(f"Using vanilla HuggingFace model: {config.model}")
                checkpoint_name = config.model
            
            # Load model and tokenizer for this checkpoint
            model, tokenizer = load_model_and_tokenizer(config)
        
        # Process the dataset for this checkpoint
        checkpoint_results = []
        
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
                
                # Store the result with dataset columns first, then response column
                result = {}
                
                # Add dataset columns first
                for col in config.dataset_columns:
                    result[col] = example.get(col, "")
                
                # Add checkpoint column if multi-checkpoint inference
                if is_multi_checkpoint:
                    checkpoint_col = getattr(config, 'checkpoint_column', 'checkpoint')
                    result[checkpoint_col] = checkpoint_name
                
                # Add response column after dataset columns
                response_col = getattr(config, 'response_column', 'response')
                result[response_col] = response
                
                checkpoint_results.append(result)
                
                print(f"Processed example {i+1}/{len(data_split)}")
                
            except Exception as e:
                print(f"Error processing example {i}: {e}")
                continue
        
        # Add results from this checkpoint to overall results
        all_results.extend(checkpoint_results)
        print(f"Completed checkpoint {checkpoint_idx + 1}/{len(checkpoint_paths)}: {len(checkpoint_results)} examples processed")
        
        # Clean up model to free memory before loading next checkpoint
        if model is not None:
            del model
            del tokenizer
            torch.cuda.empty_cache()
    
    # Use all_results as the final results
    results = all_results
    
    # Create output directory if it doesn't exist
    output_dir = os.path.dirname(config.output_file)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    
    # Save results to CSV file
    df = pd.DataFrame(results)
    df.to_csv(config.output_file, index=False, encoding='utf-8', quoting=csv.QUOTE_ALL)
    
    print(f"\nResults saved to: {config.output_file}")
    print(f"Processed {len(results)} examples successfully")
    
    # Determine model info for summary
    if use_api:
        model_info = config.model
        inference_type = 'api'
    elif is_multi_checkpoint:
        model_info = checkpoint_paths  # List of all checkpoints
        inference_type = 'multi_checkpoint'
    elif hasattr(config, 'model_paths') and config.model_paths:
        model_info = checkpoint_paths  # List of checkpoints (single or multiple)
        inference_type = 'local_finetuned'
    else:
        model_info = config.model
        inference_type = 'huggingface_vanilla'
    
    # Calculate total expected examples (dataset size * number of checkpoints)
    total_expected = len(data_split) * len(checkpoint_paths)
    
    # Create summary
    summary = {
        'total_examples': len(data_split),
        'num_checkpoints': len(checkpoint_paths),
        'total_expected_results': total_expected,
        'successful_examples': len(results),
        'failed_examples': total_expected - len(results),
        'config': config.to_dict(),
        'inference_type': inference_type,
        'model_info': model_info,
        'dataset_name': config.dataset_name,
        'dataset_columns_used': config.dataset_columns,
        'system_prompt': config.system_prompt
    }
    
    # Save summary (keep as JSON, base filename on CSV output)
    summary_file = config.output_file.replace('.csv', '_summary.json')
    with open(summary_file, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    
    print(f"Summary saved to: {summary_file}")


def load_inference_recipe_with_overrides(args):
    """Load inference recipe from file and/or command-line arguments.
    
    Priority (highest to lowest):
    1. Command-line overrides
    2. Recipe file values
    3. Default values
    """
    if not args.recipe:
        raise ValueError("--recipe argument is required")
    
    # Load base recipe from YAML
    recipe_dict = load_recipe_from_yaml(args.recipe)
    
    # Apply command-line overrides
    if args.overrides:
        recipe_dict = apply_overrides_to_recipe(recipe_dict, args.overrides)
    
    # Write temporary recipe file with overrides applied
    import tempfile
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as tmp_file:
        yaml.dump(recipe_dict, tmp_file)
        tmp_recipe_path = tmp_file.name
    
    try:
        # Load using existing config loader
        config = ExperimentConfig.load_inference_config(tmp_recipe_path)
    finally:
        # Clean up temporary file
        os.unlink(tmp_recipe_path)
    
    return config


def main():
    """Main inference function."""
    global args
    args = parse_args()
    
    # Handle nohup mode
    if args.nohup:
        # Create logs directory if it doesn't exist
        os.makedirs("logs", exist_ok=True)
        
        # Generate log filename with timestamp
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = f"logs/Inference_{timestamp}.log"
        
        print(f"Running inference in background with nohup. Output will be saved to: {log_file}")
        
        # Build command to run the script without --nohup
        script_path = os.path.abspath(__file__)
        cmd_args = [sys.executable, script_path, "--recipe", args.recipe]
        if args.debug:
            cmd_args.append("--debug")
        
        # Add overrides to command
        if args.overrides:
            cmd_args.extend(args.overrides)
        
        # Prepare nohup command: nohup <command> > log_file 2>&1 &
        cmd_str = " ".join(cmd_args) + f" > {log_file} 2>&1 &"
        full_cmd = f"nohup {cmd_str}"
        
        print(f"Executing: {full_cmd}")
        
        # Execute with shell to handle redirection and background
        result = subprocess.call(full_cmd, shell=True)
        
        if result == 0:
            print(f"Inference started in background. Monitor progress with: tail -f {log_file}")
        
        return result
    
    # Load recipe with overrides
    config = load_inference_recipe_with_overrides(args)
    
    # Determine inference type
    use_api = hasattr(config, 'api_key') and config.api_key and \
              hasattr(config, 'model') and config.model
    
    print("Starting inference with the following configuration:")
    if use_api:
        print(f"  Model (API): {config.model}")
        print(f"  Inference type: API-based")
    elif hasattr(config, 'model_paths') and config.model_paths:
        print(f"  Number of checkpoints: {len(config.model_paths)}")
        print(f"  Checkpoints:")
        for i, path in enumerate(config.model_paths, 1):
            print(f"    {i}. {path}")
        if len(config.model_paths) > 1:
            print(f"  Inference type: Multi-checkpoint inference")
        else:
            print(f"  Inference type: Local fine-tuned model")
    elif hasattr(config, 'model') and config.model:
        print(f"  Model: {config.model}")
        print(f"  Inference type: HuggingFace vanilla model")
    else:
        raise ValueError("Either model_paths or model must be specified in config")
    
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