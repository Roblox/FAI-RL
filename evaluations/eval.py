"""
Comprehensive evaluation system that leverages the existing inference pipeline.
"""

import argparse
import os
import sys
import re
import json, csv
import yaml
import pandas as pd
import warnings
import subprocess
import datetime
from typing import Dict, List, Tuple, Any, Optional
try:
    # Prefer the HuggingFace 'datasets' library. If a conflicting local module existed
    # (e.g., a folder named 'datasets'), it could shadow the external package. That
    # folder has been renamed to 'eval_datasets' to avoid ImportError.
    from datasets import load_dataset  # type: ignore
except Exception as _import_err:  # pragma: no cover
    raise ImportError(
        f"Failed to import 'load_dataset' from HuggingFace datasets library: {_import_err}. "
        "Ensure 'datasets' is installed (pip install datasets) and no local 'datasets' package shadows it."
    )
import numpy as np

# Suppress Pydantic warnings from dependencies (TRL/transformers)
warnings.filterwarnings("ignore", category=UserWarning, module="pydantic._internal._generate_schema")
warnings.filterwarnings("ignore", message=".*'repr' attribute.*has no effect.*")

# Add project root to path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Import existing modules
from core.config import ExperimentConfig
from inference.inference import run_inference, load_model_and_tokenizer, generate_response
from utils.api_utils import generate_response_by_api
from utils.recipe_overrides import apply_overrides_to_recipe, load_recipe_from_yaml

# Import dataset-specific evaluation utilities
from evaluations.eval_datasets import mmlu


def extract_predicted_answer(text, dataset_name, choice_labels=None):
    """
    Extract predicted answer from model response based on dataset.
    
    Args:
        text: Model response text
        dataset_name: Name of the dataset being evaluated
        choice_labels: List of choice labels for multiple choice
    
    Returns:
        Extracted answer or None if not found
    """
    if dataset_name == "cais/mmlu":
        return mmlu.extract_predicted_answer(text, choice_labels)
    else:
        # Original numerical extraction logic
        regex_pattern = "(-?[$0-9.,]{2,})|(-?[0-9]+)"
        regexes_to_ignore = [",", "\\$", "(?s).*#### ", "\\.$"]
        
        match = re.findall(regex_pattern, text)
        if match:
            match = match[-1]
            if isinstance(match, tuple):
                match = [m for m in match if m][0]
            text = match.strip()

            for regex in regexes_to_ignore:
                text = re.sub(regex, "", text)
            return text
        else:
            return None


def extract_ground_truth(text, dataset_name, choice_labels=None):
    """
    Extract ground truth answer based on dataset.
    
    Args:
        text: Ground truth text or index
        dataset_name: Name of the dataset being evaluated
        choice_labels: List of choice labels for multiple choice
    
    Returns:
        Processed ground truth answer
    """
    if dataset_name == "cais/mmlu":
        # For MMLU, ground truth is typically a numerical index
        return mmlu.extract_ground_truth(text, choice_labels)
    else:
        # Original logic for numerical answers
        return text.split('####')[-1].strip()


def load_evaluation_dataset(dataset_name: str, split: str = "test", subset: str = None) -> pd.DataFrame:
    """
    Load evaluation dataset and return as DataFrame.
    
    Args:
        dataset_name: Name of the dataset to load
        split: Dataset split to use (default: "test")
        subset: Dataset subset/config name (for datasets like MMLU with multiple subjects)
        
    Returns:
        DataFrame containing the evaluation dataset
    """
    print(f"Loading evaluation dataset: {dataset_name}")
    if subset:
        print(f"Dataset subset: {subset}")
    print(f"Split: {split}")
    
    # Load dataset with or without subset
    if subset:
        dataset = load_dataset(dataset_name, subset)
    else:
        dataset = load_dataset(dataset_name)
    
    # Get the appropriate split
    data_split = dataset[split] if split in dataset else dataset[list(dataset.keys())[0]]
    
    # Convert to DataFrame for easier processing
    df = pd.DataFrame(data_split)
    print(f"Loaded {len(df)} examples from evaluation dataset")
    
    return df


def run_inference_for_evaluation(config: ExperimentConfig, debug: bool = False) -> pd.DataFrame:
    """
    Run inference using the existing inference system and return results.
    
    Args:
        config: Experiment configuration
        
    Returns:
        DataFrame containing inference results
    """
    print("Running inference for evaluation...")
    
    # Run inference using existing system
    run_inference(config, debug)
    
    # Load the inference results
    results_df = pd.read_csv(config.output_file)
    print(f"Loaded {len(results_df)} inference results from {config.output_file}")
    
    return results_df


def calculate_accuracy_metrics(predictions: List[Optional[str]], 
                             ground_truths: List[str]) -> Dict[str, float]:
    """
    Calculate accuracy metrics by comparing predictions with ground truths.
    
    Args:
        predictions: List of predicted answers (can contain None values)
        ground_truths: List of ground truth answers
        
    Returns:
        Dictionary containing accuracy metrics
    """
    if len(predictions) != len(ground_truths):
        raise ValueError(f"Mismatch in lengths: {len(predictions)} predictions vs {len(ground_truths)} ground truths")
    
    total_examples = len(predictions)
    correct_count = 0
    valid_predictions = 0
    invalid_predictions = 0
    
    detailed_results = []
    
    for i, (pred, truth) in enumerate(zip(predictions, ground_truths)):
        if pred is not None:
            valid_predictions += 1
            is_correct = pred.strip().lower() == truth.strip().lower()
            if is_correct:
                correct_count += 1
            
            detailed_results.append({
                'prediction': pred,
                'ground_truth': truth,
                'correct': is_correct
            })            
        else:
            invalid_predictions += 1
            detailed_results.append({
                'prediction': None,
                'ground_truth': truth,
                'correct': False
            })
    
    # Calculate metrics
    overall_accuracy = correct_count / total_examples if total_examples > 0 else 0.0
    valid_accuracy = correct_count / valid_predictions if valid_predictions > 0 else 0.0
    extraction_success_rate = valid_predictions / total_examples if total_examples > 0 else 0.0
    
    metrics = {
        'overall_accuracy': overall_accuracy,
        'valid_accuracy': valid_accuracy,
        'extraction_success_rate': extraction_success_rate,
        'total_examples': total_examples,
        'correct_predictions': correct_count,
        'valid_predictions': valid_predictions,
        'invalid_predictions': invalid_predictions
    }
    
    return metrics, detailed_results


def load_eval_recipe_with_overrides(recipe_path: str, overrides: list):
    """Load evaluation recipe from file and/or command-line arguments.
    
    Priority (highest to lowest):
    1. Command-line overrides
    2. Recipe file values
    3. Default values
    """
    if not recipe_path:
        raise ValueError("recipe_path is required")
    
    # Load base recipe from YAML
    recipe_dict = load_recipe_from_yaml(recipe_path)
    
    # Apply command-line overrides
    if overrides:
        recipe_dict = apply_overrides_to_recipe(recipe_dict, overrides)
    
    # Write temporary recipe file with overrides applied
    import tempfile
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as tmp_file:
        yaml.dump(recipe_dict, tmp_file)
        tmp_recipe_path = tmp_file.name
    
    try:
        # Load using existing config loader
        config = ExperimentConfig.load_eval_config(tmp_recipe_path)
    finally:
        # Clean up temporary file
        os.unlink(tmp_recipe_path)
    
    return config


def run_comprehensive_evaluation(recipe_path: str, 
                                debug: bool = False,
                                overrides: list = None) -> Dict[str, Any]:
    """
    Run comprehensive evaluation by:
    1. Loading evaluation dataset
    2. Running inference using existing system
    3. Extracting predicted and ground truth answers
    4. Calculating accuracy metrics
    
    Args:
        recipe_path: Path to evaluation recipe file
        debug: Enable debug logging
        overrides: List of recipe overrides
        
    Returns:
        Dictionary containing evaluation results
    """
    print("Starting comprehensive evaluation...")
    print(f"Recipe: {recipe_path}")
    
    try:
        # Load recipe to get dataset name and other settings with overrides
        config = load_eval_recipe_with_overrides(recipe_path, overrides or [])
        choice_labels = getattr(config, 'choice_labels', None)
        dataset_subset = getattr(config, 'dataset_subset', None)
        evaluation_split = getattr(config, 'dataset_split', 'test')
        ground_truth_column = getattr(config, 'ground_truth_column', 'answer')
        dataset_columns = getattr(config, 'dataset_columns', [])
        
        # Use dataset name from config
        evaluation_dataset_name = config.dataset_name
        print(f"Evaluation dataset: {evaluation_dataset_name}")
        if dataset_subset:
            print(f"Dataset subset: {dataset_subset}")
        
        # Load evaluation dataset to get ground truths
        eval_dataset = load_evaluation_dataset(evaluation_dataset_name, evaluation_split, dataset_subset)
        
        # Run inference
        inference_results = run_inference_for_evaluation(config, debug)
        
        # Extract predicted answers
        print("Extracting predicted answers...")
        response_col = getattr(config, 'response_column', 'response')
        predicted_answers = []
        for response in inference_results[response_col]:
            pred_answer = extract_predicted_answer(
                response, 
                dataset_name=evaluation_dataset_name,
                choice_labels=choice_labels
            )
            predicted_answers.append(pred_answer)
        
        # Extract ground truth answers
        print("Extracting ground truth answers...")
        ground_truth_answers = []
        for truth_text in eval_dataset[ground_truth_column]:
            gt_answer = extract_ground_truth(
                str(truth_text), 
                dataset_name=evaluation_dataset_name,
                choice_labels=choice_labels
            )
            ground_truth_answers.append(gt_answer)
        
        # Calculate accuracy metrics with dataset columns
        print("Calculating accuracy metrics...")
        metrics, detailed_results = calculate_accuracy_metrics_with_dataset_columns(
            predicted_answers, 
            ground_truth_answers,
            eval_dataset,
            dataset_columns
        )
        
        # Print summary results
        print("Evaluation Results Summary:")
        print(f"Overall Accuracy: {metrics['overall_accuracy']:.4f} ({metrics['correct_predictions']}/{metrics['total_examples']})")
        print(f"Valid Accuracy: {metrics['valid_accuracy']:.4f} ({metrics['correct_predictions']}/{metrics['valid_predictions']})")
        print(f"Extraction Success Rate: {metrics['extraction_success_rate']:.4f} ({metrics['valid_predictions']}/{metrics['total_examples']})")
        
        # Save detailed results
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        evaluation_results = {
            'recipe_path': recipe_path,
            'evaluation_dataset': evaluation_dataset_name,
            'evaluation_split': evaluation_split,
            'ground_truth_column': ground_truth_column,
            'dataset_columns': dataset_columns,
            'system_prompt': config.system_prompt,
            'metrics': metrics,
            'detailed_results': detailed_results,
            'timestamp': timestamp
        }

        def convert_to_python_types(obj):
            if isinstance(obj, dict):
                return {k: convert_to_python_types(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_to_python_types(v) for v in obj]
            elif isinstance(obj, (np.integer, np.int64)):
                return int(obj)
            elif isinstance(obj, (np.floating, np.float32, np.float64)):
                return float(obj)
            else:
                return obj

        eval_summary_file = config.output_file.replace('.csv', '-eval-summary.json')
        os.makedirs(os.path.dirname(eval_summary_file), exist_ok=True)
        with open(eval_summary_file, 'w') as f:
            json.dump(convert_to_python_types(evaluation_results), f, indent=2)

        # Save detailed results as CSV too
        detailed_df = pd.DataFrame(detailed_results)
        csv_eval_output_file = config.output_file.replace('.csv', '-eval.csv')
        detailed_df.to_csv(csv_eval_output_file, index=False, encoding='utf-8', quoting=csv.QUOTE_ALL)
        print(f"Detailed results CSV saved to: {csv_eval_output_file}")
        
        print("\n" + "="*50)
        print("EVALUATION COMPLETED SUCCESSFULLY")
        print("="*50)
        print(f"Overall Accuracy: {metrics['overall_accuracy']:.4f}")
        print(f"Valid Accuracy: {metrics['valid_accuracy']:.4f}")
        print(f"Extraction Success Rate: {metrics['extraction_success_rate']:.4f}")
        print("="*50)
        
        return evaluation_results
        
    except Exception as e:
        print(f"Evaluation failed: {str(e)}")
        if debug:
            import traceback
            traceback.print_exc()
        raise


def calculate_accuracy_metrics_with_dataset_columns(predictions: List[Optional[str]], 
                                                   ground_truths: List[str],
                                                   eval_dataset: pd.DataFrame,
                                                   dataset_columns: List[str]) -> Dict[str, float]:
    """
    Calculate accuracy metrics by comparing predictions with ground truths,
    and include dataset columns in the detailed results.
    
    Args:
        predictions: List of predicted answers (can contain None values)
        ground_truths: List of ground truth answers
        eval_dataset: DataFrame containing the evaluation dataset
        dataset_columns: List of dataset column names to include in output
        
    Returns:
        Dictionary containing accuracy metrics and detailed results with dataset columns
    """
    if len(predictions) != len(ground_truths):
        raise ValueError(f"Mismatch in lengths: {len(predictions)} predictions vs {len(ground_truths)} ground truths")
    
    if len(predictions) != len(eval_dataset):
        raise ValueError(f"Mismatch in lengths: {len(predictions)} predictions vs {len(eval_dataset)} dataset rows")
    
    total_examples = len(predictions)
    correct_count = 0
    valid_predictions = 0
    invalid_predictions = 0
    
    detailed_results = []
    
    for i, (pred, truth) in enumerate(zip(predictions, ground_truths)):
        result_row = {
            'prediction': pred,
            'ground_truth': truth
        }
        
        # Add dataset columns to the result row
        for column in dataset_columns:
            if column in eval_dataset.columns:
                result_row[column] = eval_dataset.iloc[i][column]
            else:
                print(f"Warning: Column '{column}' not found in dataset")
                result_row[column] = None
        
        if pred is not None:
            valid_predictions += 1
            is_correct = pred.strip().lower() == truth.strip().lower()
            if is_correct:
                correct_count += 1
            result_row['correct'] = is_correct
        else:
            invalid_predictions += 1
            result_row['correct'] = False
        
        detailed_results.append(result_row)
    
    # Calculate metrics
    overall_accuracy = correct_count / total_examples if total_examples > 0 else 0.0
    valid_accuracy = correct_count / valid_predictions if valid_predictions > 0 else 0.0
    extraction_success_rate = valid_predictions / total_examples if total_examples > 0 else 0.0
    
    metrics = {
        'overall_accuracy': overall_accuracy,
        'valid_accuracy': valid_accuracy,
        'extraction_success_rate': extraction_success_rate,
        'total_examples': total_examples,
        'correct_predictions': correct_count,
        'valid_predictions': valid_predictions,
        'invalid_predictions': invalid_predictions
    }
    
    return metrics, detailed_results


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Run comprehensive model evaluation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Using recipe file:
  fai-rl-eval --recipe recipes/evaluation/mmlu/llama3_3B.yaml
  
  # Mix recipe file with overrides:
  fai-rl-eval --recipe recipe.yaml model_path='./my_model' temperature=0.7
  
  # Override evaluation parameters:
  fai-rl-eval --recipe recipe.yaml dataset_split='validation' max_new_tokens=256
"""
    )
    parser.add_argument(
        "--recipe",
        type=str,
        default=None,
        help="Path to evaluation recipe YAML file"
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug mode with verbose logging"
    )
    parser.add_argument(
        "--nohup",
        action="store_true",
        help="Run evaluation in background with nohup (output redirected to evaluation_<timestamp>.log)"
    )
    parser.add_argument(
        "overrides",
        nargs="*",
        help="Recipe overrides in key=value format (e.g., model_path='./output' dataset_split='test')"
    )
    
    args = parser.parse_args()
    
    # Add this check: if no arguments provided at all, show help and exit
    if len(sys.argv) == 1:
        parser.print_help(sys.stderr)
        sys.exit(1)
    
    return args


def main():
    """Main evaluation function."""
    args = parse_args()
    
    # Handle nohup mode
    if args.nohup:
        # Generate log filename with timestamp
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = f"evaluation_{timestamp}.log"
        
        print(f"Running evaluation in background with nohup. Output will be saved to: {log_file}")
        
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
            print(f"Evaluation started in background. Monitor progress with: tail -f {log_file}")
        
        return result
    
    # Create necessary directories
    os.makedirs("logs", exist_ok=True)
    os.makedirs("outputs", exist_ok=True)
    
    print("Starting Comprehensive Model Evaluation")
    print(f"Recipe: {args.recipe}")
    print()
    
    try:
        results = run_comprehensive_evaluation(
            recipe_path=args.recipe,
            debug=args.debug,
            overrides=args.overrides
        )
        
        return results
        
    except Exception as e:
        print(f"Evaluation failed with error: {str(e)}")
        if args.debug:
            import traceback
            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()