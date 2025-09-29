"""
Comprehensive evaluation system that leverages the existing inference pipeline.
"""

import argparse
import os
import sys
import re
import json, csv
import pandas as pd
from datetime import datetime
from typing import Dict, List, Tuple, Any, Optional
from datasets import load_dataset
import numpy as np

# Add project root to path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Import existing modules
from core.config import ExperimentConfig
from inference.inference import run_inference, load_model_and_tokenizer, generate_response, generate_response_by_api


def extract_multiple_choice_answer(text, choice_labels=None):
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

    # Pattern 0: JSON format, e.g., "answer": "B" or “answer”: “B”
    pattern0 = r'[“"]ANSWER[”"]\s*:\s*[“"]([' + ''.join(choice_labels) + r'])[”"]'

    match = re.search(pattern0, text_upper)
    if match:
        return match.group(1)
    
    # Pattern 1: "The best/final answer/option is A" or "The answer/option is A"
    pattern1 = r'(?:THE\s+)?(?:BEST|FINAL)?\s*(?:ANSWER|OPTION)\s+IS\s+([' + ''.join(choice_labels) + r'])'
    match = re.search(pattern1, text_upper)
    if match:
        return match.group(1)
    
    # Pattern 2: "Answer: A" (with optional additional text)
    pattern2 = r'ANSWER:\s*([' + ''.join(choice_labels) + r'])'
    match = re.search(pattern2, text_upper)
    if match:
        return match.group(1)

    # Pattern 3 for "option/answer C is the best/final answer/option"
    pattern3 = r'(?:OPTION|ANSWER)\s+([' + ''.join(choice_labels) + r'])\s+IS\s+(?:THE\s+)?(?:BEST|FINAL)?\s*(?:ANSWER|OPTION)'
    match = re.search(pattern3, text_upper)
    if match:
        return match.group(1)

    # Pattern 4: "A."
    first_char = text.strip().upper()[0]
    
    return first_char if first_char in choice_labels else None


def convert_answer_index_to_letter(answer_index, choice_labels=None):
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


def extract_predicted_answer(text, output_type="numerical", choice_labels=None):
    """
    Extract predicted answer from model response based on output type.
    
    Args:
        text: Model response text
        output_type: Type of answer format ("numerical", "multiple_choice")
        choice_labels: List of choice labels for multiple choice
    
    Returns:
        Extracted answer or None if not found
    """
    if output_type == "multiple_choice":
        return extract_multiple_choice_answer(text, choice_labels)
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


def extract_ground_truth(text, output_type="numerical", choice_labels=None):
    """
    Extract ground truth answer based on output type.
    
    Args:
        text: Ground truth text or index
        output_type: Type of answer format
        choice_labels: List of choice labels for multiple choice
    
    Returns:
        Processed ground truth answer
    """
    if output_type == "multiple_choice":
        # For MMLU, ground truth is typically a numerical index
        return convert_answer_index_to_letter(text, choice_labels)
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


def run_comprehensive_evaluation(config_path: str, 
                                debug: bool = False) -> Dict[str, Any]:
    """
    Run comprehensive evaluation by:
    1. Loading evaluation dataset
    2. Running inference using existing system
    3. Extracting predicted and ground truth answers
    4. Calculating accuracy metrics
    
    Args:
        config_path: Path to evaluation configuration file
        debug: Enable debug logging
        
    Returns:
        Dictionary containing evaluation results
    """
    print("Starting comprehensive evaluation...")
    print(f"Config: {config_path}")
    
    try:
        # Load configuration to get dataset name and other settings
        config = ExperimentConfig.load_eval_config(config_path)
        output_type = getattr(config, 'output_type', 'numerical')
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
        predicted_answers = []
        for response in inference_results['response']:
            pred_answer = extract_predicted_answer(
                response, 
                output_type=output_type,
                choice_labels=choice_labels
            )
            predicted_answers.append(pred_answer)
        
        # Extract ground truth answers
        print("Extracting ground truth answers...")
        ground_truth_answers = []
        for truth_text in eval_dataset[ground_truth_column]:
            gt_answer = extract_ground_truth(
                str(truth_text), 
                output_type=output_type,
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
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        evaluation_results = {
            'config_path': config_path,
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
    parser = argparse.ArgumentParser(description="Run comprehensive model evaluation")
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


def main():
    """Main evaluation function."""
    args = parse_args()
    
    # Create necessary directories
    os.makedirs("logs", exist_ok=True)
    os.makedirs("outputs", exist_ok=True)
    
    print("Starting Comprehensive Model Evaluation")
    print(f"Config: {args.config}")
    print()
    
    try:
        results = run_comprehensive_evaluation(
            config_path=args.config,
            debug=args.debug
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