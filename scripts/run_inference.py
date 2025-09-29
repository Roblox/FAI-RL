"""
Main script for running model inference.
"""

import argparse
import time
from pathlib import Path
import sys, os

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from core.config import ExperimentConfig
from utils.logging_utils import TrainingLogger, log_system_info


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


def main():
    """Main inference function."""
    args = parse_args()
    
    # Load configuration
    config = ExperimentConfig.load_inference_config(args.config)
    
    # Setup logging
    inference_logger = TrainingLogger("model_inference")
    
    # Log system information
    log_system_info()
    
    # Log inference configuration
    inference_logger.log_experiment_start({
        "inference_config": config.to_dict(),
    })
    
    start_time = time.time()
    
    try:
        # Import and run the inference
        from inference.inference import run_inference
        
        print("Starting inference with the following configuration:")
        
        # Check if we should use API-based inference
        use_api = hasattr(config, 'mlp_api_key') and config.mlp_api_key
        
        if use_api:
            print(f"  Model (API): {config.model}")
            print(f"  Inference type: API-based")
        else:
            print(f"  Model path: {config.model_path}")
            print(f"  Inference type: Local model")
        
        print(f"  Dataset: {config.dataset_name}")
        print(f"  Output file: {config.output_file}")
        print(f"  Temperature: {config.temperature}")
        print(f"  Top-p: {config.top_p}")
        print(f"  Max new tokens: {config.max_new_tokens}")
        print(f"  Do sample: {config.do_sample}")
        print()
        
        # Run inference
        run_inference(config, debug=args.debug)
        
        inference_logger.logger.info("Inference completed successfully!")
        
    except Exception as e:
        inference_logger.logger.error(f"Inference failed with error: {str(e)}")
        if args.debug:
            import traceback
            traceback.print_exc()
        raise
        
    finally:
        # Log experiment end
        end_time = time.time()
        duration = end_time - start_time
        inference_logger.log_experiment_end(duration)


if __name__ == "__main__":
    main()
