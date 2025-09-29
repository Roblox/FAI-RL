#!/bin/bash

# Simple script to run evaluations with common datasets

# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# Get project root (parent directory of scripts)
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

# Default values
CONFIG_PATH=""
DEBUG_FLAG=""

# Function to display usage
usage() {
    echo "Usage: $0 --config <config_path> [--debug]"
    echo ""
    echo "Options:"
    echo "  --config    Path to evaluation config (required)"
    echo "  --debug     Enable debug mode"
    echo "  --help      Show this help message"
    exit 1
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --config)
            CONFIG_PATH="$2"
            shift 2
            ;;
        --debug)
            DEBUG_FLAG="--debug"
            shift
            ;;
        --help)
            usage
            ;;
        *)
            echo "Unknown option: $1"
            usage
            ;;
    esac
done

# Check if required arguments are provided
if [[ -z "$CONFIG_PATH" ]]; then
    echo "Error: --config argument is required"
    echo ""
    usage
fi

# Check if config file exists
if [[ ! -f "$CONFIG_PATH" ]]; then
    echo "Error: Config file '$CONFIG_PATH' does not exist"
    exit 1
fi

echo "Running evaluation with config: $CONFIG_PATH"
echo ""

# Change to project root directory
cd "$PROJECT_ROOT" || {
    echo "Error: Could not change to project directory: $PROJECT_ROOT"
    exit 1
}

# Add project root to PYTHONPATH
export PYTHONPATH="$PROJECT_ROOT:$PYTHONPATH"

# Run the evaluation
python3 -c "
import sys
from evaluations.eval import run_comprehensive_evaluation

try:
    results = run_comprehensive_evaluation(
        config_path='$CONFIG_PATH',
        debug=$([[ -n "$DEBUG_FLAG" ]] && echo "True" || echo "False")
    )
    print('Evaluation completed successfully!')
except Exception as e:
    print(f'Evaluation failed: {str(e)}')
    sys.exit(1)
" || {
    echo "Evaluation failed"
    exit 1
}