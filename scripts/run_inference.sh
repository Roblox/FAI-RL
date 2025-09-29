#!/bin/bash
# scripts/run_inference.sh

set -e  # Exit on any error

# Default values
CONFIG=""
LOG_DIR="logs"
NOHUP_MODE=0  # Whether to run via nohup
DEBUG_MODE=0  # Whether to enable debug mode

# Parse command line arguments
while [[ $# -gt 0 ]]; do
  case $1 in
    --config)
      CONFIG="$2"
      shift 2
      ;;

    --nohup)
      NOHUP_MODE=1
      shift
      ;;
    --debug)
      DEBUG_MODE=1
      shift
      ;;
    -h|--help)
      echo "Usage: $0 --config CONFIG_FILE [OPTIONS]"
      echo ""
      echo "Options:"
      echo "  --config CONFIG_FILE              Path to inference configuration YAML file (required)"
      echo "  --nohup                           Run in background with nohup"
      echo "  --debug                           Enable debug mode with verbose logging"
      echo "  -h, --help                        Show this help message"
      echo ""
      echo "Examples:"
      echo "  $0 --config configs/inference/llama3_3B_inference.yaml"
      echo "  $0 --config configs/inference/llama3_3B_inference.yaml --nohup"
      echo "  $0 --config configs/inference/llama3_3B_inference.yaml --debug"
      exit 0
      ;;
    *)
      echo "Unknown option $1"
      exit 1
      ;;
  esac
done

# Check required arguments
if [ -z "$CONFIG" ]; then
    echo "Error: --config is required"
    echo "Use --help for usage information"
    exit 1
fi

if [ ! -f "$CONFIG" ]; then
    echo "Error: Config file '$CONFIG' not found"
    exit 1
fi

mkdir -p "$LOG_DIR"

# Build command arguments
SCRIPT_ARGS="--config $CONFIG"
[ "$DEBUG_MODE" -eq 1 ] && SCRIPT_ARGS="$SCRIPT_ARGS --debug"

TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_FILE="$LOG_DIR/inference_${TIMESTAMP}.log"

echo "Starting model inference with the following configuration:"
echo "  Config: $CONFIG"
[ "$DEBUG_MODE" -eq 1 ] && echo "  Debug mode: enabled"
echo "  Log file: $LOG_FILE"
echo ""

CMD="python scripts/run_inference.py $SCRIPT_ARGS"

if [ "$NOHUP_MODE" -eq 1 ]; then
    echo "Running in background with nohup..."
    nohup bash -c "$CMD" > "$LOG_FILE" 2>&1 &
    echo "Inference started in background. Use 'tail -f $LOG_FILE' to monitor logs."
else
    echo "Running in foreground..."
    $CMD 2>&1 | tee "$LOG_FILE"
    if [ $? -eq 0 ]; then
        echo ""
        echo "Inference completed successfully!"
        echo "Log file: $LOG_FILE"
    else
        echo ""
        echo "Inference failed. Check the log file for details: $LOG_FILE"
        exit 1
    fi
fi
