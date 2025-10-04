#!/bin/bash
# scripts/run_training.sh - Unified training script for DPO, GRPO, GSPO, PPO, and SFT

set -e  # Exit on any error

# Default values
CONFIG=""
NUM_GPUS=8
OUTPUT_DIR=""
LOG_DIR="logs"
NOHUP_MODE=0  # Whether to run via nohup
DEEPSPEED_CONFIG=""
NO_DEEPSPEED=0  # Whether to disable DeepSpeed

# Function to extract algorithm from config file
extract_algorithm_from_config() {
    local config_path="$1"
    if [ -f "$config_path" ]; then
        # Use python to parse YAML and extract algorithm
        python3 -c "
import yaml
import sys
try:
    with open('$config_path', 'r') as f:
        config = yaml.safe_load(f)
    algorithm = config.get('training', {}).get('algorithm', '')
    print(algorithm.lower())
except Exception as e:
    sys.exit(1)
"
    else
        echo ""
    fi
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
  case $1 in
    --config)
      CONFIG="$2"
      shift 2
      ;;
    --num-gpus)
      NUM_GPUS="$2"
      shift 2
      ;;
    --deepspeed-config)
      DEEPSPEED_CONFIG="$2"  # Allow manual override
      shift 2
      ;;
    --no-deepspeed)
      NO_DEEPSPEED=1
      shift
      ;;
    --nohup)
      NOHUP_MODE=1
      shift
      ;;
    -h|--help)
      echo "Usage: $0 --config CONFIG_FILE [OPTIONS]"
      echo ""
      echo "Options:"
      echo "  --config CONFIG_FILE                   Path to configuration YAML file (required)"
      echo "  --num-gpus NUM_GPUS                    Number of GPUs to use (default: 8)"
      echo "  --deepspeed-config DEEPSPEED_CONFIG    Override ZeRO config file (auto-selected if not provided)"
      echo "  --nohup                                Run in background with nohup"
      echo "  -h, --help                             Show this help message"
      echo ""
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

# Extract algorithm from config file
ALGORITHM=$(extract_algorithm_from_config "$CONFIG")
if [ -z "$ALGORITHM" ]; then
    echo "Error: Could not extract algorithm from config file. Make sure 'training.algorithm' is specified in the YAML."
    exit 1
fi

# Validate algorithm
if [[ "$ALGORITHM" != "dpo" && "$ALGORITHM" != "grpo" && "$ALGORITHM" != "gspo" && "$ALGORITHM" != "ppo" && "$ALGORITHM" != "sft" ]]; then
    echo "Error: Algorithm in config must be 'dpo', 'grpo', 'gspo', 'ppo', or 'sft', found: $ALGORITHM"
    exit 1
fi

echo "Detected algorithm from config: $ALGORITHM"

# Check if config uses QLoRA (4-bit or 8-bit quantization)
# QLoRA is incompatible with DeepSpeed, so automatically disable it
USES_QUANTIZATION=$(python3 -c "
import yaml
import sys
try:
    with open('$CONFIG', 'r') as f:
        config = yaml.safe_load(f)
    model = config.get('model', {})
    if model.get('load_in_4bit', False) or model.get('load_in_8bit', False):
        print('true')
    else:
        print('false')
except Exception:
    print('false')
" 2>/dev/null)

if [ "$USES_QUANTIZATION" = "true" ]; then
    # QLoRA (bitsandbytes quantization) is incompatible with DeepSpeed
    # Automatically disable DeepSpeed and use torchrun for multi-GPU training
    echo "Detected quantization (QLoRA) - DeepSpeed is incompatible with bitsandbytes."
    echo "Automatically disabling DeepSpeed and using torchrun for distributed training."
    NO_DEEPSPEED=1
    DEEPSPEED_CONFIG=""
fi

# Auto-select deepspeed config if not provided and not disabled
if [ "$NO_DEEPSPEED" -eq 0 ]; then
    if [ -z "$DEEPSPEED_CONFIG" ]; then
        # Use Zero Stage 3 for full fine-tuning
        DEEPSPEED_CONFIG="configs/deepspeed/zero3_config_gpu${NUM_GPUS}.json"
        echo "Auto-selected deepspeed config: $DEEPSPEED_CONFIG"
    fi

    if [ ! -f "$DEEPSPEED_CONFIG" ]; then
        echo "Error: Deepspeed config file '$DEEPSPEED_CONFIG' not found"
        exit 1
    fi
else
    echo "DeepSpeed disabled - using torchrun for distributed training"
    DEEPSPEED_CONFIG=""
fi

mkdir -p "$LOG_DIR"

# Build command arguments
SCRIPT_ARGS="--config $CONFIG"
[ -n "$DEEPSPEED_CONFIG" ] && SCRIPT_ARGS="$SCRIPT_ARGS --deepspeed-config $DEEPSPEED_CONFIG"
[ -n "$OUTPUT_DIR" ] && SCRIPT_ARGS="$SCRIPT_ARGS --output-dir $OUTPUT_DIR"

TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_FILE="$LOG_DIR/${ALGORITHM}_training_${TIMESTAMP}.log"

echo "Starting ${ALGORITHM^^} training with the following configuration:"
echo "  Algorithm: $ALGORITHM (from config)"
echo "  Config: $CONFIG"
echo "  Deepspeed Config: ${DEEPSPEED_CONFIG:-disabled}"
echo "  GPUs: $NUM_GPUS"
echo "  Log file: $LOG_FILE"
echo "  Additional args: $SCRIPT_ARGS"
echo ""

if [ -n "$DEEPSPEED_CONFIG" ]; then
    CMD="deepspeed --num_gpus=$NUM_GPUS scripts/train.py $SCRIPT_ARGS"
else
    # Run without DeepSpeed (use torchrun for multi-GPU or python for single GPU)
    if [ "$NUM_GPUS" -gt 1 ]; then
        CMD="torchrun --nproc_per_node=$NUM_GPUS scripts/train.py $SCRIPT_ARGS"
    else
        CMD="python scripts/train.py $SCRIPT_ARGS"
    fi
fi

if [ "$NOHUP_MODE" -eq 1 ]; then
    echo "Running in background with nohup..."
    nohup bash -c "$CMD" > "$LOG_FILE" 2>&1 &
    echo "Training started in background. Use 'tail -f $LOG_FILE' to monitor logs."
else
    echo "Running in foreground..."
    $CMD 2>&1 | tee "$LOG_FILE"
    if [ $? -eq 0 ]; then
        echo ""
        echo "${ALGORITHM^^} training completed successfully!"
        echo "Log file: $LOG_FILE"
    else
        echo ""
        echo "Training failed. Check the log file for details: $LOG_FILE"
        exit 1
    fi
fi
