#!/bin/bash
# Basic training launch script
# Usage: ./launch_basic.sh [experiment_name] [config_file]

set -e  # Exit on error

# Default parameters
EXPERIMENT_NAME=${1:-"basic_experiment"}
CONFIG_FILE=${2:-"configs/train_basic.yaml"}
SEED=${3:-42}

# Set up environment
export PYTHONPATH=/share/fandixiaLab/suguangsheng/PycharmProjects/Sparse2Full:$PYTHONPATH
export PYTHONHASHSEED=0
export CUDA_LAUNCH_BLOCKING=1  # For better error messages

# Create output directory
OUTPUT_DIR="runs/${EXPERIMENT_NAME}_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$OUTPUT_DIR"

# Log experiment info
echo "Starting basic training experiment"
echo "Experiment: $EXPERIMENT_NAME"
echo "Config: $CONFIG_FILE"
echo "Seed: $SEED"
echo "Output: $OUTPUT_DIR"
echo "Date: $(date)"

# Record environment information
{
    echo "Environment Information:"
    echo "Python: $(python --version)"
    echo "PyTorch: $(python -c 'import torch; print(torch.__version__)')"
    echo "CUDA Available: $(python -c 'import torch; print(torch.cuda.is_available())')"
    echo "CUDA Version: $(python -c 'import torch; print(torch.version.cuda if torch.cuda.is_available() else \"N/A\")')"
    echo "Git Commit: $(git rev-parse HEAD 2>/dev/null || echo 'N/A')"
    echo "Git Branch: $(git rev-parse --abbrev-ref HEAD 2>/dev/null || echo 'N/A')"
} > "$OUTPUT_DIR/experiment_info.txt"

# Run training with logging
python tools/train.py \
    --config-path=$(dirname "$CONFIG_FILE") \
    --config-name=$(basename "$CONFIG_FILE" .yaml) \
    experiment.name="$EXPERIMENT_NAME" \
    experiment.seed=$SEED \
    experiment.output_dir="$OUTPUT_DIR" \
    2>&1 | tee "$OUTPUT_DIR/training.log"

# Check if training was successful
if [ $? -eq 0 ]; then
    echo "Training completed successfully!"
    echo "Results saved to: $OUTPUT_DIR"
    
    # Generate summary
    if [ -f "$OUTPUT_DIR/paper_package/metrics/summary.csv" ]; then
        echo "Performance Summary:"
        cat "$OUTPUT_DIR/paper_package/metrics/summary.csv"
    fi
    
    # Create experiment completed marker
    echo "$(date): Training completed successfully" > "$OUTPUT_DIR/COMPLETED"
else
    echo "Training failed! Check logs: $OUTPUT_DIR/training.log"
    exit 1
fi