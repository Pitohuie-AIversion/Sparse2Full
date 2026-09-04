#!/bin/bash
# Curriculum learning launch script
# Usage: ./launch_curriculum.sh [experiment_name] [stages]

set -e  # Exit on error

# Default parameters
EXPERIMENT_NAME=${1:-"curriculum_experiment"}
STAGES=${2:-"all"}  # all, foundation, short, medium, long, hybrid, nar
CONFIG_FILE="configs/train_curriculum.yaml"
SEED=${3:-42}

# Set up environment
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
export PYTHONPATH="$PROJECT_ROOT${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONHASHSEED=0
export CUDA_LAUNCH_BLOCKING=1

# Create output directory with timestamp
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
OUTPUT_DIR="runs/${EXPERIMENT_NAME}_curriculum_${TIMESTAMP}"
mkdir -p "$OUTPUT_DIR"

# Log experiment info
echo "Starting curriculum learning experiment"
echo "Experiment: $EXPERIMENT_NAME"
echo "Stages: $STAGES"
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
    echo "Git Commit: $(git rev-parse HEAD 2>/dev/null || echo 'N/A')"
    echo "Git Branch: $(git rev-parse --abbrev-ref HEAD 2>/dev/null || echo 'N/A')"
} > "$OUTPUT_DIR/experiment_info.txt"

# Configure curriculum stages based on selection
case "$STAGES" in
    "foundation")
        CURRICULUM_CONFIG="curriculum.stages=['foundation']"
        ;;
    "short")
        CURRICULUM_CONFIG="curriculum.stages=['foundation','short_term']"
        ;;
    "medium")
        CURRICULUM_CONFIG="curriculum.stages=['foundation','short_term','medium_term']"
        ;;
    "long")
        CURRICULUM_CONFIG="curriculum.stages=['foundation','short_term','medium_term','long_term']"
        ;;
    "hybrid")
        CURRICULUM_CONFIG="curriculum.stages=['foundation','short_term','medium_term','long_term','hybrid']"
        ;;
    "nar")
        CURRICULUM_CONFIG="curriculum.stages=['foundation','short_term','medium_term','long_term','hybrid','nar_refinement']"
        ;;
    *)
        CURRICULUM_CONFIG=""  # Use default from config file
        ;;
esac

# Run curriculum training
echo "Starting curriculum training with stages: $STAGES"

if [ -n "$CURRICULUM_CONFIG" ]; then
    python tools/train.py \
        --config-path=$(dirname "$CONFIG_FILE") \
        --config-name=$(basename "$CONFIG_FILE" .yaml) \
        experiment.name="$EXPERIMENT_NAME" \
        experiment.seed=$SEED \
        experiment.output_dir="$OUTPUT_DIR" \
        $CURRICULUM_CONFIG \
        2>&1 | tee "$OUTPUT_DIR/training.log"
else
    python tools/train.py \
        --config-path=$(dirname "$CONFIG_FILE") \
        --config-name=$(basename "$CONFIG_FILE" .yaml) \
        experiment.name="$EXPERIMENT_NAME" \
        experiment.seed=$SEED \
        experiment.output_dir="$OUTPUT_DIR" \
        2>&1 | tee "$OUTPUT_DIR/training.log"
fi

# Check if training was successful
if [ $? -eq 0 ]; then
    echo "Curriculum training completed successfully!"
    echo "Results saved to: $OUTPUT_DIR"
    
    # Generate curriculum progression report
    if [ -f "$OUTPUT_DIR/curriculum_progress.json" ]; then
        echo "Curriculum Progression:"
        cat "$OUTPUT_DIR/curriculum_progress.json" | python -m json.tool
    fi
    
    # Create experiment completed marker
    echo "$(date): Curriculum training completed successfully" > "$OUTPUT_DIR/COMPLETED"
else
    echo "Curriculum training failed! Check logs: $OUTPUT_DIR/training.log"
    exit 1
fi
