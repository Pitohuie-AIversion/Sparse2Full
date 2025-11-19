#!/bin/bash
# Benchmark training launch script
# Usage: ./launch_benchmark.sh [experiment_name] [models] [obs_modes]

set -e  # Exit on error

# Default parameters
EXPERIMENT_NAME=${1:-"benchmark_experiment"}
MODELS=${2:-"SwinUNet"}  # comma-separated: SwinUNet,SwinTemporalNAR,UNet
OBS_MODES=${3:-"SRx2,SRx4"}  # comma-separated: SRx2,SRx4,Crop40,Crop20
CONFIG_FILE="configs/train_benchmark.yaml"
SEED=${4:-42}

# Set up environment
export PYTHONPATH=/share/fandixiaLab/suguangsheng/PycharmProjects/Sparse2Full:$PYTHONPATH
export PYTHONHASHSEED=0
export CUDA_LAUNCH_BLOCKING=1

# Create output directory with timestamp
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
OUTPUT_DIR="runs/${EXPERIMENT_NAME}_benchmark_${TIMESTAMP}"
mkdir -p "$OUTPUT_DIR"

# Log experiment info
echo "Starting benchmark experiment"
echo "Experiment: $EXPERIMENT_NAME"
echo "Models: $MODELS"
echo "Observation Modes: $OBS_MODES"
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

# Parse model and observation mode lists into arrays
IFS=',' read -ra MODEL_ARRAY <<< "$MODELS"
IFS=',' read -ra OBS_ARRAY <<< "$OBS_MODES"

# Create benchmark configuration
echo "Running benchmark with models: ${MODEL_ARRAY[*]} and observation modes: ${OBS_ARRAY[*]}"

# Run benchmark training
python tools/train.py \
    --config-path=$(dirname "$CONFIG_FILE") \
    --config-name=$(basename "$CONFIG_FILE" .yaml) \
    experiment.name="$EXPERIMENT_NAME" \
    experiment.seed=$SEED \
    experiment.output_dir="$OUTPUT_DIR" \
    benchmark.models="[$MODELS]" \
    benchmark.observation_modes="[$OBS_MODES]" \
    2>&1 | tee "$OUTPUT_DIR/training.log"

# Check if training was successful
if [ $? -eq 0 ]; then
    echo "Benchmark training completed successfully!"
    echo "Results saved to: $OUTPUT_DIR"
    
    # Generate benchmark summary report
    if [ -f "$OUTPUT_DIR/benchmark_summary.json" ]; then
        echo "Benchmark Summary:"
        cat "$OUTPUT_DIR/benchmark_summary.json" | python -m json.tool
        
        # Extract key results
        echo "Key Results:"
        python -c "
import json
with open('$OUTPUT_DIR/benchmark_summary.json', 'r') as f:
    summary = json.load(f)

print('Model Performance Comparison:')
for model, results in summary.get('results', {}).items():
    print(f'  {model}:')
    for obs_mode, metrics in results.items():
        rel_l2 = metrics.get('rel_l2', {}).get('mean', 'N/A')
        psnr = metrics.get('psnr', {}).get('mean', 'N/A')
        print(f'    {obs_mode}: Rel-L2={rel_l2:.4f}, PSNR={psnr:.2f}')
"
    fi
    
    # Create benchmark completed marker
    echo "$(date): Benchmark training completed successfully" > "$OUTPUT_DIR/COMPLETED"
else
    echo "Benchmark training failed! Check logs: $OUTPUT_DIR/training.log"
    exit 1
fi