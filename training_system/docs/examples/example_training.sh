#!/bin/bash
# Comprehensive example script demonstrating all training modes
# Usage: ./example_training.sh

set -e

echo "=== PDEBench Training Launch System Examples ==="
echo "This script demonstrates various training modes and configurations"
echo ""

# Set up environment
export PYTHONPATH=/share/fandixiaLab/suguangsheng/PycharmProjects/Sparse2Full:$PYTHONPATH
export PYTHONHASHSEED=0

# Create examples directory
EXAMPLES_DIR="runs/examples"
mkdir -p "$EXAMPLES_DIR"

echo "1. Basic Training Example (Quick Test)"
echo "======================================="
./tools/launch_basic.sh \
    "quick_test" \
    "configs/quick_test.yaml" \
    42

echo ""
echo "2. Basic Training with Super-Resolution (SRx4)"
echo "==============================================="
./tools/launch_basic.sh \
    "srx4_baseline" \
    "configs/train_basic.yaml" \
    42

echo ""
echo "3. Basic Training with Cropping (40% observation)"
echo "=================================================="
python tools/train.py \
    --config-path=configs \
    --config-name=train_basic \
    experiment.name="crop40_baseline" \
    experiment.seed=42 \
    experiment.output_dir="$EXAMPLES_DIR/crop40_baseline" \
    data.observation_mode="Crop" \
    data.observation_params.crop_size=102 \
    training.epochs=50

echo ""
echo "4. Curriculum Learning Example"
echo "==============================="
./tools/launch_curriculum.sh \
    "curriculum_example" \
    "foundation,short,medium" \
    42

echo ""
echo "5. Benchmark Comparison (Small Scale)"
echo "========================================"
./tools/launch_benchmark.sh \
    "mini_benchmark" \
    "SwinUNet,UNet" \
    "SRx2,SRx4" \
    42

echo ""
echo "6. Multi-Seed Experiment (Reproducibility)"
echo "=========================================="
for seed in 42 123 456; do
    echo "Running experiment with seed: $seed"
    python tools/train.py \
        --config-path=configs \
        --config-name=train_basic \
        experiment.name="multiseed_s${seed}" \
        experiment.seed=$seed \
        experiment.output_dir="$EXAMPLES_DIR/multiseed_s${seed}" \
        training.epochs=30
done

echo ""
echo "7. Custom Loss Function Experiment"
echo "==================================="
python tools/train.py \
    --config-path=configs \
    --config-name=train_basic \
    experiment.name="custom_loss_test" \
    experiment.seed=42 \
    experiment.output_dir="$EXAMPLES_DIR/custom_loss_test" \
    loss.reconstruction_weight=1.0 \
    loss.spectral_weight=0.8 \
    loss.dc_weight=1.2 \
    loss.loss_types.reconstruction="L1" \
    training.epochs=25

echo ""
echo "8. High-Performance Configuration"
echo "=================================="
python tools/train.py \
    --config-path=configs \
    --config-name=high_performance_config \
    experiment.name="high_perf_test" \
    experiment.seed=42 \
    experiment.output_dir="$EXAMPLES_DIR/high_perf_test" \
    training.batch_size=32 \
    training.num_workers=8

echo ""
echo "9. Generate Paper Package from Existing Run"
echo "============================================"
if [ -d "$EXAMPLES_DIR/quick_test" ]; then
    python tools/generate_paper_package.py \
        --run-dir="$EXAMPLES_DIR/quick_test" \
        --output-dir="$EXAMPLES_DIR/quick_test/paper_package_manual"
else
    echo "Skipping paper package generation (no existing run found)"
fi

echo ""
echo "10. Data Consistency Validation"
echo "================================"
python tools/check_dc_equivalence.py \
    --config=configs/train_basic.yaml \
    --num-samples=50 \
    --output-dir="$EXAMPLES_DIR/consistency_check"

echo ""
echo "=== All Examples Completed ==="
echo ""
echo "Results are saved in: $EXAMPLES_DIR"
echo ""
echo "To view results:"
echo "  ls -la $EXAMPLES_DIR"
echo ""
echo "To reproduce any experiment:"
echo "  cd <experiment_directory>"
echo "  ./scripts/reproduce.sh"
echo ""
echo "To generate paper packages:"
echo "  python tools/generate_paper_package.py --run-dir=<experiment_directory>"
echo ""
echo "Example training launch system demonstration complete!"