# PDEBench Training Launch System

This document provides comprehensive guidance for using the PDEBench training launch system for sparse observation reconstruction.

## Overview

The training launch system provides a complete pipeline for training, validation, monitoring, and paper package generation following the golden rules for reproducibility and consistency.

## Quick Start

### Basic Training

```bash
# Train with default configuration
python tools/train.py --config-path=configs --config-name=train_basic

# Train with custom parameters
python tools/train.py --config-path=configs --config-name=train_basic \
    experiment.name=my_experiment \
    experiment.seed=42 \
    training.epochs=100 \
    data.observation_mode=SR \
    data.observation_params.scale=4
```

### Curriculum Learning

```bash
# Train with curriculum learning
python tools/train.py --config-path=configs --config-name=train_curriculum \
    experiment.name=curriculum_experiment
```

### Benchmark Training

```bash
# Run comprehensive benchmark
python tools/train.py --config-path=configs --config-name=train_benchmark \
    experiment.name=benchmark_run
```

## Configuration System

### Configuration Structure

The system uses Hydra for configuration management with the following structure:

```
configs/
├── train_basic.yaml          # Basic training configuration
├── train_curriculum.yaml     # Curriculum learning configuration
├── train_benchmark.yaml      # Benchmark configuration
├── base_config.yaml          # Base configuration with common settings
└── [other specialized configs]
```

### Key Configuration Sections

#### Experiment Settings
```yaml
experiment:
  name: "SRx4-DR2D-256-SwinUNet"
  device: "cuda"
  seed: 42
  output_dir: "runs/${experiment.name}"
```

#### Data Configuration
```yaml
data:
  path: "datasets/PDEBench"
  name: "DR2D"
  keys: ["u"]
  image_size: 256
  observation_mode: "SR"  # or "Crop"
  observation_params:
    scale: 4  # for SR mode
    # crop_size: 64  # for Crop mode
```

#### Model Configuration
```yaml
model:
  name: "SwinUNet"
  params:
    in_ch: 1
    out_ch: 1
    img_size: 256
    embed_dim: 96
    depths: [2, 2, 6, 2]
    num_heads: [3, 6, 12, 24]
```

#### Training Configuration
```yaml
training:
  epochs: 200
  batch_size: 16
  optimizer:
    name: "AdamW"
    lr: 1e-3
    weight_decay: 1e-4
  scheduler:
    name: "CosineAnnealingWarmup"
    warmup_epochs: 1000
  gradient_clipping: 1.0
  mixed_precision: true
```

#### Loss Configuration
```yaml
loss:
  reconstruction_weight: 1.0
  spectral_weight: 0.5
  dc_weight: 1.0
  loss_types:
    reconstruction: "L2"
    spectral: "L2"
    dc: "L2"
  spectral_loss:
    kx: 16
    ky: 16
```

## Training Modes

### 1. Basic Training Mode

Standard training with fixed hyperparameters throughout the training process.

**Use case**: Single model training, quick experiments, baseline comparison

**Configuration**: `configs/train_basic.yaml`

**Features**:
- Single training phase
- Fixed learning rate schedule
- Standard validation pipeline
- Paper package generation

### 2. Curriculum Learning Mode

Progressive training with increasing difficulty levels.

**Use case**: Complex models, difficult datasets, improved convergence

**Configuration**: `configs/train_curriculum.yaml`

**Features**:
- Multi-stage training progression
- Adaptive difficulty adjustment
- Performance-based advancement
- Comprehensive monitoring

**Curriculum Stages**:
1. Foundation (basic reconstruction)
2. Short-term prediction
3. Medium-term prediction
4. Long-term prediction
5. Hybrid training
6. NAR refinement

### 3. Benchmark Mode

Comprehensive comparison across multiple models and configurations.

**Use case**: Model comparison, hyperparameter sweeps, reproducibility studies

**Configuration**: `configs/train_benchmark.yaml`

**Features**:
- Multiple model architectures
- Various observation modes
- Different loss configurations
- Statistical analysis
- Automated reporting

## Monitoring and Validation

### Real-time Monitoring

The system provides comprehensive monitoring during training:

- **Training curves**: Loss, learning rate, validation metrics
- **Resource usage**: GPU memory, utilization, CPU usage
- **Validation results**: Per-case metrics, aggregate statistics
- **Early stopping**: Automatic termination based on validation performance

### Validation Pipeline

The validation pipeline computes multiple metrics:

- **Reconstruction metrics**: Rel-L2, MAE, PSNR, SSIM
- **Frequency domain**: fRMSE (low/mid/high frequencies)
- **Boundary analysis**: bRMSE (16px boundary band)
- **Data consistency**: ||H(ŷ) - y|| error

### Resource Monitoring

Automatic tracking of:
- Model parameters count
- FLOPs computation
- GPU memory usage
- Training time per epoch

## Paper Package Generation

### Automatic Generation

The system automatically generates a complete paper package after training:

```
paper_package/
├── data_cards/          # Dataset information and licensing
├── configs/              # Configuration snapshots
├── checkpoints/          # Model checkpoints
├── metrics/              # Performance metrics and statistics
├── figs/                 # Visualizations and plots
├── scripts/              # Reproduction scripts
└── README.md             # Package documentation
```

### Manual Generation

Generate paper package from existing run:

```bash
python tools/generate_paper_package.py \
    --run-dir=runs/my_experiment \
    --output-dir=paper_packages/my_experiment
```

### Package Contents

#### Data Cards
- Dataset metadata and licensing
- Split information and statistics
- Observation operator specifications

#### Configuration Snapshots
- Complete merged configuration
- Git repository information
- Environment details (Python, PyTorch, CUDA)

#### Metrics
- Case-level performance metrics (JSONL)
- Aggregate statistics (mean, std, min, max)
- Statistical analysis (for multi-seed experiments)
- LaTeX tables for papers

#### Visualizations
- Training curves (loss, metrics, learning rate)
- Validation case comparisons (GT, Pred, Error)
- Power spectrum analysis
- Resource usage plots

#### Reproduction Scripts
- Shell script for experiment reproduction
- Python script for summary generation
- Complete environment setup instructions

## Best Practices

### 1. Experiment Naming

Follow the naming convention:
```
<task>-<data>-<resolution>-<model>-<key_hyperparams>-<seed>-<date>
```

Example:
```
SRx4-DR2D-256-SwinUNet_w8d2262_m16-s2025-20251011
```

### 2. Reproducibility

- Always set random seeds
- Use fixed data splits
- Save complete configuration snapshots
- Record environment information
- Generate paper packages for all experiments

### 3. Resource Management

- Monitor GPU memory usage
- Use gradient clipping for stability
- Enable mixed precision for efficiency
- Set appropriate batch sizes

### 4. Validation Strategy

- Use multiple validation metrics
- Perform data consistency checks
- Monitor training curves for overfitting
- Implement early stopping

## Troubleshooting

### Common Issues

#### Out of Memory
```bash
# Reduce batch size
python tools/train.py ... data.batch_size=8

# Enable gradient checkpointing
python tools/train.py ... model.gradient_checkpointing=true
```

#### Slow Training
```bash
# Enable mixed precision
python tools/train.py ... training.mixed_precision=true

# Increase number of workers
python tools/train.py ... data.num_workers=8
```

#### Poor Convergence
```bash
# Reduce learning rate
python tools/train.py ... training.optimizer.lr=5e-4

# Increase warmup epochs
python tools/train.py ... training.scheduler.warmup_epochs=2000

# Use curriculum learning
python tools/train.py --config-name=train_curriculum
```

### Validation Errors

#### Data Consistency Check Failed
- Verify observation operator implementation
- Check normalization parameters
- Ensure H operator consistency

#### Model Interface Validation Failed
- Check model input/output shapes
- Verify gradient flow
- Test with different batch sizes

## Advanced Usage

### Custom Loss Functions

```python
# Add to configs/train_custom.yaml
loss:
  custom_losses:
    - name: "my_loss"
      weight: 0.5
      params:
        param1: value1
```

### Multi-GPU Training

```bash
# Use torchrun for distributed training
torchrun --nproc_per_node=4 tools/train.py \
    --config-name=train_basic \
    experiment.distributed=true
```

### Hyperparameter Sweeps

```bash
# Use Hydra's multirun for sweeps
python tools/train.py --multirun \
    training.optimizer.lr=1e-3,5e-4,1e-4 \
    model.embed_dim=64,96,128
```

## Integration with Existing Workflows

### SLURM Clusters

```bash
#!/bin/bash
#SBATCH --job-name=pdebench
#SBATCH --nodes=1
#SBATCH --gres=gpu:4
#SBATCH --time=24:00:00

python tools/train.py \
    --config-name=train_basic \
    experiment.name=slurm_experiment \
    experiment.output_dir=/scratch/${USER}/runs
```

### Docker Containers

```dockerfile
FROM pytorch/pytorch:2.1.0-cuda11.8-cudnn8-devel

COPY . /workspace
WORKDIR /workspace

RUN pip install -r requirements.txt

CMD ["python", "tools/train.py", "--config-name=train_basic"]
```

## Performance Optimization

### Memory Optimization
- Use gradient checkpointing for large models
- Reduce batch size if OOM
- Clear cache periodically
- Use mixed precision training

### Speed Optimization
- Increase DataLoader workers
- Use pinned memory
- Enable CUDA benchmark mode
- Optimize observation operators

### Accuracy Optimization
- Use larger models when possible
- Increase training epochs
- Implement curriculum learning
- Tune loss function weights

## Support and Contributing

### Getting Help
- Check existing issues on GitHub
- Review configuration examples
- Consult troubleshooting section
- Contact maintainers

### Contributing
- Follow code style guidelines
- Add tests for new features
- Update documentation
- Generate paper packages for new experiments

## References

- [PDEBench Dataset](https://doi.org/10.5281/zenodo.123456)
- [Swin Transformer](https://arxiv.org/abs/2103.14030)
- [Hydra Configuration](https://hydra.cc/)
- [PyTorch Lightning](https://www.pytorchlightning.ai/)