# PDEBench Training System

A comprehensive training framework for sparse observation reconstruction in PDEBench, supporting multiple training modes, automated paper package generation, and extensive monitoring capabilities.

## 🚀 Quick Start

### Basic Training
```bash
# Quick system validation
./launchers/basic/quick_test.sh

# Standard training
./launchers/basic/launch_basic.sh
```

### Curriculum Learning
```bash
# Launch curriculum training
./launchers/curriculum/launch_curriculum.sh
```

### Benchmark Comparison
```bash
# Run comprehensive benchmark
./launchers/benchmark/launch_benchmark.sh
```

## 📁 Directory Structure

```
training_system/
├── scripts/                    # Main training scripts
│   └── train.py               # Primary training script with Hydra config
├── configs/                    # Configuration files
│   ├── basic/                 # Basic training configurations
│   ├── curriculum/            # Curriculum learning configs
│   ├── benchmark/             # Benchmark comparison configs
│   └── quick_test/            # Quick validation configs
├── utils/                      # Training utilities
│   ├── trainers/              # Trainer implementations
│   │   ├── trainer.py         # Main PDEBench trainer
│   │   ├── benchmark_trainer.py # Benchmark comparison trainer
│   │   └── curriculum_trainer.py # Curriculum learning trainer
│   ├── monitoring/            # Monitoring and validation
│   │   └── monitoring.py      # Training monitoring pipeline
│   └── paper_package/         # Paper package generation
│       └── generate_paper_package.py
├── launchers/                  # Launch scripts
│   ├── basic/                 # Basic training launchers
│   ├── curriculum/            # Curriculum learning launchers
│   └── benchmark/             # Benchmark launchers
├── docs/                       # Documentation
│   ├── guides/                # User guides
│   │   └── TRAINING_LAUNCH_GUIDE.md
│   └── examples/              # Example scripts
│       └── example_training.sh
├── tests/                      # Test suite
│   └── integration/           # Integration tests
│       └── test_comprehensive_framework.py
└── paper_package/             # Generated paper packages (runtime)
```

## 🎯 Training Modes

### 1. Basic Training (`training_mode=basic`)
- Standard single-model training
- Supports all PDEBench models
- Comprehensive monitoring and validation
- Automatic paper package generation

### 2. Curriculum Learning (`training_mode=curriculum`)
- Progressive difficulty training
- Multi-stage learning (e.g., SR: ×2 → ×4)
- Automatic stage progression
- Enhanced convergence

### 3. Benchmark Comparison (`training_mode=benchmark`)
- Multi-model comparison
- Statistical analysis across seeds
- Comprehensive performance reporting
- Resource usage tracking

## ⚙️ Configuration System

The system uses Hydra for configuration management with hierarchical YAML files:

```yaml
# Example: Basic training config
defaults:
  - base_config
  - /data: pdebench_sparse
  - /model: swin_unet
  - /loss: combined
  - _self_

experiment:
  name: "SRx4-DR2D-256-SwinUNet"
  task: "sr_x4"
  seed: 42

training:
  epochs: 300
  batch_size: 16
  learning_rate: 1e-3
```

## 📊 Monitoring & Validation

### Real-time Monitoring
- Training/validation loss tracking
- Resource usage (GPU/CPU/memory)
- Learning rate scheduling
- Early stopping

### Validation Pipeline
- Multiple metrics (Rel-L2, MAE, PSNR, SSIM)
- Data consistency checks
- Statistical analysis
- Visualization generation

### Resource Tracking
- Model parameters count
- FLOPs computation
- Memory usage profiling
- Inference latency measurement

## 📦 Paper Package Generation

Automatic generation of comprehensive paper packages including:

- **Data Cards**: Dataset information and licensing
- **Configuration Snapshots**: Complete training configs
- **Checkpoints**: Key model checkpoints
- **Metrics**: Statistical analysis and comparison tables
- **Visualizations**: Training curves, validation cases, error analysis
- **Reproduction Scripts**: One-command reproduction
- **Documentation**: Complete experiment documentation

### Manual Package Generation
```bash
# Generate package from existing run
python utils/paper_package/generate_paper_package.py \
  --run_dir runs/SRx4-DR2D-256-SwinUNet-20251011 \
  --output_dir paper_package/manual_package
```

## 🧪 Testing

### Comprehensive Test Suite
```bash
# Run integration tests
python -m pytest tests/integration/test_comprehensive_framework.py -v

# Test specific components
python -m pytest tests/ -k "test_data_consistency"
python -m pytest tests/ -k "test_model_interface"
python -m pytest tests/ -k "test_reproducibility"
```

### Validation Checks
- Data consistency (observation operator)
- Model interface compliance
- Reproducibility verification
- Configuration validation
- Resource monitoring

## 🔧 Advanced Usage

### Custom Loss Functions
```yaml
loss:
  type: "custom"
  components:
    reconstruction: {weight: 1.0}
    spectral: {weight: 0.5, low_freq: 16}
    data_consistency: {weight: 1.0}
```

### Multi-GPU Training
```bash
# DDP training
torchrun --nproc_per_node=4 scripts/train.py \
  --config configs/basic/train_multi_gpu.yaml
```

### Hyperparameter Sweeps
```bash
# Grid search
python scripts/train.py \
  --config configs/basic/train.yaml \
  --multirun \
  training.learning_rate=1e-4,1e-3,1e-2 \
  training.batch_size=8,16,32
```

## 📈 Performance Optimization

### Memory Optimization
- Gradient checkpointing
- Mixed precision training (AMP)
- Dynamic batch sizing
- Memory-efficient data loading

### Speed Optimization
- Multi-threaded data loading
- GPU memory pre-allocation
- Optimized CUDA kernels
- Distributed training support

## 🔍 Troubleshooting

### Common Issues
1. **CUDA Out of Memory**: Reduce batch size or enable gradient checkpointing
2. **Slow Training**: Increase number of workers or enable mixed precision
3. **Poor Convergence**: Check learning rate scheduling and loss weights
4. **Reproducibility Issues**: Set all seeds and enable deterministic mode

### Debug Mode
```bash
# Enable debug mode
python scripts/train.py --config configs/debug_config.yaml
```

## 📚 Documentation

- [Training Launch Guide](docs/guides/TRAINING_LAUNCH_GUIDE.md)
- [Configuration Reference](docs/guides/CONFIG_REFERENCE.md)
- [API Documentation](docs/api/)
- [Examples](docs/examples/)

## 🤝 Contributing

1. Follow the golden rules and coding standards
2. Add comprehensive tests for new features
3. Update documentation
4. Ensure CI passes (lint, type check, tests)

## 📄 License

This training system is part of the PDEBench project and follows the same licensing terms.

## 🆘 Support

For issues and questions:
- Check the troubleshooting section
- Review the documentation
- Run validation tests
- Check existing issues

---

**Note**: This training system implements the golden rules for PDEBench sparse observation reconstruction, ensuring consistency, reproducibility, and comprehensive evaluation.