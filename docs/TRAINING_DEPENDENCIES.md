# Training Dependencies

This document provides a comprehensive analysis of the project's training dependencies, including Python packages, datasets, configurations, and environment variables. It serves as a guide for setting up the training environment and understanding the system's requirements.

## 1. Python Dependencies

### Core Libraries
The project relies on the following core Python libraries, as defined in `requirements.txt` and `setup.py`.

| Package | Version | Purpose |
| :--- | :--- | :--- |
| `torch` | `>=2.0.0` | Deep learning framework (Core). Supports AMP and CUDA. |
| `numpy` | - | Numerical computing and array manipulation. |
| `h5py` | - | Handling HDF5 datasets (PDEBench, Real Diffusion-Reaction). |
| `omegaconf` | - | Hierarchical configuration management (YAML). |
| `pytorch-lightning` | - | High-level training framework (Trainer, LightningModule). |
| `pytest` | - | Unit and integration testing. |

### Implied & Optional Dependencies
These dependencies are used in the codebase or configurations but may not be explicitly listed in `requirements.txt`.

| Package | Usage Context | Purpose |
| :--- | :--- | :--- |
| `hydra-core` | `configs/` | Configuration composition and management. |
| `tensorboard` | `configs/train/default.yaml` | Experiment logging and visualization (`use_tensorboard: true`). |
| `wandb` | `configs/train/default.yaml` | Experiment tracking (Optional, default `false`). |

---

## 2. Data Dependencies

The project supports multiple datasets, primarily stored in HDF5 format.

### PDEBench (Core)
- **Format**: HDF5 (`.h5`)
- **Default Path**: `data/pdebench/sample_data_0.h5`
- **Config**: `configs/data/pdebench.yaml`
- **Structure**:
    - Key: `tensor`
    - Shape: `(10000, 1, 128, 128)` (Time, Channels, Height, Width)
- **Usage**: Used for standard benchmarks (e.g., DarcyFlow).

### Real Diffusion-Reaction
- **Format**: HDF5
- **Context**: Mentioned in `setup.py` description ("real diffusion-reaction data").
- **Usage**: Targeted for autoregressive training tasks.

### Directory Structure
```
data/
├── pdebench/
│   └── sample_data_0.h5
└── PDEBench/ (Alternative path referenced in params)
```

---

## 3. Configuration System

The project uses **Hydra** for configuration management, allowing hierarchical composition and overriding.

### Structure (`configs/`)
- **`config.yaml`**: Main entry point. Defines defaults (`data`, `model`, `train`) and experiment settings.
- **`data/`**: Dataset-specific configs (e.g., `pdebench.yaml`).
- **`model/`**: Model architecture configs (e.g., `swin_unet.yaml`).
- **`train/`**: Training hyperparameters (e.g., `default.yaml`).
- **`experiment/`**: Full experiment overrides.

### Key Configuration Files

| File | Purpose | Key Parameters |
| :--- | :--- | :--- |
| `configs/config.yaml` | Global settings | `experiment.name`, `experiment.device`, `experiment.precision` (AMP) |
| `configs/data/pdebench.yaml` | Data loading | `data_path`, `keys`, `image_size` (128), `dataloader.batch_size` |
| `configs/model/swin_unet.yaml` | Model params | `embed_dim` (96), `depths`, `num_heads`, `window_size` (8) |
| `configs/train/default.yaml` | Training loop | `epochs` (1000), `optimizer` (AdamW), `scheduler` (Cosine+Warmup), `loss_weights` |

---

## 4. Environment Variables

Environment variables are used for distributed training, hardware configuration, and reproducibility.

| Variable | Purpose | Typical Value / Context |
| :--- | :--- | :--- |
| `CUDA_VISIBLE_DEVICES` | GPU Selection | `0,1` (Select specific GPUs) |
| `MASTER_ADDR` | Distributed | Master node IP (DDP) |
| `MASTER_PORT` | Distributed | Master node port |
| `WORLD_SIZE` | Distributed | Total number of processes |
| `RANK` | Distributed | Process rank |
| `PYTORCH_CUDA_ALLOC_CONF` | Memory | `max_split_size_mb:128` (Fix fragmentation) |
| `PDEBENCH_DATA_PATH` | Data | Override path to specific HDF5 file |
| `PDEBENCH_DATA_ROOT` | Data | Override root directory for datasets |
| `CUBLAS_WORKSPACE_CONFIG` | Reproducibility | `:4096:8` (For deterministic cuBLAS) |
| `PYTHONHASHSEED` | Reproducibility | Fixed integer (e.g., `2025`) |
| `OMP_NUM_THREADS` | Performance | CPU thread limit (e.g., `4`) |

---

## 5. Dependency Tree & Architecture

A high-level view of how dependencies flow in the system.

```mermaid
graph TD
    subgraph Environment
        ENV[Env Vars] --> CFG[Hydra Configs]
        HW[Hardware (GPU/CPU)] --> RUN[Runtime]
    end

    subgraph Data
        H5[HDF5 Files] --> DL[DataLoaders]
        DL --> |h5py| NP[NumPy Arrays]
        NP --> |ToTensor| TEN[PyTorch Tensors]
    end

    subgraph Model_Training
        CFG --> |Instantiate| MOD[Model (SwinUNet/FNO)]
        CFG --> |Configure| OPT[Optimizer/Scheduler]
        TEN --> MOD
        MOD --> |Forward| OUT[Output]
        OUT --> |Loss| LOSS[Loss Functions]
    end

    subgraph Core_Libs
        TOR[PyTorch] --> MOD
        TOR --> OPT
        PL[PyTorch Lightning] --> RUN
        PL --> |Manages| MOD
        PL --> |Manages| OPT
    end

    ENV --> RUN
    DL --> PL
```

### Critical Paths
1.  **Config Loading**: `config.yaml` -> `hydra` -> `omegaconf` -> `DictConfig`.
2.  **Data Loading**: Path (Env/Config) -> `h5py` -> `numpy` -> `torch.utils.data.DataLoader`.
3.  **Training Loop**: `pytorch-lightning.Trainer` -> `Model` -> `Optimizer` -> `AMP` -> `GPU`.
