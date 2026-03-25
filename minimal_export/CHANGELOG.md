# Changelog

## [0.1.1] - 2026-03-25

### Added
- **Tests**: Created a `tests` directory with unit tests for models (`test_models.py`) and datasets (`test_datasets.py`) to ensure basic initialization and forward passes work.
- **Models**: Added `models/base.py` containing `BaseModel` which inherits from `torch.nn.Module` and handles default attribute initialization (`in_channels`, `out_channels`, `img_size`).
- **Models**: Added `models/registry.py` with `register_model` and `create_model` to manage model instantiation.
- **Models**: Added `models/encoders/sparse_input_encoder.py` containing a dummy `SparseInputEncoder` to resolve import errors in `swin_t_with_encoder.py`.

### Changed
- **Code Style**: Ran `ruff` to fix common linter errors and `black` to format all Python files, unifying the code style across the project.
- **Dependencies**: Updated `requirements.txt` and `setup.py` to reflect actual project dependencies (`torch`, `numpy`, `h5py`, `omegaconf`, `pytorch-lightning`, `matplotlib`, `tqdm`, `pandas`, `scipy`, `einops`, `timm`, `scikit-image`, `fvcore`, `thop`, `psutil`, `tensorboard`, `pytest`). Removed invalid or unused local package declarations.
- **Models**: Updated `tests/test_models.py` to correctly import and use `ARWrapper`.
- **Documentation**: Formatted `README.md` spacing and added minor clarifications.

### Fixed
- Fixed `AttributeError` during `UNet` initialization by appropriately capturing and assigning constructor arguments in `BaseModel`.
- Fixed `TypeError` in `register_model` by supporting `aliases` as an argument in the decorator.
