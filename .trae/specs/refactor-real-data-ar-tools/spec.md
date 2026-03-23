# Real Data AR Tools Refactoring Spec

## Why
The current training tools for real data AR (`tools/training/train_real_data_ar*.py`) are scattered, contain inline class definitions that duplicate existing logic, and lack a standardized library structure. This makes maintenance, testing, and reuse difficult. A unified, well-structured library is needed to improve code quality and developer experience.

## What Changes
- Create a new Python package named `real_data_ar`.
- Refactor `tools/training/train_real_data_ar_refactored.py` into modular components.
- Integrate `datasets/real_diffusion_reaction_dataset.py` into the package or ensure robust dependency management.
- Establish a clear directory structure adhering to the Single Responsibility Principle.
- Add `setup.py`, `requirements.txt`, and `README.md`.
- Add comprehensive docstrings and type hints (PEP 8/484).
- Implement unit tests with 100% coverage for core logic.

## Impact
- **Affected Code**: `tools/training/` scripts will be superseded by this library.
- **New Structure**:
    ```
    real_data_ar/
    ├── config/
    ├── data/
    ├── models/
    ├── training/
    └── utils/
    ```

## ADDED Requirements
### Requirement: Standardized Package Structure
The system SHALL be organized as a pip-installable Python package with `setup.py`.

### Requirement: Modular Design
The code SHALL be split into modules based on functionality (Config, Data, Model, Training) to ensure separation of concerns.

### Requirement: Documentation and Typing
All public functions and classes SHALL have docstrings and type annotations.

### Requirement: Testing
The core logic SHALL be covered by unit tests with a target of 100% pass rate.

## MODIFIED Requirements
### Requirement: Training Logic
The monolithic training scripts SHALL be refactored into a `Trainer` class within the library, preserving the existing three-stage training logic (Spatial, Temporal, Joint).
