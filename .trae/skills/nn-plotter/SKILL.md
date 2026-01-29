---
name: nn-plotter
description: Automates the generation of neural network architecture diagrams using PlotNeuralNet. Use this skill when the user asks to "generate figures", "plot architecture", "update diagrams", or "compile latex figures" for the neural networks.
---

# Neural Network Architecture Plotter

This skill helps generate and compile TikZ/LaTeX figures for neural network architectures defined in the codebase.

## Capabilities

1.  **Generate TikZ Code**: Exports PyTorch models to ONNX (optional) and generates `.tex` files using `PlotNeuralNet`.
2.  **Compile PDFs**: Compiles the generated `.tex` files into PDFs using `tectonic`.

## Usage

### 1. Generate and Compile All Figures

To generate TikZ code for all supported models and compile them to PDF, run the bundled script:

```bash
bash .trae/skills/nn-plotter/scripts/generate_all.sh
```

### 2. Generate for Specific Models

To generate for a specific list of models (e.g., `swin_unet` and `deeponet`):

```bash
python thesis_paper/figures_nn/export_and_gen_tikz.py --models swin_unet,deeponet --compile
```

## Supported Models

The following models are currently supported in `export_and_gen_tikz.py`:
- `swin_unet` (SwinUNet)
- `mlp_mixer` (MLPMixer)
- `ufno` (UFNOUNet)
- `liif` (LIIF)
- `deeponet` (DeepONet)
- `edsr`
- `unet`
- `fno2d`
- `segformer`
- `swint`
- `hybrid`

## Troubleshooting

- **Missing Model**: If a model is not found, ensure it is imported in `thesis_paper/figures_nn/export_and_gen_tikz.py` or registered in `models/registry.py`.
- **Compile Errors**: Check `thesis_paper/figures_nn/build_all.sh` logs. Ensure `tectonic` is installed or available in the environment.
