---
name: tikz-generator
description: Generates high-quality TikZ/LaTeX neural network architecture diagrams for project models (Spatial & Temporal). Handles summary generation, ONNX export, and PDF compilation automatically. Use when the user wants to visualize model architectures like SwinIR, UNet, VideoSwin, ConvLSTM, etc.
---

# TikZ Architecture Generator

This skill wraps the `export_and_gen_tikz.py` script to generate publication-ready neural network diagrams.

## Capabilities

1.  **3D Diagrams**: Uses `PlotNeuralNet` style (stacked blocks) for spatial/spatiotemporal models.
2.  **2D Diagrams**: Uses standard TikZ for flat, abstract flowcharts.
3.  **Auto-Summary**: Generates `torchinfo` summaries for models.
4.  **Auto-Compile**: Compiles `.tex` to `.pdf` using `tectonic`.

## Usage

Run the generator script from the project root.

### Command

```bash
python thesis_paper/figures_nn/export_and_gen_tikz.py --models <model_list> [options]
```

### Parameters

-   `--models`: Comma-separated list of model names (e.g., `SwinIR,VideoSwin`).
    -   **Supported Models**: `swinir`, `nafnet`, `restormer`, `unet`, `swin_unet`, `videoswin`, `convlstm`, `physics_transformer`, `sequential`, `deeponet`, `fno`, `ufno`, etc.
-   `--model`: Single model name (alternative to `--models`).
-   `--compile`: **Recommended**. Automatically compiles the generated `.tex` files to `.pdf`.
-   `--in_ch`: Input channels (default: 1).
-   `--img`: Input image size (default: 128).
-   `--upscale`: Upscale factor (default: 4, for SR models).

### Examples

**Generate diagrams for a list of models:**
```bash
python thesis_paper/figures_nn/export_and_gen_tikz.py --models SwinIR,NAFNet,VideoSwin --compile
```

**Generate for a physics model with specific input size:**
```bash
python thesis_paper/figures_nn/export_and_gen_tikz.py --model PhysicsTransformer --img 64 --compile
```

## Output

Artifacts are saved in `thesis_paper/figures_nn/build_export/<ModelName>/`:

-   `fig_<model>_auto.pdf`: 3D Architecture Diagram
-   `fig_<model>_2d_auto.pdf`: 2D Architecture Diagram
-   `<model>_summary.txt`: Layer-wise summary
-   `<model>.onnx`: ONNX export

## Troubleshooting

-   **Dimension Errors**: The script automatically tries to detect if a model needs 4D (Spatial) or 5D (Temporal) inputs. If it fails, check the `process_model` logic in the script.
-   **Missing Styles**: The script automatically copies `nn_blocks.tex` and `.sty` files to the output directory.
