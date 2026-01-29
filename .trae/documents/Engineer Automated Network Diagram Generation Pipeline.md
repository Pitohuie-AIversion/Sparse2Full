I will implement the revised engineering plan to create a robust and reproducible automated network diagram generation pipeline.

**1. File Organization & Cleanup (Task A)**

* Create `thesis_paper/figures_nn/build_export/`.

* Update `export_and_gen_tikz.py` to save `*_summary.txt` and `*.onnx` to `build_export/`, while keeping `.tex` files in `thesis_paper/figures_nn/` and PDFs in `build/`.

* **Do not move existing artifacts** to ensure backward compatibility.

**2. Robust Model Loading (Task B)**

* Update `build_model` in `export_and_gen_tikz.py` to prioritize the project registry (`models.registry`) for model resolution.

* Implement a secondary fallback search in `models.spatial` with case-insensitive matching and alias support (e.g., `edsr` == `EDSR`).

* Ensure clear error messages listing available models if lookup fails.

**3. Implement New TikZ Generators (Task C)**

* **Preserve** existing `gen_tikz_edsr` and `gen_tikz_unet`.

* **Add FNO2d Generator (`gen_tikz_fno2d`)**: Visualize `Input -> Lift -> K x SpectralConv2d -> Projection -> Output`. Extract `K`, `width`, `modes` if possible.

* **Add SegFormer Generator (`gen_tikz_segformer`)**: Visualize `Input -> Overlap Patch Embedding -> Stage1-4 (MiT) -> Head -> Output`.

* **Add SwinT Generator (`gen_tikz_swint`)**: Visualize `Input -> Patch Embedding -> Stage1-4 (Swin blocks) -> Head -> Output`. **Crucially, implement this as a hierarchical encoder (backbone), NOT a U-Net.**

* **Generic Fallback**: Use `gen_tikz_generic` for models without specialized generators.

**4. Batch Processing & Reporting (Task D)**

* Update CLI to accept `--models "EDSR,UNet,..."` (comma-separated list) or a single `--model`.

* Implement a robust batch loop where failure in one model does not stop the process.

* Generate `build_export/report.json` summarizing status, paths, generator used, and error messages for each model.

**5. Reproducible Compilation (Task E)**

* Update compilation logic to strictly use `conda run -n latex tectonic ...` regardless of the current shell environment.

* Capture and report compilation errors in `report.json`.

**6. Verification**

* Run the batch command for all 5 target models: `EDSR`, `UNet`, `FNO2d`, `SegFormer`, `SwinT`.

* Verify the existence of all 5 PDF outputs in `build/` and artifacts in `build_export/`.

* Check `report.json` for correct status reporting.

