I will implement the automated workflow for generating LaTeX TikZ figures and PDFs as requested.

**1. Update `export_and_gen_tikz.py`**
*   **Enhance Arguments**: Add `--compile` (boolean) and `--latex_env` (string) arguments.
*   **ONNX Export**: Add functionality to export the model to ONNX format (`torch.onnx.export`) alongside the summary and TikZ file.
*   **Compilation Logic**: Update the compilation step to use the specified conda environment for `tectonic`. Since the `latex` environment exists and has `tectonic` installed (verified), I will use the absolute path `/share/fandixiaLab/suguangsheng/anaconda3/envs/latex/bin/tectonic` or `conda run -n latex tectonic` to ensure it works reliably.

**2. Create `build_all.sh`**
*   Create the shell script in `thesis_paper/figures_nn/` to iterate over all `fig_*.tex` files and compile them using `tectonic`.

**3. Execution & Verification**
*   **Run for EDSR**: Execute the updated script for EDSR with compilation enabled.
    ```bash
    python thesis_paper/figures_nn/export_and_gen_tikz.py --model EDSR --in_ch 1 --out_ch 1 --img 128 --upscale 4 --compile --latex_env latex
    ```
*   **Run for UNet**: Execute the script for UNet to verify extensibility.
    ```bash
    python thesis_paper/figures_nn/export_and_gen_tikz.py --model UNet --in_ch 1 --out_ch 1 --img 128 --upscale 4 --compile --latex_env latex
    ```
*   **Verify Outputs**: Check for `EDSR_summary.txt`, `EDSR.onnx`, `fig_edsr_auto.tex`, `build/fig_edsr_auto.pdf` (and similarly for UNet).

**Environment Note**:
The `latex` conda environment already exists and contains `tectonic` 0.15.0. No installation is required. I will utilize this existing environment.