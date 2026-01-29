I will automate the process of extracting the model structure and generating the LaTeX TikZ figure.

**1. Install Dependencies**

* Install `tectonic` via pip (required for rendering PDF).

* `torchinfo`, `onnx`, `torchviz` are already installed in the environment.

**2. Create Necessary Files**

* **Create** **`thesis_paper/figures_nn/nn_blocks.tex`**: This file is missing but required by the script. I will create it with the necessary TikZ style definitions (`block`, `conv`, `res`, `sum`, `op`, `up`) to ensure the generated LaTeX compiles correctly.

* **Create** **`thesis_paper/figures_nn/export_and_gen_tikz.py`**: I will use the code you provided, adding `sys.path` adjustments to ensure it can correctly import the `models` module from the project root.

**3. Execute Automation Script**

* Run the script to generate the structure summary and TikZ figure for the **EDSR** model.

* Command: `python thesis_paper/figures_nn/export_and_gen_tikz.py --model EDSR --in_ch 1 --out_ch 1 --img 128 --upscale 4`

**4. Verification**

* Verify that `EDSR_summary.txt`, `fig_edsr_x4_auto.tex`, and `fig_edsr_x4_auto.pdf` are generated successfully.

