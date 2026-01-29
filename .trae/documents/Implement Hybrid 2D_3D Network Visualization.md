I will implement a **hybrid 2D/3D visualization pipeline** that combines high-level 2D flowchart logic with PlotNeuralNet's detailed 3D blocks for complex operations (like Convolutions and Encoders).

**1. Update** **`nn_blocks.tex`** **(Style Foundation)**

* **Enable PlotNeuralNet**: Uncomment the `\input{vendor/PlotNeuralNet/layers/init.tex}` line to activate 3D capabilities.

* **Fix Paths**: Ensure the path to `init.tex` is correct relative to the generated `.tex` files (likely `vendor/PlotNeuralNet/layers/init.tex`).

* **Add 3D Colors**: Define standard PlotNeuralNet colors (`\definecolor{ConvColor}{rgb}{...}`) compatible with our existing 2D palette.

**2. Update** **`export_and_gen_tikz.py`** **(Hybrid Generators)**

* **Modify** **`gen_tikz_edsr`**:

  * Use **3D Blocks** for: `Head` (Conv), `Body` (ResBlock Stack), `Tail` (Conv).

  * Use **2D Nodes** for: `Input`, `Upsampler` (PixelShuffle is abstract), `Output`, and `Sum` operations.

  * Connect them using TikZ arrows, handling the coordinate offsets between 2D nodes and 3D anchors.

* **Modify** **`gen_tikz_unet`**:

  * Use **3D Blocks** for: `Encoder` stages (stack of convs), `Bottleneck`, `Decoder` stages.

  * Use **2D Nodes** for: `Input`, `Output`, `Skip Connections` (arrows).

  * Visualize the "U" shape by manually positioning the 3D blocks.

* **Modify** **`gen_tikz_fno2d`**:

  * Use **3D Blocks** for: `Lift` (Channel expansion), `Spectral Layers` (Stack of blocks), `Projection` (Channel compression).

  * Use **2D Nodes** for: `Input`, `Output`.

* **Modify** **`gen_tikz_segformer`** **/** **`gen_tikz_swint`**:

  * Use **3D Blocks** for: `Patch Embed`, `Transformer Stages` (representing feature map volumes).

  * Use **2D Nodes** for: `Input`, `Head`, `Output`.

**3. Verification**

* Run the batch generation command:

  ```bash
  python thesis_paper/figures_nn/export_and_gen_tikz.py --models "EDSR,UNet,FNO2d,SegFormer,SwinT" --compile --latex_env latex
  ```

* Verify the PDFs contain mixed 2D/3D elements and compile without errors.

**Why this approach?**

* **Visual Appeal**: 3D blocks intuitively show channel expansion/compression (thickness) and resolution changes (height/width).

* **Clarity**: 2D nodes are better for abstract operations (Sum, PixelShuffle, Input/Output labels) where 3D adds unnecessary clutter.

* **Technical Feasibility**: Since PlotNeuralNet is just TikZ macros, they can coexist in the same `tikzpicture` environment.

