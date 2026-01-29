Based on the analysis of the `models` directory and the detailed review of key model files, I have gathered sufficient information to create the systematic model architecture documentation as requested.

**Plan:**

1.  **Create Directory**: Create the target directory `模型架构说明` (Model Architecture Descriptions) inside `/share/fandixiaLab/suguangsheng/PycharmProjects/Sparse2Full/models`.

2.  **Draft Documentation**: I will write a comprehensive Markdown document named `MODEL_ARCHITECTURE_GUIDE.md` (or similar, but the user asked for files *in* the subdirectory, so I will name it `README.md` inside that new folder or multiple files. Let's stick to a single well-structured file `01_Model_Architectures.md` inside the new folder for clarity, or split them. Given the request for "systematic organization", a single index file + detailed chapters in one file is efficient). I will structure the content as follows:

    *   **Chapter 1: Spatial Reconstruction Models (空间重建模型)**
        *   Swin-UNet (with FNO bottleneck)
        *   FNO 2D (Fourier Neural Operator)
        *   Classic U-Net
        *   DeepONet
        *   SegFormer
        *   UNetFormer
        *   (Brief mention of other variants like MLP, LIIF based on common patterns)
    *   **Chapter 2: Temporal Prediction Models (时序预测模型)**
        *   SwinTemporal Wrapper
        *   PhysicsTransformer
        *   Temporal Components (TemporalBlock, etc.)
    *   **Chapter 3: Autoregressive Framework (自回归框架)**
        *   ARWrapper
    *   **Chapter 4: Hybrid & Other Models (混合与其他模型)**
        *   HybridModel
        *   MLPModel

3.  **Content Details**: For each model, I will include:
    *   **Name & Version**: e.g., "Swin-UNet v1.0 (Spatial)"
    *   **Schematic**: Text-based representation of layers (Encoder -> Bottleneck -> Decoder).
    *   **IO Specs**: Input `[B, C, H, W]`, Output `[B, C, H, W]`.
    *   **Parameters**: Key config args (channels, heads, depth).
    *   **Characteristics**: Performance notes (e.g., "Good for high-freq details", "Efficient for global dependencies").

4.  **Formatting**: Use H1 for main title, H2 for Chapters, H3 for Models. Uniform font and tables.

5.  **Index**: Add a Table of Contents at the beginning.

**Action**: I will execute the creation of the directory and the file writing. No further searching is required.