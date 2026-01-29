Based on the list of files provided by the user and the current contents of `01_Model_Architectures.md`, I need to identify which spatial models are missing documentation.

**Files in `models/spatial/` (from user input):**
*   `cnn_attn_lite.py` (Missing)
*   `conv_gate_lite.py` (Missing)
*   `conv_unet_lite.py` (Missing)
*   `coordinate_encoder.py` (Missing, likely component)
*   `deeponet.py` (Documented: 1.4)
*   `edsr.py` (Missing)
*   `factory.py` (Factory, not model)
*   `fno2d_stable.py` (Variant of FNO, likely covered by 1.2 or needs note)
*   `fno2d.py` (Documented: 1.2)
*   `hybrid.py` (Documented: 4.1)
*   `liif_head.py` (Component)
*   `liif.py` (Missing)
*   `mlp_mixer.py` (Missing)
*   `mlp.py` (Documented: 4.2)
*   `nafnet.py` (Missing)
*   `partialconv_unet.py` (Missing)
*   `perceiverio.py` (Missing)
*   `rcan.py` (Missing)
*   `rdn.py` (Missing)
*   `resnet.py` (Missing)
*   `restormer.py` (Missing)
*   `segformer_unetformer.py` (Missing, likely variant)
*   `segformer.py` (Documented: 1.5)
*   `sparse_attention_encoder.py` (Documented: 1.8)
*   `swin_t_with_encoder.py` (Missing)
*   `swin_t.py` (Missing)
*   `swin_unet.py` (Documented: 1.1)
*   `swinir.py` (Missing)
*   `transformer.py` (Missing, generic)
*   `ufno_unet_bottleneck.py` (Component, likely FNO Bottleneck in 1.1)
*   `unet_plus_plus.py` (Missing)
*   `unet.py` (Documented: 1.3)
*   `unetformer.py` (Documented: 1.6)
*   `uno.py` (Missing)
*   `vit.py` (Missing)

**Missing Spatial Models:**
1.  **Lite/Mobile Variants**: `cnn_attn_lite`, `conv_gate_lite`, `conv_unet_lite`.
2.  **Super-Resolution (SR) Classics**: `edsr`, `rcan`, `rdn`, `swinir`, `liif`.
3.  **Modern Restoration**: `nafnet`, `restormer`.
4.  **Other Architectures**: `mlp_mixer`, `partialconv_unet`, `perceiverio`, `resnet`, `unet_plus_plus`, `uno`, `vit`, `swin_t`.

**Plan:**
1.  **Group Missing Models**: I will group these into logical sections (e.g., "Lightweight Models", "Image Restoration/SR Models", "Other Architectures").
2.  **Analyze & Document**: I will read the key files for these missing models to understand their structure (Input/Output, Layers).
3.  **Update Documentation**: I will add new sections to `01_Model_Architectures.md` for these models, including ASCII diagrams and Mermaid flowcharts.
4.  **Check Common Modules**: I will look for re-used blocks (like `ResidualBlock` in ResNet/EDSR/RCAN) to define common diagram patterns.

**Immediate Action**: Read a representative subset of the missing files to draft their sections. I'll focus on the "SR Classics" and "Modern Restoration" first as they are significant groups. Then "Lite" models.

**Batch 1 Reading**: `edsr.py`, `rcan.py`, `nafnet.py`, `restormer.py`, `swinir.py`.
**Batch 2 Reading**: `cnn_attn_lite.py`, `conv_unet_lite.py`, `unet_plus_plus.py`.

I will start by reading Batch 1.