Based on the file list in `runs/`, the output directories from your **previous batch run** (the one that accidentally wrote to `runs/` instead of `run_sw_4x`) follow this naming pattern:

`AR-SW-10M-{model_name}`

Specifically, these are the directories created:

*   `AR-SW-10M-bilinear3x3decoder`
*   `AR-SW-10M-CNNAttnLite`
*   `AR-SW-10M-ConvGateLite`
*   `AR-SW-10M-ConvUNetLite`
*   `AR-SW-10M-DeepONet` (and variants like `deeponet`, `deeponet2d`)
*   `AR-SW-10M-EDSR` (and `edsr`, `edsrnet`)
*   `AR-SW-10M-fno` / `fno2d` / `FNO2d`
*   `AR-SW-10M-HybridModel` (and `hybrid`, `hybridmodel`)
*   `AR-SW-10M-LIIF`
*   `AR-SW-10M-MLP` / `MLPMixer` / `mlpmodel`
*   `AR-SW-10M-NAFNet` (and `nafnet`)
*   `AR-SW-10M-PartialConvUNet` (and `partialconvunet`, `pconvunet`)
*   `AR-SW-10M-PerceiverIO` (and `perceiverio`, `perceiverio2d`)
*   `AR-SW-10M-RCAN` (and `rcan`, `rcannet`)
*   `AR-SW-10M-RDN` (and `rdn`)
*   `AR-SW-10M-ResNetLite` (and `resnetlite`)
*   `AR-SW-10M-Restormer` (and `restormer`, `restormernet`)
*   `AR-SW-10M-SegFormer` (and `segformer`)
*   `AR-SW-10M-SegFormerUNetFormer`
*   `AR-SW-10M-SparseSwinUNet`
*   `AR-SW-10M-stablefno2d` / `stablefnomodel`
*   `AR-SW-10M-SwinIRLite`
*   `AR-SW-10M-SwinT-model_SwinT-s2025-20251229`
*   `AR-SW-10M-SwinTWithEncoder-model_SwinTWithEncoder-s2025-20251229`
*   `AR-SW-10M-SwinUNet`
*   `AR-SW-10M-Transformer`
*   `AR-SW-10M-UFNOUNet`
*   `AR-SW-10M-UformerLite`
*   `AR-SW-10M-UNetFormer`
*   `AR-SW-10M-UNet-model_UNet-s2025-20251228`
*   `AR-SW-10M-UNetPlusPlus-model_UNetPlusPlus-s2025-20251228`
*   `AR-SW-10M-UNO`
*   `AR-SW-10M-ViT-model_ViT-s2025-20251229`

**Note on Duplicates**: You will notice some redundancy (e.g., `AR-SW-10M-DeepONet` and `AR-SW-10M-deeponet`). This is likely due to the case-sensitive dynamic scanning picking up both the class name and file name or slight variations in how the model loader reports names.

These are the directories you should look for if you want to retrieve results from that specific run. The new run (if successfully started) will be in `run_sw_4x/` with similar names.