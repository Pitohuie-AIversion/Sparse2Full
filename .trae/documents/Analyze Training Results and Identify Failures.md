I have analyzed the `runs/` directory to identify successful and failed/incomplete runs from your previous batch.

**Criteria:**
*   **Success**: A run is considered "successful" if it generated a `test_results.json` file, indicating the training loop completed and the final test phase finished.
*   **Incomplete/Failed**: A run is flagged if it has a `training.log` but *no* `test_results.json`, or if the log does not contain "Finished training".

**Analysis of `AR-SW-10M-*` Runs (Previous Batch):**

**✅ Successful Runs (Completed & Tested):**
The following models successfully finished training and testing:
1.  `AR-SW-10M-bilinear3x3decoder`
2.  `AR-SW-10M-ConvUNetLite`
3.  `AR-SW-10M-DeepONet` (including `deeponet`, `deeponet2d`)
4.  `AR-SW-10M-EDSR` (including `edsr`, `edsrnet`)
5.  `AR-SW-10M-FNO2d` (including `fno`, `fno2d`)
6.  `AR-SW-10M-MLP` / `MLPMixer` / `mlpmodel`
7.  `AR-SW-10M-NAFNet` (including `nafnet`)
8.  `AR-SW-10M-PartialConvUNet` (including `partialconvunet`, `pconvunet`)
9.  `AR-SW-10M-ResNetLite` (including `resnetlite`)
10. `AR-SW-10M-SegFormer` (including `segformer`)
11. `AR-SW-10M-stablefno2d` (Note: `stablefnomodel` is still running)
12. `AR-SW-10M-SwinT` / `SwinTWithEncoder`
13. `AR-SW-10M-UNet` / `UNetPlusPlus`
14. `AR-SW-10M-ViT`

**⚠️ Incomplete/Running/Failed Runs:**
The following directories exist (have logs) but lack `test_results.json`, meaning they either failed, were interrupted, or are currently running:
1.  **`AR-SW-10M-stablefnomodel`**: **Currently Running** (Confirmed by process check).
2.  `AR-SW-10M-CNNAttnLite`: Training started but no test results.
3.  `AR-SW-10M-ConvGateLite`: Training started but no test results.
4.  `AR-SW-10M-HybridModel` (including `hybrid`, `hybridmodel`): Training started but no test results.
5.  `AR-SW-10M-LIIF`: Training started but no test results.
6.  `AR-SW-10M-PerceiverIO` (including `perceiverio`, `perceiverio2d`): Training started but no test results.
7.  `AR-SW-10M-RCAN` (including `rcan`, `rcannet`): Training started but no test results.
8.  `AR-SW-10M-RDN` (including `rdn`): Training started but no test results.
9.  `AR-SW-10M-Restormer` (including `restormer`, `restormernet`): Training started but no test results.
10. `AR-SW-10M-SegFormerUNetFormer`: Training started but no test results.
11. `AR-SW-10M-SparseSwinUNet`: Training started but no test results.
12. `AR-SW-10M-SwinIRLite`: Training started but no test results.
13. `AR-SW-10M-SwinUNet`: Training started but no test results.
14. `AR-SW-10M-Transformer`: Training started but no test results.
15. `AR-SW-10M-UFNOUNet`: Training started but no test results.
16. `AR-SW-10M-UformerLite`: Training started but no test results.
17. `AR-SW-10M-UNetFormer`: Training started but no test results.
18. `AR-SW-10M-UNO`: Training started but no test results.

**Action Plan:**
I will proceed to **debug the failed models**. Since there are many, I will pick a representative failure (`AR-SW-10M-ConvGateLite` or `AR-SW-10M-CNNAttnLite`) to investigate the logs and determine the root cause (e.g., OOM, architecture error, divergence).

**Proposed Step**: Check the last 50 lines of `runs/AR-SW-10M-ConvGateLite/training.log` (or similar) to diagnose the failure.