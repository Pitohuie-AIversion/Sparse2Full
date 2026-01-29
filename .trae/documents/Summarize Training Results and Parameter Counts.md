I have collected the parameter counts and test metrics for the completed models. I will now present a summary table of the results.

**Completed Models:**
1.  **UNet**: 9.89M params, Rel-L2: 0.1985
2.  **ConvUNetLite**: 10.09M params, Rel-L2: 0.1729
3.  **NAFNet**: 8.15M params, Rel-L2: 0.1562 (Best so far)
4.  **UNO**: 28.05M params (Over budget), Rel-L2: 0.7337 (Poor performance)
5.  **EDSR**: 1.22M params (Under budget), Rel-L2: 0.3449
6.  **UformerLite**: 10.09M params, Rel-L2: 0.1834
7.  **RestormerLite**: 10.03M params, (Training incomplete/No test results yet)
8.  **StableFNO2d**: 10.66M params, Rel-L2: 0.2137

**Next Steps:**
I will generate a formatted Markdown table summarizing these statistics, including Parameter Count, Rel-L2, MAE, PSNR, and SSIM for easy comparison. I will also highlight models that deviated significantly from the 10M parameter target.
