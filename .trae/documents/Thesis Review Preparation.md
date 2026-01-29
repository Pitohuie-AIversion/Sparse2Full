# Symbol Table Verification and Completion Plan

Based on the user's request to "check if my thesis has a symbol table, if not please supplement", I have reviewed the existing files.

## Findings

1. **Existence**: A symbol table already exists in `chapter0_notation.md`.
2. **Content**: The existing `chapter0_notation.md` is comprehensive and follows the strict "Observational Protocol" defined in `symbol_checklist.md`. It correctly defines:

   * Basic variables ($u$, $y$, $\mathbf{x}$, $t$)

   * Noise and parameters ($\sigma\_{\mathrm{blur}}$, $\sigma\_n$, $\sigma\_z$) - **Crucially, these are already disambiguated**.

   * Operators ($H$, $DC$, $f\_\theta$)

   * Loss functions and Metrics ($\mathcal{L}_{rec}$, $H_{\mathrm{err}}$, etc.)
3. **Formatting**: It uses a clear Markdown table format.

## Conclusion

The thesis **already has a high-quality symbol table** in `chapter0_notation.md`. No new file needs to be created. However, to ensure it is fully "supplemented" as requested, I will:

1. **Verify completeness**: I will double-check if any key symbols used in the main chapters (like $G\_{\sigma\_{\mathrm{blur}}}$ for the Gaussian kernel or specific sets like $\mathcal{K}\_{\mathrm{low}}$) are missing from the table.

   * *Self-Correction*: I noticed `G_{\sigma_{\mathrm{blur}}}` is mentioned in the description of $H$ but not as a separate entry. Adding it explicitly might be helpful.

   * $\mathcal{K}\_{\mathrm{low}}$ (Low-frequency set) is used in Chapter 3/6 but might be missing from the table.

## Action Plan

1. **Refine** **`chapter0_notation.md`**:

   * Add $\mathcal{K}\_{\mathrm{low}}$ (Low-frequency index set).

   * Add $G\_{\sigma\_{\mathrm{blur}}}$ (Gaussian kernel).

   * Add $D\_s$ (Downsampling operator).

   * Add $C\_{h\_c, w\_c}$ (Cropping operator).

   * Ensure the file title and structure are consistent with the rest of the thesis.

2. **Report to User**: Inform the user that the symbol table exists in `chapter0_notation.md` and that I have supplemented it with a few missing operator symbols to make it perfect.

