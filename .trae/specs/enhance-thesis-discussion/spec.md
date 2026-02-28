# Enhance Thesis Discussion Spec

## Why
While the thesis logic is robust, two minor gaps remain:
1.  The observed performance drop at $16 \times 16$ resolution (Chapter 4) lacks a theoretical explanation linking it to physical sampling limits (Nyquist-Shannon).
2.  Figures (e.g., Fig 3-1, Fig 4-1) are referenced in text but missing actual Markdown image links, which will cause issues during PDF generation.

## What Changes
- **Chapter 5 (Discussion)**: Add a theoretical discussion paragraph linking the $16 \times 16$ empirical limit to the Nyquist sampling theorem and the characteristic length scale of the physical fields (SWE/DRD).
- **Chapter 3 & 4**: Insert standard Markdown image placeholders (e.g., `![图 3-1: 总体框架图](../images/fig3-1.png)`) where figures are referenced.

## Impact
- **Affected Files**:
  - `chapter3_implementation_setup.md`
  - `chapter4_results_verification.md`
  - `chapter5_discussion_conclusion.md`

## ADDED Requirements
### Requirement: Theoretical Interpretation of Sparsity Limit
The discussion SHALL explicitly connect the empirical $16 \times 16$ breakdown point to the Nyquist-Shannon sampling theorem, explaining that this resolution likely falls below the critical sampling rate required to capture the smallest active physical scales (e.g., shock wave width or Turing pattern wavelength).

### Requirement: Figure Placeholders
All figure references in the text (e.g., "如图 3-1 所示") SHALL be accompanied by a valid Markdown image link syntax `![Caption](Path)` to ensure correct rendering.

## MODIFIED Requirements
N/A

## REMOVED Requirements
N/A
