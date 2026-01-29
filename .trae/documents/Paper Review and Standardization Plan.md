# Paper Review and Standardization Plan

Based on the comprehensive review of your manuscript against the provided checklists (`writing_checklist.md`, `symbol_checklist.md`) and template (`template.md`), I have identified specific consistency issues, particularly regarding symbol usage.

## Review Findings

1.  **Symbol Ambiguity ($\sigma$)**:
    -   **Issue**: The symbol $\sigma$ is used ambiguously across chapters, violating **Mandatory Rule 4** of the `symbol_checklist.md` ("$\sigma$" alone is unqualified).
    -   **Blur Scale**: Referred to as $\sigma$ or $G_\sigma$ in **Chapters 2, 3, 4, 5**.
    -   **Noise Level**: Referred to as $\sigma$ (e.g., $\sigma \in \{0.0, 0.01\}$) in **Chapter 6** and **Chapter 0 (Notation)**.
    -   **Z-score Std**: Referred to as $\sigma$ in some contexts (though $\sigma_z$ is defined in checklist).

2.  **Observational Protocol ($H$ vs $DC$)**:
    -   The text generally adheres to the $H \equiv DC$ principle, but the notation for the Gaussian kernel ($G_\sigma$) needs to be strictly unified to $G_{\sigma_{\mathrm{blur}}}$ to match the protocol card.

3.  **Structure & Formatting**:
    -   The chapter structure (Abstract + 1-9) aligns well with `template.md`.
    -   Content coverage appears comprehensive.

## Proposed Implementation Steps

I propose to standardize the notation across all markdown files to meet the strict "Observational Protocol" requirements:

1.  **Update Chapter 0 (Notation)**:
    -   Explicitly define $\sigma_{\mathrm{blur}}$ (Gaussian kernel size), $\sigma_n$ (Noise std), and $\sigma_z$ (Data normalization std).
    -   Remove ambiguous "$\sigma$" definitions.

2.  **Global Symbol Replacement (Chapters 1-9)**:
    -   **Blur**: Replace $G_\sigma$, $\sigma$ (in kernel context) $\rightarrow$ $\sigma_{\mathrm{blur}}$.
    -   **Noise**: Replace $\sigma$ (in noise context) $\rightarrow$ $\sigma_n$.
    -   **Normalization**: Ensure $\sigma_z$ is used consistently.

3.  **Verification**:
    -   Perform a final scan to ensure no "naked" $\sigma$ symbols remain in mathematical contexts.

This plan ensures your paper passes the "Mandatory Rule 4" and eliminates ambiguity for the reviewers.