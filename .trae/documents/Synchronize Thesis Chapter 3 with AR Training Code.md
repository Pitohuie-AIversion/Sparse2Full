Based on the analysis of `train_real_data_ar.py` and `ops/losses.py`, I will update **Chapter 3 (Methodology)** to strictly align with the actual code implementation.

**Key Synchronization Points:**

1.  **Training Strategy (Section 3.4.4)**:
    *   **Update**: Explicitly specify the **20-step autoregressive rollout** in the "Joint Fine-tuning" phase, as implemented in the code's main loop.
    *   **Refinement**: Clarify the "Spatial -> Temporal -> Joint" execution flow matches the code's `SequentialSpatiotemporalTrainer` class structure.

2.  **Loss Function (Section 3.5)**:
    *   **Addition**: Add **Section 3.5.6 Temporal Consistency Regularization (AR-Specific)** to describe the `derivative_consistency` (time derivative error) and `energy_consistency` (energy evolution error) terms found in `compute_ar_total_loss`. These are critical for AR stability but missing from the current text.
    *   **Clarification**: Note that these terms are weighted by a dynamic factor linked to teacher forcing decay (as seen in lines 1070-1076 of `losses.py`).

3.  **Data Consistency (Section 3.3)**:
    *   **Verification**: The text already accurately describes the Z-score/Original domain split and `INTER_AREA` interpolation, which matches `ops/degradation.py` and `RealDiffusionReactionDataModule`. No major changes needed here, just minor reinforcement of the "blocking audit" relying on the external script `check_dc_equivalence.py`.

**Action Plan:**
*   **Edit `thesis_paper/manuscript_gpt_review/chapter3.md`**:
    *   Insert details about the 20-step rollout in Section 3.4.4.
    *   Add the new subsection 3.5.6 for Temporal Consistency.
    *   Ensure the "Three-Stage" description perfectly matches the code's logic.

I will only edit `chapter3.md` as `chapter4.md` (Theoretical Analysis) remains consistent with the high-level method and requires no code-specific engineering updates.