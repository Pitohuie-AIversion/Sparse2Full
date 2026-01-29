I will perform the following edits to ensure narrative consistency across the thesis, specifically bridging the gap between the code's **Autoregressive (AR)** features and the introductory/concluding chapters.

1. **Update Chapter 1 (Introduction)**:

   * **Section 1.6.3 (Loss Design)**: Add a specific note about **"Temporal Consistency Regularization"** (Derivative/Energy terms) for long-term prediction, matching the new Section 3.5.5.

   * **Section 1.7 (Innovations)**: Explicitly list the **"Sequential Training Strategy"** (Spatial -> Temporal -> Joint) as a contribution, as it is a key engineering solution implemented in the code to solve convergence issues.

2. **Update Chapter 9 (Conclusion)**:

   * **Section 9.2 (Innovations)**: Expand the innovation summary to include **"Robust Long-term Prediction Strategy"**, highlighting the combination of Sequential Training and Temporal Consistency Regularization.

**Reasoning**:

* The code heavily relies on `SequentialSpatiotemporalTrainer` and AR-specific losses (`compute_ar_total_loss`).

* Chapter 3 (Methodology) now details these.

* Chapter 1 and 9 currently miss these aspects, creating a "surprise" in the middle chapters. These edits will ensure the "AR" and "Temporal" narrative is consistent from start to finish.

