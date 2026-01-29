Based on the strict comparison between the source code (`swin_unet.py`, `fno2d.py`, `unetformer.py`) and the ASCII flowcharts in `01_Model_Architectures.md`, I have identified the following points:

1.  **Swin-UNet**:
    *   **Flowchart Accuracy**: Generally accurate.
    *   **Refinement**: The code shows `PatchMerging` happens *after* Swin blocks in the encoder (Code: `layer = BasicLayer(..., downsample=PatchMerging if ... else None)`). The flowchart correctly depicts this. The optional `FNOBottleneck` is applied on the deepest tokens, which matches the code (`if self.fno_bottleneck is not None: tokens = self.fno_bottleneck(...)`). The symmetric decoder structure (fuse -> swin -> expand) is also correctly represented.
    *   **Action**: No major changes needed, but I will ensure the "Output Head" explicitly mentions `final_activation` if configured (e.g., Tanh/Sigmoid) as per lines 1022-1027 in `swin_unet.py`.

2.  **FNO 2D**:
    *   **Flowchart Accuracy**: Accurate.
    *   **Refinement**: The code uses a `Linear` projector for input (`self.fc0`) and `MLP` (Linear->Act->Linear) for output (`self.fc1`, `self.fc2`). The spectral layers involve `SpectralConv2d` and a parallel `w` (Conv1x1) branch, summed together (`x1 + x2`). The flowchart currently shows `Activation(分支1 + 分支2)`, which is correct (`x = self.activation(x)` in the loop).
    *   **Action**: Confirm the grid concatenation is clearly shown (it is: `[Input] + [Grid]`).

3.  **UNetFormer**:
    *   **Flowchart Accuracy**: Needs refinement.
    *   **Refinement**: The code structure is `ConvBlock` -> `TransformerConvBlock` stack. The flowchart simplifies this as "GL Block". I should clarify that the encoder stages are composed of *multiple* blocks (`TransformerConvBlock` which contains both CNN and Transformer branches), not just a single "GL Block". Also, the skip connection logic involves `_align_like` and `concat`, which is standard but good to note.
    *   **Action**: Update UNetFormer flowchart to reflect the `ConvBlock` + `TransformerConvBlock` sequence more precisely, and mention the `TransformerConvBlock` internal hybrid structure (CNN branch + Transformer branch sum).

**Conclusion**: The documentation is largely consistent, but UNetFormer's description can be more precise regarding its internal block structure. I will update `01_Model_Architectures.md` to reflect these fine-grained details for better prompt generation.

**Plan**:
1.  **Refine UNetFormer Flowchart**: Update Section 1.6 to show the `TransformerConvBlock` internal parallel structure (CNN + Transformer).
2.  **Verify Swin-UNet**: Add a small note about `final_activation`.

I will execute the update now.