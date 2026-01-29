# 论文级技术原理配图材料生成计划

基于对 `models/spatial/` 下所有注册模型的深度代码审计，我已准备好完整的生成材料。以下是执行计划摘要：

1.  **Registry 索引**：基于 `models/spatial/__init__.py` 建立完整的模型列表（共 22 个条目）。
2.  **结构解析**：对每个模型，严格依据 `forward` 方法和 `__init__` 参数，推导其输入注入方式、主干结构、分支融合策略及输出头形式。
    - *特别追踪*：对 `SwinTWithEncoder` 使用的 `SparseInputEncoder` 进行了跨文件追踪。
    - *缺失标记*：确认 `ResNetLite` 在注册表中存在但对应文件缺失，将在报告中标记为 RED/MISSING。
3.  **产出生成**：
    - **ArchitectureSpec**：YAML 格式的精确结构描述，包含 `evidence_anchors`（代码证据锚点）。
    - **DiagramCode**：可直接渲染的 Mermaid 流程图代码。
    - **ImagePrompt**：用于生成学术级矢量图的详细提示词。
4.  **覆盖率报告**：最终输出 Coverage Report，统计 N vs M 及可信度评级。

**模型覆盖列表**：
- **CNN**: UNet, UNet++, FNO2d, UFNOUNet
- **Transformer**: SegFormer, UNetFormer, SegFormerUNetFormer, ViT, SwinT, Transformer, SwinUNet
- **MLP/Implicit**: MLP, MLPMixer, LIIF
- **Hybrid/Sparse**: HybridModel, SwinTWithEncoder, SparseAttentionEncoder, SparseSwinUNet
- **Lite**: ConvUNetLite, CNNAttnLite, ConvGateLite, (ResNetLite - Missing)

我将现在开始逐一输出每个模型的详细技术材料。