# 检查训练结果计划 (Check Training Results Plan)

## 1. 摘要 (Summary)
用户要求检查之前部署的 `Bicubic-CNN` 和 `RBF-CNN` 基线模型的训练结果。目前 `RBF-CNN` 已经训练并测试完成，而 `Bicubic-CNN` 仍在后台训练中。本计划旨在提取已完成实验的指标，并在未完成的实验结束后更新相应的论文表格。

## 2. 当前状态分析 (Current State Analysis)
- **RBF-CNN (DRD 数据集, Crop 16x16)**:
  - 训练已完成。
  - 测试结果已保存在 `runs_rbf_cnn_drd/test_results.json` 中。
  - 核心指标：Rel-L2 = 0.9745, PSNR = 16.65 dB, SSIM = 0.0633, DC Error = 0.0020。
  - 资源消耗：Params = 0.37 M, FLOPs = 6.06 G, Latency = 1.05 ms。
  - 论文表 4-3 中的 RBF-CNN 结果已更新。
- **Bicubic-CNN (SWE 数据集, SRx4)**:
  - 正在后台训练中（当前进度约 40/300 Epochs）。
  - 输出目录为 `runs_bicubic_cnn_swe/`，但最终的 `test_results.json` 尚未生成。

## 3. 具体执行步骤 (Proposed Changes)
1. **监控 Bicubic-CNN 训练进度**：
   - 持续监控 `nohup_bicubic_cnn.log` 的输出，直到训练完成。
   - 解析 `runs_bicubic_cnn_swe/` 目录下生成的 `test_results.json`。
2. **更新论文表格**：
   - 将 Bicubic-CNN 提取到的 Rel-L2、PSNR、SSIM 和 $H_{\mathrm{err}}$ 填入 `thesis_paper/manuscript_5_chapter/chapter4_results_verification.md` 的表 4-3 中，替换“待填入”。
3. **补充结果分析**：
   - 根据两者的实际指标（特别是 RBF-CNN 在极度稀疏下的表现，以及 Bicubic-CNN 在超分下的表现），完善论文 4.2.1 节中的“混合插值基线分析”段落。

## 4. 假设与决策 (Assumptions & Decisions)
- RBF-CNN 在 16x16 极度稀疏的 Crop 任务中（保留率约 1.5%），表现出极高的误差（Rel-L2=0.9745），这与之前纯 UNet/EDSR 在此极限下的失效现象一致，证明传统插值在极度稀疏下无法提供有效的初始场。我们将在论文中如实反映这一点。
- Bicubic-CNN 预计将在 SRx4 任务中取得优于纯 Bicubic（Rel-L2=0.1480）但可能弱于 EDSR（Rel-L2=0.0023）的结果。

## 5. 验证步骤 (Verification Steps)
- 确认所有待填入的指标已准确更新到 `.md` 论文文件中。
- 确保没有引入格式错误（如 Markdown 表格对齐问题）。