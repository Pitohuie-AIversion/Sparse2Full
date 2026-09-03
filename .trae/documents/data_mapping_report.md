# 第四章实验数据目录映射报告

通过扫描项目根目录下的 `runs*` 文件夹，现将第四章各类实验所对应的具体数据路径梳理如下：

## 1. 空间重建主实验 (Spatial Reconstruction)
**对应图表**：表 4-2、表 4-3，图 4-1 (收敛曲线), 图 4-3 (重建可视化), 图 4-4 (能谱), 图 4-8 (Pareto)
- **SWE 数据集 (runs/ 目录)**:
  - EDSR (Ours): `runs/AR-SW-10M-edsr`
  - UNet (Baseline): `runs/AR-SW-10M-unet`
  - FNO: `runs/AR-SW-10M-fno2d`
  - NAFNet: `runs/AR-SW-10M-nafnet`
  - ResNetLite: `runs/AR-SW-10M-resnetlite`
  - SegFormer: `runs/AR-SW-10M-segformer`
  - UNO: `runs/AR-SW-10M-uno`
  - MLP-Model: `runs/AR-SW-10M-mlpmodel`
- **DRD 数据集 (runs_drd/ 目录)**:
  - EDSR: `runs_drd/AR-DR2D-EDSR-SRx4-10M-300ep-model_EDSR-s2025-20260103`
  - UNet: `runs_drd/AR-DR2D-10M-UNet-model_UNet-s2025-20251228`
  - FNO: `runs_drd/AR-DR2D-10M-FNO2d-model_FNO2d-s2025-20251228`

## 2. 视野受限下的空间重建 (Crop/Inpainting)
**对应图表**：表 4-4 (不同 Crop Size 下的重建性能)
- **Crop 模型目录**:
  - `runs/AR-DR2D-Crop-Inpainting-PartialConvUNet-Size80-*`
  - `runs/AR-DR2D-Crop-Inpainting-PartialConvUNet-Size64-*`
  - `runs/AR-DR2D-Crop-Inpainting-PartialConvUNet-Size48-*`
  - `runs/AR-DR2D-Crop-Inpainting-PartialConvUNet-Size32-*`
  - (UNet Crop): `runs/UNet_Crop_Scan` 系列

## 3. 时空演化性能 (Spatiotemporal Evolution / Rollout)
**对应图表**：表 4-5, 图 4-2 (Rollout Error)
- **时序自回归模型目录**:
  - Seq-EDSR (VideoSwin): `runs_drd_paper/AR-DR2D-Sequential-EDSR-VideoSwin-SRx4-model_unknown-s2025-20260116`
  - 其他时序测试基准存放在 `runsold2/AR-DR2D-Test-SequentialTemporal-FNO2D-*`

## 4. 消融实验 (Ablation Study)
**对应图表**：表 4-6, 图 4-6 (损失消融曲线)
- **三项损失消融**:
  - Baseline (MSE Only): `runs_3loss_ablation_unet/A0_Baseline` 或 `runs_3loss_ablation/A0_Baseline`
  - Full Loss: `runs_3loss_ablation_unet/A3_Full` 或 `runs_3loss_ablation/A3_Full`

## 5. 序列化课程学习演进 (Sequential Training)
**对应图表**：图 4-7 (演进曲线)
- **阶段化训练日志**:
  - Stage 1: `runs_drd_paper/AR-DR2D-Stage1-EDSR-SRx4-model_EDSR-s2025-20260116`
  - Stage 2: `runs_drd_paper/AR-DR2D-Stage2-VideoSwin-SRx4-model_unknown-s2025-20260116`
  - Stage 3: `runs_drd_paper/AR-DR2D-Stage3-VideoSwin-SRx4-JointFineTune-model_unknown-s2025-20260226`
