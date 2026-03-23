# Chapter 4 Results Figures Arrangement Spec

## Why
用户希望开始梳理第四章（实验结果与分析）的配图。由于现有的结果目录包含独立的观测（ob）、预测（prd）、真实值（gt）和误差（err）图像，需要将它们组合、排版并嵌入到论文第四章中，以直观展示所提方法与基线模型的对比效果。

## What Changes
- 搜集并筛选具有代表性的实验结果图（ob, prd, gt, err）。
- 编写或使用现有的可视化脚本，将这些独立图像拼接成论文所需的复合图（如：多行多列的对比图，包含 colorbar 和标签）。
- 更新 `chapter4_results_verification.md`，使用最新生成的对比图替换现有的占位图或旧图。
- 确保图注（Captions）和图编号与论文正文中的引用保持一致。

## Impact
- Affected specs: Thesis manuscript formatting and visual representation.
- Affected code: `thesis_paper/manuscript_5_chapter/chapter4_results_verification.md`, `thesis_paper/manuscript_5_chapter/images/`

## ADDED Requirements
### Requirement: 实验结果组图生成
The system SHALL provide a script or process to combine individual experimental images (ob, gt, prd, err) into a unified grid figure suitable for thesis publication.

#### Scenario: Success case
- **WHEN** user provides the paths to individual ob, gt, prd, err images
- **THEN** a combined PNG/PDF figure is generated with proper titles, labels, and colorbars.

## MODIFIED Requirements
### Requirement: 第四章插图更新
将 `chapter4_results_verification.md` 中的占位图替换为实际生成的拼接结果图，并确保文字描述与图片内容相符。