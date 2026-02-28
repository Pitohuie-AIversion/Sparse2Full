# Chapter 4 Restructuring Spec

## Why
The current structure of Chapter 4 scatters related results (e.g., qualitative analysis is at the end of 4.2, while main quantitative results are at the beginning). Mechanism analysis and ablation studies are separated but cover overlapping topics (e.g., consistency and loss functions). A restructured layout will improve logical flow, grouping "Performance" (What), "Mechanism" (Why), and "Robustness" (When/Cost).

## What Changes
- **Reorganize Sections**:
    - **4.1 Experimental Setup**: Keep as is.
    - **4.2 Main Reconstruction Performance**: Combine SWE (Spatial), DRD (Spatiotemporal), and Qualitative Analysis.
        - Move **Qualitative Analysis** (currently 4.2.6) into this section to support quantitative claims immediately.
        - Move **Crop Capability** (currently 4.2.5) here as a spatial sub-task.
    - **4.3 Mechanism Analysis & Ablation**: Merge "Mechanism Analysis" (4.3) and "Ablation Study" (4.4).
        - Group **Loss Function** (4.4.1) with **Consistency Analysis** (4.3.1) as they both relate to physical constraints.
        - Group **Sequential Training** (4.3.2) with **Spatial Necessity** (4.3.3).
    - **4.4 Robustness & Efficiency**: Group remaining sections.
        - **Noise Sensitivity** (4.4.3).
        - **Extreme Sparsity** (4.5).
        - **Cross-Equation Generalization** (4.4.4).
        - **Resource Efficiency** (4.6).
- **Update Text**: Adjust transitional text to fit the new flow.
- **Renumber Tables/Figures**: Ensure all artifacts are correctly numbered in the new order.

## Impact
- **Affected Code**: `thesis_paper/manuscript_5_chapter/chapter4_results_verification.md`.
- **No New Experiments**: This is purely a content reorganization.

## REORGANIZED Structure
### 4.1 实验设置 (Experimental Setup)
- 4.1.1 Datasets
- 4.1.2 Training Setup
- 4.1.3 Baselines
- 4.1.4 Consistency Protocol
- 4.1.5 Metrics
- 4.1.6 Statistics

### 4.2 稀疏场重建主结果 (Main Reconstruction Results)
- **4.2.1 空间重建性能 (Spatial Reconstruction)**
    - SWE Results (Table 4-2, 4-3)
    - Convergence (Fig 4-4)
    - Crop Inpainting (Table 4-5) - *Moved from 4.2.5*
- **4.2.2 时空演化性能 (Spatiotemporal Evolution)**
    - DRD Results (Table 4-4)
    - Rollout Error (Fig 4-7)
- **4.2.3 定性与谱分析 (Qualitative & Spectral Analysis)** - *Moved from 4.2.6*
    - Visual Comparison (Fig 4-1)
    - Power Spectrum (Fig 4-2)
    - Failure Cases (Fig 4-8)

### 4.3 核心机制与消融实验 (Mechanism & Ablation)
- **4.3.1 物理约束的有效性 (Physical Constraints)**
    - Loss Ablation (Table 4-9 -> **Table 4-6**) - *Renumber*
    - Loss Curves (Fig 4-6)
    - Consistency Mismatch (Table 4-6 -> **Table 4-7**) - *Renumber*
- **4.3.2 序列化训练的必要性 (Sequential Training Strategy)**
    - Stage Evolution (Table 4-7 -> **Table 4-8**, Fig 4-5)
    - Spatial Necessity (Table 4-8 -> **Table 4-9**)

### 4.4 鲁棒性、边界与效率 (Robustness, Boundaries & Efficiency)
- **4.4.1 噪声与跨域鲁棒性 (Noise & Generalization)**
    - Noise Sensitivity (Table 4-10)
    - Cross-Equation (Text)
- **4.4.2 极度稀疏边界 (Extreme Sparsity)**
    - Extreme Scan (Table 4-10 -> **Table 4-11**)
- **4.4.3 资源效率分析 (Resource Efficiency)**
    - Resource Table (Table 4-11 -> **Table 4-12**)
    - Pareto Frontier (Fig 4-3)

### 4.5 本章小结 (Summary)
