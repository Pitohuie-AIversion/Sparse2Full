# 附录：理论验证与实验审计协议 (Verification Protocols)

> **说明**：本附录收录了第4章中因篇幅限制未详细展开的工程验证细节，包括评测一致性审计脚本逻辑、结构稳健性扫描区间、跨域鲁棒性诊断流程以及详细的材料归档标准。这些内容构成了本研究“可复现性”与“理论闭环”的工程证据链。

---

## A.1 评测一致性验证 (Proposition 1 Verification)

### A.1.1 阻断式审计机制 (Blocking Audit Mechanism)

**目的**：在统计汇总之前，证明训练端退化算子 $\mathrm{DC}$ 与数据观测算子 $H$ 满足硬约束：
$$
\mathrm{DC} \equiv H
\quad \text{（同一入口、同一实现、同一参数镜像、同一边界/插值/对齐策略）}.
$$

**脚本工具**：`tools/check_dc_equivalence.py`

**执行方法**：
随机抽样 $N \ge 100$ 个样本 $u^{(i)}$，在**关闭观测噪声**（$n=0$）的条件下，分别计算：
1.  算子输出：$y_H^{(i)} = H(u^{(i)})$
2.  退化输出：$y_{DC}^{(i)} = DC(u^{(i)})$

记录误差统计量：
$$
e^{(i)}=\mathrm{MSE}\!\left(y_H^{(i)},\,y_{DC}^{(i)}\right),\quad
\bar e=\frac{1}{N}\sum_{i=1}^N e^{(i)},\quad
e_{\max}=\max_i e^{(i)}.
$$

**验收阈值**：
*   **Pass**：$\bar e < 10^{-8}$ 且 $e_{\max} < 10^{-7}$。
*   **Fail**：否则终止实验，并将差异归档至 `runs/<exp>/consistency_report.json`。

> **工程备注**：当 $H$ 内含浮点插值、FFT 或 GPU 非确定性算子时，阈值需与数值精度匹配。

### A.1.2 负例构造与反证 (Negative Controls)

为证明一致性的必要性，设计如下“故意错配”的负例条件：

*   **操作层错配**：
    *   SR 任务：`INTER_AREA → INTER_LINEAR` 或 $\sigma_{\mathrm{blur}} \to \sigma_{\mathrm{blur}}+\Delta\sigma_{\mathrm{blur}}$
    *   Crop 任务：`mirror → zero` 边界或 `center → corner` 对齐偏移

**统计诊断**：
计算 Rel-L2 与 $H_{\mathrm{err}}$ 的相关性：
$$
r=\mathrm{corr}_{\text{Pearson}}(\mathrm{Rel\text{-}L2}_j,\,H_{\mathrm{err},j})
$$
**判定准则**：
*   **正例**：$r$ 显著正相关，Rel-L2 下降时 $H_{\mathrm{err}}$ 同步下降。
*   **负例**：出现“Rel-L2 改善但 $H_{\mathrm{err}}$ 恶化”的“指标断裂”现象。

---

## A.2 结构稳健性验证 (Proposition 2 Verification)

### A.2.1 敏感性扫描参数空间

为验证频域约束（$L_{\mathrm{spec}}$）的稳健性，对关键超参进行网格扫描：

$$
k_{\max} \in \{8,12,16,20,24\},\qquad
\lambda_s \in \{10^{-4},10^{-3},10^{-2}\}.
$$

**验收结论逻辑**：
不寻求单一“最优解”，而是寻找“稳定区间”：
*   $k_{\max} \le 12$：低频过于平滑，细节丢失。
*   $k_{\max} \ge 24$：引入高频噪声，训练不稳定。
*   **稳定区间**：$k_{\max} \in [14, 18]$，在此区间内 Rel-L2 与 $fRMSE_{low}$ 均保持低位且波动极小。

---

## A.3 跨域鲁棒性验证 (Proposition 3 Verification)

### A.3.1 跨分辨率别名诊断流程

当出现“训练分辨率（256）表现良好，但推理解析度（512）性能崩塌”时，执行以下诊断流程：

1.  **口径复核**：重新运行 `check_dc_equivalence.py`，排除 $H$ 实现中的硬编码尺寸 bug。
2.  **别名/混叠诊断**：计算预测结果的功率谱 $P(k)$。
    *   若在 Nyquist 频率附近出现异常能量堆积（Energy Pile-up），判定为**混叠（Aliasing）**。
3.  **修复策略**：
    *   **阈值自适应**：将 $L_{\mathrm{spec}}$ 的 $k_{\max}$ 从“固定索引”改为“按 Nyquist 比例自适应”。
    *   **滤波器修正**：在解码器末端显式加入抗混叠低通滤波器（Anti-aliasing Filter）。

---

## A.4 材料归档与审计 (Archiving & Auditing)

为满足可复现性要求，所有主实验必须生成以下审计材料。

### A.4.1 环境指纹 (Environment Fingerprint)

文件路径：`runs/<exp>/env_fingerprint.json`
必须记录的字段：
*   **Random Seed**：Python, NumPy, PyTorch (CPU/CUDA)
*   **Determinism**：`torch.use_deterministic_algorithms` 状态
*   **Hardware**：GPU 型号、驱动版本、CUDA 版本
*   **Software**：PyTorch 版本、OS 信息
*   **Codebase**：Git Commit Hash, Dirty Flag

### A.4.2 交付物清单 (Deliverables)

每次实验必须产出的完整包：
1.  **Config Snapshot**：`config_merged.yaml`（运行时真实配置）
2.  **Audit Report**：`consistency_report.json`（一致性审计结果）
3.  **Metrics**：`metrics.jsonl`（逐样本指标）与 `results.md`（汇总表）
4.  **Visuals**：`figs/` 目录下的标准图组（GT/Pred/Err, Spectrum, Zoom-in）

---

## A.5 参考文献

[1] Bartolucci, F., et al. (2023). Representation equivalent neural operators: A framework for alias-free operator learning. arXiv:2305.19913.
[2] PyTorch Documentation. Reproducibility. https://pytorch.org/docs/stable/notes/randomness.html
