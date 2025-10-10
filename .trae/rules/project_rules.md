
---

# 0. 黄金法则（Golden Rules）

1. **一致性优先**：观测算子 **H** 与训练 **DC** 必须**复用同一实现与配置**（核/σ/插值/对齐/边界）。
2. **可复现**：同一 YAML + 种子，验证指标方差 ≤ **1e-4**。
3. **统一接口**：所有模型 `forward(x[B,C_in,H,W])→y[B,C_out,H,W]`；输入打包 `[baseline, coords, mask, (fourier_pe?)]`。
4. **可比性**：横向对比必须报告 **均值±标准差（≥3 种子）** + **资源成本**（Params/FLOPs/显存/时延）。
5. **文档先行**：新增任务/算子/模型前，先提交 PRD/技术文档补丁。

---

# 1. 代码与风格

* **语言与版本**：Python 3.10+，PyTorch ≥ 2.1。
* **风格**：`ruff + black + isort`；类型标注 `mypy --strict` 通过。
* **结构**：`configs/、datasets/、models/、ops/、utils/、tools/、runs/、paper_package/`。
* **提交信息**：`<scope>: <summary>`（如 `data: add PDEBenchCrop dataset`）。
* **测试**：`pytest -q` 必须通过；核心算子（H/DC/频域损失）有**单元测试**。

**CI 检查**：lint/type/test 全绿才可合并。
**DoD**：新增模块带最小单测；接口与现有训练脚本联通。

---

# 2. 数据与观测（PDEBench）

* **数据卡**：来源版本、变量键名、边界条件、许可；存 `paper_package/data_cards/`。
* **切分**：固定 `splits/{train,val,test}.txt`；**不得**临时随机切分。
* **标准化**：逐通道 `z-score`，`norm_stat.npz` 随实验产出。
* **观测生成**：

  * SR：`GaussianBlur(σ,k=5)+INTER_AREA downsample×s`；
  * Crop：`(h_c,w_c)` 与中心**对齐 patch_size 倍数**；边界策略明确（mirror/zero/wrap）。
* **一致性**：训练 DC 与数据观测 **H** 完全一致（同实现，同配置）。

**CI 检查**：`tools/check_dc_equivalence.py` 随机抽样 100 个 case，验证 `MSE(H(GT), y) < 1e-8`。
**DoD**：通过一致性脚本；`datasets/` 与 `ops/degradation.py` 参数一一对应。

---

# 3. 配置与命名

* **Hydra YAML**：数据/模型/训练/损失分层；不可在代码硬编码关键超参。
* **实验名**：`<task>-<data>-<res>-<model>-<keyhyper>-<seed>-<date>`
  例：`SRx4-DR2D-256-SwinFNO_w8d2262_m16-s2025-20251011`。
* **快照**：训练开始时，将合并后的 YAML 写入 `runs/<exp>/config_merged.yaml`。

**DoD**：每个 `runs/<exp>/` 都有完整 YAML 快照与版本信息（git commit、env 指纹）。

---

# 4. 训练与资源

* **优化**：AdamW(lr=1e-3, wd=1e-4)、Cosine+1k warmup、AMP、梯度裁剪 1.0。
* **课程**：SR 先 ×2 再 ×4；Crop 先大窗后小窗（如 40%→20%）。
* **采样**（Crop）：均匀 40% + 边界 30% + 高梯度 30%。
* **分布式**：DDP 优先；固定随机种子与确定性开关。
* **资源统计**：记录 Params(M)、FLOPs(G@256²)、显存峰值(GB)、推理延迟(ms)。

**DoD**：训练日志包含资源四项；`torch.cuda.max_memory_allocated()` 与 FLOPs 工具记录齐全。

---

# 5. 模型与接口

* **最小集合**：Swin-UNet（可选 FNO 瓶颈）、Hybrid(Attn∥FNO∥UNet)、通用 MLP（坐标/patch）。
* **扩展基线**：U-Net/U-Net++、FNO、U-FNO、SegFormer/UNetFormer、MLP-Mixer、LIIF-Head。
* **统一签名**：`__init__(in_ch, out_ch, img_size, **kwargs)`；`forward(x)->y`。
* **解码**：优先“**双线性 + 3×3**”，减少棋盘格。

**DoD**：新增模型能在相同 `Dataset`/损失下训练并产出完整日志与指标。

---

# 6. 损失与值域

* **三件套**：`L = L_rec + λ_s L_spec(low-freq) + λ_dc L_dc`（默认权重 1.0/0.5/1.0）。
* **频域**：仅比较 `kx=ky=16` 低频模；非周期可镜像延拓。
* **值域**：模型输出在 z-score 域；**DC 与频域损失在原值域**计算（反归一化 `μ/σ`）。
* **可选**：梯度损失、PDE 残差（低频加权）。

**CI 检查**：单测验证“z-score→原值域→H”管线无偏差。
**DoD**：训练曲线中 `||H(ŷ)−y||` 与 Rel-L2 同步下降。

---

# 7. 评测与对比（论文口径）

* **指标**：Rel-L2、MAE、PSNR、SSIM、fRMSE-low/mid/high、bRMSE（边界带 16px 比例缩放）、cRMSE、`||H(ŷ)−y||`。
* **聚合**：每通道先算，后**等权平均**（物理权重需另列）。
* **统计**：≥3 种子，**均值±标准差**；对主基线做 **paired t-test**（Rel-L2）+ **Cohen’s d**。
* **资源表**：Params/FLOPs/显存/延迟。

**工具**：`eval.py` 产出 `metrics.jsonl`（case 级）；`tools/summarize_runs.py` 自动生成 `results.md/tex` 与显著性报告。
**DoD**：`paper_package/metrics/` 自动产出主表 + 资源表 + 显著性。

---

# 8. 可视化与诊断

* **标准图**：GT/Pred/Err 热图（统一色标）、功率谱（log）、边界带局部放大。
* **失败案例**：标注失败类型（边界层溢出/相位漂移/振铃/能量偏差）与改进建议。

**DoD**：每个主实验至少 3 个代表 case 的成套图，放在 `paper_package/figs/`。

---

# 9. 论文材料（paper_package/）

```
paper_package/
├─ data_cards/（来源/许可/切分）
├─ configs/（最终 YAML 快照）
├─ checkpoints/（关键 ckpt，走 LFS）
├─ metrics/（主表/显著性/CSV/每 case JSONL）
├─ figs/（代表图/失败案例/谱图）
├─ scripts/（一键复现与汇总）
└─ README.md（环境/命令/结果重现）
```

* **盲审模式**：支持匿名化导出（隐藏作者/路径）。
  **DoD**：`make paper_package` 命令产出完整材料包，可独立审阅与复现。

---

# 10. 安全与合规

* **数据许可**：仅使用拥有再发布许可的数据；`data_cards/` 声明来源与限制。
* **隐私**：不上传原始受限数据到公共仓库；使用 Git LFS 管理 ckpt/ONNX/npz。
* **开源协议**：代码 MIT/Apache-2.0；模型权重与数据按各自许可。

**DoD**：仓库含 LICENSE、NOTICE；`paper_package` 不含受限素材。

---

# 11. 分支与发布

* **Git Flow**：`main`（稳定）、`dev`（集成）、`feat/*`（功能）、`exp/*`（实验）。
* **Release**：语义化版本 `vX.Y.Z`；release 附变更日志、关键 ckpt 与表格。

**DoD**：发布前 CI 全绿；`paper_package/` 与 `runs/` 指向的结果一致。

---

# 12. 受理与验收（Definition of Done）

* **功能/模块 DoD**：代码+文档+单测+CI 通过。
* **实验 DoD**：3 种子、主表+资源表、可视化、`H` 一致性脚本通过。
* **论文 DoD**：`paper_package/` 完整，盲审版可直接打包；复现命令能跑通并复现主表（≤5% 误差）。

---
