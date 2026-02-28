# 论文写作检查清单（五章版）

## 写作前准备（总控门禁）
- [ ] 口径锁定：H 与 DC 同源复用（同入口/同参数镜像）
- [ ] 实验门禁：未通过 `tools/check_dc_equivalence.py` 不得进入主表与结论
- [ ] 复现门禁：`runs/<exp>/config_merged.yaml` + `runs/<exp>/env_fingerprint.json` 必须存在
- [ ] 材料门禁：`paper_package/`（metrics/figs/scripts/README）齐全

---

## 前置部分：摘要（不计入五章）
- [ ] 背景/问题/挑战（口径断裂、谱偏置、复现不可比）
- [ ] 方法一句话（H/DC 复用 + 三件套损失 + 统一协议）
- [ ] 关键结果（至少2–3个指标：Rel-L2、H_err、PSNR/SSIM 或 fRMSE-low）
- [ ] 统计证据一句话（≥3 seeds + paired t-test + Cohen’s d）
- [ ] 关键词 3–6 个
- [ ] 摘要不出现章节号；不引入未验证新主张

---

## 第1章 绪论与相关工作（原1+2）
- [ ] 背景与应用动机清晰
- [ ] 问题定义与核心矛盾：训练/退化/评测口径不一致 → 评测断裂
- [ ] 相关工作覆盖：稀疏观测重建/神经算子/频谱约束/复现协议
- [ ] 研究空白定位：缺少“口径可控 + 审计门禁 + 材料闭环”
- [ ] 本文贡献列表与第5章结论一致

---

## 第2章 问题定义与统一框架（原3+4）
- [ ] 符号与口径：u、ŷ(z)、ũ、H、DC、H_err 定义一致
- [ ] 三件套损失：L_rec / L_spec / L_dc（域、权重、作用机理）
- [ ] 顺序训练策略：空间→时序→联合（切换条件与目标）
- [ ] 可检验命题/假设（3条）与后续验证协议一一对应
- [ ] 复杂度与资源口径定义（Params/FLOPs@256²/显存/延迟）

---

## 第3章 工程实现与实验设计（原5 + 原6的设置/协议）
- [ ] 统一接口契约：init/forward 约定明确
- [ ] 数据与退化实现：SR/Crop 口径签名写清（σ、k、interp、align、boundary）
- [ ] `tools/check_dc_equivalence.py`：输入/输出/阈值/报告字段完整
- [ ] 实验设置：数据切分、标准化、seed、确定性、AMP、硬件环境
- [ ] 指标集固定：Rel-L2、MAE、PSNR、SSIM、fRMSE-low/mid/high、bRMSE、cRMSE、H_err
- [ ] 对比基线与消融配置清晰（A0–A3）
- [ ] 资源四项测量方式统一

---

## 第4章 实验结果与理论验证（原6结果 + 原7验证）
- [ ] 主结果表：≥3 seeds 均值±标准差
- [ ] 显著性：paired t-test（样本配对）+ Cohen’s d + 置信区间
- [ ] 消融：A0–A3（证明三件套互补）
- [ ] 扫描：k_max 与 λ_s（稳定区间+拐点+资源代价）
- [ ] 一致性验证：mean/max MSE 阈值 Pass/Fail（不通过则阻断）
- [ ] 负例反证：错配导致“断裂率”上升（H_err vs Rel-L2 相关性破坏）
- [ ] 跨分辨率：128/256/512 外推评测 + 诊断日志（口径→别名→阈值）
- [ ] 长时预测：≥20 steps 稳定性/能量漂移率（如有）
- [ ] 资源四项：Params/FLOPs/显存/延迟同设备同batch报告
- [ ] 所有图表与表格落地到 `paper_package/figs` 与 `paper_package/metrics`

---

## 第5章 讨论与结论（原8+9）
- [ ] 优势：评测口径可控、三件套互补、复现可比
- [ ] 物理意义：能谱/尺度一致性（解释“为什么变好”）
- [ ] 局限性边界：复杂边界、强噪声、极端稀疏、工程成本
- [ ] 工程建议：先锁口径再调模型；λ_dc 与噪声联动；边界带诊断
- [ ] 结论不引入新主张；与第1章贡献列表完全一致
- [ ] 未来工作可落地：主动采样/动态频谱权重/弱式约束/基础模型

---

## 最终交付材料（强制）
- [ ] `runs/<exp>/config_merged.yaml`
- [ ] `runs/<exp>/env_fingerprint.json`
- [ ] `runs/<exp>/consistency_report.json`
- [ ] `paper_package/scripts/`（复现/汇总/显著性/画图）
- [ ] `paper_package/metrics/`（主表/显著性/资源表/诊断日志）
- [ ] `paper_package/figs/`（代表/失败/谱图/边界带）
- [ ] `paper_package/README.md`（一键复现说明）
