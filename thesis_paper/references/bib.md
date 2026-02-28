[1] RAISSI M, PERDIKARIS P, KARNIADAKIS G E. Physics-informed neural networks: A deep learning framework for solving forward and inverse problems involving nonlinear partial differential equations[J]. Journal of Computational Physics, 2019, 378:686-707.

[2] WANG S, YU X, PERDIKARIS P. When and why PINNs fail to train: A neural tangent kernel perspective[J]. Journal of Computational Physics, 2022, 449:110768.

[3] WANG S, TENG Y, PERDIKARIS P. Respecting causality is all you need for training physics-informed neural networks[EB/OL]. arXiv:2203.07404, 2022[2026-01-26].

[4] EVENSEN G. The ensemble Kalman filter: theoretical formulation and practical implementation[J]. Ocean Dynamics, 2003, 53:343-367.

[5] EVERSON R, SIROVICH L. Karhunen–Loève procedure for gappy data[J]. Journal of the Optical Society of America A, 1995, 12(8):1657-1664.

[6] LI Z, KOVACHKI N B, AZIZZADENESHELI K, et al. Fourier neural operator for parametric partial differential equations[C]//Proceedings of the International Conference on Learning Representations (ICLR 2021). [S.l.]:[出版者不祥], 2021.

[7] LU L, JIN P, PANG G, et al. Learning nonlinear operators via DeepONet based on the universal approximation theorem of operators[J]. Nature Machine Intelligence, 2021, 3(3):218-229.

[8] BARTOLUCCI F, DE BÉZENAC E, RAONIĆ B, et al. Representation equivalent neural operators: A framework for alias-free operator learning[C]//Advances in Neural Information Processing Systems 37 (NeurIPS 2023). [S.l.]:Curran Associates, Inc., 2023.

[9] TAKAMOTO M, PRADITIA T, BORTOLATO G, et al. PDEBench: An extensive benchmark for scientific machine learning[C]//Advances in Neural Information Processing Systems 35 (NeurIPS 2022): Datasets and Benchmarks Track. [S.l.]:Curran Associates, Inc., 2022.

[10] TAKAMOTO M, PRADITIA T, BORTOLATO G, et al. PDEBench datasets[EB/OL]. DaRUS Data Repository, 2022[2026-01-26]. DOI:10.18419/darus-2986.

[11] RAHAMAN N, BARATIN A, ARPIT D, et al. On the spectral bias of neural networks[C]//Proceedings of the 36th International Conference on Machine Learning (ICML 2019). Proceedings of Machine Learning Research, 2019, 97:5301-5310.

[12] TANCIK M, SRINIVASAN P P, MILDENHALL B, et al. Fourier features let networks learn high frequency functions in low dimensional domains[C]//Advances in Neural Information Processing Systems 33 (NeurIPS 2020). [S.l.]:Curran Associates, Inc., 2020.

[13] SHI X, CHEN Z, WANG H, et al. Convolutional LSTM network: A machine learning approach for precipitation nowcasting[C]//Advances in Neural Information Processing Systems 28 (NIPS 2015). [S.l.]:MIT Press, 2015:802-810.

[14] LIU Y, CAI Z, XU Z. Multi-scale deep neural network (MscaleDNN) for solving Poisson–Boltzmann equation in a molecular region[J]. Communications in Computational Physics, 2020, 28(5):1718-1740.

[15] OPENCV. Interpolation methods[EB/OL]. OpenCV, [2026-01-26].

[16] OPENCV. Gaussian filter (GaussianBlur)[EB/OL]. OpenCV Documentation, [2026-01-26].
## 第3章引用插入建议（按小节）

- 3.1（问题定义/逆问题范式：观测一致性 + 正则/先验）
  在“argmin ||H(u)-y|| + R(u)”附近插入：[1][2]

- 3.2.1（SR口径：先低通再降采样；INTER_AREA 缩小时更合适）
  在“抗混叠原则”“固定采用 INTER_AREA 插值”附近插入：[8][9]

- 3.2.2（Crop口径：中心对齐、patch倍数、边界/对齐显式声明）
  口径写法属于工程规范性表述，可不强制引用；如需要“裁剪/对齐导致伪影”的依据，可在第2章放文献更合适（第3章可只保留实现规范）

- 3.4.1（Fourier 特征编码）
  在“Fourier 特征编码”首次出现处插入：[4]

- 3.5.2（频谱偏置与谱域约束动机）
  在“频谱偏置”“低频更易学，高频更难稳定学习”的动机处插入：[3]
  如果还要补充“用谱域损失/频域约束常用于重建类任务”，可选加：[7]

- 3.4.2 / 训练策略（三阶段、Teacher Forcing、Teacher Forcing Decay、Exposure Bias）
  在“Teacher Forcing”“逐步减少真值引导/衰减”处插入：[5]
  在“Exposure Bias/滚动推理误差累积”处插入：[6]

- 3.3（DC ≡ H、阻断式审计）
  这一条属于论文的自定义方法论与工程机制，通常不要求外文献；除非你要强调“可审计性/可复现协议”，那更适合在第6章评测协议里集中引用（例如可复现实验实践类文献/指南）

---

## 第3章参考文献（建议清单，可直接并入总参考文献表）

[1] KAIPIO J, SOMERSALO E. Statistical and Computational Inverse Problems[M]. New York: Springer, 2005.

[2] VOGEL C R. Computational Methods for Inverse Problems[M]. Philadelphia: Society for Industrial and Applied Mathematics (SIAM), 2002.

[3] RAHAMAN N, BARATIN A, ARPIT D, et al. On the Spectral Bias of Neural Networks[J]. Proceedings of Machine Learning Research, 2019.

[4] TANCIK M, SRINIVASAN P P, MILDENHALL B, et al. Fourier Features Let Networks Learn High Frequency Functions in Low Dimensional Domains[C]//Advances in Neural Information Processing Systems (NeurIPS). 2020.

[5] BENGIO S, VINYALS O, JAITLY N, et al. Scheduled Sampling for Sequence Prediction with Recurrent Neural Networks[C]//Advances in Neural Information Processing Systems (NeurIPS). 2015.

[6] RANZATO M, CHOPRA S, AULI M, et al. Sequence Level Training with Recurrent Neural Networks[C]//International Conference on Learning Representations (ICLR). 2016.

[7] (可选) 频域损失/频率聚焦类：如果第6章会做频谱指标与消融，这里再补更合适；第3章可先不放。

[8] OpenCV. Resizing and Rescaling Images with OpenCV[EB/OL]. [2026-01-26]. https://opencv.org/blog/resizing-and-rescaling-images-with-opencv/

[9] OpenCV Documentation. Smoothing Images (Gaussian Blur)[EB/OL]. [2026-01-26]. https://docs.opencv.org/4.x/d4/d13/tutorial_py_filtering.html
第4章建议补齐的引用文献（按你给的“大连海事大学格式”示例）

下列条目用于第4章的“欠定逆问题/正则化、课程学习/Teacher Forcing、频谱偏置”等位置；其中 arXiv 预印本按“电子文献”类型给出，便于你从第1章开始统一补齐。

逆问题/正则化（建议放在 4.0 或 4.1）
[1] KAIPIO J, SOMERSALO E. Statistical and Computational Inverse Problems[M]. New York: Springer, 2005.
[2] VOGEL C R. Computational Methods for Inverse Problems[M]. Philadelphia: SIAM, 2002.

算子范数/Lipschitz 基础（建议放在 4.2.1）
[3] KREYSZIG E. Introductory Functional Analysis with Applications[M]. New York: Wiley, 1978.

课程学习与 Teacher Forcing/Exposure Bias（建议放在 4.5）
[4] BENGIO Y, LOURADOUR J, COLLOBERT R, et al. Curriculum learning[C]//Proceedings of the 26th Annual International Conference on Machine Learning (ICML). New York: ACM, 2009.
[5] BENGIO S, VINYALS O, JAITLY N, et al. Scheduled Sampling for Sequence Prediction with Recurrent Neural Networks[EB/OL]. (2015-06-10)[2026-01-26]. arXiv:1506.03099.
[6] LAMB A M, GOYAL A G, ZHANG Y, et al. Professor Forcing: A New Algorithm for Training Recurrent Networks[EB/OL]. (2016-10-28)[2026-01-26]. arXiv:1610.09038.

频谱偏置（建议放在 4.4 第3点）
[7] RAHAMAN N, BARATIN A, ARPIT D, et al. On the Spectral Bias of Neural Networks[EB/OL]. (2018-06-22)[2026-01-26]. arXiv:1806.08734.