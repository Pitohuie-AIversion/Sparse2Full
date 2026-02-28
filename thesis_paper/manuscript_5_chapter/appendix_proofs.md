# 附录 D：理论命题证明 (Theoretical Proofs)

本附录为正文第 2 章中提出的理论命题提供详细的数学证明。

## D.1 定理 2.3 证明：一致性误差上界

**定理重述**：
设 $H: \mathcal{U} \to \mathcal{Y}$ 为有界线性观测算子，$\mathcal{F}_\theta$ 为重建模型。令 $\hat{u} = \mathcal{F}_\theta(y)$ 为重建解，$u$ 为真值。若观测生成过程为 $y = H(u) + \eta$，且模型满足重建误差界 $\|\hat{u} - u\|_\mathcal{U} \le \epsilon$，则观测一致性误差满足：
$$
\|H(\hat{u}) - y\|_\mathcal{Y} \le \|H\|_{\text{op}} \cdot \epsilon + \|\eta\|_\mathcal{Y}
$$

**证明**：

1.  根据观测模型定义，观测数据 $y$ 与真值 $u$ 的关系为：
    $$ y = H(u) + \eta $$

2.  我们将观测一致性误差 $H_{\text{err}} = H(\hat{u}) - y$ 展开：
    $$
    \begin{aligned}
    H(\hat{u}) - y &= H(\hat{u}) - (H(u) + \eta) \\
                   &= H(\hat{u}) - H(u) - \eta
    \end{aligned}
    $$

3.  利用算子 $H$ 的线性性质（Linearity），有 $H(\hat{u}) - H(u) = H(\hat{u} - u)$。代入上式：
    $$ H(\hat{u}) - y = H(\hat{u} - u) - \eta $$

4.  对等式两边取范数，并应用三角不等式（Triangle Inequality, $\|a+b\| \le \|a\| + \|b\|$）：
    $$
    \begin{aligned}
    \|H(\hat{u}) - y\|_\mathcal{Y} &= \|H(\hat{u} - u) + (-\eta)\|_\mathcal{Y} \\
                                   &\le \|H(\hat{u} - u)\|_\mathcal{Y} + \|-\eta\|_\mathcal{Y} \\
                                   &= \|H(\hat{u} - u)\|_\mathcal{Y} + \|\eta\|_\mathcal{Y}
    \end{aligned}
    $$

5.  根据算子范数（Operator Norm）的定义 $\|H\|_{\text{op}} = \sup_{v \neq 0} \frac{\|Hv\|}{\|v\|}$，对于任意向量 $v$，有 $\|Hv\| \le \|H\|_{\text{op}} \|v\|$。
    令 $v = \hat{u} - u$，则：
    $$ \|H(\hat{u} - u)\|_\mathcal{Y} \le \|H\|_{\text{op}} \cdot \|\hat{u} - u\|_\mathcal{U} $$

6.  结合步骤 4 和 5，并代入已知条件 $\|\hat{u} - u\|_\mathcal{U} \le \epsilon$，得：
    $$
    \begin{aligned}
    \|H(\hat{u}) - y\|_\mathcal{Y} &\le \|H\|_{\text{op}} \cdot \|\hat{u} - u\|_\mathcal{U} + \|\eta\|_\mathcal{Y} \\
                                   &\le \|H\|_{\text{op}} \cdot \epsilon + \|\eta\|_\mathcal{Y}
    \end{aligned}
    $$

**证毕。** $\blacksquare$

---

## D.2 命题 2.2 证明：口径错配误差下界

**命题重述**：
若训练阶段使用退化算子 $DC$ 进行约束，且 $DC \neq H$（存在口径错配），则评测口径误差下界为：
$$
\|H(\hat{u}) - y\|_2 \ge \left| \|DC(\hat{u}) - y\|_2 - \|(H - DC)(\hat{u})\|_2 \right|
$$

**证明**：

1.  我们将观测一致性误差项 $H(\hat{u}) - y$ 进行分解，引入中间项 $DC(\hat{u})$：
    $$
    \begin{aligned}
    H(\hat{u}) - y &= H(\hat{u}) - DC(\hat{u}) + DC(\hat{u}) - y \\
                   &= (H - DC)(\hat{u}) + (DC(\hat{u}) - y)
    \end{aligned}
    $$
    记向量 $A = H(\hat{u}) - y$，向量 $B = (H - DC)(\hat{u})$，向量 $C = DC(\hat{u}) - y$。
    则有 $A = B + C$，即 $C = A - B$。

2.  我们要寻找 $\|A\|$ 的下界。
    考察向量 $C = DC(\hat{u}) - y$。根据三角不等式：
    $$
    \|C\| = \|A - B\| \le \|A\| + \|-B\| = \|A\| + \|B\|
    $$
    即 $\|DC(\hat{u}) - y\| \le \|H(\hat{u}) - y\| + \|(H - DC)(\hat{u})\|$。
    移项得：
    $$
    \|H(\hat{u}) - y\| \ge \|DC(\hat{u}) - y\| - \|(H - DC)(\hat{u})\| \quad \cdots\cdots (1)
    $$

3.  同理，考察向量 $B = (H - DC)(\hat{u})$。
    $$
    B = A - C
    $$
    根据三角不等式：
    $$
    \|B\| = \|A - C\| \le \|A\| + \|C\|
    $$
    即 $\|(H - DC)(\hat{u})\| \le \|H(\hat{u}) - y\| + \|DC(\hat{u}) - y\|$。
    移项得：
    $$
    \|H(\hat{u}) - y\| \ge \|(H - DC)(\hat{u})\| - \|DC(\hat{u}) - y\| \quad \cdots\cdots (2)
    $$

4.  结合不等式 (1) 和 (2)，我们得到：
    $$
    \|H(\hat{u}) - y\| \ge \max\left( \|DC(\hat{u}) - y\| - \|(H - DC)(\hat{u})\|, \; \|(H - DC)(\hat{u})\| - \|DC(\hat{u}) - y\| \right)
    $$
    
    利用绝对值性质 $\max(x-y, y-x) = |x-y|$，上式可写为：
    $$
    \|H(\hat{u}) - y\| \ge \left| \|DC(\hat{u}) - y\| - \|(H - DC)(\hat{u})\| \right|
    $$

**证毕。** $\blacksquare$

**物理意义补充**：
该不等式表明，真实观测误差 $\|H(\hat{u}) - y\|$ 不可能小于算子错配带来的系统性偏差 $\|(H - DC)(\hat{u})\|$ 与训练拟合残差 $\|DC(\hat{u}) - y\|$ 之差的绝对值。当模型在训练任务上完美过拟合（即 $\|DC(\hat{u}) - y\| \to 0$）时，测试误差将被算子错配项 $\|(H - DC)(\hat{u})\|$ 锁定，无法进一步下降。
