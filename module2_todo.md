# Module 2 TODO：Prior Evidence Alignment / Visual Consistency Gating

> 固定基线：Module 2 只建立在正式 **PBridge-B** 上。B 使用 PPR-MoE
> 预测分数同时完成 Top-3 排序和 `softmax(score/0.5)` prefix 加权，
> 固定测试结果为 285/413（69.01%）。基线身份记录在
> `module2_baseline.json`。从本阶段开始，禁止修改、重训或替换 Module 1。

## 0. 研究目标

上一轮 probing 已基本排除：

> “PBridge 的主要收益来自把视觉注意力移动到更正确的位置。”

因此下一步不再继续深挖 attention head，而验证一个更直接的科学问题：

> **当前检索到的 pediatric prior 是否真的被当前图像证据支持？如果不支持，是否应该降低它对 Lingshu 的影响？**

Module 1 解决：

```text
哪些 pediatric prior 值得送进模型？
```

Module 2 拟解决：

```text
这些 prior 对当前病例是否适用？应该信多少？
```

核心假设：

\[
\boxed{
\text{Prior helpfulness depends on prior–visual evidence consistency}
}
\]

---

# 1. 工程约束（必须遵守）

本阶段优先做验证，不直接重构现有 PBridge。

- [x] 固定使用 `module2_baseline.json` 指定的 PBridge-B。
- [x] 不修改当前 Module 1 原有源码。
- [x] 不修改现有 checkpoint。
- [ ] 不修改 Lingshu 正式 SDPA forward。
- [ ] 不修改 MRoPE / mask / input token 顺序。
- [ ] 不重新设计 retrieval。
- [x] 不重新训练 Module 1。
- [ ] 所有实验代码新建到独立目录。
- [ ] 允许 import / wrapper / hook 复用当前 PBridge 代码。
- [ ] Module 2 必须是小模块，不能靠大规模新增参数“硬涨点”。

建议目录：

```text
module2_alignment/
├── build_counterfactual_targets.py
├── extract_features.py
├── consistency_model.py
├── train_consistency.py
├── eval_consistency.py
├── eval_gated_pbridge.py
├── analyze_results.py
└── outputs/
```

---

# 2. 首先停止继续使用 413 题做算法选择

当前 413 题已经反复用于：

- attention probing；
- layer/head 选择；
- patching；
- 机制假设形成。

因此：

> **后续不能再把这 413 题当成严格意义上的独立验证集来调 Module 2。**

TODO：

- [ ] 从原训练数据中重新划分 Module 2 的 train/dev。
- [ ] 所有模型结构、阈值、loss 权重、层位选择都只在 train/dev 上决定。
- [ ] 413 题仅可继续作为 exploratory reference，不能再据此宣称 confirmatory significance。
- [ ] 如果条件允许，另外保留一批从未参与任何设计的新测试集，作为最终一次性评估。

---

# 3. 先构造“Prior 是否有帮助”的监督信号

不能直接把：

```text
Top-3 prior = positive
Random prior = negative
```

当作真值，因为当前 Top-3 中本身可能有不适用于当前病例的 prior。

应使用 frozen PBridge / Lingshu 做 counterfactual measurement。

## 3.1 单条 prior 的 counterfactual helpfulness

对每个训练病例的 Top-K 候选知识，单独测试：

```text
No-Prior
vs
Only-Prior-i
```

记录正确答案 margin：

\[
m_0 = margin(I,Q)
\]

\[
m_i = margin(I,Q,P_i)
\]

定义：

\[
\Delta m_i = m_i - m_0
\]

解释：

```text
Δm_i > 0：该 prior 对当前病例有帮助
Δm_i < 0：该 prior 对当前病例有伤害
Δm_i ≈ 0：影响很小
```

TODO：

- [ ] 对训练集预计算每条候选 prior 的 `Δm_i`。
- [ ] 保存 sample id / prior id / utility / PPR score / `Δm_i`。
- [ ] 不在 Module 2 推理时使用 gold label。
- [ ] gold 只用于离线构造训练 target。

## 3.2 Case-level Top-3 helpfulness

同时计算：

\[
\Delta m_{top3}
=
margin(I,Q,P_{1:3})
-
margin(I,Q)
\]

TODO：

- [ ] 保存 `Δm_top3`。
- [ ] 统计 helpful / harmful / neutral 比例。
- [ ] 检查训练集是否同样存在明显的“纠错”和“干扰”两类病例。

如果训练集中几乎不存在 harmful prior：

> Module 2 的问题定义需要重新检查。

---

# 4. 提取 Visual Evidence 表示

第一版不要做复杂 cross-attention。

只使用 frozen Lingshu 已有视觉表示。

### Option A：vision encoder / merger 输出

优点：

```text
最纯粹的图像证据
不受 prior 影响
```

### Option B：early LLM visual state

优点：

```text
与 Lingshu 后续语义空间更一致
```

TODO：

- [ ] 优先测试 Option A。
- [ ] 再测试一个 early LLM visual state。
- [ ] 不一次扫很多层。
- [ ] 对 visual tokens 做 mean pooling：

\[
v \in \mathbb{R}^{d_v}
\]

---

# 5. Prior 表示

直接复用 Module 1 当前已经生成的 3 个 soft prefix：

\[
p_i \in \mathbb{R}^{3584}
\]

TODO：

- [ ] 使用现有 prefix projection 输出。
- [ ] 不重新编码 raw prior text。
- [ ] 不设计第二套 knowledge encoder。
- [ ] Module 1 checkpoint 全部冻结。

---

# 6. 第一版 Consistency Scorer

目标：

\[
c_i = f(v,p_i)
\]

其中：

```text
c_i 高：当前图像证据支持该 prior
c_i 低：当前图像证据不支持 / prior 可能有害
```

## 6.1 最简单版本

先投影到同一低维空间：

\[
\hat v = W_v v
\]

\[
\hat p_i = W_p p_i
\]

推荐：

```text
d = 128 或 256
```

计算 cosine compatibility：

\[
c_i
=
\frac{\hat v^\top \hat p_i}
{\|\hat v\|\|\hat p_i\|}
\]

只训练：

```text
W_v
W_p
temperature / scale
```

第一版不要上 Transformer / Cross-Attention。

## 6.2 小型增强版

如果纯 cosine 明显不足，再测试：

\[
x_i =
[\hat v,\hat p_i,\hat v\odot\hat p_i,|\hat v-\hat p_i|]
\]

接一个很小的 MLP：

```text
Linear → GELU → Linear → scalar
```

要求：

- [ ] 参数量明确报告。
- [ ] 必须与 cosine baseline 比较。
- [ ] 如果 MLP 没有明显收益，保留简单版本。

---

# 7. Scorer 的训练目标

核心 target 来自真实 counterfactual：

\[
\Delta m_i
\]

## 7.1 Regression

\[
\hat{\Delta m}_i=f(v,p_i)
\]

Loss：

\[
L_{reg}
=
Huber(\hat{\Delta m}_i,\Delta m_i)
\]

- [ ] 作为主 loss 候选。

## 7.2 Helpful / Harmful Classification

定义：

```text
Helpful: Δm_i > δ
Harmful: Δm_i < -δ
Neutral: 其余
```

`δ` 只能在 dev set 确定。

- [ ] 二分类 helpful vs harmful。
- [ ] 或三分类 helpful / neutral / harmful。
- [ ] 报告 AUROC / AUPRC / calibration。

## 7.3 Pairwise Ranking

同一个病例内：

如果：

\[
\Delta m_i > \Delta m_j
\]

要求：

\[
c_i > c_j
\]

可用：

\[
L_{rank}
=
\log(1+\exp(-(c_i-c_j)))
\]

第一版最多：

\[
L=L_{reg}+\lambda L_{rank}
\]

不要一次堆很多 loss。

---

# 8. Random / Shuffled 只作为辅助负对照

Random / Shuffled 的作用不是定义 ground truth，而是验证：

> scorer 是否真的利用了病例与 prior 的匹配关系。

TODO：

- [ ] Correct vs Random consistency score。
- [ ] Correct vs Shuffled consistency score。
- [ ] 同一个 prior 配到错误病例后，score 是否下降。
- [ ] 同一个病例换成无关 prior 后，score 是否下降。

如果 scorer 连这个都分不开：

> Prior–Visual Consistency 假设暂时不成立。

---

# 9. Module 2 的最小干预：Gate，而不是新 Residual

只有 scorer 在 held-out dev 上有效后，才进入这一阶段。

固定 PBridge-B 的 PPR 预测分数权重：

\[
\alpha_i=\operatorname{softmax}(s_i/0.5)
\]

加入 visual-consistency gate：

\[
g_i = \sigma(c_i)
\]

然后：

\[
\tilde{\alpha}_i
\propto
\alpha_i g_i
\]

重新归一化：

\[
\beta_i
=
\frac{\alpha_i g_i}
{\sum_j \alpha_j g_j+\epsilon}
\]

最终仍然使用原来的三个 prefix：

\[
P_i' = \beta_i P_i
\]

核心原则：

> **Module 2 不产生新知识，只决定现有 prior 应该被信多少。**

---

# 10. 必须保留“完全关闭 prior”的能力

只在 3 条知识之间重新归一化有一个问题：

```text
如果 3 条 prior 都不适用，
模型仍会被迫选择其中一条。
```

因此增加 case-level gate：

\[
g_{case}
=
\sigma(\max_i c_i)
\]

或：

\[
g_{case}
=
\sigma(f(c_1,c_2,c_3))
\]

最终：

\[
P_i'
=
g_{case}\cdot\beta_iP_i
\]

要求：

```text
所有 prior 都不可信 → g_case ≈ 0
存在可靠 prior → g_case ≈ 1
```

---

# 11. 必须做的 Baselines

### B0 — No Prior

```text
原始 Lingshu
```

### B1 — Module 1

```text
固定 PBridge-B
```

### B2 — Utility-only Gate

只使用：

```text
Utility / PPR score
```

不看图像。

用途：

> 证明 Module 2 的收益不是简单重新利用 Utility。

### B3 — Visual Consistency Gate

```text
image + prior
```

核心方法。

### B4 — Image + Question + Prior

如果 B3 不够，再测试：

```text
image + question + prior
```

如果 B4 有效而 B3 无效：

> 机制应称为 case-level consistency，而不能只称 visual consistency。

### B5 — Oracle Gate

直接用真实：

\[
\Delta m_i
\]

决定是否启用 prior。

**只作为 upper bound，不可作为正式方法。**

---

# 12. 先做 Oracle，再决定值不值得继续

在训练 consistency scorer 前，先做：

```text
Oracle helpfulness gate
```

例如：

```text
如果 Δm_top3 > 0：
    使用 Module 1 prior
否则：
    使用 No-Prior
```

统计：

- [ ] accuracy upper bound
- [ ] fix 数
- [ ] harm 数
- [ ] net gain

如果 Oracle Gate 相比 Module 1 几乎没有提升：

> “什么时候信 prior”不是值得单独做 Module 2 的问题，直接 No-Go。

如果 Oracle 有明显提升：

> 说明这个问题真实存在，值得学习 consistency predictor。

---

# 13. 最关键的评价指标

不能只看 Accuracy。

### Accuracy

\[
Acc
\]

### Fix Rate

```text
No-Prior wrong → Method correct
```

### Harm Rate

```text
No-Prior correct → Method wrong
```

### Net Benefit

```text
Fix - Harm
```

Module 2 的目标是：

\[
\boxed{
\text{保留 Module 1 的纠错，同时减少 prior-induced harm}
}
\]

---

# 14. Scorer 本身必须单独评估

对 held-out dev/test：

- [ ] Spearman(`predicted consistency`, `Δm_i`)
- [ ] Helpful/Harmful AUROC
- [ ] AUPRC
- [ ] calibration curve
- [ ] Brier score
- [ ] Correct vs Random score
- [ ] Correct vs Shuffled score

如果 scorer 连 prior helpfulness 都预测不了：

> 即使最终 VQA 偶然涨点，也不能宣称 Prior Evidence Alignment 被验证。

---

# 15. Ablation

至少做：

```text
Module 1
Module 1 + Utility-only gate
Module 1 + visual consistency
Module 1 + visual consistency + case gate
Module 1 + image-question-prior consistency
Oracle gate
```

另外：

- [ ] 去掉 image feature。
- [ ] 去掉 prior feature。
- [ ] random prior。
- [ ] shuffled prior。
- [ ] 固定 gate=1。
- [ ] 固定 gate=0。

---

# 16. Go / No-Go 标准

## Go

只有同时满足：

1. Oracle Gate 明显优于固定 PBridge-B；
2. consistency scorer 在独立 dev 上能预测 `Δm_i`；
3. visual scorer 优于 Utility-only baseline；
4. gated PBridge 明显降低 Harm Rate；
5. 同时基本保留 Module 1 的 Fix Rate；
6. Random / Shuffled prior 得分明显更低；
7. 最终收益不是靠大量新增参数得到。

此时 Module 2 可以定义为：

```text
Prior Evidence Alignment
```

或：

```text
Pediatric Prior–Visual Consistency Gating
```

## No-Go

如果出现以下任一情况，应停止：

- Oracle Gate 几乎没有额外空间；
- scorer 无法预测 helpful / harmful；
- visual consistency 不优于 Utility-only；
- gate 降低 harm 的同时也大量丢掉 fix；
- Random/Shuffled 和 Correct scorer 无明显区别；
- 只有复杂大模型 scorer 才能产生很小收益。

---

# 17. 推荐实验顺序

```text
Step 1
训练集预计算单 prior / Top-3 的 Δmargin
        ↓
Step 2
先算 Oracle Gate upper bound
        ↓
Step 3
提取 frozen visual feature + soft prior feature
        ↓
Step 4
训练最简单 cosine consistency scorer
        ↓
Step 5
验证 scorer 能否预测 helpful / harmful
        ↓
Step 6
加入 per-prior gate + case-level gate
        ↓
Step 7
比较 Module 1 / Utility-only / Visual Gate
        ↓
Step 8
再决定是否需要更复杂 scorer
```

---

# 18. 第一阶段最小可行实验（MVP）

优先只完成：

- [ ] 在训练集上构造 `Δm_top3`。
- [ ] 计算 Oracle Gate。
- [ ] 判断理论上能减少多少 `对→错`。
- [ ] 提取 frozen visual pooled feature。
- [ ] 复用现有 3 个 soft prior token。
- [ ] 训练一个 128/256 维 cosine scorer。
- [ ] 在独立 dev 上预测 prior helpfulness。
- [ ] 与 Utility-only baseline 比较。
- [ ] 如果 scorer 有效，再接 gate 做 VQA。

**Oracle 无明显空间或 scorer 无法泛化时，直接停止，不继续堆复杂模块。**
