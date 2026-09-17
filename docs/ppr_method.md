# PBridge Module 1: Prior-aware Passage Reranking

## 1. 总体架构

```
GME 检索 Top-20 候选知识
    │
    ├── 纯文本 WHO 知识 ──→ GME 编码 ──→ 1536 维向量
    │
    └── Pediatric Imaging 知识 ──→ GME 编码 ──→ 1536 维向量
                                        │
                          Top-20 每条知识 → 1 个 1536 维向量
                                        │
                          ┌─────────────▼─────────────┐
                          │  MoE 多视角打分            │
                          │  每个专家输出 20 维评分向量 │
                          │  Router：q, z_i, u_i       │
                          └─────────────┬─────────────┘
                                        │
                              每条知识 → 最终分数 s_i
                                        │
                          按 s_i 排序，取 Top-3
                                        │
                          ┌─────────────▼─────────────┐
                          │  Utility 缩放 + 拼接       │
                          │  线性投影 → MedMO 嵌入空间  │
                          └─────────────┬─────────────┘
                                        │
                                   MedMO 推理
```

GME 文本塔、GME 视觉塔、MedMO 主干全部冻结。可训练模块只有三个：MoE 的 Router、MoE 的专家双线性矩阵、连接知识向量到 MedMO 的线性投影。


## 2. 异构知识编码（统一 GME）

### 2.1 纯文本知识（WHO）

GME 编码器（冻结）：

\[
\mathbf{z}_i = \text{GME}(k_{\text{who}}) \in \mathbb{R}^{1536}
\]

### 2.2 图文知识（Pediatric Imaging）

GME 编码器（冻结）：

\[
\mathbf{z}_i = \text{GME}([k_{\text{ped}}; I_{\text{ref}}]) \in \mathbb{R}^{1536}
\]

GME 内部完成文本与视觉的跨模态融合与池化，输出统一的 1536 维向量。纯文本知识和图文知识在 GME 输出空间中天然同质，无需额外对齐。

### 2.3 Top-20 堆叠

\[
\mathbf{Z} = [\mathbf{z}_1; \dots; \mathbf{z}_{20}] \in \mathbb{R}^{20 \times 1536}
\]


## 3. MoE 多视角打分

### 3.1 专家网络

专家数量 \(N_e = 6\)。每个专家是一个独立的双线性打分函数，从自己的视角对 Top-20 所有知识同时打分：

\[
\mathbf{s}^{(e)} = \mathbf{Z} \, \mathbf{W}_e \, \mathbf{q} \in \mathbb{R}^{20}
\]

- \(\mathbf{W}_e \in \mathbb{R}^{1536 \times 1536}\)：专家 \(e\) 的可学习打分矩阵
- \(\mathbf{q} \in \mathbb{R}^{1536}\)：GME query 向量
- \(\mathbf{s}^{(e)}\) 的第 \(i\) 个分量 \(s_i^{(e)}\) 是专家 \(e\) 给知识 \(i\) 的分数

每个专家输出 20 维向量，对所有候选知识同时打分，形成相对比较。不同专家在训练中自然分化出不同视角：解剖匹配度、病理相关性、视觉征象一致性、数值信息相关性、治疗预后关联、鉴别诊断支持度。这些视角不需要预先定义，通过 Router 的路由分化和 pairwise ranking loss 的监督自然涌现。

### 3.2 Router：知识级路由

\[
\mathbf{r}_i = \text{Router}\left([\mathbf{q}; \mathbf{z}_i; u_i]\right) \in \mathbb{R}^{N_e}
\]

- \(\mathbf{q}\)：GME query 向量
- \(\mathbf{z}_i\)：知识 \(i\) 的 1536 维向量
- \(u_i\)：Counterfactual Utility

Router 是一个 2 层 MLP，输入维度 \(1536 + 1536 + 1\)，隐藏维度 \(1024\)，输出维度 \(N_e\)。

取 Top-2 专家：

\[
\text{TopK}(\mathbf{r}_i) = \{e_{i,1}, e_{i,2}\}
\]

归一化路由权重：

\[
g_{i,e} = \frac{\exp(r_{i,e})}{\sum_{e' \in \text{TopK}(\mathbf{r}_i)} \exp(r_{i,e'})}
\]

### 3.3 多视角打分融合

知识 \(i\) 的最终分数为被选中专家对知识 \(i\) 的打分加权组合：

\[
s_i = \sum_{e \in \text{TopK}(\mathbf{r}_i)} g_{i,e} \cdot s_i^{(e)}
\]

其中 \(s_i^{(e)}\) 是专家 \(e\) 评分向量 \(\mathbf{s}^{(e)}\) 的第 \(i\) 个分量。


## 4. 带效用边际的 Pairwise Ranking Loss

### 4.1 损失形式

\[
\mathcal{L}_{\text{rank}} = \text{mean}_{u_i > u_j} \text{softplus}\left(-\left[(s_i - s_j) - (u_i - u_j)\right]\right)
\]

### 4.2 内在结构

效用 \(u\) 定义在 log-odds 空间，scorer 分数 \(s\) 也在 log-odds 空间。理想打分器应满足：

\[
s_i - s_j = u_i - u_j
\]

即 scorer 分数差应复现效用差。改进后的损失把这个理想作为 softplus 的零点：

- 当 \(s_i - s_j > u_i - u_j\)：损失趋近 0，梯度小。
- 当 \(s_i - s_j < u_i - u_j\)：损失增大，梯度推动分数差扩大。
- 当 \(s_i - s_j = u_i - u_j\)：损失为 \(\log 2\)，梯度为 0.5，处于“刚好满足边际”的临界点。

效用从“配对选择器”升级为“目标边际”。softplus 的梯度为 sigmoid，始终有界，不会爆炸。当分数差距远超效用边际时梯度自然消失。

### 4.3 效用长尾处理

若效用差距分布长尾，用 tanh 温和压平：

\[
\mathcal{L}_{\text{rank}} = \text{mean}_{u_i > u_j} \text{softplus}\left(-\left[(s_i - s_j) - \tanh(u_i - u_j)\right]\right)
\]

tanh 把效用差距压到 \((-1, 1)\)，消除极端值主导梯度，同时保留单调性和零点。


## 5. 融合与 MedMO 输入

按 \(s_i\) 排序取 Top-3 知识，得到 \(\mathbf{z}_1, \mathbf{z}_2, \mathbf{z}_3 \in \mathbb{R}^{1536}\)。

### 5.1 Utility 缩放

\[
\alpha_i = \frac{\exp(u_i / \tau)}{\sum_{j=1}^{3} \exp(u_j / \tau)}, \quad \tau = 0.5
\]

\[
\tilde{\mathbf{z}}_i = \alpha_i \cdot \mathbf{z}_i \in \mathbb{R}^{1536}
\]

标量对整条知识向量统一缩放。高 Utility 知识的向量数值更大，MedMO 的注意力自然更容易被吸引。\(\sum_i \alpha_i = 1\)，不改变整体数值范围，只重新分配三条知识的相对强度。

### 5.2 拼接

\[
\mathbf{Z}_{\text{cat}} = [\tilde{\mathbf{z}}_1; \tilde{\mathbf{z}}_2; \tilde{\mathbf{z}}_3] \in \mathbb{R}^{3 \times 1536}
\]

沿 token 维度拼接。加入知识位置 embedding：

\[
\mathbf{Z}_{\text{cat}} \leftarrow \mathbf{Z}_{\text{cat}} + \mathbf{e}_{\text{pos}}, \quad \mathbf{e}_{\text{pos}} \in \mathbb{R}^{3 \times 1536}
\]

### 5.3 线性投影到 MedMO 嵌入空间

\[
\mathbf{Z}_{\text{medmo}} = \mathbf{Z}_{\text{cat}} \mathbf{W}_p \in \mathbb{R}^{3 \times d_{\text{medmo}}}, \quad \mathbf{W}_p \in \mathbb{R}^{1536 \times d_{\text{medmo}}}
\]

单层线性投影，参数量约 6.3M（\(d_{\text{medmo}}=4096\)）。由 \(\mathcal{L}_{\text{ce}}\) 直接监督，梯度路径短，训练稳定。

### 5.4 输入 MedMO

\[
[\mathbf{Z}_{\text{medmo}} \in \mathbb{R}^{3 \times d_{\text{medmo}}};\ \mathbf{E}_{\text{image}};\ \mathbf{E}_{\text{question}};\ \mathbf{E}_{\text{options}}]
\]

3 个知识前缀 token 在最前面。MedMO 主干冻结，输出四个选项的概率。


## 6. 损失函数

\[
\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{ce}} + \mathcal{L}_{\text{rank}} + \lambda_b \mathcal{L}_{\text{balance}}
\]

- \(\mathcal{L}_{\text{ce}}\)：MedMO 选择正确答案的交叉熵，监督 \(\mathbf{W}_p\) 和知识位置 embedding
- \(\mathcal{L}_{\text{rank}}\)：带效用边际的 pairwise ranking loss，监督 MoE 专家和 Router

\[
\mathcal{L}_{\text{rank}} = \text{mean}_{u_i > u_j} \text{softplus}\left(-\left[(s_i - s_j) - (u_i - u_j)\right]\right)
\]

- \(\mathcal{L}_{\text{balance}}\)：MoE 负载均衡

\[
\mathcal{L}_{\text{balance}} = N_e \sum_{e=1}^{N_e} f_e \cdot P_e, \quad \lambda_b = 0.01
\]

\(f_e\)：专家 \(e\) 被选中的频率；\(P_e\)：专家 \(e\) 的平均路由概率。


## 7. 可训练参数

| 模块 | 参数量 |
|---|---|
| 知识位置 embedding \(\mathbf{e}_{\text{pos}}\) | 约 18K |
| Router MLP | 约 4.7M |
| 专家双线性矩阵（6 个，\(1536 \times 1536\)） | 约 14.2M |
| 线性投影 \(\mathbf{W}_p\) | 约 6.3M |
| GME 文本塔 | 冻结 |
| GME 视觉塔 | 冻结 |
| MedMO 主干 | 冻结 |

总可训练参数约 25M。所有可训练模块都有直接的监督信号：MoE 由 \(\mathcal{L}_{\text{rank}}\) 监督，\(\mathbf{W}_p\) 由 \(\mathcal{L}_{\text{ce}}\) 监督。


## 8. 阶段化训练

### 阶段 1：MoE 多视角打分训练

冻结 GME、MedMO、\(\mathbf{W}_p\)。只训练 Router + 专家双线性矩阵。损失为 \(\mathcal{L}_{\text{rank}} + \lambda_b \mathcal{L}_{\text{balance}}\)。知识表示由 GME 直接给出，MoE 学习从多视角对 Top-20 知识打分并排序。

### 阶段 2：投影与端到端微调

冻结 GME、MedMO。解冻 \(\mathbf{W}_p\)、知识位置 embedding、MoE。损失为 \(\mathcal{L}_{\text{ce}} + \mathcal{L}_{\text{rank}} + \lambda_b \mathcal{L}_{\text{balance}}\)。\(\mathbf{W}_p\) 学习将知识向量投影到 MedMO 嵌入空间，MoE 的打分与 MedMO 的选择协同优化。


## 9. 推理流程

1. GME 根据患者图像和问题检索 Top-20 候选知识。
2. 对每条候选知识，GME 编码为 1536 维向量。纯文本知识和图文知识走同一编码器，输出同质。
3. Top-20 堆叠为 \(\mathbf{Z} \in \mathbb{R}^{20 \times 1536}\)。
4. Router 根据问题、知识向量和 Utility 决定每条知识的 Top-2 专家。
5. 6 个专家分别对 Top-20 打分，输出 6 个 20 维评分向量。
6. 按 Router 权重加权融合，得到每条知识的最终分数 \(s_i\)。
7. 按 \(s_i\) 排序取 Top-3。
8. 用 Utility softmax 权重缩放 Top-3 的 1536 维向量。
9. 拼接为 3 个 token，加入知识位置 embedding。
10. 通过单层线性投影到 MedMO 嵌入维度。
11. 与患者图像、问题、选项一起输入 MedMO，选择概率最高的选项。


## 10. 差异化的来源

模型对不同重要程度先验知识的差异化对待，由三个机制共同实现：

| 机制 | 作用层面 | 监督信号 |
|---|---|---|
| Utility 边际 Ranking Loss | MoE 专家打分 | Utility |
| Router 将 Utility 作为输入 | 专家选择 | Utility |
| Utility softmax 缩放 | MedMO 输入数值 | Utility |

三个机制都直接或间接由 Utility 驱动。Utility 来自 Counterfactual Utility Learning，是从 VQA 多选题的选项概率中计算出的、无需额外标注的监督信号。整个框架不引入任何需要额外标注的模块，所有可训练参数都有明确的监督来源。
