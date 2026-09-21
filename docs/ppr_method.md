# PBridge: Prior-aware Passage Reranking

## 1. Candidate representation

GME 先检索 20 条儿科知识候选，并把当前图像—问题与每条候选编码到统一的
1,536 维空间：

\[
\mathbf q,\mathbf z_i\in\mathbb R^{1536},\qquad i=1,\ldots,20.
\]

## 2. Counterfactual Utility

对候选 \(i\)，PBridge 比较“无知识”和“只加入该知识”时冻结 Lingshu 的答案
边际变化。训练目标仅使用训练集标签；推理时以 Lingshu 的 No-RAG 预测作为
参照，不读取测试集答案。

## 3. Prior-aware MoE reranking

六个双线性专家分别从不同参数子空间评价候选：

\[
s_i^{(e)}=\mathbf z_i^\top\mathbf W_e\mathbf q.
\]

路由器同时接收问题向量、候选知识向量和 Utility：

\[
\mathbf r_i=\operatorname{Router}([\mathbf q;\mathbf z_i;u_i]).
\]

每条候选确定性激活分数最高的两个专家，再按归一化门控权重融合：

\[
s_i=\sum_{e\in\operatorname{Top2}(\mathbf r_i)}
g_{i,e}s_i^{(e)}.
\]

最终按 \(s_i\) 排序，保留 Top-3 知识。

## 4. Prior-prefix construction

Top-3 候选使用 PPR-MoE 的预测分数分配权重，并加入知识位置向量：

\[
\alpha_i=\operatorname{softmax}(s_i/0.5),\qquad
\widetilde{\mathbf z}_i=\alpha_i\mathbf z_i+\mathbf e_i.
\]

因此同一组预测分数同时决定 Top-3 排序和三条知识的相对强度；
gold-conditioned Utility 只监督训练损失，不进入 prefix 构造。

随后用可学习线性层把三个向量投影到 Lingshu 的 3,584 维词嵌入空间：

\[
\mathbf p_i=\mathbf W_p\widetilde{\mathbf z}_i
\in\mathbb R^{3584}.
\]

这三个向量作为先验 prefix 放在原生图像—问题序列之前，同时保留 Qwen2.5-VL
的多模态旋转位置编码；Lingshu 全程冻结。

## 5. Optimization

Stage 1 只训练路由器和双线性专家：

\[
\mathcal L_1=\mathcal L_{\mathrm{rank}}
+0.01\mathcal L_{\mathrm{balance}}.
\]

排序损失要求 scorer 的候选分差复现 Utility 的相对大小：

\[
\mathcal L_{\mathrm{rank}}=
\underset{u_i>u_j}{\operatorname{mean}}
\operatorname{softplus}
\left(-[(s_i-s_j)-\tanh(u_i-u_j)]\right).
\]

Stage 2 在继续约束排序的同时，用答案交叉熵训练完整 PPR 模块；Lingshu 仍然
冻结：

\[
\mathcal L_2=\mathcal L_{\mathrm{CE}}
+\mathcal L_{\mathrm{rank}}
+0.01\mathcal L_{\mathrm{balance}}.
\]

## 6. Current model

| Component | Status |
|---|---|
| GME retriever | Frozen |
| Lingshu-7B | Frozen |
| Six bilinear experts | Trainable |
| Top-2 router | Trainable |
| Three position embeddings | Trainable |
| \(1536\rightarrow3584\) projection | Trainable |

PPR 模块共有 22,819,334 个可训练参数。Stage 1 在第 1,950 轮满足收敛条件，
Stage 2 在第 46 轮满足收敛条件；测试使用各阶段最低损失 checkpoint。
