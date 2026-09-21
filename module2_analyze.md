# Module 2 当前状态

Module 2 的唯一基线已固定为 PBridge-B：

- Lingshu-7B 与 GME 冻结；
- PPR-MoE checkpoint 固定；
- PPR 预测分数同时用于 Top-3 排序和 prefix 加权；
- Module 1 源码、权重、检索资产和输入协议均不再由 Module 2 修改。

基线身份及 SHA-256 保存在 `module2_baseline.json`。所有 Module 2 入口在
加载时校验该文件；若 Module 1 源码或 checkpoint 变化，实验直接停止。

此前 Module 2 的 Oracle、attention 和 patching 结果基于旧 Module 1
prefix 加权逻辑，不能作为 PBridge-B 的结论，相关中间结果已清理。
后续 Module 2 验证必须在固定 B 上重新生成结果。
