# Module 2 验证报告

## 验证范围

使用当前唯一正式方案 Lingshu-PBridge，按 `module2_todo.md` 对
PediatricsMQA 固定测试集 413 题重新执行图像预处理检查，并复测 Module 1
指标。

## 验证过程

1. 读取 Lingshu checkpoint 的 Processor 与本地配置文件。
2. 抽样 20 张图，对比原始尺寸、PIL 加载尺寸和送入 Processor 前的尺寸。
3. 对 413 张测试图逐张统计 `image_grid_thw`、视觉 token 数和输入序列中的
   图像 token 数。
4. 对同一张 500×500 控制图比较默认预算、checkpoint 显式预算、TODO 预算和
   强制单 token 预算。
5. 用最终代码重新评测 Lingshu-7B No-RAG 与 Lingshu-PBridge。

## Processor 配置

| 项目 | 数值 |
|---|---|
| Processor | Qwen2.5-VL Processor |
| 运行时 Transformers | 4.57.6 |
| Checkpoint Transformers 元数据 | 4.51.3 |
| `min_pixels` | 3,136 |
| `max_pixels` | 12,845,056 |
| Patch size | 14 |
| Merge size | 2 |

Tokenizer 中没有第二套冲突的像素预算。

## 图像与视觉 token

| 统计项 | 结果 |
|---|---:|
| 原始宽度 | 306–600，中位数 500 |
| 原始高度 | 133–664，中位数 450 |
| Processor 前尺寸抽查 | 20/20 未改变 |
| 完整测试集检查 | 413 张 |
| 最终视觉 token 数 | 90–432 |
| 保留多个视觉 token | 413/413 |

完整测试集共有 21 种视觉网格。最常见网格为 `[1,32,42]`，对应 336 个最终
视觉 token，共 122 题。

## 像素预算对照

500×500 控制图结果：

| 设置 | 网格 | 最终视觉 token |
|---|---:|---:|
| 当前 PBridge 默认设置 | `[1,36,36]` | 324 |
| Checkpoint 显式预算 | `[1,36,36]` | 324 |
| TODO 参考预算 | `[1,36,36]` | 324 |
| 强制 `28×28` 诊断预算 | `[1,2,2]` | 1 |

只有人为强制最低预算时才会复现单视觉 token。

## Module 1 稳定性复测

| 方法 | 正确数 | Accuracy |
|---|---:|---:|
| Lingshu-7B No-RAG | 253/413 | 61.26% |
| Lingshu-PBridge | 272/413 | 65.86% |

PBridge 比相同 Lingshu backbone 的 No-RAG 多答对 19 题，提高 4.60 个百分点；
与切换后已有记录完全一致。

## 结论

当前 Lingshu-PBridge 不存在单视觉 token 缺陷。数据加载没有提前缩图，
checkpoint 与 Processor 配置一致，413/413 张测试图均保留空间视觉序列。
因此不需要因图像分辨率问题调整配置或重新训练 Module 1。

完整中间结果位于 `module2_probe/outputs/image_preprocess_check/`。
