# Module 1 Accuracy Comparison

## Same-Split Comparison

所有模型均按 PBridge 的固定留出题目统计：PediatricsMQA 为 413 题，
VinDr-PCXR 为 129 题。

| Model | PediatricsMQA | Accuracy | VinDr-PCXR | Accuracy |
|---|---:|---:|---:|---:|
| Lingshu-7B | 251/413 | 60.77% | 45/129 | 34.88% |
| HuatuoGPT-Vision-7B | 250/413 | 60.53% | 20/129 | 15.50% |
| Qwen3-VL-8B | 240/413 | 58.11% | 54/129 | 41.86% |
| InternVL3-8B | 243/413 | 58.84% | 36/129 | 27.91% |
| MedMO-4B-Next | 231/413 | 55.93% | 48/129 | 37.21% |
| MedMO-8B-Next | 212/413 | 51.33% | 60/129 | 46.51% |
| MedVLM-R1-2B | 208/413 | 50.36% | 54/129 | 41.86% |
| BiomedGPT | 133/413 | 32.20% | 35/129 | 27.13% |
| LLaVA-Med-7B | 107/413 | 25.91% | 24/129 | 18.60% |
| Med-R1-2B | 93/413 | 22.52% | 31/129 | 24.03% |
| **PBridge** | **265/413** | **64.16%** | **94/129** | **72.87%** |

## Full-Test Reference

普通模型原始全量测试结果如下。PBridge 必须在各数据集的 80% 训练集上训练，
因此只报告独立 20% 留出集，不存在对应的全量无泄露结果。

| Model | PediatricsMQA | Accuracy | VinDr-PCXR | Accuracy |
|---|---:|---:|---:|---:|
| Lingshu-7B | 1292/2067 | 62.51% | 234/638 | 36.68% |
| HuatuoGPT-Vision-7B | 1265/2067 | 61.20% | 130/638 | 20.38% |
| Qwen3-VL-8B | 1234/2067 | 59.70% | 282/638 | 44.20% |
| InternVL3-8B | 1224/2067 | 59.22% | 200/638 | 31.35% |
| MedMO-4B-Next | 1163/2067 | 56.27% | 243/638 | 38.09% |
| MedMO-8B-Next | 1053/2067 | 50.94% | 332/638 | 52.04% |
| MedVLM-R1-2B | 1030/2067 | 49.83% | 321/638 | 50.31% |
| BiomedGPT | 637/2067 | 30.82% | 168/638 | 26.33% |
| LLaVA-Med-7B | 469/2067 | 22.69% | 128/638 | 20.06% |
| Med-R1-2B | 451/2067 | 21.82% | 181/638 | 28.37% |
| **PBridge** | **265/413** | **64.16%** | **94/129** | **72.87%** |

## Training Status

学习 prefix 现在保留 Qwen-VL 原生图像 MRoPE，并将原序列位置整体后移三个
prefix token。

- PediatricsMQA：Stage 1 在 1230 epochs 收敛，Stage 2 在 29 epochs 收敛。
- VinDr-PCXR：Stage 1 在 2063 epochs 收敛，Stage 2 在 21 epochs 收敛。
