# Results

All current results use the fixed PediatricsMQA 8:2 split and the same A/B/C/D
next-token-logit protocol.

| Method | Test correct | Accuracy | NLL | Brier |
|---|---:|---:|---:|---:|
| Lingshu-7B No-RAG | 253/413 | 61.26% | 1.1034 | 0.5543 |
| **PBridge** | **285/413** | **69.01%** | **2.8160** | **0.5836** |

PBridge uses PPR predicted scores for both Top-3 selection and prefix weighting.
It improves accuracy by 32 questions, or 7.75 percentage points, over No-RAG.

Training convergence:

- Stage 1: converged at epoch 2,156; saved epoch 2,115.
- Stage 2: converged at epoch 31; saved epoch 28.
- Lowest Stage 2 loss: 0.20606.

Historical MedMO results are not part of the current code path:

| Historical configuration | Accuracy |
|---|---:|
| MedMO-8B No-RAG, single visual token | 52.06% |
| MedMO-8B PBridge, single visual token | 64.16% |
