# Results

All current results use the fixed PediatricsMQA 8:2 split and the same A/B/C/D
next-token-logit protocol.

| Method | Test correct | Accuracy | NLL | Brier |
|---|---:|---:|---:|---:|
| Lingshu-7B No-RAG | 253/413 | 61.26% | 1.1034 | 0.5543 |
| **PBridge** | **272/413** | **65.86%** | 2.9933 | 0.6453 |

PBridge improves accuracy by 19 questions, or 4.60 percentage points. NLL and
Brier are worse because the learned prefix makes predictions more confident,
including the remaining errors; the accuracy result should therefore not be
described as a calibration improvement.

Training convergence:

- Stage 1: converged at epoch 1,950; saved epoch 1,943.
- Stage 2: converged at epoch 46; saved epoch 43.
- Lowest Stage 2 loss: 0.20595.

Historical MedMO results are not part of the current code path:

| Historical configuration | Accuracy |
|---|---:|
| MedMO-8B No-RAG, single visual token | 52.06% |
| MedMO-8B PBridge, single visual token | 64.16% |
