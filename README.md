# PBridge

PBridge is a pediatric visual question-answering pipeline built on a frozen
Lingshu-7B reader and a prior-aware passage reranking MoE.

## Current configuration

- Backbone: `/data/cyf/codes/YYY/VQA/Lingshu-7B/checkpoint`
- Dataset: fixed PediatricsMQA 8:2 split
- Train/test questions: 1,654 / 413
- Reader input: native multiple visual tokens
- Frozen modules: Lingshu-7B and GME
- Trainable module: PPR-MoE and three projected prior-prefix tokens

The converged checkpoint reaches:

| Method | Correct | Accuracy |
|---|---:|---:|
| Lingshu-7B No-RAG | 253/413 | 61.26% |
| **PBridge** | **272/413** | **65.86%** |

## Structure

- `module1_ppr/`: retrieval assets, Utility construction, training and
  evaluation.
- `module2_probe/`: Lingshu image-preprocessing validation.
- `docs/ppr_method.md`: algorithm.
- `docs/results.md`: results.
- `module2_analyze.md`: Module 2 validation report.

## Run

Commands are executed from the repository root:

```bash
python module1_ppr/gme_assets.py encode-knowledge --device cuda:0
python module1_ppr/gme_assets.py build-candidates --split train --device cuda:0
python module1_ppr/gme_assets.py build-candidates --split test --device cuda:0
python module1_ppr/reader_assets.py --split train --device cuda:0
python module1_ppr/reader_assets.py --split test --device cuda:0
python module1_ppr/run.py train --device cuda:0
python module1_ppr/run.py evaluate --device cuda:0
python module1_ppr/run.py baseline --device cuda:0
```

Validate the image preprocessing used by the active backbone:

```bash
python -m module2_probe.check_image_preprocess
```

Local Lingshu and GME model weights are not committed.
