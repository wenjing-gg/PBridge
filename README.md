# PBridge

PBridge is a pediatric visual question answering pipeline built on a frozen
MedMO-8B reader and a prior-aware passage reranking module.

## Structure

- `module1_ppr/`: retrieval assets, counterfactual utility construction,
  PPR-MoE training, and evaluation.
- `data/pediatric_knowledge/`: knowledge metadata. Knowledge images are local
  assets and are not committed.
- `models/gme/`: local GME checkpoint location. Model weights are not committed.
- `docs/ppr_method.md`: algorithm description.
- `docs/results.md`: evaluation results.
- `prepare_knowledge.py`: builds the pediatric knowledge bank.
- `prepare_split.py`: creates the fixed PediatricsMQA 80:20 split.

## Required Local Models

Place the frozen GME checkpoint under:

```text
models/gme/
```

MedMO-8B is loaded from:

```text
/data/cyf/codes/YYY/VQA/MedMO/checkpoint/MedMO-8B-Next
```

## Pipeline

Run commands from the repository root.

```bash
python module1_ppr/gme_assets.py encode-knowledge --device cuda:0
python module1_ppr/gme_assets.py build-candidates --split train --device cuda:0
python module1_ppr/gme_assets.py build-candidates --split test --device cuda:0
python module1_ppr/reader_assets.py --split train --device cuda:0
python module1_ppr/reader_assets.py --split test --device cuda:0
python module1_ppr/run.py train --device cuda:0
python module1_ppr/run.py evaluate --device cuda:0
```

The retained PediatricsMQA result is `265/413` (`64.16%`). Both training stages
of the retained checkpoint converged.
