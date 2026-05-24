# DFFNDDS

A concise guide for dataset layout and training in this project.

## Data Layout (SynergyX-style)

Project root:

```text
DFFNDDS/
└── MyDataset/
    ├── new-cell/
    ├── LLM_data/
    ├── long-tail/
    ├── merged_data.csv
    ├── cell_map.csv
    ├── context_set_m.json
    └── drug_set.json
```
The data can be downloaded at [HuggingFace](https://huggingface.co/datasets/Matthewmtf/LaCo/tree/main/DFFNDDS).

## Required Files

Core files used by current training flow:
- `MyDataset/merged_data.csv`
- `MyDataset/cell_map.csv`
- `MyDataset/context_set_m.json`
- `MyDataset/drug_set.json`
- `MyDataset/new-cell/db-train.csv`
- `MyDataset/new-cell/db-valid.csv`
- `MyDataset/new-cell/db-test.csv`
- `MyDataset/LLM_data/drug_map.csv`
- `MyDataset/LLM_data/cell_map.csv`
- `MyDataset/LLM_data/drug_random_embedding.npy`
- `MyDataset/LLM_data/cell_random_embedding.npy`

## Notes on LLM Embedding Names

SynergyX commonly uses:
- `5_l_drug_embeddings.npy`
- `5_l_cell_embeddings.npy`

Current DFFNDDS code reads:
- `drug_random_embedding.npy`
- `cell_random_embedding.npy`

If you want SynergyX naming, update file names in `dataset.py` accordingly.

## Optional Split Directories

`MyDataset/new-cell/` (recommended):
- `v1-train.csv`, `v1-valid.csv`, `v1-test.csv`
- `db-train.csv`, `db-valid.csv`, `db-test.csv`

`MyDataset/long-tail/` (for long-tail experiments):
- `drugcomb_train_top_cells.csv`
- `drugcomb_test_tail_cells.csv`
- `drugcombdb_train_top_cells.csv`
- `drugcombdb_test_tail_cells.csv`

## Run

From `feature_based/DFFNDDS`:

```bash
python main-split.py
```

## Entry Files

- `main-split.py`
- `dataset.py`
- `model_h.py`
- `head.py`
- `contrastive.py`
- `data_split_standard.py`
- `output/simcsesqrt-model/` (full directory)
