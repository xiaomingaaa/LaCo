# DeepDDs 3

DeepDDs 3 is used for drug-combination synergy prediction.
This README provides a complete, runnable workflow with the current normalized data layout.

## 1. Project Entry

Main training script:
- `trianing_GCN.py`

Related core scripts:
- `utils_test.py` (dataset class + feature assembly)
- `creat_data_DC.py` (builds processed `.pt` files)
- `models/` (model definitions)

## 2. Environment Setup

Recommended Python version: 3.8+

Install dependencies (example):

```bash
cd "/Users/hyq/LACO/DeepDDs 3"
python -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install torch torchvision torchaudio
pip install torch-geometric
pip install numpy pandas scikit-learn scipy rdkit networkx
```

If you already have a working environment, use your existing one.

## 3. Data Layout (Normalized)

All data is under `./data/`. You can download the data at [HuggingFace](https://huggingface.co/datasets/Matthewmtf/LaCo/tree/main/DeepDDS). 

```text
data/
├── LLM_DATA/
├── drugcomb/
│   └── inductive/
├── drugcombdb/
│   └── inductive/
├── independent_set/
├── leave_cell/
├── leave_comb/
├── leave_drug/
└── CCLE_RNAseq_rsem_transcripts_tpm_20180929/
```

### 3.1 LLM Representation Files

Put LLM embeddings and mapping files in:
- `data/LLM_DATA/`

Required files:
- `5_l_drug_embeddings.npy`
- `5_l_cell_embeddings.npy`
- `drug_map_with_smiles.csv`
- `cell_map.csv`

### 3.2 DrugComb Task Files

Folder:
- `data/drugcomb/`

Typical files used by pipeline:
- `filtered_updated_drugcomb_v1.csv`
- `drug_map_with_smiles.csv`
- `cell_map.csv`
- `inductive/train_idx.txt`
- `inductive/valid_idx.txt`
- `inductive/test_idx.txt`
- (optional split CSVs like `drugcomb-train.csv`, `drugcomb-valid.csv`, `drugcomb-test.csv`)

### 3.3 DrugCombDB Task Files

Folder:
- `data/drugcombdb/`

Typical files used by pipeline:
- `filtered_updated_drugcombdb.csv`
- `drug_map_with_smiles.csv`
- `cell_map.csv`
- `inductive/train_idx.txt`
- `inductive/valid_idx.txt`
- `inductive/test_idx.txt`
- `inductive/db-train.csv`
- `inductive/db-valid.csv`
- `inductive/db-test.csv`

### 3.4 DB Preprocessing Notebooks

All DB preprocessing notebooks are under:
- `data/drugcombdb/inductive/`

Current notebooks:
- `db_inductive_filtering.ipynb`
- `db_valid_split_mapping.ipynb`

## 4. Prepare Processed Graph Data (.pt)

Before training, make sure processed files exist in `data/processed/...`.
If missing, run:

```bash
cd "/Users/hyq/LACO/DeepDDs 3"
python creat_data_DC.py
```

This script reads raw CSV and generates graph `.pt` files used by the trainer.

## 5. Train Command

Run from project root:

```bash
cd "/Users/hyq/LACO/DeepDDs 3"
python trianing_GCN.py
```

Current `trianing_GCN.py` behavior:
- runs 5 random seeds per execution
- uses train/valid/test indices from `data/drugcombdb/inductive/`
- reads LLM embeddings from `data/LLM_DATA/`

## 6. Switch Between DrugComb and DrugCombDB

The key setting is in `trianing_GCN.py`:
- `datafile = 'drugcomb/...` or `drugcombdb/...`
- index files path in `np.loadtxt(...)`

If you switch task source, update both:
1. dataset file prefix (`datafile`)
2. corresponding index files under the matching `inductive/` folder

## 7. Quick Sanity Check

Run a syntax check:

```bash
python -m py_compile trianing_GCN.py utils_test.py creat_data_DC.py
```

Check that key paths exist:

```bash
ls data/LLM_DATA
ls data/drugcombdb/inductive
ls data/drugcomb/inductive
```

## 8. Common Issues

1. `FileNotFoundError` for LLM files
- Ensure `.npy` files are in `data/LLM_DATA/`.

2. Index file not found
- Ensure `train_idx.txt`, `valid_idx.txt`, `test_idx.txt` exist in the selected task folder.

3. CUDA device mismatch
- `trianing_GCN.py` sets `CUDA_VISIBLE_DEVICES` internally.
- Adjust it to your local GPU setup if needed.

4. Empty or outdated processed files
- Re-run `python creat_data_DC.py` to rebuild processed data.

## 9. Current Naming Conventions

- Use `drugcomb` instead of `v1`
- Use `drugcombdb` instead of `db`
- Use `LLM_DATA` as the shared LLM embedding folder (parallel to task folders)
