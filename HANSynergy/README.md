# HANSynergy 2

A streamlined setup guide for training drug-combination synergy models.

## Entry Points

- Train: `main.py`
- Model: `model.py`
- Dataset loader: `dataset.py`
- Utilities: `utils.py`

## Environment

```bash
conda create -n hansynergy2 python=3.8.17
conda activate hansynergy2
pip install -r requirements.txt
```

## Data Layout
The data can be downloaded at [HuggingFace](https://huggingface.co/datasets/Matthewmtf/LaCo/tree/main/HANSynergy).

Default split files:
- `data/drugcombdb/db/db-train_final.csv`
- `data/drugcombdb/db/db-valid_final.csv`
- `data/drugcombdb/db/db-test_final.csv`

Additional files used by `dataset.py`:
- `data/cid_smile.json`
- `data/<data_type>/context_set_m.json`
- `data/<data_type>/id_dict.json`
- `data/drugcombdb/db/drug_map_with_cid.csv`
- `data/drugcombdb/db/cell_map.csv`

LLM embedding files (required):
- `data/LLM_data/5_l_drug_embeddings.npy`
- `data/LLM_data/5_l_cell_embeddings.npy`

Place downloaded `.npy` LLM embeddings into `data/LLM_data/`.

## Run

```bash
python main.py
```

Recommended explicit command:

```bash
python main.py \
  --data_type drugcombdb \
  --train_file data/drugcombdb/db/db-train_final.csv \
  --valid_file data/drugcombdb/db/db-valid_final.csv \
  --test_file data/drugcombdb/db/db-test_final.csv \
  --batch_size 8 \
  --n_epochs 200 \
  --lr 1e-3 \
  --gpu_index 0
```

## Notes for Custom Datasets

Update these args as needed:
- `--data_type`
- `--train_file`, `--valid_file`, `--test_file`
- `--col_drug1`, `--col_drug2`, `--col_context`, `--col_label`

Your CSV should contain (or be mapped to):
- `drug_1`
- `drug_2`
- `context`
- `label`
