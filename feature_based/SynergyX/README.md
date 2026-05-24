# SynergyX

A concise guide for data layout and execution.

## Entry Files

- Main: `main-5.py`
- Data loader: `utlis.py`
- Dataset class: `dataset/My_inMemory_dataset.py`

## Data Layout

Use `SynergyX/` as root. The data can be downloaded at [HuggingFace](https://huggingface.co/datasets/Matthewmtf/LaCo/tree/main/SynergyX).

### 1) LLM embeddings (required)

Place under `MyDataset/LLM_data/`:
- `5_l_drug_embeddings.npy`
- `5_l_cell_embeddings.npy`
- `drug_map.csv`
- `cell_map.csv`

### 2) Structure/omics data

Place under `MyDataset/`:
- `drugSmile_drugSubEmbed_db.npy`
- `1024079_genes_norm.npy`
- `drug_map_with_smiles.csv`
- `cell.csv`

### 3) Split files

Place under `MyDataset/new-cell/` or `MyDataset/long-tail/`:
- `new-cell/v1-train.csv`, `v1-valid.csv`, `v1-test.csv`
- `new-cell/db-train.csv`, `db-valid.csv`, `db-test.csv`
- `long-tail/drugcomb_train_top_cells.csv`, `drugcomb_test_tail_cells.csv`
- `long-tail/drugcombdb_train_top_cells.csv`, `drugcombdb_test_tail_cells.csv`

## Run

From `feature_based/SynergyX`.

### Train

```bash
python main-5.py --mode train --dataset-split v1
```

Available splits:
- `v1`
- `db`
- `longtail_v1`
- `longtail_db`

Example:

```bash
python main-5.py --mode train --dataset-split db --batch_size 128 --lr 1e-4 --epochs 500
```

### Test

```bash
python main-5.py \
  --mode test \
  --saved-model ./experiment/<time>/<k>_fold_early_stop.pth \
  --dataset-split v1
```

### Inference

```bash
python main-5.py \
  --mode infer \
  --saved-model ./experiment/<time>/<k>_fold_early_stop.pth \
  --infer-path ./MyDataset/long-tail/drugcomb_test_tail_cells.csv \
  --output-attn 1
```

## Custom Dataset Notes

Prefer changing runtime args first:
- `--dataset-split`
- `--infer-path`
- `--workdir`

If your split logic differs, update `load_dataloader()` in `utlis.py`.

Your CSV should include:
- `drug_a`
- `drug_b`
- `cell`
- `synergy`
