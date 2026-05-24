# KG4DB

A graph-based drug combination prediction framework built with PyTorch and DGL.

## Overview

This repository supports training drug combination classifiers with several GNN variants and optional LLM-derived embeddings. The main entrypoint is `dcb_main.py`, which builds a DGL graph, runs training, logs metrics, and saves the best checkpoint.

## Repository structure

- `dcb_main.py` — main training and evaluation script
- `utils.py` — model saving, evaluation, and logging helpers
- `model.py` — GNN model definitions (`GCN`, `HeteroGAT`, `KGNN`)
- `layers.py` — regularization and layer utilities
- `dataloader.py` — data loader and batching logic
- `preprocess.py` — dataset preprocessing utilities
- `generate_embed_bge.py` — BGE embedding generation helper
- `run.sh`, `run_base.sh` — example commands
- `datasets/` — dataset input files and prepared datasets
- `ckpts/` — checkpoint output root
- `logs/` — runtime logs
- `wandb/` — optional W&B output

## Installation

Recommended Python version: `3.8+`.

Install required packages:

```bash
pip install -r requeirments.txt
```

If you use a custom environment, ensure `torch` and `dgl` are compatible for the same CUDA version.

## Usage

Example training command:

```bash
python dcb_main.py --gpu 0 --model SAGE --dataset drugcombdb --debug --aug --llm gpt-4o-mini --setting S1
```

Common models:

- `GCN`
- `HGAT`
- `KGNN`

Common LLM options:

- `gpt-4o-mini`
- `gpt-3.5-turbo`
- `gpt-5`
- `llama3-8b-chat`
- `Baichuan2-chat`
- `llama`
- `qwen`
- `base`

### Example commands

```bash
python dcb_main.py --gpu 0 --model GCN --dataset drugcombdb --debug
python dcb_main.py --gpu 0 --model SAGE --dataset drugcombdb --debug --aug --llm gpt-5 --setting S1
python dcb_main.py --gpu 0 --model HGAT --dataset drugcombdb --debug --aug --llm gpt-3.5-turbo --setting S1
```

## Checkpoints and logs

- Checkpoints are saved to `ckpts/model/` by default.
- Logs are written to `logs/`.
- W&B logging is enabled unless `--debug` is specified, in which case it runs in dryrun mode.

This version of the repository automatically creates missing directories when saving checkpoints and logs.

## Data preparation

The graph construction currently expects:

- `datasets/kg/entities.dict`
- `datasets/kg/relations.dict`
- `datasets/kg/train_new.tsv`

The dataset files are available from the Hugging Face dataset:

- https://huggingface.co/datasets/Matthewmtf/LaCo

For a public release, the `datasets/` and `ckpts/` folders are provided separately as compressed archives.

Please download the archives or dataset files and extract them into the repository root so the extracted folder structure matches:

- `datasets/`
  - `kg/`
    - `entities.dict`
    - `relations.dict`
    - `train_new.tsv`
  - other dataset files used by the code
- `ckpts/`
  - `llms/`
    - pretrained LLM embedding files and maps

Example extraction commands:

```bash
unzip datasets.zip -d .
unzip ckpts.zip -d .
```

After extraction, verify that `datasets/kg/` and `ckpts/llms/` exist before running training.

Adjust data paths in `dcb_main.py` and `dataloader.py` if your dataset layout differs.

## Public release notes

Large runtime artifacts such as model checkpoints, logs, and W&B output are ignored by `.gitignore`.

## Notes

- `dcb_main.py` runs repeated experiments and writes average results to the log file.
- `generate_embed_bge.py` already creates output directories before saving embeddings.
- Use a clean Python environment for best reproducibility.
