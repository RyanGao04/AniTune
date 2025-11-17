# AniTune
Robust anime face recognition via ViT fine-tuning with LoRA on iCartoonFace.

## Quickstart
1) Create env and install deps:
```bash
python -m venv .venv && source .venv/bin/activate
python -m pip install -r requirements.txt
```
2) Prepare data in ImageFolder layout (e.g., `data/icartoonface/train/<id>/*.jpg`). Leave a held-out split for validation or rely on the built-in splitter.
3) Train LoRA baseline:
```bash
python scripts/train.py --config configs/lora_vitb16.yaml --data-root data/icartoonface/train
```
4) Evaluate a checkpoint:
```bash
python scripts/eval.py --config configs/lora_vitb16.yaml --checkpoint runs/lora_vitb16/best.pt --data-root data/icartoonface/train
```

## Project Structure
- `configs/`: YAML configs for LoRA and full fine-tuning variants.
- `scripts/`: Entry points for training/eval.
- `src/anitune/`: Library code (models, LoRA injection, data, train loop).
- `runs/`: Default output directory for checkpoints and logs (created at runtime).
- `data/`: Expected location for iCartoonFace assets (gitignored; not included).

## Goals (from proposal)
- Compare LoRA vs full fine-tuning on ViT backbones (MAE/DINOv2 variants).
- Report identification and verification metrics with parameter/latency costs.
- Include ablations on LoRA rank and input resolution.
