# AniTune
Robust anime face recognition via ViT fine-tuning with LoRA on iCartoonFace.

## Quickstart
1) Create env and install deps (conda + pip):
```bash
conda env create -f environment.yml  # or: python -m venv .venv && source .venv/bin/activate
conda activate anitune
pip install -r requirements.txt
pip install -e .  # install anitune package for clean imports
```
2) Prepare data in ImageFolder layout (e.g., `data/personai_icartoonface_rectrain/icartoonface_rectrain/<id>/*.jpg`) and generate manifests:
```bash
python scripts/prepare_icartoonface.py --source data/personai_icartoonface_rectrain/icartoonface_rectrain --output data/icartoonface --val-ratio 0.1 --seed 42
```
3) Train LoRA baseline:
```bash
python scripts/train.py --config configs/lora_vitb16.yaml --data-root data/personai_icartoonface_rectrain/icartoonface_rectrain
```
4) Evaluate a checkpoint:
```bash
python scripts/eval.py --config configs/lora_vitb16.yaml --checkpoint runs/lora_vitb16/best.pt --data-root data/personai_icartoonface_rectrain/icartoonface_rectrain
```
5) (Optional) Enable Weights & Biases logging:
```bash
python scripts/train.py --config configs/lora_vitb16.yaml --data-root data/personai_icartoonface_rectrain/icartoonface_rectrain --wandb --wandb-project AniTune
```

## Quick smoke test (small subset)
- Create small manifests (e.g., 20 IDs, max 10 images each) to validate the pipeline quickly:
```bash
python scripts/prepare_icartoonface.py --source data/personai_icartoonface_rectrain/icartoonface_rectrain --output data/icartoonface --val-ratio 0.1 --seed 42 --max-ids 20 --max-per-id 10
```
- Then run the smoke config (1 epoch, tiny batch, no pretrained download):
```bash
python scripts/train.py --config configs/smoke.yaml --data-root data/personai_icartoonface_rectrain/icartoonface_rectrain
```

## Project Structure
- `configs/`: YAML configs for LoRA and full fine-tuning variants.
- `scripts/`: Entry points for training/eval.
- `src/anitune/`: Library code (models, LoRA injection, data, train loop).
- `runs/`: Default output directory for checkpoints and logs (created at runtime).
- `data/`: Expected location for iCartoonFace assets (gitignored; not included).

## Dataset (iCartoonFace)
- Recognition split download (mirrors from upstream):  
  - iQIYI: https://fft.cloud.iqiyi.com/s/bUbdw5A (pwd: 5Kv2M1)  
  - Google Drive: https://drive.google.com/drive/folders/1m6pAL9Wbn8B1td0hFUj9RVRrSweNKskW?usp=sharing
- After extracting, keep the structure `data/personai_icartoonface_rectrain/icartoonface_rectrain/<identity_id>/*.jpg` (5013 IDs, ~389k images).
- Run the manifest prep step (see Quickstart) to write `data/icartoonface/splits/{train,val}.txt` and `stats.json` without copying files; the default config consumes these manifests.
- Optional: keep detection set separate (`data/personai_icartoonface_rectest`) if you plan to add detection; current code only uses the recognition portion.

## Goals (from proposal)
- Compare LoRA vs full fine-tuning on ViT backbones (MAE/DINOv2 variants).
- Report identification and verification metrics with parameter/latency costs.
- Include ablations on LoRA rank and input resolution.
