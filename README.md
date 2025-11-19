# AniTune
ViT fine-tuning with LoRA for robust anime face recognition on iCartoonFace.

## Setup
```bash
conda env create -f environment.yml && conda activate anitune
python -m pip install -r requirements.txt  # ensures torch/timm versions match
```
Tip: set `PYTHONPATH=src` when running scripts from the repo root. The venv route works too, but conda is the tested setup.

## Data
1) Download iCartoonFace recognition split (mirrors from the paper):
   - iQIYI: https://fft.cloud.iqiyi.com/s/bUbdw5A (pwd: 5Kv2M1)
   - Google Drive: https://drive.google.com/drive/folders/1m6pAL9Wbn8B1td0hFUj9RVRrSweNKskW
2) Extract to `data/personai_icartoonface_rectrain/icartoonface_rectrain/<identity_id>/*.jpg` (5013 IDs).
3) Generate manifests (no file copying):
```bash
python scripts/prepare_icartoonface.py \
  --source data/personai_icartoonface_rectrain/icartoonface_rectrain \
  --output data/icartoonface \
  --val-ratio 0.1 --seed 42
```

## Training
Baseline LoRA ViT-B/16 (uses manifests by default):
```bash
PYTHONPATH=src python scripts/train.py \
  --config configs/lora_vitb16.yaml \
  --data-root data/personai_icartoonface_rectrain/icartoonface_rectrain
```
Helpful flags:
- `--num-workers 0` if shared memory is restricted (CPU runs).
- `--no-lora` for full fine-tune; `--head-only` to freeze backbone.
- `--wandb --wandb-project AniTune` to log to Weights & Biases.

If offline, pretrained weights will be skipped automatically and the model will start from random init.

## Evaluation
```bash
PYTHONPATH=src python scripts/eval.py \
  --config configs/lora_vitb16.yaml \
  --checkpoint runs/lora_vitb16/best.pt \
  --data-root data/personai_icartoonface_rectrain/icartoonface_rectrain
```
To score against the official recognition test split (`personai_icartoonface_rectest`), point the evaluator at the test directory/manifest:
```bash
PYTHONPATH=src python scripts/eval.py \
  --config configs/lora_vitb16.yaml \
  --checkpoint runs/lora_vitb16/best.pt \
  --eval-split test \
  --test-root data/personai_icartoonface_rectest/icartoonface_rectest
# optionally add: --test-manifest data/icartoonface/splits/test.txt
```

## Project Structure
- `configs/`: YAML configs (LoRA and full FT).
- `scripts/`: Entry points (`train.py`, `eval.py`, `prepare_icartoonface.py`).
- `src/anitune/`: Library code (models, LoRA injection, data, train loop).
- `runs/`: Checkpoints and logs (created at runtime).
- `data/`: Expected dataset location (gitignored).

## Notes
- Default config: batch size 64, 10 epochs, 224² crops, LoRA rank 8 on ViT-B/16.
- Keep detection split (`data/personai_icartoonface_rectest`) separate if you plan to extend to detection; current code is recognition-only.
