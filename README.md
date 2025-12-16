# AniTune

A Vision Transformer (ViT) fine-tuning framework for anime/cartoon face recognition, featuring parameter-efficient training with LoRA (Low-Rank Adaptation).

## Overview

AniTune provides a comprehensive experimental framework for comparing different fine-tuning strategies on the iCartoonFace dataset (5,013 anime character classes). The project demonstrates how LoRA can achieve near full fine-tuning performance while training less than 5% of the parameters.

## Features

- **Four Training Modes**: Head-only, Full fine-tuning, LoRA-only (recommended), and LoRA+Full
- **LoRA Implementation**: Custom Low-Rank Adaptation for Vision Transformers
- **Parameter Efficiency**: Achieve 93%+ accuracy with only 0.35% trainable parameters
- **Experiment Framework**: Automated experiment running and result analysis
- **W&B Integration**: Real-time training monitoring with Weights & Biases
- **Mixed Precision**: Automatic mixed precision (AMP) for faster training

## Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/Ryan-GRY/AniTune.git
cd AniTune

# Create conda environment
conda create -n anitune python=3.10
conda activate anitune

# Install dependencies
pip install -r requirements.txt

# For A100 GPU setup
bash setup_a100.sh
```

### Data Preparation

```bash
# Download iCartoonFace dataset and place in data/personai_icartoonface_rectrain/

# Generate train/val manifests
python scripts/prepare_icartoonface.py \
    --source data/personai_icartoonface_rectrain/icartoonface_rectrain \
    --output data/icartoonface \
    --val-ratio 0.1 \
    --seed 42
```

### Training

```bash
# Run LoRA-only training (recommended)
./experiments/run_single_experiment.sh lora_only 8

# Run all experiments for comparison
./experiments/run_all_experiments.sh

# Manual training with full control
PYTHONPATH=src python experiments/train_experiments.py \
    --config experiments/configs/base_experiment.yaml \
    --data-root data/personai_icartoonface_rectrain/icartoonface_rectrain \
    --mode lora_only \
    --lora-rank 8 \
    --device cuda \
    --wandb
```

## Training Modes

| Mode | Backbone | LoRA | Head | Trainable Params | Use Case |
|------|----------|------|------|------------------|----------|
| `head_only` | Frozen | No | Train | ~10K (0.01%) | Baseline |
| `full_ft` | Train | No | Train | ~86M (100%) | Possible upper bound |
| `lora_only` | Frozen | Train | Train | ~300K (0.35%) | **Recommended** |
| `lora_full` | Train | Train | Train | ~86M+ | Experimental |

## Expected Results

Based on ViT-B/16 with 5,013 anime character classes:

| Method | Trainable Params | Accuracy | GPU Memory |
|--------|------------------|--------------|------------|
| Head-only | 10K (0.01%) | 75-80% | 8GB |
| Full FT | 86M (100%) | 92-95% | 24GB |
| LoRA r=8 | 300K (0.35%) | 91-94% | 12GB |
| LoRA r=16 | 600K (0.70%) | 92-95% | 14GB |

## Project Structure

```
AniTune/
├── src/anitune/           # Core library
│   ├── models.py          # Model building with 4 training modes
│   ├── lora.py            # LoRA implementation
│   ├── data.py            # Data loading utilities
│   ├── train_loop.py      # Training and evaluation
│   └── utils.py           # Helper utilities
├── experiments/           # Experiment framework
│   ├── train_experiments.py
│   ├── run_all_experiments.sh
│   ├── run_single_experiment.sh
│   ├── analyze_results.py
│   └── configs/
├── scripts/               # Utility scripts
│   ├── prepare_icartoonface.py
│   ├── eval.py
│   └── error_analysis.py
├── configs/               # Model configurations
├── doc/                   # Documentation
└── data/                  # Dataset directory
```

## Configuration

Key configuration options in `experiments/configs/base_experiment.yaml`:

```yaml
model:
  name: vit_base_patch16_224
  num_classes: 5013
  pretrained: true

data:
  img_size: 224
  batch_size: 512

optim:
  lr: 2.0e-4
  weight_decay: 0.05
  epochs: 10
  amp: true
```

## LoRA Rank Ablation

Test different LoRA ranks to find the optimal balance:

```bash
for RANK in 4 8 16 24 32; do
    ./experiments/run_single_experiment.sh lora_only $RANK
done
```

| Rank | Trainable Params | Accuracy |
|------|------------------|--------------|
| 4 | 150K (0.17%) | ~91.8% |
| 8 | 300K (0.35%) | ~93.1% |
| 16 | 600K (0.70%) | ~93.8% |
| 24 | 900K (1.05%) | ~94.0% |

## Requirements

- Python 3.8+
- PyTorch 1.12+ (with CUDA support recommended)
- NVIDIA GPU with 12GB+ VRAM (24GB+ for full fine-tuning)

### Dependencies

- timm >= 0.9.12
- pyyaml >= 6.0
- wandb >= 0.16.0
- tqdm >= 4.66.0
- pandas >= 2.1.0
- matplotlib >= 3.8.0
- Pillow >= 10.1.0
- numpy >= 1.24.0

## Evaluation

```bash
# Evaluate a trained model
PYTHONPATH=src python scripts/eval.py \
    --config configs/lora_vitb16.yaml \
    --checkpoint runs/vit_experiment_lora_only_r8/best.pt

# Analyze experiment results
python experiments/analyze_results.py
```

## Documentation

Detailed documentation is available in the [doc/](doc/) directory:

- [Quick Start Guide](doc/QUICK_START.md)
- [Configuration Guide](doc/CONFIG_GUIDE_CN.md)
- [A100 Setup Guide](doc/A100_SETUP_GUIDE.md)
- [Error Analysis](doc/ERROR_ANALYSIS.md)

## Citation

If you use this code in your research, please cite:

```bibtex
@software{anitune2025,
  title = {AniTune: Parameter-Efficient Fine-tuning for Anime Face Recognition},
  author = {Tingting Du, Ryan Gao, Evelyn Liu},
  year = {2025},
  url = {https://github.com/Ryan-GRY/AniTune}
}
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- [iCartoonFace](https://github.com/luxiangju-PersonAI/iCartoonFace) dataset
- [timm](https://github.com/huggingface/pytorch-image-models) for Vision Transformer implementations
- [LoRA](https://arxiv.org/abs/2106.09685) for the parameter-efficient fine-tuning method
