#!/usr/bin/env python
"""Test pretrained ViT base model (before fine-tuning) on iCartoonFace.

This establishes a baseline to compare against fine-tuned models.
Note: The pretrained ImageNet model won't perform well on anime faces,
but it's useful to know the starting point.
"""
import argparse
from pathlib import Path

import torch
import torch.nn.functional as F
from tqdm import tqdm

from anitune.data import DataConfig, build_dataloaders
from anitune.models import ModelConfig, build_model


def parse_args():
    parser = argparse.ArgumentParser(description="Test pretrained ViT base model (no fine-tuning)")
    parser.add_argument("--config", type=Path, required=True, help="Config file (for data settings)")
    parser.add_argument("--data-root", type=Path, help="Override data root directory")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument(
        "--split",
        choices=["train", "val"],
        default="val",
        help="Which split to evaluate (train or val)",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=None,
        help="Limit evaluation to first N samples (for quick testing)",
    )
    return parser.parse_args()


def evaluate_random_baseline(num_classes, num_samples):
    """Random guessing baseline."""
    return 1.0 / num_classes


def evaluate_pretrained_features(model, loader, device, num_samples=None):
    """
    Evaluate pretrained ViT features using nearest-centroid classification.
    
    Strategy:
    1. Extract features from all training samples and compute class centroids
    2. For each test sample, classify by nearest centroid
    
    This gives a better estimate of feature quality than random guessing.
    """
    model.eval()
    
    all_features = []
    all_labels = []
    
    print("Extracting features...")
    with torch.no_grad():
        for batch_idx, (images, labels) in enumerate(tqdm(loader)):
            if num_samples and batch_idx * loader.batch_size >= num_samples:
                break
            
            images = images.to(device)
            
            # Extract features before final classification layer
            # For ViT: use the [CLS] token representation
            if hasattr(model, 'head'):
                # Remove the classification head temporarily
                original_head = model.head
                model.head = torch.nn.Identity()
                features = model(images)
                model.head = original_head
            else:
                # Fallback: use model output as features
                features = model(images)
            
            all_features.append(features.cpu())
            all_labels.append(labels.cpu())
    
    all_features = torch.cat(all_features, dim=0)
    all_labels = torch.cat(all_labels, dim=0)
    
    # Compute class centroids
    unique_labels = torch.unique(all_labels)
    centroids = []
    centroid_labels = []
    
    print(f"Computing centroids for {len(unique_labels)} classes...")
    for label in unique_labels:
        mask = all_labels == label
        class_features = all_features[mask]
        centroid = class_features.mean(dim=0)
        centroids.append(centroid)
        centroid_labels.append(label.item())
    
    centroids = torch.stack(centroids)  # [num_classes, feature_dim]
    
    # Classify by nearest centroid
    print("Classifying by nearest centroid...")
    correct = 0
    total = 0
    
    for i in range(len(all_features)):
        feature = all_features[i]
        true_label = all_labels[i].item()
        
        # Compute distances to all centroids
        distances = torch.cdist(feature.unsqueeze(0), centroids, p=2)
        pred_idx = distances.argmin().item()
        pred_label = centroid_labels[pred_idx]
        
        if pred_label == true_label:
            correct += 1
        total += 1
    
    accuracy = correct / total if total > 0 else 0
    return accuracy, total


def simple_evaluation(model, loader, device, num_samples=None):
    """
    Simple evaluation: use the model's classification head directly.
    Note: This will give very low accuracy since the head is initialized randomly.
    """
    model.eval()
    
    correct_top1 = 0
    correct_top5 = 0
    total = 0
    
    print("Evaluating model predictions...")
    with torch.no_grad():
        for batch_idx, (images, labels) in enumerate(tqdm(loader)):
            if num_samples and batch_idx * loader.batch_size >= num_samples:
                break
            
            images = images.to(device)
            labels = labels.to(device)
            
            outputs = model(images)
            
            # Top-1 accuracy
            _, pred = outputs.max(1)
            correct_top1 += (pred == labels).sum().item()
            
            # Top-5 accuracy
            _, pred_top5 = outputs.topk(5, 1, largest=True, sorted=True)
            correct_top5 += sum([labels[i].item() in pred_top5[i].tolist() for i in range(len(labels))])
            
            total += labels.size(0)
    
    top1_acc = correct_top1 / total if total > 0 else 0
    top5_acc = correct_top5 / total if total > 0 else 0
    
    return top1_acc, top5_acc, total


def main():
    args = parse_args()
    
    # Load config (for data settings)
    from anitune.utils import load_config
    cfg = load_config(args.config)
    
    data_cfg = DataConfig(**cfg["data"])
    if args.data_root:
        data_cfg.root = args.data_root
    
    # Build dataloaders
    print(f"Loading data from {data_cfg.root}...")
    train_loader, val_loader = build_dataloaders(data_cfg)
    
    # Get number of classes
    train_ds = train_loader.dataset
    base_ds = getattr(train_ds, "dataset", train_ds)
    if hasattr(base_ds, "classes"):
        num_classes = len(base_ds.classes)
    else:
        raise SystemExit("Unable to infer number of classes from dataset.")
    
    print(f"Number of classes: {num_classes}")
    
    # Build pretrained model (WITHOUT loading any checkpoint)
    model_cfg = ModelConfig(**cfg["model"])
    model_cfg.num_classes = num_classes
    model_cfg.pretrained = True  # Load ImageNet pretrained weights
    model_cfg.use_lora = False   # Don't use LoRA for baseline
    
    print(f"Building pretrained {model_cfg.name} model...")
    model = build_model(model_cfg)
    
    device = torch.device(args.device)
    model.to(device)
    
    # Choose evaluation split
    loader = train_loader if args.split == "train" else val_loader
    print(f"\nEvaluating on {args.split} split...")
    
    # Random baseline
    random_acc = evaluate_random_baseline(num_classes, len(loader.dataset))
    print(f"\n{'='*60}")
    print(f"Random Guessing Baseline:")
    print(f"  Top-1 Accuracy: {random_acc*100:.4f}%")
    print(f"{'='*60}")
    
    # Pretrained model with random head (will be very low)
    print(f"\n{'='*60}")
    print(f"Pretrained ViT with Random Classification Head:")
    print(f"{'='*60}")
    top1, top5, total = simple_evaluation(model, loader, device, args.num_samples)
    print(f"  Samples evaluated: {total}")
    print(f"  Top-1 Accuracy: {top1*100:.4f}%")
    print(f"  Top-5 Accuracy: {top5*100:.4f}%")
    print(f"\nNote: Low accuracy is expected since the classification head")
    print(f"      is randomly initialized (not trained on anime faces).")
    
    print(f"\n{'='*60}")
    print(f"Summary:")
    print(f"{'='*60}")
    print(f"Dataset: {num_classes} classes, {len(loader.dataset)} images ({args.split} split)")
    print(f"Random baseline: {random_acc*100:.4f}%")
    print(f"Pretrained ViT (untrained head): {top1*100:.4f}%")
    print(f"\nThis baseline will improve significantly after fine-tuning!")
    print(f"Expected after LoRA fine-tuning: 85-90% (10 epochs)")


if __name__ == "__main__":
    main()

