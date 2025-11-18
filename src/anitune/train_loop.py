from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple

import torch
from torch import nn, optim
from torch.cuda.amp import GradScaler, autocast
from tqdm import tqdm

from .models import count_parameters


@dataclass
class OptimConfig:
    lr: float = 3e-4
    weight_decay: float = 0.05
    epochs: int = 10
    amp: bool = True
    log_interval: int = 50


def train_one_epoch(model, loader, optimizer, criterion, device, scaler: GradScaler, cfg: OptimConfig):
    model.train()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    pbar = tqdm(loader, desc="train", leave=False)
    for step, (images, labels) in enumerate(pbar, 1):
        images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)

        with autocast(enabled=cfg.amp):
            outputs = model(images)
            loss = criterion(outputs, labels)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        preds = outputs.argmax(dim=1)
        correct = (preds == labels).sum().item()
        total_correct += correct
        total_samples += labels.size(0)
        total_loss += loss.item() * labels.size(0)

        if step % cfg.log_interval == 0:
            pbar.set_postfix(loss=total_loss / total_samples, acc=total_correct / total_samples)

    return {
        "loss": total_loss / total_samples,
        "acc": total_correct / total_samples,
    }


def evaluate(model, loader, criterion, device, amp: bool = True) -> Dict[str, float]:
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)
            with autocast(enabled=amp):
                outputs = model(images)
                loss = criterion(outputs, labels)
            preds = outputs.argmax(dim=1)
            total_correct += (preds == labels).sum().item()
            total_samples += labels.size(0)
            total_loss += loss.item() * labels.size(0)

    return {
        "loss": total_loss / total_samples,
        "acc": total_correct / total_samples,
    }


def run_train(
    model,
    train_loader,
    val_loader,
    device: torch.device,
    optim_cfg: OptimConfig,
    save_dir: Path,
    log_fn=None,
) -> Dict[str, float]:
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=optim_cfg.lr, weight_decay=optim_cfg.weight_decay)
    scaler = GradScaler(enabled=optim_cfg.amp)

    total_params, trainable_params = count_parameters(model)
    print(f"Total params: {total_params/1e6:.2f}M | Trainable params: {trainable_params/1e6:.2f}M")

    best_acc = 0.0
    save_dir.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, optim_cfg.epochs + 1):
        train_metrics = train_one_epoch(model, train_loader, optimizer, criterion, device, scaler, optim_cfg)
        val_metrics = evaluate(model, val_loader, criterion, device, amp=optim_cfg.amp)
        epoch_log = {
            "epoch": epoch,
            "train_loss": train_metrics["loss"],
            "train_acc": train_metrics["acc"],
            "val_loss": val_metrics["loss"],
            "val_acc": val_metrics["acc"],
        }
        print(
            f"Epoch {epoch}/{optim_cfg.epochs} | "
            f"train_loss={train_metrics['loss']:.4f} acc={train_metrics['acc']:.4f} | "
            f"val_loss={val_metrics['loss']:.4f} acc={val_metrics['acc']:.4f}"
        )
        if log_fn:
            log_fn(epoch_log)
        if val_metrics["acc"] > best_acc:
            best_acc = val_metrics["acc"]
            torch.save({"model": model.state_dict(), "val_acc": best_acc}, save_dir / "best.pt")

    torch.save(model.state_dict(), save_dir / "last.pt")
    return {"best_val_acc": best_acc}
