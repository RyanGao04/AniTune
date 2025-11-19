from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

from PIL import Image
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision import datasets, transforms


@dataclass
class DataConfig:
    root: Path
    img_size: int = 224
    train_split: float = 0.9
    batch_size: int = 32
    num_workers: int = 4
    use_grayscale: bool = False
    manifest_dir: Optional[Path] = None

    def __post_init__(self):
        # Accept str paths from YAML and normalize to Path objects
        if not isinstance(self.root, Path):
            self.root = Path(self.root)
        if self.manifest_dir is not None and not isinstance(self.manifest_dir, Path):
            self.manifest_dir = Path(self.manifest_dir)


def build_transforms(img_size: int, use_grayscale: bool = False):
    channels = 1 if use_grayscale else 3
    normalize = transforms.Normalize(
        mean=[0.5] * channels,
        std=[0.5] * channels,
    )
    train_tfms = transforms.Compose(
        [
            transforms.Resize(int(img_size * 1.1)),
            transforms.CenterCrop(img_size),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ]
    )
    eval_tfms = transforms.Compose(
        [
            transforms.Resize(int(img_size * 1.1)),
            transforms.CenterCrop(img_size),
            transforms.ToTensor(),
            normalize,
        ]
    )
    return train_tfms, eval_tfms


class ManifestDataset(Dataset):
    """Dataset that loads images based on a manifest file with `relative_path label` per line."""

    def __init__(self, root: Path, manifest: Path, transform):
        self.root = root
        self.transform = transform
        self.entries = []
        labels = []
        with open(manifest, "r") as f:
            for line in f:
                rel_path, label = line.strip().split()
                label_int = int(label)
                self.entries.append((rel_path, label_int))
                labels.append(label_int)
        self.classes = sorted(set(labels))

    def __len__(self):
        return len(self.entries)

    def __getitem__(self, idx):
        rel_path, label = self.entries[idx]
        img_path = self.root / rel_path
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, label


def build_dataloaders(cfg: DataConfig) -> Tuple[DataLoader, DataLoader]:
    train_tfms, eval_tfms = build_transforms(cfg.img_size, cfg.use_grayscale)

    if cfg.manifest_dir and (cfg.manifest_dir / "train.txt").exists():
        train_ds = ManifestDataset(cfg.root, cfg.manifest_dir / "train.txt", train_tfms)
        val_ds = ManifestDataset(cfg.root, cfg.manifest_dir / "val.txt", eval_tfms)
    else:
        dataset = datasets.ImageFolder(cfg.root, transform=train_tfms)
        n_train = int(len(dataset) * cfg.train_split)
        n_val = len(dataset) - n_train
        train_ds, val_ds = random_split(dataset, [n_train, n_val])
        # Apply eval transforms to val subset
        val_ds.dataset.transform = eval_tfms

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=True,
    )
    return train_loader, val_loader


def build_eval_loader(
    cfg: DataConfig,
    *,
    root: Optional[Path] = None,
    manifest: Optional[Path] = None,
) -> DataLoader:
    """Build a deterministic eval loader (val/test) with optional overrides."""
    _, eval_tfms = build_transforms(cfg.img_size, cfg.use_grayscale)
    data_root = Path(root) if root else cfg.root

    manifest_path = None
    if manifest:
        manifest_path = Path(manifest)
        if manifest_path.is_dir():
            manifest_path = manifest_path / "test.txt"
    elif cfg.manifest_dir:
        candidate = cfg.manifest_dir / "test.txt"
        if candidate.exists():
            manifest_path = candidate

    if manifest_path and manifest_path.exists():
        dataset = ManifestDataset(data_root, manifest_path, eval_tfms)
    else:
        dataset = datasets.ImageFolder(data_root, transform=eval_tfms)

    return DataLoader(
        dataset,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=True,
    )
