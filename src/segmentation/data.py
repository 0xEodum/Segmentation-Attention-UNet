from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset


@dataclass(frozen=True)
class ImageMaskPair:
    image_path: Path
    mask_path: Path
    patient_id: str
    sample_id: str


def find_image_mask_pairs(root: str | Path) -> list[ImageMaskPair]:
    dataset_root = Path(root)
    if not dataset_root.exists():
        raise FileNotFoundError(f"Dataset root does not exist: {dataset_root}")

    pairs: list[ImageMaskPair] = []
    for image_path in sorted(dataset_root.glob("TCGA_*/*.tif")):
        if image_path.name.endswith("_mask.tif"):
            continue
        mask_path = image_path.with_name(f"{image_path.stem}_mask.tif")
        if not mask_path.exists():
            raise FileNotFoundError(f"Missing mask for {image_path}: expected {mask_path}")
        pairs.append(
            ImageMaskPair(
                image_path=image_path,
                mask_path=mask_path,
                patient_id=image_path.parent.name,
                sample_id=image_path.stem,
            )
        )

    if not pairs:
        raise ValueError(f"No image/mask pairs found under {dataset_root}")
    return pairs


def split_by_patient(
    pairs: Iterable[ImageMaskPair],
    *,
    val_fraction: float = 0.15,
    test_fraction: float = 0.15,
    seed: int = 42,
) -> dict[str, list[ImageMaskPair]]:
    pair_list = list(pairs)
    if not 0 <= val_fraction < 1 or not 0 <= test_fraction < 1:
        raise ValueError("val_fraction and test_fraction must be in [0, 1)")
    if val_fraction + test_fraction >= 1:
        raise ValueError("val_fraction + test_fraction must be less than 1")

    patients = sorted({pair.patient_id for pair in pair_list})
    rng = random.Random(seed)
    shuffled = patients[:]
    rng.shuffle(shuffled)

    def fraction_count(fraction: float) -> int:
        if fraction == 0:
            return 0
        return max(1, int(round(len(shuffled) * fraction)))

    n_test = fraction_count(test_fraction)
    n_val = fraction_count(val_fraction)
    if n_test + n_val >= len(shuffled):
        raise ValueError("Not enough patients for requested split fractions")

    test_patients = frozenset(shuffled[:n_test])
    val_patients = frozenset(shuffled[n_test : n_test + n_val])
    train_patients = frozenset(shuffled[n_test + n_val :])

    def select(patient_ids: frozenset[str]) -> list[ImageMaskPair]:
        return [pair for pair in pair_list if pair.patient_id in patient_ids]

    return {
        "train": select(train_patients),
        "val": select(val_patients),
        "test": select(test_patients),
    }


def pair_to_record(pair: ImageMaskPair) -> dict[str, str]:
    return {
        "image_path": str(pair.image_path),
        "mask_path": str(pair.mask_path),
        "patient_id": pair.patient_id,
        "sample_id": pair.sample_id,
    }


def record_to_pair(record: dict[str, str]) -> ImageMaskPair:
    return ImageMaskPair(
        image_path=Path(record["image_path"]),
        mask_path=Path(record["mask_path"]),
        patient_id=record["patient_id"],
        sample_id=record["sample_id"],
    )


class BrainTumorDataset(Dataset[tuple[torch.Tensor, torch.Tensor, dict[str, str]]]):
    def __init__(
        self,
        pairs: Iterable[ImageMaskPair],
        *,
        image_size: int = 256,
        augment: bool = False,
    ) -> None:
        self.pairs = list(pairs)
        if not self.pairs:
            raise ValueError("BrainTumorDataset requires at least one image/mask pair")
        self.image_size = image_size
        self.augment = augment

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, dict[str, str]]:
        pair = self.pairs[index]
        image = self._load_image(pair.image_path)
        mask = self._load_mask(pair.mask_path)
        if self.augment:
            image, mask = self._augment(image, mask)
        meta = {
            "image_path": str(pair.image_path),
            "mask_path": str(pair.mask_path),
            "patient_id": pair.patient_id,
            "sample_id": pair.sample_id,
        }
        return image, mask, meta

    def _load_image(self, path: Path) -> torch.Tensor:
        with Image.open(path) as image:
            image = image.convert("RGB").resize((self.image_size, self.image_size), Image.Resampling.BILINEAR)
            array = np.asarray(image, dtype=np.float32) / 255.0
        return torch.from_numpy(array).permute(2, 0, 1).contiguous()

    def _load_mask(self, path: Path) -> torch.Tensor:
        with Image.open(path) as mask:
            mask = mask.convert("L").resize((self.image_size, self.image_size), Image.Resampling.NEAREST)
            array = (np.asarray(mask) > 0).astype(np.float32)
        return torch.from_numpy(array).unsqueeze(0).contiguous()

    def _augment(self, image: torch.Tensor, mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if torch.rand(()) < 0.5:
            image = torch.flip(image, dims=(2,))
            mask = torch.flip(mask, dims=(2,))
        if torch.rand(()) < 0.5:
            image = torch.flip(image, dims=(1,))
            mask = torch.flip(mask, dims=(1,))
        k = int(torch.randint(low=0, high=4, size=()).item())
        if k:
            image = torch.rot90(image, k=k, dims=(1, 2))
            mask = torch.rot90(mask, k=k, dims=(1, 2))
        if torch.rand(()) < 0.25:
            scale = 0.9 + 0.2 * torch.rand(())
            shift = -0.05 + 0.1 * torch.rand(())
            image = torch.clamp(image * scale + shift, 0.0, 1.0)
        return image.contiguous(), mask.contiguous()


def worker_seed_init(worker_id: int) -> None:
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed + worker_id)
    random.seed(worker_seed + worker_id)
