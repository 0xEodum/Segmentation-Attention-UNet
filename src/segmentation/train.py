from __future__ import annotations

import argparse
import json
import math
import random
import shutil
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from .data import (
    BrainTumorDataset,
    find_image_mask_pairs,
    pair_to_record,
    split_by_patient,
    worker_seed_init,
)
from .losses import bce_dice_loss, confusion_from_logits, metrics_from_confusion
from .models import build_model


@dataclass(frozen=True)
class TrainConfig:
    architecture: str
    data_root: str
    output_dir: str
    checkpoint_dir: str
    epochs: int
    batch_size: int
    image_size: int
    base_channels: int
    learning_rate: float
    weight_decay: float
    val_fraction: float
    test_fraction: float
    seed: int
    num_workers: int
    positive_weight: float | None
    threshold: float
    amp: bool
    channels_last: bool
    compile_model: bool
    resume: str | None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train U-Net or Attention U-Net on kaggle_3m brain MRI masks.")
    parser.add_argument("--architecture", choices=["unet", "attention_unet"], required=True)
    parser.add_argument("--data-root", default="kaggle_3m")
    parser.add_argument("--output-dir", default="runs")
    parser.add_argument("--checkpoint-dir", default="checkpoints")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--base-channels", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--val-fraction", type=float, default=0.15)
    parser.add_argument("--test-fraction", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=6)
    parser.add_argument("--positive-weight", type=float, default=4.0)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--no-amp", action="store_true")
    parser.add_argument("--no-channels-last", action="store_true")
    parser.add_argument("--compile", action="store_true", dest="compile_model")
    parser.add_argument("--resume")
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def make_grad_scaler(enabled: bool) -> torch.amp.GradScaler:
    try:
        return torch.amp.GradScaler("cuda", enabled=enabled)
    except TypeError:
        return torch.cuda.amp.GradScaler(enabled=enabled)


def autocast_context(device: torch.device, enabled: bool):
    if device.type == "cuda":
        return torch.amp.autocast(device_type="cuda", dtype=torch.float16, enabled=enabled)
    return torch.amp.autocast(device_type="cpu", enabled=False)


def make_loader(
    dataset: BrainTumorDataset,
    *,
    batch_size: int,
    shuffle: bool,
    num_workers: int,
    seed: int,
    device: torch.device,
) -> DataLoader:
    generator = torch.Generator()
    generator.manual_seed(seed)
    loader_kwargs: dict[str, Any] = {
        "batch_size": batch_size,
        "shuffle": shuffle,
        "num_workers": num_workers,
        "pin_memory": device.type == "cuda",
        "worker_init_fn": worker_seed_init if num_workers > 0 else None,
        "generator": generator,
    }
    if num_workers > 0:
        loader_kwargs["persistent_workers"] = True
        loader_kwargs["prefetch_factor"] = 4
    return DataLoader(dataset, **loader_kwargs)


def tensor_to_device(images: torch.Tensor, masks: torch.Tensor, device: torch.device, channels_last: bool) -> tuple[torch.Tensor, torch.Tensor]:
    if channels_last and device.type == "cuda":
        images = images.to(device=device, non_blocking=True, memory_format=torch.channels_last)
    else:
        images = images.to(device=device, non_blocking=True)
    masks = masks.to(device=device, non_blocking=True)
    return images, masks


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
    device: torch.device,
    config: TrainConfig,
) -> dict[str, float]:
    model.train()
    loss_sum = torch.zeros((), device=device)
    tp = torch.zeros((), device=device)
    fp = torch.zeros((), device=device)
    fn = torch.zeros((), device=device)
    tn = torch.zeros((), device=device)
    samples = 0
    start = time.perf_counter()

    for images, masks, _ in loader:
        images, masks = tensor_to_device(images, masks, device, config.channels_last)
        optimizer.zero_grad(set_to_none=True)
        with autocast_context(device, config.amp):
            logits = model(images)
            loss = bce_dice_loss(logits, masks, positive_weight=config.positive_weight)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        batch_size = images.shape[0]
        loss_sum += loss.detach() * batch_size
        batch_tp, batch_fp, batch_fn, batch_tn = confusion_from_logits(logits.detach(), masks, threshold=config.threshold)
        tp += batch_tp
        fp += batch_fp
        fn += batch_fn
        tn += batch_tn
        samples += batch_size

    elapsed = time.perf_counter() - start
    metrics = metrics_from_confusion(tp, fp, fn, tn)
    metrics["loss"] = float((loss_sum / max(samples, 1)).detach().cpu())
    metrics["images_per_second"] = samples / max(elapsed, 1e-9)
    return metrics


@torch.inference_mode()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    config: TrainConfig,
) -> dict[str, float]:
    model.eval()
    loss_sum = torch.zeros((), device=device)
    tp = torch.zeros((), device=device)
    fp = torch.zeros((), device=device)
    fn = torch.zeros((), device=device)
    tn = torch.zeros((), device=device)
    samples = 0
    start = time.perf_counter()

    for images, masks, _ in loader:
        images, masks = tensor_to_device(images, masks, device, config.channels_last)
        with autocast_context(device, config.amp):
            logits = model(images)
            loss = bce_dice_loss(logits, masks, positive_weight=config.positive_weight)
        batch_size = images.shape[0]
        loss_sum += loss.detach() * batch_size
        batch_tp, batch_fp, batch_fn, batch_tn = confusion_from_logits(logits, masks, threshold=config.threshold)
        tp += batch_tp
        fp += batch_fp
        fn += batch_fn
        tn += batch_tn
        samples += batch_size

    elapsed = time.perf_counter() - start
    metrics = metrics_from_confusion(tp, fp, fn, tn)
    metrics["loss"] = float((loss_sum / max(samples, 1)).detach().cpu())
    metrics["images_per_second"] = samples / max(elapsed, 1e-9)
    return metrics


def save_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def save_checkpoint(
    path: Path,
    *,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    scaler: torch.amp.GradScaler,
    epoch: int,
    best_val_dice: float,
    config: TrainConfig,
    history: list[dict[str, Any]],
    split_records: dict[str, list[dict[str, str]]],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "epoch": epoch,
            "best_val_dice": best_val_dice,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "scheduler_state": scheduler.state_dict(),
            "scaler_state": scaler.state_dict(),
            "config": asdict(config),
            "history": history,
            "splits": split_records,
        },
        path,
    )


def load_resume(
    path: str,
    *,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    scaler: torch.amp.GradScaler,
    device: torch.device,
) -> tuple[int, float, list[dict[str, Any]]]:
    checkpoint = torch.load(path, map_location=device)
    model.load_state_dict(checkpoint["model_state"])
    optimizer.load_state_dict(checkpoint["optimizer_state"])
    scheduler.load_state_dict(checkpoint["scheduler_state"])
    scaler.load_state_dict(checkpoint["scaler_state"])
    return int(checkpoint["epoch"]) + 1, float(checkpoint["best_val_dice"]), list(checkpoint.get("history", []))


def train(config: TrainConfig) -> Path:
    set_seed(config.seed)
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
        torch.set_float32_matmul_precision("high")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp_enabled = config.amp and device.type == "cuda"

    pairs = find_image_mask_pairs(config.data_root)
    split = split_by_patient(
        pairs,
        val_fraction=config.val_fraction,
        test_fraction=config.test_fraction,
        seed=config.seed,
    )
    train_dataset = BrainTumorDataset(split["train"], image_size=config.image_size, augment=True)
    val_dataset = BrainTumorDataset(split["val"], image_size=config.image_size, augment=False)
    test_dataset = BrainTumorDataset(split["test"], image_size=config.image_size, augment=False)

    train_loader = make_loader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        seed=config.seed,
        device=device,
    )
    val_loader = make_loader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        seed=config.seed + 1,
        device=device,
    )
    test_loader = make_loader(
        test_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        seed=config.seed + 2,
        device=device,
    )

    model = build_model(config.architecture, base_channels=config.base_channels)
    model = model.to(device)
    if config.channels_last and device.type == "cuda":
        model = model.to(memory_format=torch.channels_last)
    if config.compile_model:
        model = torch.compile(model)

    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(config.epochs, 1), eta_min=config.learning_rate * 0.05)
    scaler = make_grad_scaler(enabled=amp_enabled)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = Path(config.output_dir) / f"{timestamp}_{config.architecture}"
    run_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir = Path(config.checkpoint_dir)
    split_records = {name: [pair_to_record(pair) for pair in values] for name, values in split.items()}
    save_json(run_dir / "config.json", asdict(config))
    save_json(run_dir / "splits.json", split_records)

    start_epoch = 1
    best_val_dice = -math.inf
    history: list[dict[str, Any]] = []
    if config.resume:
        start_epoch, best_val_dice, history = load_resume(
            config.resume,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            device=device,
        )

    print(
        json.dumps(
            {
                "event": "start",
                "architecture": config.architecture,
                "device": str(device),
                "cuda_name": torch.cuda.get_device_name(0) if device.type == "cuda" else None,
                "train_images": len(train_dataset),
                "val_images": len(val_dataset),
                "test_images": len(test_dataset),
                "run_dir": str(run_dir),
            },
            ensure_ascii=False,
        ),
        flush=True,
    )

    best_path = run_dir / "best.pt"
    last_path = run_dir / "last.pt"
    exported_best_path = checkpoint_dir / f"{config.architecture}_best.pt"

    for epoch in range(start_epoch, config.epochs + 1):
        train_metrics = train_one_epoch(model, train_loader, optimizer, scaler, device, config)
        val_metrics = evaluate(model, val_loader, device, config)
        scheduler.step()
        record = {
            "epoch": epoch,
            "learning_rate": scheduler.get_last_lr()[0],
            "train": train_metrics,
            "val": val_metrics,
        }
        history.append(record)
        save_json(run_dir / "history.json", history)

        is_best = val_metrics["dice"] > best_val_dice
        if is_best:
            best_val_dice = val_metrics["dice"]
            save_checkpoint(
                best_path,
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                scaler=scaler,
                epoch=epoch,
                best_val_dice=best_val_dice,
                config=config,
                history=history,
                split_records=split_records,
            )
            checkpoint_dir.mkdir(parents=True, exist_ok=True)
            shutil.copy2(best_path, exported_best_path)

        save_checkpoint(
            last_path,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            epoch=epoch,
            best_val_dice=best_val_dice,
            config=config,
            history=history,
            split_records=split_records,
        )
        print(json.dumps(record, ensure_ascii=False), flush=True)

    if best_path.exists():
        checkpoint = torch.load(best_path, map_location=device)
        model.load_state_dict(checkpoint["model_state"])
    test_metrics = evaluate(model, test_loader, device, config)
    summary = {
        "best_checkpoint": str(best_path),
        "exported_best_checkpoint": str(exported_best_path),
        "last_checkpoint": str(last_path),
        "best_val_dice": best_val_dice,
        "test": test_metrics,
        "history": history,
    }
    save_json(run_dir / "summary.json", summary)
    print(json.dumps({"event": "done", **summary}, ensure_ascii=False), flush=True)
    return exported_best_path


def main() -> None:
    args = parse_args()
    config = TrainConfig(
        architecture=args.architecture,
        data_root=args.data_root,
        output_dir=args.output_dir,
        checkpoint_dir=args.checkpoint_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        image_size=args.image_size,
        base_channels=args.base_channels,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        val_fraction=args.val_fraction,
        test_fraction=args.test_fraction,
        seed=args.seed,
        num_workers=args.num_workers,
        positive_weight=args.positive_weight,
        threshold=args.threshold,
        amp=not args.no_amp,
        channels_last=not args.no_channels_last,
        compile_model=args.compile_model,
        resume=args.resume,
    )
    train(config)


if __name__ == "__main__":
    main()
