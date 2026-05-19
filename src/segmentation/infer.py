from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
from PIL import Image

from .models import build_model


IMAGE_EXTENSIONS = {".tif", ".tiff", ".png", ".jpg", ".jpeg", ".bmp"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run segmentation inference with a trained U-Net checkpoint.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--input", required=True, help="Image file or directory of images.")
    parser.add_argument("--output-dir", default="predictions")
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--recursive", action="store_true")
    parser.add_argument("--save-overlay", action="store_true")
    return parser.parse_args()


def load_image(path: Path, image_size: int) -> tuple[torch.Tensor, Image.Image]:
    with Image.open(path) as raw:
        original = raw.convert("RGB")
        resized = original.resize((image_size, image_size), Image.Resampling.BILINEAR)
        array = np.asarray(resized, dtype=np.float32) / 255.0
    tensor = torch.from_numpy(array).permute(2, 0, 1).unsqueeze(0).contiguous()
    return tensor, original


def iter_input_images(path: Path, recursive: bool) -> Iterable[Path]:
    if path.is_file():
        yield path
        return
    pattern = "**/*" if recursive else "*"
    for candidate in sorted(path.glob(pattern)):
        if candidate.is_file() and candidate.suffix.lower() in IMAGE_EXTENSIONS and not candidate.name.endswith("_mask.tif"):
            yield candidate


def save_mask(mask: np.ndarray, output_path: Path, original_size: tuple[int, int]) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    image = Image.fromarray((mask * 255).astype(np.uint8), mode="L")
    if image.size != original_size:
        image = image.resize(original_size, Image.Resampling.NEAREST)
    image.save(output_path)


def save_overlay(original: Image.Image, mask: np.ndarray, output_path: Path) -> None:
    overlay_mask = Image.fromarray((mask * 180).astype(np.uint8), mode="L")
    if overlay_mask.size != original.size:
        overlay_mask = overlay_mask.resize(original.size, Image.Resampling.NEAREST)
    red = Image.new("RGBA", original.size, (255, 32, 32, 0))
    red.putalpha(overlay_mask)
    composed = Image.alpha_composite(original.convert("RGBA"), red)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    composed.save(output_path)


@torch.inference_mode()
def run_inference(args: argparse.Namespace) -> list[dict[str, str | float]]:
    device = torch.device(args.device)
    checkpoint = torch.load(args.checkpoint, map_location=device)
    config = checkpoint.get("config", {})
    architecture = config.get("architecture", "unet")
    image_size = int(config.get("image_size", 256))
    base_channels = int(config.get("base_channels", 32))

    model = build_model(architecture, base_channels=base_channels)
    model.load_state_dict(checkpoint["model_state"])
    model.to(device)
    model.eval()

    output_dir = Path(args.output_dir)
    records: list[dict[str, str | float]] = []
    for image_path in iter_input_images(Path(args.input), recursive=args.recursive):
        tensor, original = load_image(image_path, image_size)
        tensor = tensor.to(device=device, non_blocking=True)
        logits = model(tensor)
        probs = torch.sigmoid(logits)[0, 0].detach().cpu().numpy()
        mask = (probs >= args.threshold).astype(np.uint8)

        relative_name = image_path.stem
        mask_path = output_dir / f"{relative_name}_pred_mask.png"
        save_mask(mask, mask_path, original.size)
        record: dict[str, str | float] = {
            "image": str(image_path),
            "mask": str(mask_path),
            "mean_probability": float(probs.mean()),
            "positive_pixel_ratio": float(mask.mean()),
        }
        if args.save_overlay:
            overlay_path = output_dir / f"{relative_name}_overlay.png"
            save_overlay(original, mask, overlay_path)
            record["overlay"] = str(overlay_path)
        records.append(record)

    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "predictions.json").write_text(json.dumps(records, indent=2), encoding="utf-8")
    return records


def main() -> None:
    args = parse_args()
    records = run_inference(args)
    print(json.dumps({"predictions": records}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
