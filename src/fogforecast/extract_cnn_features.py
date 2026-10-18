"""Extract frozen ImageNet ResNet-18 embeddings from a timestamped image manifest."""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import torch
from PIL import Image
from torchvision.models import ResNet18_Weights, resnet18


def extract(manifest: Path, output: Path, root: Path) -> None:
    frame = pd.read_csv(manifest)
    required = {"timestamp", "image_path"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Manifest is missing required columns: {sorted(missing)}")
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], errors="raise", utc=True)
    if frame["timestamp"].duplicated().any():
        raise ValueError("Manifest must contain at most one image per timestamp")

    weights = ResNet18_Weights.DEFAULT
    transform = weights.transforms()
    model = resnet18(weights=weights)
    model.fc = torch.nn.Identity()
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    rows: list[dict[str, object]] = []
    with torch.inference_mode():
        for record in frame[["timestamp", "image_path"]].itertuples(index=False):
            image_file = (root / str(record.image_path)).resolve()
            if not image_file.is_file():
                raise FileNotFoundError(f"Image listed in manifest does not exist: {image_file}")
            with Image.open(image_file) as image:
                tensor = transform(image.convert("RGB")).unsqueeze(0).to(device)
            vector = model(tensor).squeeze(0).cpu().numpy()
            row: dict[str, object] = {"timestamp": record.timestamp.isoformat()}
            row.update({f"cnn_{i:03d}": float(value) for i, value in enumerate(vector)})
            rows.append(row)

    output.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(output, index=False)
    print(f"Saved {len(rows)} image embeddings ({len(rows[0]) - 1} features) to {output}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True, help="CSV with timestamp,image_path")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--root", type=Path, default=Path.cwd(), help="Base directory for image_path values")
    args = parser.parse_args()
    extract(args.manifest, args.output, args.root)


if __name__ == "__main__":
    main()
