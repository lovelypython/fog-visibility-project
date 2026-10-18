#!/usr/bin/env python3
"""识别 Foggy Cityscapes 透射率图中的雾系数 beta。

输入必须是 leftImg8bit_transmittance 灰度透射率图 (0..255 对应 0..1)，
不是普通 RGB 雾天照片。示例：
  python3 recognize_transmittance.py path/to/map.png
  python3 recognize_transmittance.py path/to/maps/ --csv results.csv
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from PIL import Image
import torch

from train_transmittance_beta import BetaCNN, BETAS


def resolve_checkpoint(value: str | None) -> Path:
    if value:
        return Path(value).expanduser().resolve()
    return Path(__file__).resolve().parent / "outputs/transmittance_beta/best_model.pt"


def load_model(checkpoint_path: Path, device: torch.device):
    if not checkpoint_path.is_file():
        raise FileNotFoundError(
            f"找不到模型权重：{checkpoint_path}\n"
            "请先训练模型，或使用 --model 指定 best_model.pt。"
        )
    # This checkpoint is the state_dict and metadata produced by this project.
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    model = BetaCNN(classes=len(checkpoint.get("betas", BETAS))).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    return model, tuple(float(x) for x in checkpoint.get("betas", BETAS)), tuple(checkpoint["image_size"])


def predict_one(path: Path, model, betas, size, device):
    with Image.open(path) as source:
        image = source.convert("L").resize((size[1], size[0]), Image.Resampling.BILINEAR)
        array = np.asarray(image, dtype=np.float32)[None, None, :, :] / 255.0
    x = torch.from_numpy(array).to(device)
    with torch.inference_mode():
        probabilities = torch.softmax(model(x), dim=1)[0].cpu().numpy()
    best = int(probabilities.argmax())
    return {
        "file": str(path),
        "beta_per_meter": betas[best],
        "fog_level": {0.005: "轻雾", 0.01: "中雾", 0.02: "浓雾"}.get(betas[best], "未知"),
        "confidence": float(probabilities[best]),
        "probabilities": {str(beta): float(probabilities[i]) for i, beta in enumerate(betas)},
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, help="透射率 PNG 文件或包含 PNG 的目录")
    parser.add_argument("--model", help="模型权重路径，默认使用项目 outputs 中的最佳模型")
    parser.add_argument("--csv", type=Path, help="将批量识别结果另存为 CSV")
    parser.add_argument("--json", type=Path, help="将识别结果保存为 JSON")
    args = parser.parse_args()

    source = args.input.expanduser().resolve()
    if source.is_file():
        paths = [source]
    elif source.is_dir():
        paths = sorted(p for p in source.rglob("*") if p.is_file() and p.suffix.lower() == ".png")
    else:
        parser.error(f"输入路径不存在：{source}")
    if not paths:
        parser.error(f"输入位置没有 PNG 文件：{source}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, betas, size = load_model(resolve_checkpoint(args.model), device)
    results = []
    for i, path in enumerate(paths, start=1):
        try:
            result = predict_one(path, model, betas, size, device)
            results.append(result)
            print(json.dumps(result, ensure_ascii=False))
        except Exception as exc:
            print(json.dumps({"file": str(path), "error": str(exc)}, ensure_ascii=False))
        if len(paths) > 1 and (i % 500 == 0 or i == len(paths)):
            print(f"进度：{i}/{len(paths)}", flush=True)

    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(results, ensure_ascii=False, indent=2) + "\n")
    if args.csv:
        args.csv.parent.mkdir(parents=True, exist_ok=True)
        with args.csv.open("w", newline="", encoding="utf-8-sig") as f:
            writer = csv.DictWriter(f, fieldnames=["file", "beta_per_meter", "fog_level", "confidence"])
            writer.writeheader()
            writer.writerows({k: row.get(k, "") for k in writer.fieldnames} for row in results)
    print(f"识别完成：{len(results)}/{len(paths)} 张；运行设备：{device}")
    print("物理模型：t = exp(-beta * d)，其中 t 为透射率、d 为深度、beta 单位为 m^-1。")


if __name__ == "__main__":
    main()
