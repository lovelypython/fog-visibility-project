# Transmittance Beta CNN — v1.0.0

Purpose: classify a Foggy Cityscapes **grayscale transmittance map** into the synthetic attenuation classes `β = 0.005`, `0.01`, or `0.02 m⁻¹`. This module estimates the filename-associated class. It does not infer metric depth or validate real-road meteorological optical range (MOR).

## Version record

- Version: `v1.0.0`
- Snapshot date: 2026-10-18
- Training run: five epochs; random seed 42; best validation checkpoint from epoch 5
- Purpose of this commit: add the training and inference code, trained checkpoint, metrics, training history, and 30-image random-check records as one independently reviewable ML update.
- Source of the run artifacts: project files supplied on 2026-10-18; no retraining was performed for this commit.

## Contents

- `train_transmittance_beta.py` — CNN training/evaluation script.
- `recognize_transmittance.py` — single-image and directory inference script.
- `LICENSE.txt` and `FOGGY_CITYSCAPES_README.md` — upstream dataset terms and documentation supplied with the project materials.
- `../../results/transmittance_beta/v1.0.0/` — checkpoint and records from this run.

## Reported result

The saved `metrics.json` reports 93.00% validation accuracy and 92.459% test accuracy on 4,575 balanced **synthetic** test maps (1,525 per class), with test loss 0.23637. The separate random check contains 30 images and is a sanity check, not independent real-world validation. See the integrated report in `paper/` for broader experiments and limitations.

## Install

Python 3.10+ is recommended. Install the required packages:

```bash
python -m pip install numpy Pillow torch
```

## Run inference

Input must be a grayscale `leftImg8bit_transmittance` PNG, not an ordinary RGB fog photograph:

```bash
python src/transmittance_beta/recognize_transmittance.py path/to/transmittance.png \
  --model results/transmittance_beta/v1.0.0/best_model.pt
```

Batch inference:

```bash
python src/transmittance_beta/recognize_transmittance.py path/to/transmittance_maps/ \
  --model results/transmittance_beta/v1.0.0/best_model.pt \
  --csv results/transmittance_beta/v1.0.0/predictions.csv \
  --json results/transmittance_beta/v1.0.0/predictions.json
```

## Retrain

Download Foggy Cityscapes transmittance maps under the dataset's terms. The training script expects the root to contain `leftImg8bit_transmittance/{train,val,test}`. The dataset itself is not included in this repository.

```bash
python src/transmittance_beta/train_transmittance_beta.py \
  --root /path/to/foggy-cityscapes \
  --epochs 5 --seed 42 \
  --output results/transmittance_beta/v1.0.0/retrained
```

The checkpoint supplied in this version is the original recorded run, not the output of this command.

## License and dataset handling

Foggy Cityscapes is supplied for non-commercial use under the included upstream terms. Those terms prohibit redistribution of the dataset. Therefore this repository includes neither the image maps nor the supplied dataset filename list. The upstream terms permit distribution of derivative works such as trained models; the included checkpoint is the project model trained on those maps. Cite both Foggy Cityscapes and Cityscapes as directed by their upstream documentation. Confirm applicable terms before any use or redistribution.
