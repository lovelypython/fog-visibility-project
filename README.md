# Fog Visibility Project

Research materials for road-fog visibility estimation and evolution forecasting. Repository updates are organized as focused commits with an explicit purpose and version where a component has a releaseable snapshot.

## Project components

| Component | Version / record | Purpose | Location |
|---|---|---|---|
| Integrated fog visibility study | Report revisions, Oct 2026 | Literature, methods, results, limitations | `paper/` |
| Transmittance Beta CNN | `v1.0.0` | Classify synthetic grayscale transmittance maps into three β classes | `src/transmittance_beta/` |
| CNN run artifacts | `v1.0.0`, original run 2026-10-04 | Preserve checkpoint, metrics, training history, and random-check outputs | `results/transmittance_beta/v1.0.0/` |
| Integrated model results | Report transcription | Machine-readable metrics for CNN, proxy forecasting, and AMOS experiments | `results/model_metrics.csv` |
| Real-image proxy observations | Report Appendix B transcription | Preserve the 100 reported derived observations | `results/real_image_proxy_observations.csv` |

## Transmittance CNN result

The v1.0.0 checkpoint achieved 92.459% test accuracy (Macro F1 0.9248 in the integrated report) on 4,575 balanced synthetic transmittance maps. This is a synthetic-domain classification result; it is not real-road visibility-meter validation. The 30-image random check is a sanity check from the same domain, not an independent external test.

Use and retraining instructions are in [`src/transmittance_beta/README.md`](src/transmittance_beta/README.md).

## Broader study: evidence limits

The integrated report combines synthetic CNN calibration, unlabeled real-image features/TOPSIS, a 100-image proxy sequence, AMOS observations, and a highway frame sequence. Their evidence is not interchangeable. Proxy forecasting metrics are not physically calibrated road forecasts; AMOS cross-event results show substantial generalization loss, and persistence outperformed the tested temporal Ridge model at 5- and 15-minute horizons.

## Dataset and licensing

The repository does not include Foggy Cityscapes or Cityscapes image data. Foggy Cityscapes' supplied terms prohibit dataset redistribution and restrict use to non-commercial purposes. The provided filename list was also excluded from this public repository. The trained checkpoint is included as a derivative model, consistent with the supplied terms; review `src/transmittance_beta/LICENSE.txt` and upstream documentation before use. The reports and project-authored code have no separate license grant in this repository.

## Versioned update convention

Each coherent update should be a separate commit. Use a component and version in the commit title, for example `feat(ml): add transmittance beta CNN v1.0.0`, and explain its purpose in that component's README. Keep raw datasets, secrets, and unrelated changes out of commits.
