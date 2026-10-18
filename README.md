# Fog Visibility Project

Research materials for road-fog visibility estimation and evolution forecasting. Work is versioned in focused commits, with source reports, experiment artifacts, and repeatable code preserved by component.

## Existing research materials

| Component | Version / record | Purpose | Location |
|---|---|---|---|
| Integrated fog visibility study | Report revisions, Oct 2026 | Literature, methods, results, limitations | `paper/` |
| Transmittance Beta CNN | `v1.0.0` | Classify synthetic grayscale transmittance maps into three β classes | `src/transmittance_beta/` |
| CNN run artifacts | `v1.0.0`, original run 2026-10-04 | Preserve checkpoint, metrics, training history, and random-check outputs | `results/transmittance_beta/v1.0.0/` |
| Integrated model results | Report transcription | Machine-readable metrics for CNN, proxy forecasting, and AMOS experiments | `results/model_metrics.csv` |
| Real-image proxy observations | Report Appendix B transcription | Preserve the 100 reported derived observations | `results/real_image_proxy_observations.csv` |
| CNN image-feature + regression forecast workflow | Current scaffold | Proposed paired-data workflow; existing proxy test uses index order | `src/fogforecast/` |
| Reviewed HiMCM-format paper | Oct 2026 | English summary sheet, audited methods/results, references and AI-use report | `paper/Fog_Evolution_Forecasting_HiMCM_Reviewed.pdf` (`.docx` source) |

## Transmittance CNN result

The v1.0.0 checkpoint achieved 92.459% test accuracy (Macro F1 0.9248 in the integrated report) on 4,575 balanced synthetic transmittance maps. This is a synthetic-domain classification result; it is not real-road visibility-meter validation. The 30-image random check is a sanity check from the same domain, not an independent external test.

Use and retraining instructions are in `src/transmittance_beta/README.md`.

## Broader study: evidence limits

The archived integrated reports preserve earlier study versions for provenance; use `paper/Fog_Evolution_Forecasting_HiMCM_Reviewed.pdf` as the current reviewed paper. The report combines synthetic CNN calibration, unlabeled real-image features/TOPSIS, a 100-image proxy sequence, AMOS observations, and a highway frame sequence. Their evidence is not interchangeable. Proxy forecasting metrics are not physically calibrated road forecasts; AMOS cross-event results show substantial generalization loss, and persistence outperformed the tested temporal Ridge model at 5- and 15-minute horizons.

## Forecast workflow added in this revision

For a future paired dataset, the workflow can extract frozen ImageNet ResNet-18 embeddings from timestamped images, join these with visibility and meteorological observations, form 1–5 step lag features, and compare SVR, Random Forest, and XGBoost on a chronological 70/30 split. Standardizers are fit only on the training partition. The scripts can output held-out predictions, MSE/MAE/RMSE, descriptive statistics, forecast and error plots, and XGBoost feature importance. The available 100-row proxy table has also been evaluated with the included 70/30 config; this is an index-ordered exploratory benchmark, not a physically timed road forecast.

To prepare a new dataset, use `data_dictionary_template.csv` and edit `config/example.yaml`. Original timestamped images and synchronized visibility/meteorological observations are not included. The derived 100-row proxy feature table is included for the exploratory benchmark. Keep restricted or sensitive raw data outside Git. The reviewed paper states which observations support each result and reports the training-mean baseline, which outperforms all three fitted models on this proxy holdout. It is a research draft in HiMCM format, not a solution to an announced contest problem; replace the control-number placeholder if it is later used for an actual submission.

From the repository root, install and run:

```bash
python -m pip install -r requirements.txt
export PYTHONPATH=src
python -m fogforecast.extract_cnn_features --manifest data/processed/observations.csv --output outputs/cnn_features.csv --root .
python -m fogforecast.run_experiment --config config/example.yaml
```

The first image-feature run downloads pretrained ResNet-18 weights using torchvision. Keep generated outputs and model weights out of Git unless a specific artifact is intended for release.

## Dataset and licensing

The repository does not include Foggy Cityscapes or Cityscapes image data. Foggy Cityscapes' supplied terms prohibit dataset redistribution and restrict use to non-commercial purposes. The provided filename list was also excluded from this public repository. The trained checkpoint is included as a derivative model, consistent with the supplied terms; review `src/transmittance_beta/LICENSE.txt` and upstream documentation before use. The reports and project-authored code have no separate license grant in this repository.

## Versioned update convention

Each coherent update should be a separate commit. Use a component and version in the commit title, for example `feat(ml): add transmittance beta CNN v1.0.0`, and explain its purpose in that component's README. Keep raw datasets, secrets, and unrelated changes out of commits.
