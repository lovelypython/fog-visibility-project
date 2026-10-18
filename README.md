# Fog Visibility Project — research archive v1.0.0

## Version and purpose

- Version: `1.0.0` (research-materials archive)
- Snapshot: 2026-10-18
- Purpose of this update: place the fog visibility papers, literature review, available inference entry point, and report-derived result tables under Git version control as the initial organized project snapshot.
- Scope note: this commit records the research-document snapshot. The transmittance CNN training code, checkpoint, and run files are intentionally added in the next, separate ML commit.

## Contents

- `paper/` — early HiMCM report, October integrated/revised reports, literature review, and project-method explanation.
- `src/recognize_transmittance.py` — inference entry point available at the time of this snapshot; it depends on the CNN training module and checkpoint that are added in the subsequent ML update.
- `results/model_metrics.csv` — reported metrics transcribed from the integrated report.
- `results/real_image_proxy_observations.csv` — the 100 derived proxy observations transcribed from Appendix B; these are not raw meter observations.

## Evidence limits

CNN results in the reports are from synthetic transmittance images. The real-image and highway sequences are proxy evidence without synchronized meter calibration. AMOS results show poor cross-event generalization; persistence outperformed the tested temporal Ridge forecasts. See the integrated report before interpreting the results.

## Data and reuse

No Foggy Cityscapes or Cityscapes image data is included. Their supplied terms prohibit dataset redistribution and restrict use to non-commercial purposes. This archive does not assert a license for the project reports or project-authored code.
