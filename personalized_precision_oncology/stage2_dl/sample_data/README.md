# Representative GitHub Sample Dataset

This directory contains a **lightweight, reproducible** subset of the full
Oncology‑treatment DL dataset. It is intended for demonstration, CI testing, and
educational purposes only – **not** for any clinical or production use.

## Contents
```
stage2_dl/sample_data/
├─ images/
│   ├─ train/
│   ├─ validation/
│   └─ test/
├─ temporal/
│   ├─ temporal_sample_10000.csv
│   ├─ progression_targets_sample.csv
│   └─ treatment_timeline_sample.csv
├─ image_labels_sample_1000.csv
└─ README.md
```

* **1 000** JPEG images (≈70 % train, 15 % validation, 15 % test)
* **10 000** temporal observation rows (split proportionally across train/validation/test)
* Patient‑level split is preserved for images; temporal rows respect the same
  split but may belong to any patients within that split.
* The sampling is **deterministic** (random seed = 42). Running the generator
  again with the same source data will recreate the exact same files.

## How it was generated
The script `stage2_dl/scripts/create_github_sample.py` performs the following steps:
1. Reads `train_validation_test_split.csv` to know each patient’s split.
2. Loads the full spatial label file and selects whole patients per split until
   the target image count is reached, copying the JPEGs into this folder.
3. Writes `image_labels_sample_1000.csv` with updated relative paths.
4. Samples **exactly** 10 000 temporal rows from `biomarker_timeseries.csv`
   while keeping the original train/validation/test proportions.
5. Filters the auxiliary temporal files (`progression_targets.csv` and
   `treatment_timeline.csv`) to the patients that appear in the sampled temporal
   rows.

## Regenerating the sample dataset
If you have the full dataset locally, you can recreate the sample by running:
```bash
python stage2_dl/scripts/create_github_sample.py
```
The script is safe – it never modifies or deletes any files under the original
`data/data/dl_oncology_dataset_v2/` directory.

---
*Generated on $(date)*
