# -*- coding: utf-8 -*-
"""create_github_sample.py

Deterministic generator that creates a lightweight representative sample of the full
DL oncology dataset for inclusion in the public GitHub repository.

* Source dataset: data/data/dl_oncology_dataset_v2/
* Output location: stage2_dl/sample_data/

The script:
  1. Reads the patient split file (train/validation/test).
  2. Loads the spatial label CSV and groups images by split.
  3. Selects whole patients (all their images) until reaching the target
     counts (≈70 % train, 15 % validation, 15 % test) for a total of 1 000 JPGs.
  4. Copies the selected images into stage2_dl/sample_data/images/<split>/
     preserving the original filenames.
  5. Writes image_labels_sample_1000.csv with updated image_path values.
  6. Samples exactly 10 000 temporal rows from biomarker_timeseries while
     preserving the train/validation/test split proportions.
  7. Filters progression_targets.csv and treatment_timeline.csv to the patients
     present in the sampled temporal rows.
  8. Generates a short README.md describing the sample dataset.

All operations are read‑only on the original dataset and use a fixed random
seed (42) so the output is fully reproducible.
"""

import os
import shutil
import random
import pandas as pd
from pathlib import Path

SEED = 42
SOURCE_ROOT = Path("data/data/dl_oncology_dataset_v2")
OUTPUT_ROOT = Path("stage2_dl/sample_data")
TARGET_IMAGE_COUNT = 1000
TARGET_TEMPORAL_ROWS = 10000
SPLIT_PROPORTIONS = {"train": 0.70, "validation": 0.15, "test": 0.15}

def load_split_file() -> dict:
    split_path = SOURCE_ROOT / "train_validation_test_split.csv"
    df = pd.read_csv(split_path)
    return dict(zip(df["patient_id"], df["split"]))

def load_image_labels() -> pd.DataFrame:
    path = SOURCE_ROOT / "spatial" / "labels.csv"
    return pd.read_csv(path)

def select_patients_for_images(labels_df: pd.DataFrame) -> dict:
    rnd = random.Random(SEED)
    patients_by_split = {"train": [], "validation": [], "test": []}
    for split in patients_by_split:
        patients = labels_df[labels_df["split"] == split]["patient_id"].unique()
        patients_by_split[split] = sorted(patients)
    selected = {"train": set(), "validation": set(), "test": set()}
    target_split = {s: int(round(TARGET_IMAGE_COUNT * p)) for s, p in SPLIT_PROPORTIONS.items()}
    diff = TARGET_IMAGE_COUNT - sum(target_split.values())
    target_split["train"] += diff
    counts = {"train": 0, "validation": 0, "test": 0}
    for split, patients in patients_by_split.items():
        shuffled = patients[:]
        rnd.shuffle(shuffled)
        for pid in shuffled:
            cnt = labels_df[labels_df["patient_id"] == pid].shape[0]
            if counts[split] + cnt > target_split[split] and counts[split] > 0:
                continue
            selected[split].add(pid)
            counts[split] += cnt
            if counts[split] >= target_split[split]:
                break
    total = sum(counts.values())
    if total != TARGET_IMAGE_COUNT:
        print(f"[WARN] Selected {total} images (target {TARGET_IMAGE_COUNT}).")
    return selected

def copy_selected_images(labels_df: pd.DataFrame, selected: dict) -> pd.DataFrame:
    rows = []
    for split, patients in selected.items():
        out_dir = OUTPUT_ROOT / "images" / split
        out_dir.mkdir(parents=True, exist_ok=True)
        subset = labels_df[labels_df["patient_id"].isin(patients) & (labels_df["split"] == split)]
        for _, row in subset.iterrows():
            src = SOURCE_ROOT / row["image_path"]
            dst = out_dir / Path(row["image_path"]).name
            shutil.copy2(src, dst)
            new_path = Path("stage2_dl/sample_data/images") / split / dst.name
            rows.append({
                "patient_id": row["patient_id"],
                "slide_id": row["slide_id"],
                "tile_id": row["tile_id"],
                "image_path": str(new_path),
                "tissue_class": row["tissue_class"],
                "label": row["label"],
                "split": split,
            })
    return pd.DataFrame(rows)

def create_image_labels_csv(df: pd.DataFrame):
    out = OUTPUT_ROOT / "image_labels_sample_1000.csv"
    df.to_csv(out, index=False)

def sample_temporal_rows(split_map: dict) -> pd.DataFrame:
    temporal_path = SOURCE_ROOT / "temporal" / "biomarker_timeseries.csv"
    df = pd.read_csv(temporal_path)
    df["split"] = df["patient_id"].map(split_map)
    rows_per_split = {s: int(round(TARGET_TEMPORAL_ROWS * p)) for s, p in SPLIT_PROPORTIONS.items()}
    diff = TARGET_TEMPORAL_ROWS - sum(rows_per_split.values())
    rows_per_split["train"] += diff
    rnd = random.Random(SEED)
    parts = []
    for split, n in rows_per_split.items():
        sub = df[df["split"] == split]
        if len(sub) < n:
            raise ValueError(f"Not enough temporal rows in split {split} ({len(sub)}) for {n} needed.")
        idx = list(sub.index)
        rnd.shuffle(idx)
        parts.append(sub.loc[idx[:n]])
    result = pd.concat(parts).drop(columns=["split"]).sort_index()
    return result

def filter_csv_by_patients(csv_path: Path, patients: set, out_path: Path):
    df = pd.read_csv(csv_path)
    df = df[df["patient_id"].isin(patients)]
    df.to_csv(out_path, index=False)

def generate_readme():
    readme_path = OUTPUT_ROOT / "README.md"
    content = """# Representative GitHub Sample Dataset\n\nThis directory contains a **lightweight, reproducible** subset of the full\nOncology‑treatment DL dataset. It is intended for demonstration, CI testing, and\neducational purposes only – **not** for any clinical or production use.\n\n## Contents\n```\nstage2_dl/sample_data/\n├─ images/\n│   ├─ train/\n│   ├─ validation/\n│   └─ test/\n├─ temporal/\n│   ├─ temporal_sample_10000.csv\n│   ├─ progression_targets_sample.csv\n│   └─ treatment_timeline_sample.csv\n├─ image_labels_sample_1000.csv\n└─ README.md\n```\n\n* **1 000** JPEG images (≈70 % train, 15 % validation, 15 % test)\n* **10 000** temporal observation rows (split proportionally across train/validation/test)\n* Patient‑level split is preserved for images; temporal rows respect the same\n  split but may belong to any patients within that split.\n* The sampling is **deterministic** (random seed = 42). Running the generator\n  again with the same source data will recreate the exact same files.\n\n## How it was generated\nThe script `stage2_dl/scripts/create_github_sample.py` performs the following steps:\n1. Reads `train_validation_test_split.csv` to know each patient’s split.\n2. Loads the full spatial label file and selects whole patients per split until\n   the target image count is reached, copying the JPEGs into this folder.\n3. Writes `image_labels_sample_1000.csv` with updated relative paths.\n4. Samples **exactly** 10 000 temporal rows from `biomarker_timeseries.csv`\n   while keeping the original train/validation/test proportions.\n5. Filters the auxiliary temporal files (`progression_targets.csv` and\n   `treatment_timeline.csv`) to the patients that appear in the sampled temporal\n   rows.\n\n## Regenerating the sample dataset\nIf you have the full dataset locally, you can recreate the sample by running:\n```bash\npython stage2_dl/scripts/create_github_sample.py\n```\nThe script is safe – it never modifies or deletes any files under the original\n`data/data/dl_oncology_dataset_v2/` directory.\n\n---\n*Generated on $(date)*\n"""
    readme_path.write_text(content, encoding="utf-8")

def main():
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    split_map = load_split_file()
    labels_df = load_image_labels()
    selected_patients = select_patients_for_images(labels_df)
    img_labels = copy_selected_images(labels_df, selected_patients)
    create_image_labels_csv(img_labels)
    temporal_df = sample_temporal_rows(split_map)
    temporal_dir = OUTPUT_ROOT / "temporal"
    temporal_dir.mkdir(parents=True, exist_ok=True)
    temporal_df.to_csv(temporal_dir / "temporal_sample_10000.csv", index=False)
    temporal_patients = set(temporal_df["patient_id"].unique())
    filter_csv_by_patients(SOURCE_ROOT / "temporal" / "progression_targets.csv",
                          temporal_patients,
                          temporal_dir / "progression_targets_sample.csv")
    filter_csv_by_patients(SOURCE_ROOT / "temporal" / "treatment_timeline.csv",
                          temporal_patients,
                          temporal_dir / "treatment_timeline_sample.csv")
    generate_readme()
    print("[INFO] Sample dataset generation complete.")

if __name__ == "__main__":
    main()

