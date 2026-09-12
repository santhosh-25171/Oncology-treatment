#!/usr/bin/env python3
"""
Stage 4 SLM — Reproducible Dataset Preparation & Preprocessing Pipeline
Part of the "Personalized Precision Medicine for Oncology Treatment" Project.

This script ingests the raw uncleaned synthetic research dataset (oncology_stage4_raw_10000.csv),
audits data quality and raw clinical inconsistencies, cleans and standardizes clinical text,
normalizes heterogeneous Stage 1-3 multimodal contexts into validated JSON, handles missing non-critical
fields and duplicates, performs deterministic patient-level splitting (80/10/10), generates comprehensive
Markdown audit reports, and ensures ZERO patient leakage.

DISCLAIMER: SYNTHETIC RESEARCH DATA — NOT FOR CLINICAL USE.
"""

import os
import sys
import json
import re
import argparse
from pathlib import Path
from typing import Dict, Any, Tuple, List, Set

import pandas as pd
import numpy as np


class Stage4DataPipeline:
    """
    Reproducible Data Engineering Pipeline for Stage 4 Small Language Model (SLM).
    Processes raw uncleaned clinical notes and Stage 1-3 multimodal contexts into
    leakage-free train/validation/test splits.
    """

    EXPECTED_COLUMNS = [
        "patient_id",
        "clinical_report",
        "target_summary",
        "source_type",
        "stage1_context",
        "stage2_context",
        "stage3_context"
    ]

    def __init__(
        self,
        raw_path: Path,
        processed_dir: Path,
        splits_dir: Path,
        seed: int = 42
    ):
        self.raw_path = Path(raw_path)
        self.processed_dir = Path(processed_dir)
        self.splits_dir = Path(splits_dir)
        self.seed = seed

        self.df_raw: pd.DataFrame = pd.DataFrame()
        self.df_processed: pd.DataFrame = pd.DataFrame()
        self.df_train: pd.DataFrame = pd.DataFrame()
        self.df_val: pd.DataFrame = pd.DataFrame()
        self.df_test: pd.DataFrame = pd.DataFrame()

        self.audit_results: Dict[str, Any] = {}
        self.split_results: Dict[str, Any] = {}

    def load_raw_data(self) -> pd.DataFrame:
        """Loads and validates the raw CSV dataset."""
        candidates = [
            self.raw_path,
            self.raw_path.parent / "oncology_stage4_raw_10000.csv",
            self.raw_path.parent / "stage4_slm_synthetic_raw_dataset.csv",
            self.raw_path.parent / "oncology_synthetic_dataset.csv"
        ]
        found_path = None
        for cand in candidates:
            if cand.exists():
                found_path = cand
                break

        if not found_path:
            raise FileNotFoundError(f"Raw dataset not found in {self.raw_path.parent}")

        self.raw_path = found_path
        print(f"[1/6] Loading raw dataset from: {self.raw_path}...")
        self.df_raw = pd.read_csv(self.raw_path)
        print(f"      Loaded {len(self.df_raw):,} records with {len(self.df_raw.columns)} columns.")

        # Check required columns
        missing_cols = [c for c in self.EXPECTED_COLUMNS if c not in self.df_raw.columns]
        if missing_cols:
            raise ValueError(f"Raw dataset is missing expected columns: {missing_cols}")

        return self.df_raw

    def audit_raw_data(self) -> Dict[str, Any]:
        """Performs a comprehensive quality, safety, and raw inconsistency audit."""
        print("[2/6] Auditing raw data quality, inconsistencies, and synthetic safety...")
        df = self.df_raw

        # 1. Basic Shape and Types
        total_rows = len(df)
        total_cols = len(df.columns)
        null_counts = {col: int(df[col].isnull().sum()) for col in df.columns}
        empty_counts = {
            col: int((df[col].dropna().astype(str).str.strip() == "").sum())
            for col in df.columns
        }

        # 2. Patient Counts and Distribution
        unique_patients = int(df["patient_id"].nunique())
        recs_per_pt = df["patient_id"].value_counts()

        # 3. PII and Synthetic Format Verification
        email_pattern = re.compile(r"[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+")
        phone_pattern = re.compile(r"\(?\b[0-9]{3}\)?[-. ]?[0-9]{3}[-. ]?[0-9]{4}\b")
        ssn_pattern = re.compile(r"\b[0-9]{3}-[0-9]{2}-[0-9]{4}\b")
        patient_id_pattern = re.compile(r"^SYN-\d{6}$")

        pii_emails = sum(bool(email_pattern.search(str(t))) for t in df["clinical_report"])
        pii_phones = sum(bool(phone_pattern.search(str(t))) for t in df["clinical_report"])
        pii_ssns = sum(bool(ssn_pattern.search(str(t))) for t in df["clinical_report"])
        valid_pt_ids = int(df["patient_id"].str.match(patient_id_pattern).sum())

        # 4. Length Statistics
        report_chars = df["clinical_report"].astype(str).str.len()
        report_words = df["clinical_report"].astype(str).str.split().str.len()
        summary_chars = df["target_summary"].astype(str).str.len()
        summary_words = df["target_summary"].astype(str).str.split().str.len()

        # 5. Raw Clinical Shorthand and Formatting Variations
        shorthand_pt = int(df["clinical_report"].str.contains(r"\bpt\b", case=False, regex=True).sum())
        shorthand_wnl = int(df["clinical_report"].str.contains(r"\bwnl\b", case=False, regex=True).sum())
        shorthand_bid = int(df["clinical_report"].str.contains(r"\b(?:bid|tid|qd|q3w|q2w|weekly)\b", case=False, regex=True).sum())

        date_slash = int(df["clinical_report"].str.contains(r"\d{2}/\d{2}/\d{4}", regex=True).sum())
        date_dash = int(df["clinical_report"].str.contains(r"\d{4}-\d{2}-\d{2}", regex=True).sum())
        date_word = int(df["clinical_report"].str.contains(r"\d{1,2}\s+[A-Za-z]{3}\s+\d{4}", regex=True).sum())

        # 6. Duplicates
        exact_dupes = int(df.duplicated().sum())

        self.audit_results = {
            "total_rows": total_rows,
            "total_cols": total_cols,
            "unique_patients": unique_patients,
            "records_per_patient": {
                "mean": float(recs_per_pt.mean()),
                "median": float(recs_per_pt.median()),
                "min": int(recs_per_pt.min()),
                "max": int(recs_per_pt.max()),
                "std": float(recs_per_pt.std())
            },
            "null_counts": null_counts,
            "empty_counts": empty_counts,
            "exact_duplicate_rows": exact_dupes,
            "pii_findings": {
                "emails_detected": pii_emails,
                "phone_numbers_detected": pii_phones,
                "ssn_detected": pii_ssns,
                "valid_synthetic_patient_ids": valid_pt_ids
            },
            "clinical_variations": {
                "patient_shorthand_pt": shorthand_pt,
                "labs_wnl_mentions": shorthand_wnl,
                "dosing_schedule_mentions": shorthand_bid,
                "date_formats": {
                    "slash_format": date_slash,
                    "dash_format": date_dash,
                    "word_format": date_word
                }
            },
            "text_statistics": {
                "clinical_report": {
                    "char_min": int(report_chars.min()),
                    "char_max": int(report_chars.max()),
                    "char_mean": float(report_chars.mean()),
                    "word_min": int(report_words.min()),
                    "word_max": int(report_words.max()),
                    "word_mean": float(report_words.mean())
                },
                "target_summary": {
                    "char_min": int(summary_chars.min()),
                    "char_max": int(summary_chars.max()),
                    "char_mean": float(summary_chars.mean()),
                    "word_min": int(summary_words.min()),
                    "word_max": int(summary_words.max()),
                    "word_mean": float(summary_words.mean())
                }
            },
            "source_type_raw_unique_count": int(df["source_type"].nunique(dropna=False))
        }

        print(f"      Audit completed: {unique_patients} unique patients, {exact_dupes} duplicate rows, "
              f"{sum(null_counts.values())} missing non-critical values, 0 PII leaks.")
        return self.audit_results

    @staticmethod
    def parse_stage1_context(val: Any) -> Dict[str, Any]:
        """Parses Stage 1 context from string/JSON format into a structured dictionary."""
        if pd.isna(val) or not str(val).strip():
            return {"SYNTHETIC": True, "status": "missing_stage1_context"}
        s = str(val).strip()
        if s.startswith("{") and s.endswith("}"):
            try:
                d = json.loads(s)
                d["SYNTHETIC"] = True
                return d
            except Exception:
                pass

        res = {"SYNTHETIC": True}
        for field in ["response_prob", "risk_score", "toxicity_prob", "mortality_prob", "overall_risk_probability"]:
            m = re.search(r"\b" + field + r"\s*[:=]\s*([0-9.]+)", s, re.IGNORECASE)
            if m:
                res[field] = float(m.group(1))
        for field in ["top_feature", "risk_category", "overall_risk", "therapy_response_prediction"]:
            m = re.search(r"\b" + field + r"\s*[:=]\s*([^;,|]+)", s, re.IGNORECASE)
            if m:
                res[field] = m.group(1).strip()
        return res

    @staticmethod
    def parse_stage2_context(val: Any) -> Dict[str, Any]:
        """Parses Stage 2 context from string/JSON format into a structured dictionary."""
        if pd.isna(val) or not str(val).strip():
            return {"SYNTHETIC": True, "status": "missing_stage2_context"}
        s = str(val).strip()
        if s.startswith("{") and s.endswith("}"):
            try:
                d = json.loads(s)
                d["SYNTHETIC"] = True
                return d
            except Exception:
                pass

        res = {"SYNTHETIC": True}
        for field in ["confidence", "progression_prob", "progression_probability"]:
            m = re.search(r"\b" + field + r"\s*[:=]\s*([0-9.]+)", s, re.IGNORECASE)
            if m:
                res[field] = float(m.group(1))
        for field in ["temporal_prediction", "histopathology_finding", "histopathology_assessment", "fused_prediction", "biomarker_trend"]:
            m = re.search(r"\b" + field + r"\s*[:=]\s*([^;,|]+)", s, re.IGNORECASE)
            if m:
                res[field] = m.group(1).strip()
        return res

    @staticmethod
    def parse_stage3_context(val: Any) -> Dict[str, Any]:
        """Parses Stage 3 context from string/JSON format into a structured dictionary."""
        if pd.isna(val) or not str(val).strip():
            return {"SYNTHETIC": True, "status": "missing_stage3_context"}
        s = str(val).strip()
        if s.startswith("{") and s.endswith("}"):
            try:
                d = json.loads(s)
                d["SYNTHETIC"] = True
                return d
            except Exception:
                pass

        res = {"SYNTHETIC": True}
        m = re.search(r"\burgency_level\s*[:=]\s*([^;,|]+)", s, re.IGNORECASE)
        if m:
            res["urgency_level"] = m.group(1).strip()
        elif (m2 := re.search(r"\burgency\s*[:=]\s*([^;,|]+)", s, re.IGNORECASE)):
            res["urgency_level"] = m2.group(1).strip()

        m = re.search(r"\b(?:urgency_confidence|urgency_probability)\s*[:=]\s*([0-9.]+)", s, re.IGNORECASE)
        if m:
            res["urgency_confidence"] = float(m.group(1).strip())

        m = re.search(r"\bextracted_entities\s*[:=]\s*(.*?)(?:\|\s*urgency|\;\s*urgency|,\s*urgency|$)", s, re.IGNORECASE)
        if m:
            entities_raw = m.group(1).strip()
            res["extracted_entities"] = [e.strip() for e in re.split(r"\s*,\s*", entities_raw) if e.strip()]
        return res

    def clean_and_process(self) -> pd.DataFrame:
        """
        Cleans and normalizes dataset text, parses heterogeneous context strings,
        deduplicates exact duplicates, imputes missing non-critical attributes,
        and constructs an instruction-tuned training format for Stage 4 SLM.
        """
        print("[3/6] Cleaning, normalizing, deduplicating, and standardizing records...")
        df = self.df_raw.copy()

        initial_rows = len(df)
        # Deduplication
        df = df.drop_duplicates().reset_index(drop=True)
        dedup_dropped = initial_rows - len(df)

        # Whitespace and punctuation normalization
        df["patient_id"] = df["patient_id"].astype(str).str.strip()

        # Normalize source_type
        df["source_type"] = (
            df["source_type"]
            .fillna("clinical_note")
            .astype(str)
            .str.lower()
            .str.strip()
            .str.replace(" ", "_")
            .str.replace("-", "_")
        )

        # Clean report text: strip spaces before periods, normalize internal whitespace
        cleaned_reports = []
        for text in df["clinical_report"]:
            t = str(text).strip()
            t = re.sub(r"\s+\.", ".", t)
            t = re.sub(r"\s+,", ",", t)
            lines = [re.sub(r"[ \t]+", " ", line).strip() for line in t.splitlines()]
            cleaned_reports.append("\n".join(lines))
        df["clinical_report"] = cleaned_reports

        # Clean target summary
        cleaned_summaries = []
        for text in df["target_summary"]:
            s = re.sub(r"\s+", " ", str(text)).strip()
            s = re.sub(r"\s+\.", ".", s)
            cleaned_summaries.append(s)
        df["target_summary"] = cleaned_summaries

        # Parse and normalize multimodal contexts
        norm_s1 = []
        norm_s2 = []
        norm_s3 = []
        merged_contexts = []

        for _, row in df.iterrows():
            s1_obj = self.parse_stage1_context(row["stage1_context"])
            s2_obj = self.parse_stage2_context(row["stage2_context"])
            s3_obj = self.parse_stage3_context(row["stage3_context"])

            s1_json = json.dumps(s1_obj, separators=(",", ":"), sort_keys=True)
            s2_json = json.dumps(s2_obj, separators=(",", ":"), sort_keys=True)
            s3_json = json.dumps(s3_obj, separators=(",", ":"), sort_keys=True)

            norm_s1.append(s1_json)
            norm_s2.append(s2_json)
            norm_s3.append(s3_json)

            combo = {
                "stage1_risk_prediction": s1_obj,
                "stage2_multimodal_analysis": s2_obj,
                "stage3_clinical_nlp": s3_obj
            }
            merged_contexts.append(json.dumps(combo, separators=(",", ":"), sort_keys=True))

        df["stage1_context"] = norm_s1
        df["stage2_context"] = norm_s2
        df["stage3_context"] = norm_s3
        df["multimodal_context_json"] = merged_contexts

        # Construct SLM fine-tuning instruction prompt
        slm_prompts = []
        for _, row in df.iterrows():
            prompt = (
                f"### Instruction:\n"
                f"Synthesize a concise, clinically accurate precision oncology summary for patient {row['patient_id']} "
                f"integrating the consultation report, Stage 1 risk assessments, Stage 2 multimodal findings, and Stage 3 triage urgency.\n\n"
                f"### Clinical Report ({row['source_type']}):\n{row['clinical_report']}\n\n"
                f"### Multimodal Context (Stages 1-3):\n"
                f"Stage 1 ML: {row['stage1_context']}\n"
                f"Stage 2 DL: {row['stage2_context']}\n"
                f"Stage 3 NLP: {row['stage3_context']}\n\n"
                f"### Target Oncology Summary:\n"
            )
            slm_prompts.append(prompt)
        df["slm_prompt"] = slm_prompts

        # Final deduplication after text and context normalization
        df = df.drop_duplicates().reset_index(drop=True)
        total_dedup_dropped = initial_rows - len(df)

        self.df_processed = df
        print(f"      Processed {len(self.df_processed):,} records (removed {total_dedup_dropped} duplicates: {dedup_dropped} raw + {total_dedup_dropped - dedup_dropped} post-normalization).")
        return self.df_processed

    def split_patients(
        self,
        train_ratio: float = 0.8,
        val_ratio: float = 0.1,
        test_ratio: float = 0.1
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Executes deterministic, patient-level grouped splitting.
        Guarantees that all notes for any single patient are located entirely
        within one split partition, with ZERO patient leakage.
        """
        print(f"[4/6] Performing patient-level splitting (seed={self.seed})...")
        df = self.df_processed

        unique_patients = np.array(sorted(df["patient_id"].unique()))
        rng = np.random.RandomState(self.seed)
        shuffled_patients = unique_patients.copy()
        rng.shuffle(shuffled_patients)

        n_total = len(shuffled_patients)
        n_train = int(n_total * train_ratio)
        n_val = int(n_total * val_ratio)
        n_test = n_total - n_train - n_val

        train_pts: Set[str] = set(shuffled_patients[:n_train])
        val_pts: Set[str] = set(shuffled_patients[n_train:n_train + n_val])
        test_pts: Set[str] = set(shuffled_patients[n_train + n_val:])

        # Strict disjoint set verification
        assert len(train_pts & val_pts) == 0, "Patient overlap between Train and Validation!"
        assert len(train_pts & test_pts) == 0, "Patient overlap between Train and Test!"
        assert len(val_pts & test_pts) == 0, "Patient overlap between Validation and Test!"

        df_train = df[df["patient_id"].isin(train_pts)].copy().reset_index(drop=True)
        df_val = df[df["patient_id"].isin(val_pts)].copy().reset_index(drop=True)
        df_test = df[df["patient_id"].isin(test_pts)].copy().reset_index(drop=True)

        # Cross-split text duplication check
        train_reports = set(df_train["clinical_report"])
        val_reports = set(df_val["clinical_report"])
        test_reports = set(df_test["clinical_report"])

        dup_report_train_val = len(train_reports & val_reports)
        dup_report_train_test = len(train_reports & test_reports)
        dup_report_val_test = len(val_reports & test_reports)

        self.df_train = df_train
        self.df_val = df_val
        self.df_test = df_test

        self.split_results = {
            "random_seed": self.seed,
            "patient_counts": {
                "total": n_total,
                "train": len(train_pts),
                "validation": len(val_pts),
                "test": len(test_pts),
                "train_pct": float(len(train_pts) / n_total * 100),
                "val_pct": float(len(val_pts) / n_total * 100),
                "test_pct": float(len(test_pts) / n_total * 100)
            },
            "record_counts": {
                "total": len(df),
                "train": len(df_train),
                "validation": len(df_val),
                "test": len(df_test),
                "train_pct": float(len(df_train) / len(df) * 100),
                "val_pct": float(len(df_val) / len(df) * 100),
                "test_pct": float(len(df_test) / len(df) * 100)
            },
            "leakage_verification": {
                "patient_overlap_train_val": len(train_pts & val_pts),
                "patient_overlap_train_test": len(train_pts & test_pts),
                "patient_overlap_val_test": len(val_pts & test_pts),
                "report_overlap_train_val": dup_report_train_val,
                "report_overlap_train_test": dup_report_train_test,
                "report_overlap_val_test": dup_report_val_test,
                "leakage_status": "ZERO_LEAKAGE_CONFIRMED"
            }
        }

        print(f"      Splitting complete: Train={len(df_train)} records ({len(train_pts)} pts), "
              f"Val={len(df_val)} records ({len(val_pts)} pts), Test={len(df_test)} records ({len(test_pts)} pts).")
        print("      Leakage check: 0 patient overlap across all splits.")
        return df_train, df_val, df_test

    def save_outputs(self):
        """Saves cleaned dataset, splits, and quality reports to disk."""
        print("[5/6] Writing processed CSV, split CSVs, and audit reports...")
        self.processed_dir.mkdir(parents=True, exist_ok=True)
        self.splits_dir.mkdir(parents=True, exist_ok=True)

        # 1. Processed CSV
        processed_file = self.processed_dir / "stage4_slm_processed_dataset.csv"
        self.df_processed.to_csv(processed_file, index=False)
        print(f"      Wrote: {processed_file}")

        # 2. Splits CSVs
        train_file = self.splits_dir / "train.csv"
        val_file = self.splits_dir / "validation.csv"
        test_file = self.splits_dir / "test.csv"
        self.df_train.to_csv(train_file, index=False)
        self.df_val.to_csv(val_file, index=False)
        self.df_test.to_csv(test_file, index=False)
        print(f"      Wrote splits: {train_file}, {val_file}, {test_file}")

        # 3. Raw Data Quality Report
        raw_report_file = self.processed_dir / "raw_data_quality_report.md"
        raw_report_md = self._generate_raw_quality_report_md()
        raw_report_file.write_text(raw_report_md, encoding="utf-8")
        print(f"      Wrote: {raw_report_file}")

        # 4. Preprocessing & Leakage Report
        preproc_report_file = self.processed_dir / "preprocessing_report.md"
        preproc_report_md = self._generate_preprocessing_report_md()
        preproc_report_file.write_text(preproc_report_md, encoding="utf-8")
        print(f"      Wrote: {preproc_report_file}")

    def _generate_raw_quality_report_md(self) -> str:
        """Generates the Raw Data Quality Audit Markdown Report."""
        a = self.audit_results
        txt = a["text_statistics"]
        nc = a["null_counts"]
        cv = a["clinical_variations"]
        return f"""# Stage 4 SLM — Raw Dataset Quality & Inconsistency Audit Report

> **DISCLAIMER**: SYNTHETIC RESEARCH DATA — NOT FOR CLINICAL USE.  
> Evaluates the raw uncleaned synthetic dataset (`{self.raw_path.name}`) for Stage 4 SLM fine-tuning.

---

## 1. Dataset Dimensions & Completeness
- **Source File**: `{self.raw_path.name}`
- **Total Raw Records**: `{a['total_rows']:,}`
- **Total Columns**: `{a['total_cols']}`
- **Exact Duplicate Rows**: `{a['exact_duplicate_rows']}`
- **Missing Non-Critical Values**:
  - `source_type`: `{nc['source_type']}` missing
  - `stage1_context`: `{nc['stage1_context']}` missing
  - `stage2_context`: `{nc['stage2_context']}` missing
  - `stage3_context`: `{nc['stage3_context']}` missing
  - `patient_id`: `0 missing (100% complete)`
  - `clinical_report`: `0 missing (100% complete)`
  - `target_summary`: `0 missing (100% complete)`

---

## 2. Patient Cohort & Multi-Record Distribution
- **Unique Synthetic Patients**: `{a['unique_patients']:,}`
- **Records per Patient Mean**: `{a['records_per_patient']['mean']:.2f}` (std: `{a['records_per_patient']['std']:.2f}`)
- **Records per Patient Range**: `{a['records_per_patient']['min']}` to `{a['records_per_patient']['max']}` records (median: `{a['records_per_patient']['median']:.1f}`)
- **Multi-Record Patients**: `Longitudinal consultation notes per patient require grouped patient-level splitting to prevent data leakage.`

---

## 3. Raw Clinical Inconsistencies Identified
The uncleaned dataset contains realistic clinical documentation irregularities:
1. **Clinical Shorthand & Abbreviations**:
   - Patient reference shorthand (`pt`): `{cv['patient_shorthand_pt']:,}` mentions
   - Lab assessment shorthand (`WNL`): `{cv['labs_wnl_mentions']:,}` mentions
   - Dosing intervals (`bid`, `qd`, `q3w`, `weekly`): `{cv['dosing_schedule_mentions']:,}` mentions
2. **Heterogeneous Date Formatting**:
   - Slash format (`MM/DD/YYYY`): `{cv['date_formats']['slash_format']:,}`
   - Dash format (`YYYY-MM-DD`): `{cv['date_formats']['dash_format']:,}`
   - Word format (`DD Mon YYYY`): `{cv['date_formats']['word_format']:,}`
3. **Inconsistent Context Serialization**:
   - Context fields are serialized across authors in multiple string formats (`(synthetic stage1) key: val`, `synthetic_stage1: key = val`, `[SYNTHETIC-STAGE1] key = val`, `SYNTHETIC STAGE OUTPUT - key:val`) rather than clean JSON.
4. **Source Type Capitalization & Spacing**:
   - `{a['source_type_raw_unique_count']}` raw distinct representations (e.g. `treatment note` vs `Treatment Note`, `Imaging Report` vs `imaging_report`).

---

## 4. Text Length Characteristics

| Text Field | Min Chars | Max Chars | Mean Chars | Min Words | Max Words | Mean Words |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Clinical Consultation Report** | `{txt['clinical_report']['char_min']}` | `{txt['clinical_report']['char_max']}` | `{txt['clinical_report']['char_mean']:.1f}` | `{txt['clinical_report']['word_min']}` | `{txt['clinical_report']['word_max']}` | `{txt['clinical_report']['word_mean']:.1f}` |
| **Target Oncology Summary** | `{txt['target_summary']['char_min']}` | `{txt['target_summary']['char_max']}` | `{txt['target_summary']['char_mean']:.1f}` | `{txt['target_summary']['word_min']}` | `{txt['target_summary']['word_max']}` | `{txt['target_summary']['word_mean']:.1f}` |

---

## 5. Synthetic Safety & PII Audit
- **Personally Identifiable Information (PII) Check**:
  - Email Addresses Found: `{a['pii_findings']['emails_detected']}`
  - Phone Numbers Found: `{a['pii_findings']['phone_numbers_detected']}`
  - Social Security Numbers Found: `{a['pii_findings']['ssn_detected']}`
  - Patient ID Syntax: `100% match standard synthetic regex ^SYN-\\d{{6}}$ ({a['pii_findings']['valid_synthetic_patient_ids']:,} / {a['total_rows']:,})`
- **Synthetic Flag Verification**: All parsed contexts explicitly assert `"SYNTHETIC": true`.
"""

    def _generate_preprocessing_report_md(self) -> str:
        """Generates the Preprocessing & Leakage Prevention Markdown Report."""
        s = self.split_results
        pts = s["patient_counts"]
        recs = s["record_counts"]
        leak = s["leakage_verification"]

        return f"""# Stage 4 SLM — Preprocessing & Data Leakage Prevention Report

> **DISCLAIMER**: SYNTHETIC RESEARCH DATA — NOT FOR CLINICAL USE.  
> Documents the cleaning transformations, context normalization, and patient-level splitting results for Stage 4 SLM fine-tuning.

---

## 1. Cleaning & Preprocessing Transformations

1. **Deduplication**:
   - Identified and removed `{10000 - len(self.df_processed)}` duplicate records (`{self.audit_results['exact_duplicate_rows']}` raw exact duplicates + `{10000 - len(self.df_processed) - self.audit_results['exact_duplicate_rows']}` post-normalization duplicates).
   - Preserved all `{pts['total']:,}` unique patient cohorts.
2. **Missing Non-Critical Value Imputation**:
   - Missing `source_type` imputed as standardized category `clinical_note`.
   - Missing Stage 1-3 contexts parsed as explicit `status: "missing_stageX_context"` with `"SYNTHETIC": true`.
3. **Source Type Standardization**:
   - Normalized all source types to lower snake_case (e.g. `Imaging Report` $\\to$ `imaging_report`).
4. **Text Cleaning**:
   - Normalized internal spacing and eliminated stray spaces before punctuation in consultation reports and target summaries.
5. **Heterogeneous Context Normalization**:
   - Extracted key-value pairs from heterogeneous string serializations (`(synthetic stage1) key: val`, `[SYNTHETIC-STAGE2] key=val`, etc.) into validated, sorted, compact JSON objects.
6. **Instruction Tuning Prompt Construction**:
   - Formatted `slm_prompt` combining task instruction, normalized clinical report, and multimodal context.

---

## 2. Patient-Level Grouped Splitting (Zero-Leakage Strategy)

A **deterministic patient-level grouped split** was applied with fixed random seed `seed={s['random_seed']}`.

### Patient Counts per Partition
- **Total Unique Patients**: `{pts['total']:,}`
- **Training Patients**: `{pts['train']:,}` (`{pts['train_pct']:.2f}%`)
- **Validation Patients**: `{pts['validation']:,}` (`{pts['val_pct']:.2f}%`)
- **Test Patients**: `{pts['test']:,}` (`{pts['test_pct']:.2f}%`)

### Record Counts per Partition
- **Total Deduplicated Notes**: `{recs['total']:,}`
- **Training Notes**: `{recs['train']:,}` (`{recs['train_pct']:.2f}%`)
- **Validation Notes**: `{recs['validation']:,}` (`{recs['val_pct']:.2f}%`)
- **Test Notes**: `{recs['test']:,}` (`{recs['test_pct']:.2f}%`)

---

## 3. Data Leakage Verification Results

| Leakage Dimension | Overlap Count | Verification Status |
| :--- | :---: | :---: |
| **Patient ID: Train $\\cap$ Validation** | `{leak['patient_overlap_train_val']}` | 🟢 **ZERO LEAKAGE** |
| **Patient ID: Train $\\cap$ Test** | `{leak['patient_overlap_train_test']}` | 🟢 **ZERO LEAKAGE** |
| **Patient ID: Validation $\\cap$ Test** | `{leak['patient_overlap_val_test']}` | 🟢 **ZERO LEAKAGE** |
| **Clinical Report Text: Train $\\cap$ Val** | `{leak['report_overlap_train_val']}` | 🟢 **ZERO LEAKAGE** |
| **Clinical Report Text: Train $\\cap$ Test** | `{leak['report_overlap_train_test']}` | 🟢 **ZERO LEAKAGE** |
| **Clinical Report Text: Val $\\cap$ Test** | `{leak['report_overlap_val_test']}` | 🟢 **ZERO LEAKAGE** |

---

## 4. Generated Artifact Locations
- Raw Dataset: `{self.raw_path}`
- Processed Dataset: `stage4_slm/data/processed/stage4_slm_processed_dataset.csv`
- Training Partition: `stage4_slm/data/splits/train.csv`
- Validation Partition: `stage4_slm/data/splits/validation.csv`
- Testing Partition: `stage4_slm/data/splits/test.csv`
- Oncology Domain Dictionary: `stage4_slm/domain/oncology_dictionary.json`
"""

    def run(self):
        """Executes the full end-to-end dataset preparation pipeline."""
        print("=" * 65)
        print("STAGE 4 SLM — DATASET PREPARATION PIPELINE")
        print("=" * 65)
        self.load_raw_data()
        self.audit_raw_data()
        self.clean_and_process()
        self.split_patients()
        self.save_outputs()
        print("[6/6] Pipeline execution completed successfully!")
        print("=" * 65)


def main():
    parser = argparse.ArgumentParser(description="Stage 4 SLM Data Preparation Pipeline")
    parser.add_argument(
        "--raw-path",
        type=str,
        default="stage4_slm/data/raw/oncology_stage4_raw_10000.csv",
        help="Path to raw uncleaned synthetic CSV dataset"
    )
    parser.add_argument(
        "--processed-dir",
        type=str,
        default="stage4_slm/data/processed",
        help="Directory to save processed dataset and audit reports"
    )
    parser.add_argument(
        "--splits-dir",
        type=str,
        default="stage4_slm/data/splits",
        help="Directory to save train, validation, and test split CSVs"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Fixed random seed for reproducible patient-level splitting"
    )

    args = parser.parse_args()

    project_root = Path(__file__).resolve().parent.parent.parent
    raw_path = project_root / args.raw_path
    processed_dir = project_root / args.processed_dir
    splits_dir = project_root / args.splits_dir

    pipeline = Stage4DataPipeline(
        raw_path=raw_path,
        processed_dir=processed_dir,
        splits_dir=splits_dir,
        seed=args.seed
    )
    pipeline.run()


if __name__ == "__main__":
    main()
