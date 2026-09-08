import os
import pytest
import pandas as pd
import json

BASE_DIR = os.path.dirname(os.path.dirname(__file__))

def test_inputs_exist():
    assert os.path.exists(os.path.join(BASE_DIR, "data", "cleaned", "clinical_text_cleaned.csv"))
    assert os.path.exists(os.path.join(BASE_DIR, "data", "classification", "urgency_classification.csv"))
    assert os.path.exists(os.path.join(BASE_DIR, "data", "ner", "ner_annotations.json"))

def test_eda_outputs_exist():
    # Tables
    tables_dir = os.path.join(BASE_DIR, "artifacts", "eda", "tables")
    assert os.path.exists(os.path.join(tables_dir, "urgency_distribution.csv"))
    assert os.path.exists(os.path.join(tables_dir, "text_statistics.csv"))
    assert os.path.exists(os.path.join(tables_dir, "vocabulary_by_urgency.csv"))
    assert os.path.exists(os.path.join(tables_dir, "ner_distribution.csv"))
    
    # Figures
    figures_dir = os.path.join(BASE_DIR, "artifacts", "eda", "figures")
    assert os.path.exists(os.path.join(figures_dir, "urgency_distribution.png"))
    assert os.path.exists(os.path.join(figures_dir, "text_wordcount_distribution.png"))
    assert os.path.exists(os.path.join(figures_dir, "text_length_by_urgency.png"))
    assert os.path.exists(os.path.join(figures_dir, "boxplot_wordcount_by_urgency.png"))
    
    # Report
    assert os.path.exists(os.path.join(BASE_DIR, "docs", "stage3_nlp_eda_report.md"))

def test_labels_are_valid():
    urgency_df = pd.read_csv(os.path.join(BASE_DIR, "data", "classification", "urgency_classification.csv"))
    valid_urg_labels = {"LOW", "MODERATE", "HIGH"}
    assert set(urgency_df["urgency_label"].unique()).issubset(valid_urg_labels)
    
    with open(os.path.join(BASE_DIR, "data", "ner", "ner_annotations.json"), "r") as f:
        ner_data = json.load(f)
    
    valid_ner_labels = {"GENE_MUTATION", "DRUG_NAME", "DOSAGE_LEVEL", "ADVERSE_EVENT"}
    for rec in ner_data:
        for ent in rec["entities"]:
            assert ent["label"] in valid_ner_labels

def test_stats_are_numeric():
    stats_df = pd.read_csv(os.path.join(BASE_DIR, "artifacts", "eda", "tables", "text_statistics.csv"))
    # The columns other than Metric should be numeric
    for col in ["Min", "Max", "Mean", "Median", "Std Dev"]:
        assert pd.api.types.is_numeric_dtype(stats_df[col])
