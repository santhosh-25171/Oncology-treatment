import os
import pytest
import pandas as pd
import json

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
STAGE_DIR = os.path.join(BASE_DIR, "stage3_nlp")
EVAL_ARTIFACTS = os.path.join(STAGE_DIR, "artifacts", "evaluation")

def test_classifier_evaluation_outputs_exist():
    assert os.path.exists(os.path.join(EVAL_ARTIFACTS, "classifier_metrics.json"))
    assert os.path.exists(os.path.join(EVAL_ARTIFACTS, "classifier_confusion_matrix.png"))
    assert os.path.exists(os.path.join(EVAL_ARTIFACTS, "classification_predictions.csv"))

def test_ner_evaluation_outputs_exist():
    assert os.path.exists(os.path.join(EVAL_ARTIFACTS, "ner_metrics.json"))
    assert os.path.exists(os.path.join(EVAL_ARTIFACTS, "ner_entity_metrics.csv"))
    assert os.path.exists(os.path.join(EVAL_ARTIFACTS, "ner_predictions.json"))

def test_audit_log_valid():
    log_path = os.path.join(EVAL_ARTIFACTS, "misinterpretation_audit_log.csv")
    assert os.path.exists(log_path)
    
    df = pd.read_csv(log_path)
    required_cols = ["audit_id", "record_id", "task", "text", "issue_type", "expected", "predicted", "severity", "resolution", "notes"]
    for col in required_cols:
        assert col in df.columns
        
def test_edge_cases_valid():
    edge_path = os.path.join(EVAL_ARTIFACTS, "edge_cases.csv")
    assert os.path.exists(edge_path)
    
    df = pd.read_csv(edge_path)
    required_cols = ["record_id", "text", "true_label", "predicted_label", "error_type", "confidence"]
    for col in required_cols:
        assert col in df.columns
