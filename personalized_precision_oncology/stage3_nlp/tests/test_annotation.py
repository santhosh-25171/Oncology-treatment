import sys
import os
import pytest
import pandas as pd
import json

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'annotation')))
from annotate_urgency import get_urgency_label
from annotate_ner import extract_entities

def test_urgency_labels():
    assert get_urgency_label("Patient developed severe nausea after chemotherapy.") == "HIGH"
    assert get_urgency_label("History of severe nausea, now resolved.") == "LOW"
    assert get_urgency_label("Patient denies severe nausea.") == "LOW"
    assert get_urgency_label("Moderate toxicity requiring intervention.") == "MODERATE"
    assert get_urgency_label("Patient is stable with no acute concerns.") == "LOW"

def test_ner_entities():
    text = "Patient developed severe nausea after receiving 50 mg cisplatin. EGFR L858R mutation detected."
    entities = extract_entities(text)
    
    labels = [e["label"] for e in entities]
    texts = [e["text"] for e in entities]
    
    assert "ADVERSE_EVENT" in labels
    assert "nausea" in texts
    
    assert "DOSAGE_LEVEL" in labels
    assert "50 mg" in texts
    
    assert "DRUG_NAME" in labels
    assert "cisplatin" in texts
    
    assert "GENE_MUTATION" in labels
    assert "EGFR L858R" in texts

def test_ner_offsets():
    text = "Started cisplatin 50 mg."
    entities = extract_entities(text)
    
    for ent in entities:
        assert text[ent["start"]:ent["end"]] == ent["text"]
        assert ent["start"] < ent["end"]

def test_empty_text():
    assert get_urgency_label("") is None
    assert get_urgency_label("   ") is None
    assert extract_entities("") == []
    assert extract_entities(None) == []

def test_valid_file_outputs():
    URG_CSV = os.path.join(os.path.dirname(__file__), "..", "data", "classification", "urgency_classification.csv")
    NER_JSON = os.path.join(os.path.dirname(__file__), "..", "data", "ner", "ner_annotations.json")
    
    assert os.path.exists(URG_CSV)
    assert os.path.exists(NER_JSON)
    
    df = pd.read_csv(URG_CSV)
    assert "urgency_label" in df.columns
    assert "record_id" in df.columns
    assert df["record_id"].is_unique
    
    with open(NER_JSON, "r", encoding="utf-8") as f:
        data = json.load(f)
        
    assert isinstance(data, list)
    if len(data) > 0:
        assert "record_id" in data[0]
        assert "entities" in data[0]
