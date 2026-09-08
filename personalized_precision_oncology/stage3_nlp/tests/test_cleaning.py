import sys
import os
import pytest
import pandas as pd

# Add preprocessing to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'preprocessing')))

from clean_text import clean_clinical_text
from sanitize_data import validate_raw_data

def test_missing_value_handling():
    assert clean_clinical_text(None) == ""
    assert clean_clinical_text(float('nan')) == ""

def test_whitespace_cleaning():
    messy_text = "   Patient   denies\t\t\n  nausea.   "
    expected = "Patient denies nausea."
    assert clean_clinical_text(messy_text) == expected

def test_negation_preservation():
    text = "Patient denies nausea. No evidence of fever."
    cleaned = clean_clinical_text(text)
    assert "denies" in cleaned
    assert "No evidence" in cleaned

def test_medical_term_preservation():
    text = "Patient presents with NSCLC and ECOG 1."
    cleaned = clean_clinical_text(text)
    assert "NSCLC" in cleaned
    assert "ECOG 1" in cleaned

def test_drug_preservation():
    text = "Started on osimertinib and cisplatin."
    cleaned = clean_clinical_text(text)
    assert "osimertinib" in cleaned
    assert "cisplatin" in cleaned

def test_gene_preservation():
    text = "Testing confirms EGFR L858R and BRCA1 mutation."
    cleaned = clean_clinical_text(text)
    assert "EGFR L858R" in cleaned
    assert "BRCA1" in cleaned

def test_dosage_preservation():
    text = "Dose: 50 mg/m2 or 5 mg/kg."
    cleaned = clean_clinical_text(text)
    assert "50 mg/m2" in cleaned
    assert "5 mg/kg" in cleaned

def test_adverse_event_preservation():
    text = "Reported severe neuropathy and grade 2 neutropenia."
    cleaned = clean_clinical_text(text)
    assert "severe neuropathy" in cleaned
    assert "grade 2 neutropenia" in cleaned

def test_required_columns():
    df = pd.DataFrame({
        "record_id": [1], "patient_id": [2], "note_type": ["a"],
        "timestamp": ["b"], "department": ["c"], "text": ["d"],
        "source": ["e"], "language": ["en"]
    })
    stats = validate_raw_data(df)
    assert len(stats["missing_columns"]) == 0

def test_duplicate_record_ids():
    df = pd.DataFrame({
        "record_id": [1, 1],
        "text": ["A", "B"]
    })
    stats = validate_raw_data(df)
    assert stats["duplicate_record_ids"] == 1
