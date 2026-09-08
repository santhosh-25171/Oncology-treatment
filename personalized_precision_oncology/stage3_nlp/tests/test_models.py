import os
import sys
import pytest

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
sys.path.insert(0, os.path.join(BASE_DIR, "stage3_nlp", "models"))

from predict import NLPPipeline

@pytest.fixture(scope="module")
def nlp_pipeline():
    pipeline = NLPPipeline()
    pipeline.load_models()
    return pipeline

def test_models_exist(nlp_pipeline):
    assert nlp_pipeline.clf_pipeline is not None
    assert nlp_pipeline.ner_model is not None

def test_classification_valid(nlp_pipeline):
    text = "Routine visit. Patient is stable."
    res = nlp_pipeline.predict(text)
    
    assert "urgency" in res
    assert res["urgency"] in ["LOW", "MODERATE", "HIGH"]

def test_ner_valid(nlp_pipeline):
    text = "Started 50 mg cisplatin. Developed severe nausea."
    res = nlp_pipeline.predict(text)
    
    assert "entities" in res
    valid_labels = {"GENE_MUTATION", "DRUG_NAME", "DOSAGE_LEVEL", "ADVERSE_EVENT"}
    
    for ent in res["entities"]:
        assert ent["label"] in valid_labels
        assert "text" in ent
        assert "start" in ent
        assert "end" in ent
        assert ent["start"] < ent["end"]
        # Exact span string check
        assert text[ent["start"]:ent["end"]] == ent["text"]

def test_empty_text(nlp_pipeline):
    res = nlp_pipeline.predict("")
    assert res["urgency"] == "UNKNOWN"
    assert res["entities"] == []
    
    res2 = nlp_pipeline.predict("   ")
    assert res2["urgency"] == "UNKNOWN"
    assert res2["entities"] == []
