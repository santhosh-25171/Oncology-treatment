import os
import pickle
import spacy

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))

class NLPPipeline:
    def __init__(self):
        # Paths
        self.clf_path = os.path.join(BASE_DIR, "stage3_nlp", "models", "classification", "urgency_model.pkl")
        self.ner_path = os.path.join(BASE_DIR, "stage3_nlp", "models", "ner")
        
        # Models
        self.clf_pipeline = None
        self.ner_model = None
        
    def load_models(self):
        if os.path.exists(self.clf_path):
            with open(self.clf_path, "rb") as f:
                self.clf_pipeline = pickle.load(f)
        else:
            raise FileNotFoundError(f"Classification model not found at {self.clf_path}")
            
        if os.path.exists(self.ner_path):
            self.ner_model = spacy.load(self.ner_path)
        else:
            raise FileNotFoundError(f"NER model not found at {self.ner_path}")
            
    def predict(self, text: str):
        if not text or not text.strip():
            return {"urgency": "UNKNOWN", "entities": []}
            
        if self.clf_pipeline is None or self.ner_model is None:
            self.load_models()
            
        # Urgency Prediction
        urgency_pred = self.clf_pipeline.predict([text])[0]
        
        # Probabilities if requested
        # probs = self.clf_pipeline.predict_proba([text])[0]
        
        # NER Prediction
        doc = self.ner_model(text)
        entities = []
        for ent in doc.ents:
            entities.append({
                "text": ent.text,
                "label": ent.label_,
                "start": ent.start_char,
                "end": ent.end_char
            })
            
        return {
            "urgency": urgency_pred,
            "entities": entities
        }

if __name__ == "__main__":
    pipeline = NLPPipeline()
    pipeline.load_models()
    
    sample_text = "Patient developed severe nausea after receiving 50 mg cisplatin. EGFR L858R mutation detected."
    print("Testing Pipeline on sample text:")
    print(f"Text: {sample_text}")
    print("Result:")
    import json
    print(json.dumps(pipeline.predict(sample_text), indent=2))
