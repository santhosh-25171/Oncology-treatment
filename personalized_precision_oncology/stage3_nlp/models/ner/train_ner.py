import os
import json
import random
import spacy
from spacy.training.example import Example
import pandas as pd

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))

def load_ner_data(json_path, ids_path):
    with open(json_path, "r", encoding="utf-8") as f:
        all_data = json.load(f)
    
    ids_df = pd.read_csv(ids_path)
    valid_ids = set(ids_df["record_id"].tolist())
    
    dataset = []
    for rec in all_data:
        if rec["record_id"] in valid_ids:
            text = rec["text"]
            # Convert entities to format required by Spacy: (text, {"entities": [(start, end, label)]})
            ents = []
            for ent in rec["entities"]:
                # ensure start < end
                if ent["start"] < ent["end"]:
                    ents.append((ent["start"], ent["end"], ent["label"]))
            
            # handle overlaps by keeping the first/longest or just relying on existing clean annotations
            # Spacy doesn't allow overlapping entities.
            sorted_ents = sorted(ents, key=lambda x: (x[0], -x[1]))
            clean_ents = []
            last_end = -1
            for start, end, label in sorted_ents:
                if start >= last_end:
                    clean_ents.append((start, end, label))
                    last_end = end
                    
            dataset.append((text, {"entities": clean_ents}))
            
    return dataset

def train_ner():
    print("Loading NER datasets and splits...")
    NER_JSON = os.path.join(BASE_DIR, "data", "ner", "ner_annotations.json")
    TRAIN_IDS = os.path.join(BASE_DIR, "data", "splits", "train_ids.csv")
    
    train_data = load_ner_data(NER_JSON, TRAIN_IDS)
    print(f"Loaded {len(train_data)} training examples.")
    
    # We create a blank English model
    nlp = spacy.blank("en")
    
    # Add the NER pipeline component
    if "ner" not in nlp.pipe_names:
        ner = nlp.add_pipe("ner", last=True)
    else:
        ner = nlp.get_pipe("ner")
        
    # Add labels
    for _, annotations in train_data:
        for ent in annotations.get("entities"):
            ner.add_label(ent[2])
            
    print("Labels added:", ner.labels)
    
    # Disable other pipes (if any) during training
    other_pipes = [pipe for pipe in nlp.pipe_names if pipe != "ner"]
    with nlp.disable_pipes(*other_pipes):
        optimizer = nlp.begin_training()
        
        # Train for a few iterations (baseline)
        iterations = 1
        print(f"Training NER model for {iterations} iterations...")
        for itn in range(iterations):
            random.shuffle(train_data)
            losses = {}
            # Batch the training data
            batches = spacy.util.minibatch(train_data, size=spacy.util.compounding(4.0, 32.0, 1.001))
            for batch in batches:
                examples = []
                for text, annotations in batch:
                    doc = nlp.make_doc(text)
                    example = Example.from_dict(doc, annotations)
                    examples.append(example)
                nlp.update(examples, drop=0.5, sgd=optimizer, losses=losses)
            print(f"Iteration {itn + 1} Losses: {losses}")
            
    # Save the model
    model_dir = os.path.join(BASE_DIR, "models", "ner")
    os.makedirs(model_dir, exist_ok=True)
    
    nlp.to_disk(model_dir)
    print(f"NER model saved to {model_dir}")

if __name__ == "__main__":
    train_ner()
