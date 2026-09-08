import os
import pandas as pd
import re
import json

# NER Vocabularies
GENES = [
    "EGFR exon 19 deletion", "EGFR L858R", "KRAS G12C", "BRAF V600E",
    "ALK rearrangement", "ROS1 fusion", "HER2 amplification",
    "BRCA1 mutation", "BRCA2 mutation", "TP53 mutation", "PIK3CA mutation",
    "EGFR", "KRAS", "BRAF", "ALK", "ROS1", "HER2", "BRCA1", "BRCA2", "TP53", "PIK3CA"
]

DRUGS = [
    "cisplatin", "carboplatin", "paclitaxel", "docetaxel", "doxorubicin",
    "cyclophosphamide", "pembrolizumab", "nivolumab", "trastuzumab",
    "osimertinib", "erlotinib", "gefitinib"
]

AES = [
    "febrile neutropenia", "peripheral neuropathy", "abdominal pain",
    "nausea", "vomiting", "diarrhea", "fatigue", "fever", "rash",
    "neutropenia", "anemia", "thrombocytopenia", "neuropathy",
    "mucositis", "dyspnea", "pneumonitis"
]

def build_regex(terms):
    # Sort by length descending to match longer phrases first
    sorted_terms = sorted(terms, key=len, reverse=True)
    escaped = [re.escape(t) for t in sorted_terms]
    return r'\b(' + '|'.join(escaped) + r')\b'

GENE_REGEX = build_regex(GENES)
DRUG_REGEX = build_regex(DRUGS)
AE_REGEX = build_regex(AES)
# Match dosage patterns like "50 mg", "5 mg/kg", "50 mg/m2", "500 mg twice daily"
DOSAGE_REGEX = r'\b\d+(?:\.\d+)?\s*(?:mg|mg/kg|mg/m2)(?:\s*(?:twice daily|daily))?\b'

def extract_entities(text):
    entities = []
    
    if pd.isna(text) or not str(text).strip():
        return entities
        
    text = str(text)
    
    # helper to find and append
    def find_matches(pattern, label):
        for match in re.finditer(pattern, text, flags=re.IGNORECASE):
            entities.append({
                "start": match.start(),
                "end": match.end(),
                "label": label,
                "text": text[match.start():match.end()]
            })

    find_matches(GENE_REGEX, "GENE_MUTATION")
    find_matches(DRUG_REGEX, "DRUG_NAME")
    find_matches(DOSAGE_REGEX, "DOSAGE_LEVEL")
    find_matches(AE_REGEX, "ADVERSE_EVENT")
    
    # Sort entities by start index
    entities = sorted(entities, key=lambda x: x["start"])
    
    # Remove overlaps (keep the longer one if they overlap, or just the first matched)
    filtered = []
    for ent in entities:
        overlap = False
        for f in filtered:
            # check if ent overlaps with f
            if max(ent["start"], f["start"]) < min(ent["end"], f["end"]):
                overlap = True
                break
        if not overlap:
            filtered.append(ent)
            
    return filtered

def annotate_ner(df):
    results = []
    for _, row in df.iterrows():
        text = row['text']
        ents = extract_entities(text)
        results.append({
            "record_id": row['record_id'],
            "text": text,
            "entities": ents
        })
    return results

if __name__ == "__main__":
    CLEANED_CSV = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "cleaned", "clinical_text_cleaned.csv")
    OUT_JSON = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "ner", "ner_annotations.json")
    
    df = pd.read_csv(CLEANED_CSV, keep_default_na=False)
    df = df.drop_duplicates(subset=["record_id"])
    df_valid = df[df['text'].str.strip() != ""]
    
    ner_results = annotate_ner(df_valid)
    
    os.makedirs(os.path.dirname(OUT_JSON), exist_ok=True)
    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(ner_results, f, indent=2)
    
    print(f"NER annotation complete. Annotated: {len(ner_results)}")
