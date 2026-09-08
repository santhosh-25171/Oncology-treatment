import os
import json
import pandas as pd

def validate_and_report():
    CLEANED_CSV = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "cleaned", "clinical_text_cleaned.csv")
    URGENCY_CSV = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "classification", "urgency_classification.csv")
    NER_JSON = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "ner", "ner_annotations.json")
    REPORT_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "docs", "annotation_report.md")

    # Load cleaned
    df_clean = pd.read_csv(CLEANED_CSV, keep_default_na=False)
    total_cleaned = len(df_clean)
    
    # Identify invalid/dirty rows to exclude
    # Exclude empty
    empty_count = (df_clean["text"].str.strip() == "").sum()
    
    # Exclude dupes
    dupes_count = df_clean["record_id"].duplicated().sum()
    
    df_valid = df_clean.drop_duplicates(subset=["record_id"])
    df_valid = df_valid[df_valid["text"].str.strip() != ""]
    valid_count = len(df_valid)

    # 1. Load Annotations
    df_urgency = pd.read_csv(URGENCY_CSV)
    with open(NER_JSON, "r", encoding="utf-8") as f:
        ner_data = json.load(f)
        
    # 2. Urgency Validation
    urg_valid = True
    urg_errors = []
    
    if len(df_urgency) != valid_count:
        urg_valid = False
        urg_errors.append(f"Urgency row count {len(df_urgency)} != valid count {valid_count}")
        
    invalid_labels = df_urgency[~df_urgency["urgency_label"].isin(["LOW", "MODERATE", "HIGH"])]
    if not invalid_labels.empty:
        urg_valid = False
        urg_errors.append("Found invalid urgency labels.")
        
    urgency_dist = df_urgency["urgency_label"].value_counts().to_dict()

    # 3. NER Validation
    ner_valid = True
    ner_errors = []
    
    if len(ner_data) != valid_count:
        ner_valid = False
        ner_errors.append(f"NER record count {len(ner_data)} != valid count {valid_count}")
        
    ner_counts = {"GENE_MUTATION": 0, "DRUG_NAME": 0, "DOSAGE_LEVEL": 0, "ADVERSE_EVENT": 0}
    invalid_spans = 0
    
    for rec in ner_data:
        text = rec["text"]
        for ent in rec["entities"]:
            label = ent["label"]
            start = ent["start"]
            end = ent["end"]
            ent_text = ent["text"]
            
            if label in ner_counts:
                ner_counts[label] += 1
            else:
                ner_valid = False
                ner_errors.append(f"Invalid label {label}")
                
            if start >= end or start < 0 or end > len(text):
                ner_valid = False
                invalid_spans += 1
                ner_errors.append("Invalid offsets")
                
            if text[start:end] != ent_text:
                ner_valid = False
                invalid_spans += 1
                ner_errors.append("Text mismatch at offsets")

    if invalid_spans > 0:
        ner_errors.append(f"Found {invalid_spans} invalid spans.")

    overall_valid = urg_valid and ner_valid
    
    # 4. Generate Report
    
    # Pick examples
    examples = []
    for i in range(min(5, len(ner_data))):
        rec_id = ner_data[i]["record_id"]
        urg_label = df_urgency[df_urgency["record_id"] == rec_id]["urgency_label"].values[0]
        examples.append({
            "text": ner_data[i]["text"],
            "urgency": urg_label,
            "entities": ner_data[i]["entities"]
        })

    report = f"""# Stage 03 NLP - Annotation Report

## Dataset
- **Records Processed (Cleaned Dataset):** {total_cleaned}
- **Successfully Annotated:** {valid_count}
- **Excluded Records:** {total_cleaned - valid_count} (Empty texts: {empty_count}, Duplicate IDs: {dupes_count})

## Urgency Distribution
- **LOW:** {urgency_dist.get('LOW', 0)} ({(urgency_dist.get('LOW', 0)/valid_count)*100:.1f}%)
- **MODERATE:** {urgency_dist.get('MODERATE', 0)} ({(urgency_dist.get('MODERATE', 0)/valid_count)*100:.1f}%)
- **HIGH:** {urgency_dist.get('HIGH', 0)} ({(urgency_dist.get('HIGH', 0)/valid_count)*100:.1f}%)

## NER Distribution
- **GENE_MUTATION:** {ner_counts['GENE_MUTATION']}
- **DRUG_NAME:** {ner_counts['DRUG_NAME']}
- **DOSAGE_LEVEL:** {ner_counts['DOSAGE_LEVEL']}
- **ADVERSE_EVENT:** {ner_counts['ADVERSE_EVENT']}

## Quality Checks
- **Invalid Spans:** {invalid_spans}
- **Annotation Validation Result:** {"PASS" if overall_valid else "FAIL"}

"""
    if not overall_valid:
        report += f"### Errors\nUrgency: {urg_errors}\nNER: {ner_errors}\n"
        
    report += "## Examples\n"
    for idx, ex in enumerate(examples, 1):
        report += f"### Example {idx}\n"
        report += f"**Text:** {ex['text']}\n\n"
        report += f"**Urgency:** {ex['urgency']}\n\n"
        report += "**Entities:**\n"
        for ent in ex['entities']:
            report += f"- [{ent['label']}] `{ent['text']}` (chars {ent['start']}:{ent['end']})\n"
        report += "\n---\n"

    os.makedirs(os.path.dirname(REPORT_PATH), exist_ok=True)
    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        f.write(report)
        
    print(f"Validation {'PASSED' if overall_valid else 'FAILED'}")
    print(f"Report generated at {REPORT_PATH}")

if __name__ == "__main__":
    validate_and_report()
