import os
import pandas as pd
import json

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
STAGE_DIR = os.path.join(BASE_DIR, "stage3_nlp")

def edge_case_analysis():
    print("Generating Edge Case Analysis...")
    out_dir = os.path.join(STAGE_DIR, "artifacts", "evaluation")
    
    # Classification edge cases
    clf_preds = pd.read_csv(os.path.join(out_dir, "classification_predictions.csv"))
    
    # Filter incorrectly predicted
    clf_errors = clf_preds[~clf_preds["correct"]].copy()
    
    edge_cases = []
    
    for _, row in clf_errors.iterrows():
        edge_cases.append({
            "record_id": row["record_id"],
            "text": row["text"],
            "true_label": row["true_label"],
            "predicted_label": row["predicted_label"],
            "error_type": f"MISCLASSIFIED_{row['true_label']}_AS_{row['predicted_label']}",
            "confidence": row["confidence"]
        })
        
    edge_cases_df = pd.DataFrame(edge_cases)
    edge_cases_df.to_csv(os.path.join(out_dir, "edge_cases.csv"), index=False)
    
    print(f"Generated {len(edge_cases_df)} edge cases.")

if __name__ == "__main__":
    edge_case_analysis()
