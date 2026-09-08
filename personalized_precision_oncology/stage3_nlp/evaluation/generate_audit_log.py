import os
import pandas as pd
import json

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
STAGE_DIR = os.path.join(BASE_DIR, "stage3_nlp")

def generate_audit_log():
    print("Generating Misinterpretation Audit Log...")
    out_dir = os.path.join(STAGE_DIR, "artifacts", "evaluation")
    
    audit_log = []
    audit_id_counter = 1
    
    # 1. Classification Errors
    clf_preds = pd.read_csv(os.path.join(out_dir, "classification_predictions.csv"))
    clf_errors = clf_preds[~clf_preds["correct"]].copy()
    
    for _, row in clf_errors.iterrows():
        audit_log.append({
            "audit_id": f"AUD_{audit_id_counter:05d}",
            "record_id": row["record_id"],
            "task": "URGENCY_CLASSIFICATION",
            "text": row["text"],
            "issue_type": "MISCLASSIFICATION",
            "expected": row["true_label"],
            "predicted": row["predicted_label"],
            "severity": "HIGH" if (row["true_label"] == "HIGH" and row["predicted_label"] == "LOW") else "MEDIUM",
            "resolution": "Error identified during held-out evaluation.",
            "notes": "Classical model missed context."
        })
        audit_id_counter += 1
        
    # 2. NER Errors
    with open(os.path.join(out_dir, "ner_predictions.json"), "r") as f:
        ner_preds = json.load(f)
        
    for rec in ner_preds:
        gt_ents = rec["ground_truth"]
        pred_ents = rec["predicted"]
        
        gt_tuples = [(ent["start"], ent["end"], ent["label"]) for ent in gt_ents]
        pred_tuples = [(ent["start"], ent["end"], ent["label"]) for ent in pred_ents]
        
        # We find errors
        # To identify WRONG_ENTITY_TYPE or BOUNDARY_ERROR, we check overlaps
        matched_gt = set()
        matched_pred = set()
        
        for i, pt in enumerate(pred_tuples):
            if pt in gt_tuples:
                matched_gt.add(gt_tuples.index(pt))
                matched_pred.add(i)
                continue
                
            # If not exact match, check why
            overlap_found = False
            for j, gt in enumerate(gt_tuples):
                if j in matched_gt: continue
                # Check overlap: max(start1, start2) < min(end1, end2)
                if max(pt[0], gt[0]) < min(pt[1], gt[1]):
                    overlap_found = True
                    if pt[0] == gt[0] and pt[1] == gt[1] and pt[2] != gt[2]:
                        issue = "WRONG_ENTITY_TYPE"
                    else:
                        issue = "BOUNDARY_ERROR"
                        
                    audit_log.append({
                        "audit_id": f"AUD_{audit_id_counter:05d}",
                        "record_id": rec["record_id"],
                        "task": "NER",
                        "text": rec["predicted"][i]["text"] + " vs " + gt_ents[j]["text"],
                        "issue_type": issue,
                        "expected": gt[2],
                        "predicted": pt[2],
                        "severity": "LOW",
                        "resolution": "Entity mismatch identified during held-out evaluation.",
                        "notes": ""
                    })
                    audit_id_counter += 1
                    matched_gt.add(j)
                    matched_pred.add(i)
                    break
                    
            if not overlap_found:
                audit_log.append({
                    "audit_id": f"AUD_{audit_id_counter:05d}",
                    "record_id": rec["record_id"],
                    "task": "NER",
                    "text": rec["predicted"][i]["text"],
                    "issue_type": "FALSE_POSITIVE",
                    "expected": "NONE",
                    "predicted": pt[2],
                    "severity": "LOW",
                    "resolution": "False positive identified.",
                    "notes": ""
                })
                audit_id_counter += 1
                
        # Any remaining GT are false negatives
        for j, gt in enumerate(gt_tuples):
            if j not in matched_gt:
                audit_log.append({
                    "audit_id": f"AUD_{audit_id_counter:05d}",
                    "record_id": rec["record_id"],
                    "task": "NER",
                    "text": gt_ents[j]["text"],
                    "issue_type": "FALSE_NEGATIVE",
                    "expected": gt[2],
                    "predicted": "NONE",
                    "severity": "MEDIUM",
                    "resolution": "False negative identified.",
                    "notes": ""
                })
                audit_id_counter += 1
                
    audit_df = pd.DataFrame(audit_log)
    audit_df.to_csv(os.path.join(out_dir, "misinterpretation_audit_log.csv"), index=False)
    print(f"Generated {len(audit_df)} audit log entries.")

if __name__ == "__main__":
    generate_audit_log()
