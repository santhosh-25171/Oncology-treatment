import os
import json
import pandas as pd
import spacy

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
STAGE_DIR = os.path.join(BASE_DIR, "stage3_nlp")

def evaluate_ner():
    print("Evaluating Medical NER Pipeline...")
    # Paths
    test_ids_path = os.path.join(STAGE_DIR, "data", "splits", "test_ids.csv")
    ner_path = os.path.join(STAGE_DIR, "data", "ner", "ner_annotations.json")
    model_path = os.path.join(STAGE_DIR, "models", "ner")
    out_dir = os.path.join(STAGE_DIR, "artifacts", "evaluation")
    os.makedirs(out_dir, exist_ok=True)
    
    # Load data
    test_ids = pd.read_csv(test_ids_path)
    valid_ids = set(test_ids["record_id"].tolist())
    
    with open(ner_path, "r", encoding="utf-8") as f:
        all_ner = json.load(f)
        
    test_records = [r for r in all_ner if r["record_id"] in valid_ids]
    
    # Load model
    nlp = spacy.load(model_path)
    
    labels = ["GENE_MUTATION", "DRUG_NAME", "DOSAGE_LEVEL", "ADVERSE_EVENT"]
    
    # Metrics counters
    tp = {l: 0 for l in labels}
    fp = {l: 0 for l in labels}
    fn = {l: 0 for l in labels}
    
    predictions_output = []
    
    for rec in test_records:
        text = rec["text"]
        gt_ents = rec["entities"]
        
        doc = nlp(text)
        pred_ents = [{"text": ent.text, "label": ent.label_, "start": ent.start_char, "end": ent.end_char} for ent in doc.ents]
        
        # We need exact matches for TP
        gt_tuples = [(ent["start"], ent["end"], ent["label"]) for ent in gt_ents]
        pred_tuples = [(ent["start"], ent["end"], ent["label"]) for ent in pred_ents]
        
        gt_matched = set()
        pred_matched = set()
        
        for i, pt in enumerate(pred_tuples):
            if pt in gt_tuples:
                # True Positive
                tp[pt[2]] += 1
                gt_matched.add(gt_tuples.index(pt))
                pred_matched.add(i)
            else:
                # False Positive
                # Check if label is valid, else ignore for metric if model predicted wrong label not in our set
                if pt[2] in fp:
                    fp[pt[2]] += 1
                    
        for j, gt in enumerate(gt_tuples):
            if j not in gt_matched:
                if gt[2] in fn:
                    fn[gt[2]] += 1
        
        predictions_output.append({
            "record_id": rec["record_id"],
            "ground_truth": gt_ents,
            "predicted": pred_ents
        })
        
    # Save predictions
    preds_out = os.path.join(out_dir, "ner_predictions.json")
    with open(preds_out, "w") as f:
        json.dump(predictions_output, f, indent=4)
        
    # Calculate metrics
    entity_metrics = []
    
    total_tp = 0
    total_fp = 0
    total_fn = 0
    
    for l in labels:
        p = tp[l] / (tp[l] + fp[l]) if (tp[l] + fp[l]) > 0 else 0.0
        r = tp[l] / (tp[l] + fn[l]) if (tp[l] + fn[l]) > 0 else 0.0
        f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
        support = tp[l] + fn[l]
        
        entity_metrics.append({
            "Entity": l,
            "Precision": p,
            "Recall": r,
            "F1": f1,
            "Support": support
        })
        
        total_tp += tp[l]
        total_fp += fp[l]
        total_fn += fn[l]
        
    # Overall
    overall_p = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
    overall_r = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
    overall_f1 = 2 * overall_p * overall_r / (overall_p + overall_r) if (overall_p + overall_r) > 0 else 0.0
    
    em_df = pd.DataFrame(entity_metrics)
    em_df.to_csv(os.path.join(out_dir, "ner_entity_metrics.csv"), index=False)
    
    overall_metrics = {
        "overall_precision": overall_p,
        "overall_recall": overall_r,
        "overall_f1": overall_f1
    }
    
    with open(os.path.join(out_dir, "ner_metrics.json"), "w") as f:
        json.dump(overall_metrics, f, indent=4)

    print(f"NER Evaluation complete. Evaluated {len(test_records)} records.")

if __name__ == "__main__":
    evaluate_ner()
