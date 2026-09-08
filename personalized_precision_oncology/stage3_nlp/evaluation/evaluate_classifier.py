import os
import json
import pandas as pd
import pickle
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_recall_fscore_support
import matplotlib.pyplot as plt
import seaborn as sns

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
STAGE_DIR = os.path.join(BASE_DIR, "stage3_nlp")

def evaluate_classifier():
    print("Evaluating Urgency Classifier...")
    # Paths
    test_ids_path = os.path.join(STAGE_DIR, "data", "splits", "test_ids.csv")
    urgency_path = os.path.join(STAGE_DIR, "data", "classification", "urgency_classification.csv")
    model_path = os.path.join(STAGE_DIR, "models", "classification", "urgency_model.pkl")
    out_dir = os.path.join(STAGE_DIR, "artifacts", "evaluation")
    os.makedirs(out_dir, exist_ok=True)
    
    # Load data
    test_ids = pd.read_csv(test_ids_path)
    df = pd.read_csv(urgency_path)
    
    test_df = pd.merge(test_ids, df, on="record_id", how="inner")
    
    # Load model
    with open(model_path, "rb") as f:
        model = pickle.load(f)
        
    X_test = test_df["text"]
    y_true = test_df["urgency_label"]
    
    # Predict
    y_pred = model.predict(X_test)
    
    # Probabilities if available
    try:
        y_prob = model.predict_proba(X_test)
        confidence = y_prob.max(axis=1)
    except:
        confidence = [None] * len(y_pred)
    
    # Generate predictions artifact
    test_df["predicted_label"] = y_pred
    test_df["correct"] = (test_df["urgency_label"] == test_df["predicted_label"])
    test_df["confidence"] = confidence
    
    # Save predictions
    preds_out = os.path.join(out_dir, "classification_predictions.csv")
    test_df[["record_id", "text", "urgency_label", "predicted_label", "correct", "confidence"]].rename(columns={"urgency_label": "true_label"}).to_csv(preds_out, index=False)
    
    # Metrics
    labels = ["LOW", "MODERATE", "HIGH"]
    acc = accuracy_score(y_true, y_pred)
    
    precision_mac, recall_mac, f1_mac, _ = precision_recall_fscore_support(y_true, y_pred, average="macro", labels=labels)
    precision_wt, recall_wt, f1_wt, _ = precision_recall_fscore_support(y_true, y_pred, average="weighted", labels=labels)
    
    metrics = {
        "accuracy": acc,
        "macro_precision": precision_mac,
        "macro_recall": recall_mac,
        "macro_f1": f1_mac,
        "weighted_precision": precision_wt,
        "weighted_recall": recall_wt,
        "weighted_f1": f1_wt
    }
    
    with open(os.path.join(out_dir, "classifier_metrics.json"), "w") as f:
        json.dump(metrics, f, indent=4)
        
    # Classification report
    cr = classification_report(y_true, y_pred, labels=labels, output_dict=True)
    cr_df = pd.DataFrame(cr).transpose()
    cr_df.to_csv(os.path.join(out_dir, "classifier_classification_report.csv"))
    
    # Confusion Matrix
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    plt.figure(figsize=(8,6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.title("Urgency Classification Confusion Matrix")
    plt.savefig(os.path.join(out_dir, "classifier_confusion_matrix.png"))
    plt.close()
    
    print(f"Classification Evaluation complete. Evaluated {len(test_df)} records.")

if __name__ == "__main__":
    evaluate_classifier()
