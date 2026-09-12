import os
import pandas as pd
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))

def train_classifier():
    print("Loading datasets and splits...")
    URGENCY_CSV = os.path.join(BASE_DIR, "data", "classification", "urgency_classification.csv")
    TRAIN_IDS = os.path.join(BASE_DIR, "data", "splits", "train_ids.csv")
    
    df = pd.read_csv(URGENCY_CSV)
    train_ids = pd.read_csv(TRAIN_IDS)
    
    train_df = pd.merge(train_ids, df, on="record_id", how="inner")
    
    X_train = train_df["text"]
    y_train = train_df["urgency_label"]
    
    print(f"Training on {len(X_train)} records...")
    
    # We use TF-IDF + Logistic Regression as a strong baseline
    # class_weight='balanced' handles the class imbalance (LOW=~62%, MODERATE=~16%, HIGH=~22%)
    pipeline = Pipeline([
        ('tfidf', TfidfVectorizer(max_features=10000, ngram_range=(1, 2), stop_words='english')),
        ('clf', LogisticRegression(class_weight='balanced', random_state=42, max_iter=1000))
    ])
    
    pipeline.fit(X_train, y_train)
    print("Training complete.")
    
    # Simple training check
    preds = pipeline.predict(X_train)
    print("Training metrics (Sanity Check):")
    print(classification_report(y_train, preds))
    
    # Save the pipeline
    model_dir = os.path.join(BASE_DIR, "models", "classification")
    os.makedirs(model_dir, exist_ok=True)
    model_path = os.path.join(model_dir, "urgency_model.pkl")
    
    with open(model_path, "wb") as f:
        pickle.dump(pipeline, f)
        
    print(f"Model saved to {model_path}")

if __name__ == "__main__":
    train_classifier()
