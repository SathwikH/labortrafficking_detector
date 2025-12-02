from pathlib import Path
import pandas as pd
import joblib
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score
)

def test_model(cleaned_csv: str, model_path: str, vectorizer_path: str):
    
    df = pd.read_csv(cleaned_csv)
    
    texts = df["text"].astype(str).tolist()
    labels = df["label"].astype(int).tolist()
    
    vectorizer = joblib.load(vectorizer_path)
    model = joblib.load(model_path)
    
    X = vectorizer.transform(texts)

    # prediction
    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(X)[:, 1]   
        predictions = (probabilities >= 0.5).astype(int)
    else:
        predictions = model.predict(X)
        
    # metrics
    accuracy = accuracy_score(labels, predictions)
    precision = precision_score(labels, predictions, zero_division=0)
    recall = recall_score(labels, predictions, zero_division=0)
    f1 = f1_score(labels, predictions, zero_division=0)
    
    print("\nModel Test Results:")
    print(f"Accuracy  : {accuracy:.4f}")
    print(f"Precision : {precision:.4f}")
    print(f"Recall    : {recall:.4f}")
    print(f"F1 Score  : {f1:.4f}")

if __name__ == "__main__":
    project_root = Path(__file__).resolve().parents[1]
    
    cleaned_csv = project_root / "data" / "cleaned_posts.csv"
    model_path = project_root / "outputs" / "jobpost_model.joblib"
    vectorizer_path = project_root / "outputs" / "tfidf_vectorizer.joblib"

    test_model(str(cleaned_csv), str(model_path), str(vectorizer_path))
