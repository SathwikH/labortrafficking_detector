from pathlib import Path
import pandas as pd
import joblib

def predict_from_cleaned(
    cleaned_csv: str,
    model_path: str,
    vectorizer_path: str,
    output_csv: str,
    threshold: float = 0.5,
):
    df = pd.read_csv(cleaned_csv)

    texts = df["text"].astype(str).tolist()

    # Cargar SIEMPRE primero vectorizer, luego modelo
    vectorizer = joblib.load(vectorizer_path)
    model = joblib.load(model_path)

    X = vectorizer.transform(texts)

    if hasattr(model, "predict_proba"):
        probs = model.predict_proba(X)[:, 1]
    else:
        scores = model.decision_function(X)
        probs = scores  # (si luego quieres, lo normalizas)

    preds = (probs >= threshold).astype(int)

    out_df = pd.DataFrame({
        "id": df["id"],
        "text": df["text"],
        "label": df["label"] if "label" in df.columns else None,
        "prediction": preds,
        "confidence": probs,
    })

    Path(output_csv).parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(output_csv, index=False)
    print(f"Predictions saved to {output_csv}")

if __name__ == "__main__":
    project_root = Path(__file__).resolve().parents[1]
    cleaned_csv = project_root / "data" / "cleaned_posts.csv"
    model_path = project_root / "outputs" / "jobpost_model.joblib"
    vectorizer_path = project_root / "outputs" / "tfidf_vectorizer.joblib"
    output_csv = project_root / "outputs" / "predictions.csv"
    predict_from_cleaned(str(cleaned_csv), str(model_path), str(vectorizer_path), str(output_csv))
