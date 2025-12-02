from pathlib import Path
import pandas as pd
import joblib

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

def train_model(cleaned_csv: str, out_model: str, out_vectorizer: str):
    df = pd.read_csv(cleaned_csv)

    texts = df["text"].astype(str).tolist()
    labels = df["label"].astype(int).tolist()

    vectorizer = TfidfVectorizer(
        max_features=5000,
        ngram_range=(1, 2),
        stop_words="english",
    )

    X = vectorizer.fit_transform(texts)

    # Simple baseline model
    model = LogisticRegression(max_iter=200)
    model.fit(X, labels)

    out_model = Path(out_model)
    out_vectorizer = Path(out_vectorizer)
    out_model.parent.mkdir(parents=True, exist_ok=True)

    joblib.dump(model, out_model)
    joblib.dump(vectorizer, out_vectorizer)

    print("Model saved at:", out_model)
    print("Vectorizer saved at:", out_vectorizer)

if __name__ == "__main__":
    project_root = Path(__file__).resolve().parents[1]
    cleaned_csv = project_root / "data" / "cleaned_posts.csv"
    model_path = project_root / "outputs" / "jobpost_model.joblib"
    vectorizer_path = project_root / "outputs" / "tfidf_vectorizer.joblib"

    train_model(str(cleaned_csv), str(model_path), str(vectorizer_path))
