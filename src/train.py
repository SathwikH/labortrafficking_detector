from pathlib import Path
import pandas as pd
import joblib

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_curve
import numpy as np


def train_model(cleaned_csv: str, out_model: str, out_vectorizer: str):
    df = pd.read_csv(cleaned_csv)

    texts = df["text"].astype(str).tolist()
    labels = df["label"].astype(int).tolist()

    train_texts, test_texts, train_labels, test_labels = train_test_split(
        texts,
        labels,
        test_size=0.2,
        random_state=42,
        stratify=labels
    )

    vectorizer = TfidfVectorizer(
        max_features=12000,
        ngram_range=(1, 3),
        stop_words="english",
        sublinear_tf=True,
        min_df=2
    )

    X_train = vectorizer.fit_transform(train_texts)
    X_test = vectorizer.transform(test_texts)

    model = LogisticRegression(
        penalty="l2",
        C=2.0,
        class_weight="balanced",
        solver="liblinear",
        max_iter=1500
    )

    model.fit(X_train, train_labels)

    predictions = model.predict(X_test)
    print("\nClassification Report:")
    print(classification_report(test_labels, predictions))

    probabilities = model.predict_proba(X_test)[:, 1]
    fpr, tpr, thresholds = roc_curve(test_labels, probabilities)

    best_threshold_idx = (tpr - fpr).argmax()
    best_threshold = thresholds[best_threshold_idx]

    print("\nOptimal Threshold:", best_threshold)

    print("\nTop Fraud Indicators:")
    feature_names = vectorizer.get_feature_names_out()
    coefs = model.coef_[0]
    top_index = np.argsort(coefs)[-25:]

    for i in reversed(top_index):
        print(f"{feature_names[i]:30s} coef={coefs[i]:.4f}")

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
