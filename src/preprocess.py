import re
import pandas as pd
from pathlib import Path

def basic_clean(text: str) -> str:
    text = str(text).lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

def preprocess_raw_csv(raw_csv: str, output_csv: str) -> None:
    df = pd.read_csv(raw_csv)

    text_fields = ["title", "company_profile", "description", "requirements", "benefits"]

    df["text"] = df[text_fields].fillna("").agg(" ".join, axis=1).apply(basic_clean)

    cleaned = df[["job_id", "text", "fraudulent"]].copy()
    cleaned.columns = ["id", "text", "label"]

    Path(output_csv).parent.mkdir(parents=True, exist_ok=True)
    cleaned.to_csv(output_csv, index=False)

    print(f"Saved cleaned data to {output_csv}")

if __name__ == "__main__":
    project_root = Path(__file__).resolve().parents[1]
    raw_csv = project_root / "data" / "fake_job_postings.csv"
    out_csv = project_root / "data" / "cleaned_posts.csv"
    preprocess_raw_csv(str(raw_csv), str(out_csv))
