from flask import Flask, request, jsonify, render_template
from pathlib import Path
import joblib

app = Flask(__name__, template_folder="../templates")

project_root = Path(__file__).resolve().parents[1]

model = joblib.load(project_root / "outputs" / "jobpost_model.joblib")
vectorizer = joblib.load(project_root / "outputs" / "tfidf_vectorizer.joblib")

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/api/predict", methods=["POST"])
def predict():
    data = request.get_json()
    text = data.get("text", "")
    X = vectorizer.transform([text])
    prob = float(model.predict_proba(X)[0][1])
    pred = 1 if prob >= 0.5 else 0
    return jsonify({"prediction": pred, "confidence": prob})

if __name__ == "__main__":
    app.run(debug=True)
