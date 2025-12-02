## Labor Trafficking Job Post Detector

A research prototype that analyzes job postings and predicts potential labor-trafficking risk using NLP and machine learning.
The system preprocesses job descriptions, extracts text features, trains a classifier, and provides an interactive web interface for real-time predictions.

## Project Overview and Purpose

This project demonstrates how machine learning and text processing can be used to identify potentially suspicious or fraudulent job postings that may indicate labor-trafficking risks.
Using TF-IDF features and a Logistic Regression classifier, the tool detects linguistic patterns associated with deceptive recruitment behavior and exposes them through an accessible web interface.

## Video Link of the Project

## Installation and Setup Instructions

# 1. Clone the repository
git clone https://github.com/SathwikH/labortrafficking_detector
cd labortrafficking_detector

# 2. Create and activate a virtual environment
python -m venv venv
venv\Scripts\activate   # Windows
# or:
source venv/bin/activate   # Mac/Linux

# 3. Install dependencies
pip install -r requirements.txt

## How to Run the Program and Reproduce Results

# Preprocess the dataset

Generates cleaned_posts.csv:

python src/preprocess.py

# Train the model

Creates the classifier and TF-IDF vectorizer:

python src/train.py

# Generate batch predictions

Outputs predictions.csv:

python src/predict.py

# Launch the web interface (Flask app)

python src/app.py

# Open the browser at:

http://127.0.0.1:5000/

## Technologies and Libraries Used

Python

pandas

scikit-learn

Flask

joblib

numpy

regex-based text cleaning

HTML/CSS (frontend interface)

## Authors and Contribution Summary

# Francisco Terán & Sathwik Harapanahalli

Data preprocessing

Machine learning model development

Web interface and backend API

Documentation and project structure

## Project Structure

project/
│
├── data/
│   ├── fake_job_postings.csv        # Raw dataset (from internet)
│   └── cleaned_posts.csv            # Preprocessed text dataset
│
├── outputs/
│   ├── jobpost_model.joblib         # Trained Logistic Regression model
│   ├── tfidf_vectorizer.joblib      # TF-IDF vectorizer
│   └── predictions.csv              # Batch predictions output
│
├── src/
│   ├── preprocess.py                # Cleaning + normalization script
│   ├── train.py                     # Model training pipeline
│   ├── predict.py                   # Batch prediction script
│   └── app.py                       # Flask API + Web interface backend
│
├── templates/
│   └── index.html                   # Frontend UI
│
├── report/
│   └── final_report.pdf             # Final written report (to be added)
│
├── requirements.txt
└── README.md
