# AI-Enhanced Predictive Maintenance Assistant
 
**Author:** Kevin Danh  
**Program:** Master of Data Science – Bellevue University  
**Course:** DSC 680: Applied Data Science (Project 2)  
 
## Overview

Extends the Phase 1 XGBoost predictive maintenance model by adding a Generative AI layer that translates failure predictions into plain-language maintenance recommendations using the OpenAI API.
 
## Project Phases

- **Phase 1 (model training):** https://github.com/Cronosus99/predictive-maintenance-ml
- **Phase 2 (this repo):** AI explanation layer built on top of the saved model

## How It Works

1. The XGBoost model scores each device with a failure probability
2. Flagged devices are passed to a structured prompt builder
3. OpenAI GPT-4o Mini returns a condition summary, recommended action, and urgency level

## Project Structure
 
```text
predictive-maintenance-genai/
├── genai_predictive_maintenance.ipynb   # Main notebook (AI explanation layer)
├── xgb_model.pkl                        # Trained XGBoost model from Phase 1
├── X_test.pkl                           # Held-out test features
├── y_test.pkl                           # True failure labels
├── y_prob.pkl                           # Predicted failure probabilities
├── requirements.txt
├── .env.example                         # Template for API key setup
├── .gitignore
└── README.md
```
 
## Setup

1. Clone this repository
2. Install dependencies: `pip install -r requirements.txt`
3. Copy `.env.example` to `.env` and add your OpenAI API key
4. Open `genai_predictive_maintenance.ipynb` and run all cells

## Requirements

- Python 3.10+
- OpenAI API key
- Saved `.pkl` files from Phase 1 (included in this repo)

## Future Improvements

- Replace raw feature values with SHAP values for more precise prompt inputs
- Add retrieval-augmented generation (RAG) using historical failure records
- Build a real-time monitoring pipeline with automated alerts

## References

- Danh, K. (2026). *Phase 1 — Predictive Maintenance Using Machine Learning.* https://github.com/Cronosus99/predictive-maintenance-ml
- Kaggle. (2023). *Predictive Maintenance Dataset.* https://www.kaggle.com/datasets/hiimanshuagarwal/predictive-maintenance-dataset
