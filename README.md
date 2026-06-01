# AI-Enhanced Predictive Maintenance Assistant
**DSC680 Applied Data Science — Project 2 | Bellevue University**
Author: Kevin Danh

## Overview
Extends the Phase 1 XGBoost predictive maintenance model by adding a 
Generative AI layer that translates failure predictions into plain-language 
maintenance recommendations using the OpenAI API.

## Project Structure
- Phase 1 (model training): https://github.com/Cronosus99/predictive-maintenance-ml
- Phase 2 (this repo): AI explanation layer built on top of the saved model

## How It Works
1. XGBoost model scores each device with a failure probability
2. Flagged devices are passed to a structured prompt builder
3. OpenAI GPT-4o returns a condition summary, recommended action, and urgency level

## Setup
1. Clone this repo
2. Install dependencies: `pip install -r requirements.txt`
3. Copy `.env.example` to `.env` and add your OpenAI API key
4. Open `genai_predictive_maintenance.ipynb` and run all cells

## Requirements
- Python 3.10+
- OpenAI API key
- pkl files from Phase 1 (included in this repo)

## References
- Danh, K. (2026). Phase 1 — Predictive Maintenance Using Machine Learning. 
  https://github.com/Cronosus99/predictive-maintenance-ml
- Kaggle. (2023). Predictive Maintenance Dataset.
  https://www.kaggle.com/datasets/hiimanshuagarwal/predictive-maintenance-dataset