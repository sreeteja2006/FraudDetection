# End to End Fraud Detection Inference API

A production ready machine learning microservice designed to detect fraudulent financial transactions in real time. This project encompasses the entire ML lifecycle: from raw data processing and feature engineering to training an XGBoost classifier and deploying it as a containerized FastAPI REST endpoint.~

## System Architecture

* Modeling: XGBoost Classifier optimized for severe class imbalance (scale_pos_weight).
* Evaluation Metric: AUC PR (Area Under the Precision Recall Curve), prioritizing minority class detection over misleading accuracy metrics.
* Business Logic: Implements custom decision thresholds (0.85) to minimize False Positives, significantly reducing the operational cost of manual reviews.
* Inference Engine: Asynchronous Python REST API built with FastAPI.
* Deployment: Fully containerized via Docker or Podman for immediate cloud deployment.

## Tech Stack

* Data Science: Python, Pandas, Scikit Learn, XGBoost, Joblib
* Backend: FastAPI, Uvicorn
* Infrastructure: Docker, Linux

## Project Structure

```text
FraudDetection/
  data/
    train_transaction.csv
    fraud_xgboost.json
    label_encoders.pkl
  notebooks/
    day1_eda.ipynb
    day2_feature_engineering.ipynb
    day3_modeling.ipynb
  src/
    main.py
  test_api.py
  requirements.txt
  Dockerfile