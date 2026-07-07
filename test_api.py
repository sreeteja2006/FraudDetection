import requests
import json
import pandas as pd

url = "http://127.0.0.1:8000/predict"

print("Loading a RAW fraud transaction from disk...")
df = pd.read_csv("data/train_transaction.csv")

raw_fraud_row = df[df['isFraud'] == 1].drop(columns=['isFraud', 'TransactionID']).iloc[0]

mock_transaction = raw_fraud_row.where(pd.notnull(raw_fraud_row), None).to_dict()

print("Sending real, raw fraud transaction to API...")
try:
    response = requests.post(url, json=mock_transaction)
    print(f"Status Code: {response.status_code}\n")
    print("API Response:")
    print(json.dumps(response.json(), indent=2))
except requests.exceptions.ConnectionError:
    print("Error: Could not connect to the API. Is your Uvicorn server running?")