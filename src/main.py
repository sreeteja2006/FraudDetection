import joblib
import pandas as pd
import xgboost as xgb
from fastapi import FastAPI, HTTPException
from typing import Dict, Any
from contextlib import asynccontextmanager

ml_assets = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    print("Booting up Inference Server...")
    model = xgb.Booster()
    model.load_model("data/fraud_xgboost.json")
    ml_assets["model"] = model
    
    ml_assets["encoders"] = joblib.load("data/label_encoders.pkl")
    print("Models loaded successfully. Ready for traffic.")
    
    yield
    
    ml_assets.clear()

app = FastAPI(title="Fraud Engine API", lifespan=lifespan)

@app.post("/predict")
async def predict_fraud(transaction: Dict[str, Any]):
    try:
        df = pd.DataFrame([transaction])
        
        free_domains = ['gmail.com', 'yahoo.com', 'outlook.com', 'hotmail.com', 'mail.com', 'anonymous.com']
        df['is_free_email'] = df.get('P_emaildomain', pd.Series([''])).apply(lambda x: 1 if x in free_domains else 0)
        
        cols_to_drop = ['dist2', 'D6', 'D7', 'D8', 'D9', 'D12', 'D13', 'D14', 'V138', 'V139', 'V140', 'V141', 'V142', 'V143', 'V144', 'V145', 'V146', 'V147', 'V148', 'V149', 'V150', 'V151', 'V152', 'V153', 'V154', 'V155', 'V156', 'V157', 'V158', 'V159', 'V160', 'V161', 'V162', 'V163', 'V164', 'V165', 'V166', 'V322', 'V323', 'V324', 'V325', 'V326', 'V327', 'V328', 'V329', 'V330', 'V331', 'V332', 'V333', 'V334', 'V335', 'V336', 'V337', 'V338', 'V339', 'TransactionID']
        existing_drops = [c for c in cols_to_drop if c in df.columns]
        df = df.drop(columns=existing_drops)
        
        encoders = ml_assets["encoders"]
        for col in df.columns:
            if col in encoders:
                df[col] = df[col].fillna("missing")
                df[col] = df[col].map(lambda x: encoders[col].transform([x])[0] if x in encoders[col].classes_ else -1)
            else:
                df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
                
        feature_names = ml_assets["model"].feature_names
        df = df.reindex(columns=feature_names, fill_value=0) 
        
        dmatrix = xgb.DMatrix(df)
        probability = ml_assets["model"].predict(dmatrix)[0]
        
        threshold = 0.85
        is_fraud = bool(probability >= threshold)
        
        return {
            "status": "success",
            "fraud_detected": is_fraud,
            "confidence_score": round(float(probability), 4),
            "threshold_applied": threshold,
            "recommended_action": "DECLINE_AND_REVIEW" if is_fraud else "APPROVE"
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference pipeline error: {str(e)}")