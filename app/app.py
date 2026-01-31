from fastapi import FastAPI
from pydantic import BaseModel
import mlflow
import mlflow.sklearn
import pandas as pd

app = FastAPI()

# 🔑 tell FastAPI where MLflow server is
mlflow.set_tracking_uri("http://127.0.0.1:5000")

RUN_ID = "0e906080596f455687d0d0e11a9831b6"
model = mlflow.sklearn.load_model(f"runs:/{RUN_ID}/model")

features = [
    "gender", "SeniorCitizen", "Partner", "Dependents", "tenure",
    "PhoneService", "PaperlessBilling", "MonthlyCharges", "TotalCharges",
    "is_long_term_contract", "is_auto_payment", "num_services",
    "avg_monthly_spend", "is_single",
    "MultipleLines_No phone service", "MultipleLines_Yes",
    "InternetService_Fiber optic", "InternetService_No",
    "OnlineSecurity_No internet service", "OnlineSecurity_Yes",
    "DeviceProtection_No internet service", "DeviceProtection_Yes",
    "TechSupport_No internet service", "TechSupport_Yes",
    "StreamingTV_No internet service", "StreamingTV_Yes",
    "StreamingMovies_No internet service", "StreamingMovies_Yes",
    "Contract_One year", "Contract_Two year",
    "PaymentMethod_Credit card (automatic)",
    "PaymentMethod_Electronic check",
    "PaymentMethod_Mailed check",
    "tenure_group_Mid", "tenure_group_New",
    "OnlineBackup_No internet service", "OnlineBackup_Yes"
]

default = {f: 0 for f in features}

class UserInput(BaseModel):
    tenure: float
    MonthlyCharges: float
    PaperlessBilling: int
    Contract_One_year: int = 0
    Contract_Two_year: int = 0

def prepare_dataframe(user_input: dict):
    row = default.copy()
    row.update(user_input)
    return pd.DataFrame([row], columns=features)

@app.post("/predict")
def predict(data: UserInput):
    df = prepare_dataframe(data.dict())
    prediction = model.predict(df)[0]
    return {
        "churn_prediction": "Yes" if prediction == 1 else "No"
    }
