# day4_mlflow.py
# -*- coding: utf-8 -*-

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import mlflow
import mlflow.sklearn
mlflow.set_tracking_uri("http://127.0.0.1:5000")
mlflow.set_experiment("ml_experiment")


# -------------------------------
# 1️⃣ Load dataset
# Make sure 'customer_churn_ml_ready.csv' is in the same folder as this script
df = pd.read_csv('data/customer_churn_ml_ready.csv')
print("First 5 rows of dataset:")
print(df.head())

# Drop customerID column
df.drop('customerID', axis=1, inplace=True)

# -------------------------------
# 2️⃣ Scale numeric columns
standard = ['avg_monthly_spend', 'MonthlyCharges']
minmax = ['TotalCharges', 'tenure']

scaler_standard = StandardScaler()
df[standard] = scaler_standard.fit_transform(df[standard])

scaler_minmax = MinMaxScaler()
df[minmax] = scaler_minmax.fit_transform(df[minmax])

# -------------------------------
# 3️⃣ Split features and target
X = df.drop('Churn', axis=1)
y = df['Churn']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# -------------------------------
# 4️⃣ Set MLflow experiment
mlflow.set_experiment("ml_experiment")

# -------------------------------
# 5️⃣ Train LogisticRegression model and log with MLflow
with mlflow.start_run():  # starts a run

    # Log model type as parameter
    mlflow.log_param("model_type", "LogisticRegression")
    
    # Initialize and train model
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)
    
    # Make predictions
    y_pred = model.predict(X_test)
    
    # Calculate metrics
    acc = accuracy_score(y_test, y_pred)
    mlflow.log_metric("accuracy", acc)
    
    # Optional: print confusion matrix and classification report
    print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
    print("\nClassification Report:\n", classification_report(y_test, y_pred))
    
    # Log the trained model
    mlflow.sklearn.log_model(model, "model")
    
    print(f"\n✅ Logged LogisticRegression model with accuracy: {acc}")
