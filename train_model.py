import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
import joblib

df = pd.read_csv("WA_Fn-UseC_-Telco-Customer-Churn.csv")

df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
df.fillna(0, inplace=True)

# Convert target
df["Churn"] = LabelEncoder().fit_transform(df["Churn"])

# Use only 4 columns
X = df[["SeniorCitizen", "tenure", "MonthlyCharges", "TotalCharges"]]
y = df["Churn"]

scaler = StandardScaler()
X = scaler.fit_transform(X)

model = RandomForestClassifier(random_state=42)
model.fit(X, y)

joblib.dump(model, "model.pkl")
joblib.dump(scaler, "scaler.pkl")

print("Model trained successfully")
