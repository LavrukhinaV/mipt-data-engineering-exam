import pandas as pd
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from dotenv import load_dotenv
import joblib

load_dotenv()

RAW_DATA_PATH = os.getenv("RAW_DATA_PATH")
PROCESSED_DATA_PATH = os.getenv("PROCESSED_DATA_PATH")
SCALER_PATH = "results/scaler.pkl"

def preprocess():
    df = pd.read_csv(RAW_DATA_PATH)
    X = df.drop("target", axis=1)
    y = df["target"]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    joblib.dump(scaler, SCALER_PATH)

    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

    os.makedirs("results", exist_ok=True)
    pd.DataFrame(X_train).to_csv("results/X_train.csv", index=False)
    pd.DataFrame(X_test).to_csv("results/X_test.csv", index=False)
    pd.DataFrame(y_train).to_csv("results/y_train.csv", index=False)
    pd.DataFrame(y_test).to_csv("results/y_test.csv", index=False)

    print("Preprocessing complete.")

if __name__ == "__main__":
    preprocess()
