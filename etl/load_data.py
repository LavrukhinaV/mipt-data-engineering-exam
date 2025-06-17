from sklearn.datasets import load_breast_cancer
import pandas as pd
import os
from dotenv import load_dotenv

load_dotenv()

RAW_DATA_PATH = os.getenv("RAW_DATA_PATH", "results/breast_cancer.csv")

def save_dataset():
    data = load_breast_cancer()
    df = pd.DataFrame(data.data, columns=data.feature_names)
    df["target"] = data.target  # 0 = benign, 1 = malignant
    os.makedirs(os.path.dirname(RAW_DATA_PATH), exist_ok=True)
    df.to_csv(RAW_DATA_PATH, index=False)
    print(f"Dataset saved to {RAW_DATA_PATH}")

if __name__ == "__main__":
    save_dataset()
