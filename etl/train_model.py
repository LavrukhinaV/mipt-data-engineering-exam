import pandas as pd
from sklearn.linear_model import LogisticRegression
import joblib
import os

def train():
    X_train = pd.read_csv("results/X_train.csv")
    y_train = pd.read_csv("results/y_train.csv").values.ravel()

    model = LogisticRegression(random_state=42, max_iter=1000)
    model.fit(X_train, y_train)

    joblib.dump(model, "results/model.pkl")
    print("Model saved to results/model.pkl")

if __name__ == "__main__":
    train()
