import pandas as pd
import joblib
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import json

def evaluate():
    X_test = pd.read_csv("results/X_test.csv")
    y_test = pd.read_csv("results/y_test.csv").values.ravel()

    model = joblib.load("results/model.pkl")
    y_pred = model.predict(X_test)

    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
        "f1": f1_score(y_test, y_pred),
    }

    with open("results/metrics.json", "w") as f:
        json.dump(metrics, f, indent=4)

    print("Metrics saved to results/metrics.json")

if __name__ == "__main__":
    evaluate()
