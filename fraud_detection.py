# fraud_detection.py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef,
    confusion_matrix
)

import warnings
warnings.filterwarnings("ignore")


def load_data(csv_path="https://ibm-project-bucket-19.s3.us-east-1.amazonaws.com/creditcard.csv"):
    print("Loading dataset...")
    data = pd.read_csv(csv_path)
    return data


def preprocess_data(data):
    X = data.drop(['Class'], axis=1)
    Y = data['Class']
    return X.values, Y.values


def evaluate_model(y_true, y_pred, model_name="Model"):
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred)
    rec = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    mcc = matthews_corrcoef(y_true, y_pred)
    print(f"\n{model_name} Evaluation:")
    print(f"Accuracy  : {acc:.4f}")
    print(f"Precision : {prec:.4f}")
    print(f"Recall    : {rec:.4f}")
    print(f"F1 Score  : {f1:.4f}")
    print(f"MCC       : {mcc:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred))

    LABELS = ['Normal', 'Fraud']
    conf_matrix = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(conf_matrix, xticklabels=LABELS, yticklabels=LABELS, annot=True, fmt="d")
    plt.title(f"{model_name} - Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    plt.show()


def run_isolation_forest(X_train, X_test, Y_test, outlier_fraction):
    print("\nTraining Isolation Forest...")
    model = IsolationForest(max_samples=len(X_train), contamination=outlier_fraction, random_state=1)
    model.fit(X_train)
    y_pred = model.predict(X_test)
    y_pred = np.where(y_pred == 1, 0, 1)  # Convert: 1 → 0 (normal), -1 → 1 (fraud)
    evaluate_model(Y_test, y_pred, model_name="Isolation Forest")


def run_random_forest(X_train, X_test, Y_train, Y_test):
    print("\nTraining Random Forest...")
    model = RandomForestClassifier()
    model.fit(X_train, Y_train)
    y_pred = model.predict(X_test)
    evaluate_model(Y_test, y_pred, model_name="Random Forest")


def main():
    # Load and explore data
    data = load_data()
    print(data.describe())
    print(f"\nFraudulent transactions: {data['Class'].sum()} / {len(data)}")

    # Preprocess
    X, Y = preprocess_data(data)
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

    # Get fraud ratio
    fraud_count = data[data['Class'] == 1]
    valid_count = data[data['Class'] == 0]
    outlier_fraction = len(fraud_count) / float(len(valid_count))

    # Train and evaluate models
    run_isolation_forest(X_train, X_test, Y_test, outlier_fraction)
    run_random_forest(X_train, X_test, Y_train, Y_test)


if __name__ == "__main__":
    main()
