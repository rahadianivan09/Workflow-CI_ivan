import os
import pandas as pd
import mlflow
import mlflow.sklearn

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix
)

# =========================
# CONFIG
# =========================
DATA_PATH = "data_preprocessing.csv"
TARGET = "Churn"
EXPERIMENT_NAME = "telco_churn_rf_baseline"

# =========================
# CREATE FOLDERS
# =========================
os.makedirs("artifacts", exist_ok=True)
os.makedirs("mlruns", exist_ok=True)

# =========================
# MLFLOW SETUP
# =========================
mlflow.set_tracking_uri("file:./mlruns")
mlflow.set_experiment(EXPERIMENT_NAME)

# =========================
# LOAD DATA
# =========================
df = pd.read_csv(DATA_PATH)

# =========================
# TARGET MAPPING
# =========================
if df[TARGET].dtype == "object":
    df[TARGET] = df[TARGET].map({
        "No": 0,
        "Yes": 1
    })

# =========================
# FEATURES & TARGET
# =========================
X = df.drop(TARGET, axis=1)
y = df[TARGET]

# =========================
# FIX INTEGER TYPE
# =========================
int_cols = X.select_dtypes(include=["int64"]).columns
X[int_cols] = X[int_cols].astype("float64")

# =========================
# SPLIT DATA
# =========================
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# =========================
# TRAINING
# =========================

# OPTIONAL: autolog (aman versi CI)
mlflow.sklearn.autolog(log_models=False)

# MODEL
model = RandomForestClassifier(
    n_estimators=100,
    max_depth=5,
    random_state=42,
    class_weight="balanced"
)

# TRAIN
model.fit(X_train, y_train)

# SAVE MODEL
mlflow.sklearn.save_model(
    sk_model=model,
    path="artifacts/model"
)

# PREDICTION
y_pred = model.predict(X_test)

# METRICS
acc = accuracy_score(y_test, y_pred)
prec = precision_score(y_test, y_pred, zero_division=0)
rec = recall_score(y_test, y_pred, zero_division=0)
f1 = f1_score(y_test, y_pred, zero_division=0)

# LOG METRICS
mlflow.log_metric("test_accuracy", acc)
mlflow.log_metric("test_precision", prec)
mlflow.log_metric("test_recall", rec)
mlflow.log_metric("test_f1", f1)

# OUTPUT
print("\n=== MODEL PERFORMANCE ===")
print(f"Accuracy : {acc:.4f}")
print(f"Precision: {prec:.4f}")
print(f"Recall   : {rec:.4f}")
print(f"F1 Score : {f1:.4f}")

print("\n=== CLASSIFICATION REPORT ===")
print(classification_report(y_test, y_pred, zero_division=0))

print("\n=== CONFUSION MATRIX ===")
print(confusion_matrix(y_test, y_pred))

print("\nTraining selesai dan artifact berhasil disimpan.")