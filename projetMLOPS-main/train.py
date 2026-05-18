"""
train.py — Script d'entraînement autonome avec MLflow
Usage :
    python train.py                        # entraîne tous les modèles
    python train.py --model random_forest  # un seul modèle
    python train.py --tune                 # avec GridSearchCV
"""

import argparse
import json
import joblib
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
import mlflow
import mlflow.sklearn

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from imblearn.over_sampling import SMOTE

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix,
    ConfusionMatrixDisplay, classification_report
)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ── Config MLflow ─────────────────────────────────────────────────────────────
mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_experiment("german-credit-risk")

# ── Chemins ───────────────────────────────────────────────────────────────────
DATA_PATH  = Path("german.data")
MODELS_DIR = Path("saved_models")
MODELS_DIR.mkdir(exist_ok=True)

FEATURE_COLUMNS = [
    "Status", "Duration", "CreditHistory", "Purpose", "CreditAmount",
    "Savings", "EmploymentDuration", "InstallmentRate", "PersonalStatusSex",
    "OtherDebtors", "ResidenceDuration", "Property", "Age",
    "OtherInstallmentPlans", "Housing", "ExistingCredits",
    "Job", "PeopleLiable", "Telephone", "ForeignWorker", "Target",
]

# ── Modèles ───────────────────────────────────────────────────────────────────
MODELS_CONFIG = {
    "logistic_regression": {
        "model":  LogisticRegression(max_iter=200, random_state=42),
        "params": {"C": [0.01, 0.1, 1, 10], "solver": ["lbfgs", "liblinear"]},
    },
    "random_forest": {
        "model":  RandomForestClassifier(random_state=42),
        "params": {"n_estimators": [50, 100, 200], "max_depth": [5, 10, None]},
    },
    "svm": {
        "model":  SVC(probability=True, random_state=42),
        "params": {"C": [0.1, 1, 10], "kernel": ["rbf", "linear"]},
    },
    "knn": {
        "model":  KNeighborsClassifier(),
        "params": {"n_neighbors": [3, 5, 7, 11], "weights": ["uniform", "distance"]},
    },
    "adaboost": {
        "model":  AdaBoostClassifier(random_state=42),
        "params": {"n_estimators": [50, 100, 200], "learning_rate": [0.5, 1.0, 1.5]},
    },
    "xgboost": {
        "model":  XGBClassifier(eval_metric="logloss", random_state=42),
        "params": {"n_estimators": [50, 100], "max_depth": [3, 5, 7], "learning_rate": [0.05, 0.1, 0.2]},
    },
}

# ── Chargement et préparation ─────────────────────────────────────────────────
def load_and_prepare():
    print("📂 Chargement des données...")
    df = pd.read_csv(DATA_PATH, sep=" ", header=None, names=FEATURE_COLUMNS)
    df["Target"] = df["Target"].map({1: 0, 2: 1})

    le = LabelEncoder()
    for col in df.select_dtypes(include="object").columns:
        df[col] = le.fit_transform(df[col])

    X = df.drop("Target", axis=1)
    y = df["Target"]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42, stratify=y
    )

    smote = SMOTE(random_state=42)
    X_train_bal, y_train_bal = smote.fit_resample(X_train, y_train)

    print(f"✅ Train : {X_train_bal.shape} | Test : {X_test.shape}")

    joblib.dump(scaler, MODELS_DIR / "scaler.pkl")
    return X_train_bal, y_train_bal, X_test, y_test, scaler, X_scaled, y


def compute_metrics(model, X_test, y_test):
    y_pred  = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    return {
        "accuracy":  round(accuracy_score(y_test, y_pred) * 100,  2),
        "precision": round(precision_score(y_test, y_pred, zero_division=0) * 100, 2),
        "recall":    round(recall_score(y_test, y_pred, zero_division=0) * 100,    2),
        "f1":        round(f1_score(y_test, y_pred) * 100,        2),
        "roc_auc":   round(roc_auc_score(y_test, y_proba) * 100,  2),
    }, y_pred


def log_artifacts(model, name, X_test, y_test, y_pred):
    # Matrice de confusion
    fig, ax = plt.subplots(figsize=(6, 5))
    ConfusionMatrixDisplay.from_predictions(y_test, y_pred, ax=ax)
    plt.title(f"Confusion Matrix — {name}")
    plt.tight_layout()
    cm_path = f"confusion_matrix_{name}.png"
    plt.savefig(cm_path)
    plt.close()
    mlflow.log_artifact(cm_path)

    # Rapport de classification
    report = classification_report(y_test, y_pred)
    report_path = f"classification_report_{name}.txt"
    with open(report_path, "w") as f:
        f.write(report)
    mlflow.log_artifact(report_path)


# ── Entraînement simple ───────────────────────────────────────────────────────
def train_model(name, config, X_train, y_train, X_test, y_test):
    print(f"\n🤖 Entraînement : {name}")
    model = config["model"]

    with mlflow.start_run(run_name=f"{name}_default"):
        mlflow.set_tag("model_name", name)
        mlflow.set_tag("type", "training")
        mlflow.log_params(model.get_params())

        model.fit(X_train, y_train)
        metrics, y_pred = compute_metrics(model, X_test, y_test)

        for k, v in metrics.items():
            mlflow.log_metric(k, v)

        # Matrice de confusion + rapport
        log_artifacts(model, name, X_test, y_test, y_pred)

        # Enregistrer le modèle
        mlflow.sklearn.log_model(
            model,
            artifact_path="model",
            registered_model_name=f"credit_{name}",
        )
        run_id = mlflow.active_run().info.run_id

    joblib.dump(model, MODELS_DIR / f"{name}_model.pkl")
    print(f"   Accuracy : {metrics['accuracy']}%  |  F1 : {metrics['f1']}%  |  AUC : {metrics['roc_auc']}%")
    return metrics, run_id


# ── Entraînement avec GridSearch ──────────────────────────────────────────────
def train_with_gridsearch(name, config, X_train, y_train, X_test, y_test):
    print(f"\n🔍 GridSearchCV : {name}")

    with mlflow.start_run(run_name=f"{name}_GridSearch"):
        mlflow.set_tag("model_name", name)
        mlflow.set_tag("type", "gridsearch")

        gs = GridSearchCV(
            config["model"], config["params"],
            cv=5, scoring="roc_auc", n_jobs=-1, verbose=0,
        )
        gs.fit(X_train, y_train)
        best = gs.best_estimator_

        mlflow.log_params(gs.best_params_)
        mlflow.log_metric("best_cv_roc_auc", gs.best_score_ * 100)

        metrics, y_pred = compute_metrics(best, X_test, y_test)
        for k, v in metrics.items():
            mlflow.log_metric(k, v)

        log_artifacts(best, name, X_test, y_test, y_pred)

        mlflow.sklearn.log_model(
            best,
            artifact_path="model",
            registered_model_name=f"credit_{name}_tuned",
        )
        run_id = mlflow.active_run().info.run_id

    joblib.dump(best, MODELS_DIR / f"{name}_tuned_model.pkl")
    print(f"   Meilleurs params : {gs.best_params_}")
    print(f"   AUC : {metrics['roc_auc']}%  |  F1 : {metrics['f1']}%")
    return metrics, run_id, gs.best_params_


# ── Sauvegarde résultats ──────────────────────────────────────────────────────
def save_results(results: dict):
    results_path = Path("comparison_results.json")
    existing = {}
    if results_path.exists():
        with open(results_path) as f:
            existing = json.load(f)

    for name, metrics in results.items():
        existing[name] = {
            **metrics,
            "trained_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        }

    with open(results_path, "w") as f:
        json.dump(existing, f, indent=2)
    print("💾 Résultats sauvegardés dans comparison_results.json")


# ── Tableau comparatif ────────────────────────────────────────────────────────
def print_comparison_table(results: dict):
    print("\n" + "="*75)
    print("📊 TABLEAU COMPARATIF DES MODÈLES")
    print("="*75)
    print(f"{'Modèle':<30} {'Accuracy':>9} {'F1':>8} {'AUC':>8}")
    print("-"*75)
    for name, m in results.items():
        print(f"{name:<30} {m['accuracy']:>8.1f}% {m['f1']:>7.1f}% {m['roc_auc']:>7.1f}%")
    print("="*75)
    best = max(results, key=lambda k: results[k]["roc_auc"])
    print(f"\n🏆 Meilleur modèle (AUC) : {best} — {results[best]['roc_auc']}%")


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model",   default="all")
    parser.add_argument("--tune",    action="store_true")
    parser.add_argument("--retrain", action="store_true")
    args = parser.parse_args()

    X_train, y_train, X_test, y_test, scaler, X_scaled, y = load_and_prepare()

    models_to_run = MODELS_CONFIG if args.model == "all" else {
        args.model: MODELS_CONFIG[args.model]
    }

    all_results = {}
    for name, config in models_to_run.items():
        if args.tune:
            metrics, run_id, _ = train_with_gridsearch(name, config, X_train, y_train, X_test, y_test)
        else:
            metrics, run_id = train_model(name, config, X_train, y_train, X_test, y_test)
        all_results[name] = metrics

    save_results(all_results)

    if len(all_results) > 1:
        print_comparison_table(all_results)

    print(f"\n✅ Terminé ! Visualisez : mlflow ui --port 5000")


if __name__ == "__main__":
    main()