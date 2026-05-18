"""
simulate_drift.py — Détection de Data Drift
Partie 6 de la Tâche 5 MLOps
Usage : python simulate_drift.py
"""

import numpy as np
import pandas as pd
import mlflow
import subprocess
from pathlib import Path
from sklearn.model_selection import train_test_split
from scipy import stats

# ── Config MLflow ─────────────────────────────────────────────────────────────
mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_experiment("monitoring_drift")

# ── Chargement données ────────────────────────────────────────────────────────
FEATURE_COLUMNS = [
    "Status", "Duration", "CreditHistory", "Purpose", "CreditAmount",
    "Savings", "EmploymentDuration", "InstallmentRate", "PersonalStatusSex",
    "OtherDebtors", "ResidenceDuration", "Property", "Age",
    "OtherInstallmentPlans", "Housing", "ExistingCredits",
    "Job", "PeopleLiable", "Telephone", "ForeignWorker", "Target",
]

print("📂 Chargement des données...")
df = pd.read_csv("german.data", sep=" ", header=None, names=FEATURE_COLUMNS)
df["Target"] = df["Target"].map({1: 0, 2: 1})

X = df.drop("Target", axis=1).select_dtypes(include="number")
y = df["Target"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ── Simulation du drift ───────────────────────────────────────────────────────
print("⚗️  Simulation du drift sur les données de production...")
X_prod = X_test.copy()
num_cols = X_prod.select_dtypes(include=np.number).columns

for col in num_cols[:3]:
    X_prod[col] = X_prod[col] * 1.6 + np.random.normal(0, 0.5, len(X_prod))

print(f"   Moyenne 'Duration'  — Ref: {X_train['Duration'].mean():.2f} | Prod: {X_prod['Duration'].mean():.2f}")
print(f"   Moyenne 'Age'       — Ref: {X_train['Age'].mean():.2f}      | Prod: {X_prod['Age'].mean():.2f}")

# ── KS-test par feature ───────────────────────────────────────────────────────
print("\n📊 KS-test par feature...")
ks_results = []

with mlflow.start_run(run_name="ks_test_drift_check"):
    mlflow.set_tag("type", "drift_detection")
    mlflow.set_tag("method", "KS-test")

    for col in X_train.columns:
        stat, pvalue = stats.ks_2samp(X_train[col], X_prod[col])
        drifted = pvalue < 0.05
        ks_results.append({
            "feature": col,
            "ks_stat": round(stat, 4),
            "p_value": round(pvalue, 4),
            "drifted": drifted
        })
        mlflow.log_metric(f"ks_pvalue_{col}", pvalue)
        mlflow.log_metric(f"ks_stat_{col}", stat)

    df_drift = pd.DataFrame(ks_results)
    df_drift.to_csv("ks_drift_results.csv", index=False)
    mlflow.log_artifact("ks_drift_results.csv")

    n_drifted   = df_drift["drifted"].sum()
    n_total     = len(df_drift)
    drift_share = n_drifted / n_total

    mlflow.log_metric("drift_share",     drift_share)
    mlflow.log_metric("drifted_columns", int(n_drifted))
    mlflow.log_metric("total_columns",   n_total)

    print(f"\n{'Feature':<25} {'KS Stat':>10} {'P-Value':>10} {'Drifted':>10}")
    print("-" * 55)
    for _, row in df_drift.iterrows():
        flag = "🔴 OUI" if row["drifted"] else "🟢 NON"
        print(f"{row['feature']:<25} {row['ks_stat']:>10.4f} {row['p_value']:>10.4f} {flag:>10}")

    print(f"\n📈 Drift share : {drift_share:.2%} | Colonnes driftées : {int(n_drifted)}/{n_total}")

# ── Rapport HTML manuel ───────────────────────────────────────────────────────
print("\n📊 Génération du rapport HTML de drift...")

with mlflow.start_run(run_name="drift_html_report"):
    mlflow.set_tag("type", "drift_detection")
    mlflow.set_tag("method", "KS-test HTML Report")
    mlflow.log_metric("drift_share",     drift_share)
    mlflow.log_metric("drifted_columns", int(n_drifted))
    mlflow.log_metric("total_columns",   n_total)

    rows_html = ""
    for _, r in df_drift.iterrows():
        css   = "drifted" if r["drifted"] else "ok"
        label = "🔴 Drifted" if r["drifted"] else "🟢 Stable"
        rows_html += f"""
        <tr>
            <td>{r['feature']}</td>
            <td>{r['ks_stat']}</td>
            <td>{r['p_value']}</td>
            <td class='{css}'>{label}</td>
        </tr>"""

    html = f"""<!DOCTYPE html>
<html lang="fr">
<head>
  <meta charset="utf-8">
  <title>Data Drift Report — German Credit Risk</title>
  <style>
    body      {{ font-family: Arial, sans-serif; background: #0f172a; color: #f1f5f9; padding: 2rem; }}
    h1        {{ color: #60a5fa; margin-bottom: 0.5rem; }}
    p         {{ color: #94a3b8; margin-bottom: 1.5rem; }}
    .summary  {{ display: flex; gap: 2rem; margin-bottom: 2rem; }}
    .card     {{ background: #1e293b; border: 1px solid #334155; border-radius: 12px; padding: 1.5rem 2rem; }}
    .card h2  {{ margin: 0; font-size: 2rem; }}
    .card span{{ font-size: 0.85rem; color: #94a3b8; }}
    .red      {{ color: #f87171; }}
    .green    {{ color: #4ade80; }}
    table     {{ width: 100%; border-collapse: collapse; background: #1e293b; border-radius: 12px; overflow: hidden; }}
    th        {{ background: #1e3a5f; padding: 12px 16px; text-align: left; color: #93c5fd; }}
    td        {{ padding: 10px 16px; border-bottom: 1px solid #334155; }}
    .drifted  {{ color: #f87171; font-weight: bold; }}
    .ok       {{ color: #4ade80; font-weight: bold; }}
  </style>
</head>
<body>
  <h1>📊 Data Drift Report — German Credit Risk</h1>
  <p>Méthode : Kolmogorov-Smirnov test | Seuil p-value : 0.05</p>

  <div class="summary">
    <div class="card">
      <h2 class="{'red' if drift_share > 0.30 else 'green'}">{drift_share:.1%}</h2>
      <span>Drift Share Global</span>
    </div>
    <div class="card">
      <h2 class="red">{int(n_drifted)}</h2>
      <span>Features Driftées</span>
    </div>
    <div class="card">
      <h2>{n_total}</h2>
      <span>Features Totales</span>
    </div>
    <div class="card">
      <h2 class="{'red' if drift_share > 0.30 else 'green'}">
        {'🔴 CRITIQUE' if drift_share > 0.30 else '🟡 AVERTISSEMENT' if drift_share > 0.15 else '🟢 STABLE'}
      </h2>
      <span>Statut Global</span>
    </div>
  </div>

  <table>
    <tr>
      <th>Feature</th>
      <th>KS Statistique</th>
      <th>P-Value</th>
      <th>Statut</th>
    </tr>
    {rows_html}
  </table>
</body>
</html>"""

    with open("drift_report.html", "w", encoding="utf-8") as f:
        f.write(html)

    mlflow.log_artifact("drift_report.html")
    print("   ✅ drift_report.html généré et loggé dans MLflow")

# ── Déclenchement automatique ré-entraînement ────────────────────────────────
SEUIL_DRIFT = 0.30
SEUIL_WARN  = 0.15

print(f"\n🚦 Évaluation du seuil (drift_share = {drift_share:.2%})")

with mlflow.start_run(run_name="retrain_decision"):
    mlflow.set_tag("type", "retrain_decision")
    mlflow.log_metric("drift_share",          drift_share)
    mlflow.log_metric("seuil_critique",       SEUIL_DRIFT)
    mlflow.log_metric("seuil_avertissement",  SEUIL_WARN)

    if drift_share > SEUIL_DRIFT:
        print(f"🔴 CRITIQUE : drift {drift_share:.2%} > seuil {SEUIL_DRIFT:.0%}")
        print("   → Déclenchement du ré-entraînement...")
        mlflow.log_metric("retrain_triggered", 1)
        mlflow.set_tag("decision", "RETRAIN")
        try:
            subprocess.run(["python", "train.py", "--model", "all"], check=True)
            print("   ✅ Ré-entraînement terminé !")
        except Exception as e:
            print(f"   ❌ Erreur ré-entraînement : {e}")

    elif drift_share > SEUIL_WARN:
        print(f"🟡 AVERTISSEMENT : drift {drift_share:.2%} — surveillance renforcée")
        mlflow.log_metric("retrain_triggered", 0)
        mlflow.set_tag("decision", "WARN")

    else:
        print(f"🟢 OK : drift {drift_share:.2%} — modèle stable")
        mlflow.log_metric("retrain_triggered", 0)
        mlflow.set_tag("decision", "OK")

print("\n✅ Analyse drift terminée ! Visualisez : http://localhost:5000")