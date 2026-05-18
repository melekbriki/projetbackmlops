"""
register_best_model.py — Model Registry MLflow
Partie 3 de la Tâche 5
Usage : python register_best_model.py
"""

import mlflow
from mlflow.tracking import MlflowClient

mlflow.set_tracking_uri("http://localhost:5000")
client = MlflowClient()

EXPERIMENT_NAME = "german-credit-risk"
MODEL_NAME      = "german_credit_production"
SEUIL_PRODUCTION = 0.80

# ── Trouver le meilleur run ───────────────────────────────────────────────────
print("🔍 Recherche du meilleur run...")
experiment = client.get_experiment_by_name(EXPERIMENT_NAME)

runs = client.search_runs(
    experiment_ids=[experiment.experiment_id],
    filter_string='tags.type = "training"',
    order_by=["metrics.accuracy DESC"],
    max_results=5
)

if not runs:
    print("❌ Aucun run trouvé. Lance d'abord un entraînement.")
    exit(1)

best_run = runs[0]
acc = best_run.data.metrics.get("accuracy", 0)
print(f"🏆 Meilleur run : {best_run.info.run_id}")
print(f"   Accuracy : {acc:.2f}% | F1 : {best_run.data.metrics.get('f1', 0):.2f}%")
print(f"   Params   : {best_run.data.params}")

# ── Enregistrer dans le Registry ──────────────────────────────────────────────
print(f"\n📦 Enregistrement dans le Model Registry sous '{MODEL_NAME}'...")
model_uri  = f"runs:/{best_run.info.run_id}/model"

try:
    registered = mlflow.register_model(model_uri=model_uri, name=MODEL_NAME)
    version = registered.version
    print(f"   ✅ Version {version} enregistrée")

    # Description et tags
    client.update_registered_model(
        name=MODEL_NAME,
        description="Modèle de prédiction de risque crédit allemand — pipeline MLOps"
    )
    client.set_model_version_tag(MODEL_NAME, version, "validated_by", "equipe_data")
    client.set_model_version_tag(MODEL_NAME, version, "dataset",      "german_credit")

    # ── Staging ──────────────────────────────────────────────────────────────
    print(f"\n🔄 Promotion en Staging...")
    client.transition_model_version_stage(
        name=MODEL_NAME, version=version,
        stage="Staging", archive_existing_versions=False
    )
    print(f"   ✅ Version {version} en Staging")

    # ── Production si seuil atteint ──────────────────────────────────────────
    print(f"\n🚦 Vérification du seuil production (>= {SEUIL_PRODUCTION*100:.0f}%)...")
    if acc >= SEUIL_PRODUCTION * 100:
        client.transition_model_version_stage(
            name=MODEL_NAME, version=version,
            stage="Production", archive_existing_versions=True
        )
        print(f"   🟢 Version {version} promue en PRODUCTION (accuracy={acc:.2f}%)")
    else:
        print(f"   🟡 Reste en Staging (accuracy={acc:.2f}% < seuil {SEUIL_PRODUCTION*100:.0f}%)")

except Exception as e:
    print(f"⚠️  Erreur Registry : {e}")
    print("   Vérifie que MLflow tourne sur http://localhost:5000")

print("\n✅ Terminé ! Visualisez : http://localhost:5000/#/models")