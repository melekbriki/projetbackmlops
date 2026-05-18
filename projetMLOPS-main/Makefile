# Makefile — Pipeline MLOps Local
# Usage : make <cible>
# Exemples : make train | make drift | make pipeline

.PHONY: setup train drift register serve test pipeline clean help

# ── Configuration ─────────────────────────────────────────────────────────────
PYTHON     = python
PORT_API   = 8000
PORT_MLFLOW= 5000
MODEL      = all

# ── Aide ──────────────────────────────────────────────────────────────────────
help:
	@echo ""
	@echo "========================================"
	@echo "   CreditGuard MLOps Pipeline"
	@echo "========================================"
	@echo "  make setup      — Installer les dépendances"
	@echo "  make mlflow     — Lancer MLflow UI"
	@echo "  make train      — Entraîner tous les modèles"
	@echo "  make drift      — Détecter le data drift"
	@echo "  make register   — Enregistrer le meilleur modèle"
	@echo "  make serve      — Lancer l'API FastAPI"
	@echo "  make test       — Tester l'API"
	@echo "  make pipeline   — Pipeline complet (train+drift+register)"
	@echo "  make clean      — Nettoyer les fichiers temporaires"
	@echo "========================================"
	@echo ""

# ── Setup ─────────────────────────────────────────────────────────────────────
setup:
	@echo "📦 Installation des dépendances..."
	pip install -r requirements.txt
	@echo "✅ Setup terminé !"

# ── MLflow UI ─────────────────────────────────────────────────────────────────
mlflow:
	@echo "🔬 Lancement de MLflow UI sur http://localhost:$(PORT_MLFLOW)"
	start mlflow ui --port $(PORT_MLFLOW)

# ── Entraînement ──────────────────────────────────────────────────────────────
train:
	@echo "🤖 Entraînement des modèles (model=$(MODEL))..."
	$(PYTHON) train.py --model $(MODEL)
	@echo "✅ Entraînement terminé !"

train-tune:
	@echo "🔍 Entraînement avec GridSearchCV..."
	$(PYTHON) train.py --model $(MODEL) --tune
	@echo "✅ GridSearch terminé !"

# ── Drift ─────────────────────────────────────────────────────────────────────
drift:
	@echo "📊 Détection du data drift..."
	$(PYTHON) simulate_drift.py
	@echo "✅ Analyse drift terminée !"

# ── Model Registry ────────────────────────────────────────────────────────────
register:
	@echo "📦 Enregistrement du meilleur modèle dans MLflow Registry..."
	$(PYTHON) register_best_model.py
	@echo "✅ Modèle enregistré !"

# ── API ───────────────────────────────────────────────────────────────────────
serve:
	@echo "🚀 Lancement de l'API FastAPI sur http://localhost:$(PORT_API)..."
	start uvicorn main:app --reload --port $(PORT_API)

# ── Test API ──────────────────────────────────────────────────────────────────
test:
	@echo "🧪 Test de l'API..."
	$(PYTHON) test_api.py
	@echo "✅ Tests terminés !"

# ── Pipeline complet ──────────────────────────────────────────────────────────
pipeline: train drift register
	@echo ""
	@echo "========================================"
	@echo "✅ Pipeline MLOps complet exécuté !"
	@echo "   → Entraînement    : OK"
	@echo "   → Drift detection : OK"
	@echo "   → Model Registry  : OK"
	@echo "   Visualisez : http://localhost:$(PORT_MLFLOW)"
	@echo "========================================"

# ── Nettoyage ─────────────────────────────────────────────────────────────────
clean:
	@echo "🧹 Nettoyage des fichiers temporaires..."
	-del /f confusion_matrix_*.png 2>nul
	-del /f classification_report_*.txt 2>nul
	-del /f drift_report.html 2>nul
	-del /f ks_drift_results.csv 2>nul
	@echo "✅ Nettoyage terminé !"