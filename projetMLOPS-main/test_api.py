"""
test_api.py — Tests automatiques de l'API
Usage : python test_api.py
"""

import requests
import json
import sys

BASE_URL = "http://localhost:8000"
PASSED   = 0
FAILED   = 0

def test(name, condition, detail=""):
    global PASSED, FAILED
    if condition:
        print(f"  ✅ {name}")
        PASSED += 1
    else:
        print(f"  ❌ {name} {detail}")
        FAILED += 1

print("\n🧪 Tests de l'API CreditGuard")
print("=" * 45)

# ── Test 1 : Health check ──────────────────────────────
print("\n📡 Test 1 — Health Check")
try:
    r = requests.get(f"{BASE_URL}/ping", timeout=5)
    test("GET /ping → 200",        r.status_code == 200)
    test("Réponse JSON valide",    r.json().get("status") == "ok")
except Exception as e:
    test("GET /ping",              False, str(e))

# ── Test 2 : Train ────────────────────────────────────
print("\n🤖 Test 2 — Entraînement")
try:
    payload = {"model": "random_forest", "hyperparams": {"n_estimators": 100}}
    r = requests.post(f"{BASE_URL}/train", json=payload, timeout=10)
    data = r.json()
    test("POST /train → 200",      r.status_code == 200)
    test("Champ 'accuracy' présent", "accuracy" in data)
    test("Champ 'f1' présent",     "f1" in data)
    test("Champ 'roc_auc' présent","roc_auc" in data)
    print(f"     Accuracy={data.get('accuracy')}% | F1={data.get('f1')}% | AUC={data.get('roc_auc')}%")
except Exception as e:
    test("POST /train",            False, str(e))

# ── Test 3 : Predict ──────────────────────────────────
print("\n🔮 Test 3 — Prédiction")
try:
    payload = {
        "model_id": "random_forest",
        "features": {
            "Status": 1, "Duration": 12, "CreditHistory": 2,
            "Purpose": 3, "CreditAmount": 5000, "Savings": 1,
            "EmploymentDuration": 2, "InstallmentRate": 3,
            "PersonalStatusSex": 1, "OtherDebtors": 0,
            "ResidenceDuration": 2, "Property": 1, "Age": 35,
            "OtherInstallmentPlans": 0, "Housing": 1,
            "ExistingCredits": 1, "Job": 2, "PeopleLiable": 1,
            "Telephone": 0, "ForeignWorker": 1
        }
    }
    r = requests.post(f"{BASE_URL}/predict", json=payload, timeout=10)
    data = r.json()
    test("POST /predict → 200",          r.status_code == 200)
    test("Champ 'prediction' présent",   "prediction" in data)
    test("Prédiction valide (0 ou 1)",   data.get("prediction") in [0, 1])
    test("risk_probability entre 0 et 1",0 <= data.get("risk_probability", -1) <= 1)
except Exception as e:
    test("POST /predict",                False, str(e))

# ── Test 4 : History ──────────────────────────────────
print("\n📜 Test 4 — Historique")
try:
    r = requests.get(f"{BASE_URL}/history", timeout=5)
    data = r.json()
    test("GET /history → 200",     r.status_code == 200)
    test("Champ 'runs' présent",   "runs" in data)
    test("Historique non vide",    len(data.get("runs", [])) > 0)
except Exception as e:
    test("GET /history",           False, str(e))

# ── Test 5 : Upload ───────────────────────────────────
print("\n📂 Test 5 — Upload CSV")
try:
    import io
    fake_csv = io.BytesIO(b"col1,col2\n1,2\n3,4")
    r = requests.post(
        f"{BASE_URL}/upload",
        files={"file": ("test.csv", fake_csv, "text/csv")},
        timeout=10
    )
    test("POST /upload → 200",     r.status_code == 200)
    test("Status uploaded",        r.json().get("status") == "uploaded")
except Exception as e:
    test("POST /upload",           False, str(e))

# ── Résumé ────────────────────────────────────────────
print("\n" + "=" * 45)
total = PASSED + FAILED
print(f"📊 Résultats : {PASSED}/{total} tests passés")
if FAILED == 0:
    print("🎉 Tous les tests sont passés !")
else:
    print(f"⚠️  {FAILED} test(s) échoué(s)")
print("=" * 45 + "\n")

sys.exit(0 if FAILED == 0 else 1)