import unittest
from pathlib import Path
import subprocess
import joblib

PROJECT_ROOT = Path(__file__).resolve().parents[1]
TRAIN_SCRIPT = PROJECT_ROOT / "src" / "train_model.py"
MODEL_FILE = PROJECT_ROOT / "models" / "sms_spam_clf.joblib"
METRICS_FILE = PROJECT_ROOT / "reports" / "metrics.txt"

class TestTrainModel(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # S'assure que les données sont prêtes avant l'entraînement
        prep_script = PROJECT_ROOT / "src" / "prepare_data.py"
        subprocess.run(["python", str(prep_script)], check=True, cwd=PROJECT_ROOT)
        # Entraîne le modèle
        subprocess.run(["python", str(TRAIN_SCRIPT)], check=True, cwd=PROJECT_ROOT)

    def test_artifacts_exist(self):
        self.assertTrue(MODEL_FILE.exists(), "Le modèle .joblib n'a pas été sauvegardé.")
        self.assertTrue(METRICS_FILE.exists(), "Le fichier reports/metrics.txt est absent.")

    def test_model_loads_and_predicts(self):
        pipe = joblib.load(MODEL_FILE)
        self.assertTrue(hasattr(pipe, "predict"), "L'objet chargé n'est pas un pipeline de prédiction.")
        pred = pipe.predict(["WIN a free iPhone now!"])[0]
        self.assertIn(pred, ["spam", "ham"], "La prédiction doit être 'spam' ou 'ham'.")

if __name__ == "__main__":
    unittest.main(verbosity=2)
