import unittest
from pathlib import Path
import subprocess
import json
import joblib

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PREDICT_SCRIPT = PROJECT_ROOT / "src" / "predict.py"
TRAIN_SCRIPT = PROJECT_ROOT / "src" / "train_model.py"
MODEL_FILE = PROJECT_ROOT / "models" / "sms_spam_clf.joblib"
MESSAGES_FILE = PROJECT_ROOT / "data" / "messages.txt"

class TestPredict(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Assure la présence du modèle
        if not MODEL_FILE.exists():
            # Assure d'abord la préparation des données
            prep_script = PROJECT_ROOT / "src" / "prepare_data.py"
            subprocess.run(["python", str(prep_script)], check=True, cwd=PROJECT_ROOT)
            subprocess.run(["python", str(TRAIN_SCRIPT)], check=True, cwd=PROJECT_ROOT)

    def _run_and_parse_json(self, cmd, *, use_shell=False):
        result = subprocess.run(
            cmd,
            cwd=PROJECT_ROOT,
            shell=use_shell,
            capture_output=True,
            text=True,
            check=True
        )
        try:
            return json.loads(result.stdout)
        except json.JSONDecodeError as e:
            self.fail(f"Sortie JSON invalide.\nErreur: {e}\nSTDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}")

    def test_predict_from_messages_file_if_available(self):
        # Préférence : utiliser --file si le script le supporte et si le fichier existe
        if MESSAGES_FILE.exists():
            # Tente d'abord --file
            try:
                preds = self._run_and_parse_json(
                    ["python", str(PREDICT_SCRIPT), "--file", str(MESSAGES_FILE)]
                )
            except subprocess.CalledProcessError:
                # Fallback : essayer via pipe (cat/type)
                if subprocess.os.name == "nt":
                    # Windows : type data\messages.txt | python src\predict.py
                    cmd = f'type "{MESSAGES_FILE}" | python "{PREDICT_SCRIPT}"'
                else:
                    cmd = f'cat "{MESSAGES_FILE}" | python "{PREDICT_SCRIPT}"'
                preds = self._run_and_parse_json(cmd, use_shell=True)

            self.assertIsInstance(preds, list, "La sortie doit être une liste (batch).")
            self.assertGreater(len(preds), 0, "Aucune prédiction renvoyée depuis le fichier.")

            for p in preds:
                self.assertIn("text", p)
                self.assertIn("prediction", p)
                self.assertIn("proba_spam", p)
                self.assertIn(p["prediction"], ["spam", "ham"])
                self.assertGreaterEqual(p["proba_spam"], 0.0)
                self.assertLessEqual(p["proba_spam"], 1.0)
        else:
            self.skipTest("data/messages.txt absent — test de fichier sauté.")

    def test_predict_single_message(self):
        # Test de secours : prédiction d'un seul message fonctionne
        preds = self._run_and_parse_json(
            ["python", str(PREDICT_SCRIPT), "WIN a free cruise!!!"]
        )
        # Si la sortie est un dict (notre script renvoie un dict pour single message)
        if isinstance(preds, dict):
            self.assertIn("prediction", preds)
            self.assertIn(preds["prediction"], ["spam", "ham"])
            self.assertIn("proba_spam", preds)
            self.assertGreaterEqual(preds["proba_spam"], 0.0)
            self.assertLessEqual(preds["proba_spam"], 1.0)
        else:
            # Si le script renvoie une liste même pour un message, la valider aussi
            self.assertIsInstance(preds, list)
            self.assertGreater(len(preds), 0)
            self.assertIn(preds[0]["prediction"], ["spam", "ham"])

if __name__ == "__main__":
    unittest.main(verbosity=2)
