import unittest
from pathlib import Path
import subprocess
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PREP_SCRIPT = PROJECT_ROOT / "src" / "prepare_data.py"
DATA_PROCESSED = PROJECT_ROOT / "data" / "processed" / "sms_clean.csv"
REPORT_FILE = PROJECT_ROOT / "data" / "processed" / "prep_report.txt"


class TestPrepareData(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        subprocess.run(["python", str(PREP_SCRIPT)], check=True, cwd=PROJECT_ROOT)

    def test_outputs_exist(self):
        self.assertTrue(DATA_PROCESSED.exists(), "sms_clean.csv n'a pas été créé.")
        self.assertTrue(REPORT_FILE.exists(), "prep_report.txt n'a pas été créé.")

    def test_csv_has_expected_columns(self):
        df = pd.read_csv(DATA_PROCESSED)
        self.assertTrue({"label", "text"}.issubset(df.columns))

    def test_labels_are_valid(self):
        df = pd.read_csv(DATA_PROCESSED)
        labels = set(df["label"].unique())
        self.assertTrue(labels.issubset({"spam", "ham"}))


if __name__ == "__main__":
    unittest.main(verbosity=2)
