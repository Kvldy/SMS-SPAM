from pathlib import Path
import os
import joblib
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
)


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
DATA_FILE = PROJECT_ROOT / "data" / "processed" / "sms_clean.csv"
MODELS_DIR = PROJECT_ROOT / "models"
REPORTS_DIR = PROJECT_ROOT / "reports"
MODEL_PATH = MODELS_DIR / "sms_spam_clf.joblib"

RANDOM_STATE = 42
FAST = os.environ.get("FAST", "0") == "1"


def load_data() -> pd.DataFrame:
    if not DATA_FILE.exists():
        raise FileNotFoundError(
            f"Dataset introuvable : {DATA_FILE}\n"
            "Exécute d'abord : python src/prepare_data.py"
        )
    df = pd.read_csv(DATA_FILE)
    if not {"label", "text"}.issubset(df.columns):
        raise ValueError("Le CSV doit contenir les colonnes 'label' et 'text'.")
    return df


def build_pipeline() -> Pipeline:
    if FAST:
        print("⚡ Mode rapide CI activé")
        tfidf = TfidfVectorizer(lowercase=True, strip_accents="unicode", ngram_range=(1, 1), max_features=1000)
        clf = LogisticRegression(class_weight="balanced", max_iter=100, random_state=RANDOM_STATE)
    else:
        tfidf = TfidfVectorizer(lowercase=True, strip_accents="unicode", ngram_range=(1, 2), min_df=2, max_df=0.95)
        clf = LogisticRegression(class_weight="balanced", max_iter=200, random_state=RANDOM_STATE)

    return Pipeline([("tfidf", tfidf), ("clf", clf)])


def evaluate(y_true, y_proba, y_pred) -> str:
    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, pos_label="spam")
    auc = roc_auc_score((pd.Series(y_true) == "spam").astype(int), y_proba)

    cm = confusion_matrix(y_true, y_pred, labels=["ham", "spam"])
    cm_df = pd.DataFrame(cm, index=["true_ham", "true_spam"], columns=["pred_ham", "pred_spam"])
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    cm_df.to_csv(REPORTS_DIR / "confusion_matrix.csv", index=True)

    cls_rep = classification_report(y_true, y_pred, digits=3)
    summary = (
        "=== SMS Spam – Résultats test ===\n"
        f"Accuracy : {acc:.4f}\n"
        f"F1 (classe spam) : {f1:.4f}\n"
        f"ROC AUC : {auc:.4f}\n\n"
        "=== Classification report ===\n"
        f"{cls_rep}\n"
        "Matrice de confusion enregistrée : reports/confusion_matrix.csv\n"
    )
    with open(REPORTS_DIR / "metrics.txt", "w", encoding="utf-8") as f:
        f.write(summary)
    return summary


def main() -> None:
    print("📥 Chargement des données…")
    df = load_data()

    X = df["text"].astype(str).values
    y = df["label"].astype(str).values

    print("✂️  Split train/test (stratifié)…")
    n_samples = len(y)
    n_classes = len(pd.Series(y).unique())
    safe_test_size = max(0.2, n_classes / max(n_samples, 1))

    counts = pd.Series(y).value_counts()
    use_stratify = (counts.get("ham", 0) > 0) and (counts.get("spam", 0) > 0)

    try:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=safe_test_size,
            random_state=RANDOM_STATE,
            stratify=y if use_stratify else None,
        )
    except ValueError as e:
        print(f"⚠️  Split non stratifié (fallback) : {e}")
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=safe_test_size, random_state=RANDOM_STATE, stratify=None
        )

    print("🧪 Construction du pipeline TF-IDF + LogisticRegression…")
    pipe = build_pipeline()

    print("🏋️  Entraînement…")
    pipe.fit(X_train, y_train)

    print("🔎 Évaluation…")
    class_order = list(pipe.classes_)
    spam_index = class_order.index("spam")
    y_proba = pipe.predict_proba(X_test)[:, spam_index]
    y_pred = pipe.predict(X_test)

    report = evaluate(y_test, y_proba, y_pred)
    print(report)

    print("💾 Sauvegarde du modèle…")
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipe, MODEL_PATH)
    print(f"✅ Modèle sauvegardé -> {MODEL_PATH}")


if __name__ == "__main__":
    main()
