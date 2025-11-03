from pathlib import Path
import os
import pandas as pd

# --- Chemins robustes basés sur l'emplacement du script ---
SCRIPT_DIR = Path(__file__).resolve().parent           # .../sms-spam/src
PROJECT_ROOT = SCRIPT_DIR.parent                       # .../sms-spam
RAW_DIR = PROJECT_ROOT / "data" / "raw"                # .../sms-spam/data/raw
RAW_FILE = RAW_DIR / "spam.csv"                        # fichier attendu : spam.csv
OUT_DIR = PROJECT_ROOT / "data" / "processed"          # .../sms-spam/data/processed
OUT_FILE = OUT_DIR / "sms_clean.csv"

# --- Chargement du dataset ---
def load_dataset() -> pd.DataFrame:
    """
    Charge le fichier spam.csv depuis data/raw/ et normalise (label, text).
    """
    if not RAW_FILE.exists():
        raise FileNotFoundError(
            f"❌ Fichier non trouvé : {RAW_FILE}\n"
            "Assure-toi que le fichier s'appelle bien 'spam.csv' et qu'il est dans data/raw/.\n"
            "Si ton fichier a un autre nom (ex: 'spam (1).csv'), renomme-le en 'spam.csv'."
        )

    # Lecture CSV (latin-1 pour éviter les soucis d'accents avec la version Kaggle)
    df = pd.read_csv(RAW_FILE, encoding="latin-1")

    # Version Kaggle classique : v1(label) / v2(text)
    if {"v1", "v2"}.issubset(df.columns):
        df = df[["v1", "v2"]].rename(columns={"v1": "label", "v2": "text"})
    else:
        # Tentative générique si autres noms de colonnes
        possible_label = [c for c in df.columns if c.lower() in {"label", "category", "v1"}]
        possible_text  = [c for c in df.columns if c.lower() in {"text", "message", "sms", "v2"}]
        if not possible_label or not possible_text:
            raise ValueError(
                f"Colonnes inattendues dans le CSV. Colonnes trouvées : {list(df.columns)}\n"
                "Le fichier doit avoir v1/v2 ou label/text."
            )
        df = df[[possible_label[0], possible_text[0]]]
        df.columns = ["label", "text"]

    return df

# --- Nettoyage de base ---
def basic_clean(df: pd.DataFrame) -> pd.DataFrame:
    df = df.dropna(subset=["label", "text"]).copy()
    df["text"] = df["text"].astype(str).str.strip()
    df = df[df["text"].str.len() > 0]

    df["label"] = df["label"].astype(str).str.strip().str.lower()
    df["label"] = df["label"].map({"spam": "spam", "ham": "ham"})
    df = df[df["label"].isin(["spam", "ham"])]

    df = df.drop_duplicates(subset=["label", "text"])
    return df

# --- Sauvegarde + rapport ---
def save_with_report(df: pd.DataFrame):
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT_FILE, index=False)

    report_lines = [
        f"Nombre total de lignes : {len(df)}",
        "Distribution des classes :",
        df["label"].value_counts().to_string(),
        "\nLongueur des messages (en caractères) :",
        df["text"].str.len().describe().to_string(),
    ]

    report_path = OUT_DIR / "prep_report.txt"
    with report_path.open("w", encoding="utf-8") as f:
        f.write("\n".join(report_lines))

    print(f"📊 Rapport sauvegardé -> {report_path}")

# --- Main ---
def main():
    print("📂 Lecture du dataset…")
    df = load_dataset()
    print("✨ Nettoyage des données…")
    df = basic_clean(df)
    print("💾 Sauvegarde du dataset nettoyé…")
    save_with_report(df)
    print(f"✅ Données nettoyées enregistrées ici : {OUT_FILE}")

if __name__ == "__main__":
    main()
