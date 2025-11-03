from pathlib import Path
import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
RAW_DIR = PROJECT_ROOT / "data" / "raw"
RAW_FILE = RAW_DIR / "spam.csv"
OUT_DIR = PROJECT_ROOT / "data" / "processed"
OUT_FILE = OUT_DIR / "sms_clean.csv"


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
    try:
        df = pd.read_csv(RAW_FILE, encoding="latin-1")
    except pd.errors.ParserError:
        # Fallback si CSV mal quoté (CI)
        df = pd.read_csv(RAW_FILE, encoding="latin-1", engine="python", on_bad_lines="skip")

    if {"v1", "v2"}.issubset(df.columns):
        df = df[["v1", "v2"]].rename(columns={"v1": "label", "v2": "text"})
    else:
        possible_label = [c for c in df.columns if c.lower() in {"label", "category", "v1"}]
        possible_text = [c for c in df.columns if c.lower() in {"text", "message", "sms", "v2"}]
        if not possible_label or not possible_text:
            raise ValueError(
                f"Colonnes inattendues : {list(df.columns)}. Attendu v1/v2 ou label/text."
            )
        df = df[[possible_label[0], possible_text[0]]]
        df.columns = ["label", "text"]
    return df


def basic_clean(df: pd.DataFrame) -> pd.DataFrame:
    df = df.dropna(subset=["label", "text"]).copy()
    df["text"] = df["text"].astype(str).str.strip()
    df = df[df["text"].str.len() > 0]

    df["label"] = df["label"].astype(str).str.strip().str.lower()
    df["label"] = df["label"].map({"spam": "spam", "ham": "ham"})
    df = df[df["label"].isin(["spam", "ham"])]

    df = df.drop_duplicates(subset=["label", "text"])
    return df


def save_with_report(df: pd.DataFrame) -> None:
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
    report_path.write_text("\n".join(report_lines), encoding="utf-8")
    print(f"📊 Rapport sauvegardé -> {report_path}")


def main() -> None:
    print("📂 Lecture du dataset…")
    df = load_dataset()
    print("✨ Nettoyage des données…")
    df = basic_clean(df)
    print("💾 Sauvegarde du dataset nettoyé…")
    save_with_report(df)
    print(f"✅ Données nettoyées enregistrées ici : {OUT_FILE}")


if __name__ == "__main__":
    main()
