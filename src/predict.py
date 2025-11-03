from pathlib import Path
import sys
import json
import joblib

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = PROJECT_ROOT / "models" / "sms_spam_clf.joblib"

def load_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Modèle introuvable : {MODEL_PATH}\n"
            "→ Lance d'abord : python src/train_model.py"
        )
    return joblib.load(MODEL_PATH)

def predict_one(msg: str, pipe):
    classes = list(pipe.classes_)
    spam_idx = classes.index("spam")
    proba_spam = pipe.predict_proba([msg])[0][spam_idx]
    pred = pipe.predict([msg])[0]
    return {"text": msg, "prediction": pred, "proba_spam": float(proba_spam)}

def predict_many(msgs, pipe):
    classes = list(pipe.classes_)
    spam_idx = classes.index("spam")
    probas = pipe.predict_proba(msgs)[:, spam_idx]
    preds = pipe.predict(msgs)
    return [
        {"text": m, "prediction": p, "proba_spam": float(pr)}
        for m, p, pr in zip(msgs, preds, probas)
    ]

def main():
    import sys
    from pathlib import Path
    import json

    pipe = load_model()

    # 1) Mode fichier : python src/predict.py --file data/messages.txt
    if len(sys.argv) >= 3 and sys.argv[1] == "--file":
        file_path = Path(sys.argv[2])
        if not file_path.exists():
            print(json.dumps({"error": f"Fichier introuvable: {file_path}"}))
            sys.exit(1)
        with file_path.open("r", encoding="utf-8") as f:
            lines = [l.strip() for l in f if l.strip()]
        out = predict_many(lines, pipe)
        print(json.dumps(out, ensure_ascii=False, indent=2))
        return

    # 2) Mode message direct : python src/predict.py "Your message here"
    if len(sys.argv) >= 2:
        msg = " ".join(sys.argv[1:])
        out = predict_one(msg, pipe)
        print(json.dumps(out, ensure_ascii=False, indent=2))
        return

    # 3) Mode stdin (pipe) : type data/messages.txt | python src/predict.py
    lines = [l.strip() for l in sys.stdin if l.strip()]
    if lines:
        out = predict_many(lines, pipe)
        print(json.dumps(out, ensure_ascii=False, indent=2))
    else:
        print("Usage :\n"
              "  python src/predict.py \"Your message\"\n"
              "  python src/predict.py --file data/messages.txt\n"
              "  type data\\messages.txt | python src\\predict.py")
        sys.exit(1)


if __name__ == "__main__":
    main()
