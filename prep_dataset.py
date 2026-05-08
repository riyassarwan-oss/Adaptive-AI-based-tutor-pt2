# prep_dataset.py
import pandas as pd, random, os
from pathlib import Path

BASE_DIR = Path(__file__).parent.resolve()
DATA_DIR = BASE_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
OUT_PATH = DATA_DIR / "question_bank.csv"

DATA_DIR.mkdir(parents=True, exist_ok=True)
RAW_DIR.mkdir(parents=True, exist_ok=True)

def find_sciq_csv():
    """
    Find a SciQ-like CSV in data/raw.
    Needs columns: question, correct_answer, distractor1, distractor2, distractor3
    """
    candidates = list(RAW_DIR.glob("*.csv"))
    for p in candidates:
        try:
            head = pd.read_csv(p, nrows=2)
        except Exception:
            continue
        cols = {c.strip().lower() for c in head.columns}
        required = {"question", "correct_answer", "distractor1", "distractor2", "distractor3"}
        if required.issubset(cols):
            return p
    return None

def difficulty_from_len(text: str) -> str:
    L = len(str(text or ""))
    if L < 80: return "Easy"
    if L < 160: return "Medium"
    return "Hard"

def make_bank(df: pd.DataFrame) -> pd.DataFrame:
    # Normalize column names
    colmap = {c: c.strip().lower() for c in df.columns}
    df = df.rename(columns=colmap)

    rows = []
    for i, r in df.iterrows():
        q = str(r["question"])
        correct = str(r["correct_answer"])
        d1 = str(r["distractor1"])
        d2 = str(r["distractor2"])
        d3 = str(r["distractor3"])
        # 4-option shuffle so correct isn't always A
        options = [correct, d1, d2, d3]
        random.shuffle(options)
        correct_idx = options.index(correct)  # 0..3
        correct_letter = "ABCD"[correct_idx]
        explanation = str(r.get("support", "") or "")

        rows.append({
            "id": i + 1,
            "subject": "Science",
            "topic": "General Science",
            "difficulty": difficulty_from_len(q),
            "question": q,
            "options": "|".join(options),
            "correct_option": correct_letter,
            "explanation": explanation
        })
    return pd.DataFrame(rows)

def main():
    src = find_sciq_csv()
    if not src:
        print("❌ No SciQ-like CSV found in", RAW_DIR)
        print("   Expected columns: question, correct_answer, distractor1, distractor2, distractor3")
        return

    print(f"📥 Source: {src}")
    df = pd.read_csv(src)
    bank = make_bank(df)

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    bank.to_csv(OUT_PATH, index=False, encoding="utf-8")
    print(f"✅ Wrote {len(bank):,} questions to:\n   {OUT_PATH.resolve()}")

    # Small summary
    print("\nBy topic:", bank['topic'].value_counts().to_dict())
    print("By difficulty:", bank['difficulty'].value_counts().to_dict())

if __name__ == "__main__":
    main()