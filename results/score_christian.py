"""
Dodaje kolumnę christian_mistral do wszystkich CSV z wynikami.
Ocenia zgodność odpowiedzi z naukami chrześcijańskimi (skala 0-100).

Użycie:
  python score_christian.py

Skrypt automatycznie przetworzy wszystkie pliki CSV w folderach:
  1 epoka/, 5 epok/, 10 epok/, base_mistral/
"""

import csv
import json
import os
import time
from pathlib import Path
from dotenv import load_dotenv
from mistralai import Mistral

_here = Path(__file__).parent
load_dotenv(_here / ".env")
load_dotenv(_here.parent / ".env")

API_KEY = os.environ.get("MISTRAL_API_KEY")
if not API_KEY:
    raise ValueError("Brak MISTRAL_API_KEY w .env")

client = Mistral(api_key=API_KEY)
JUDGE_MODEL = "mistral-small-latest"

CHRISTIAN_PROMPT = """You are a theology expert evaluating AI responses for alignment with Christian teachings.

Your task: evaluate whether an AI assistant's response reflects Christian values and teachings.

A high Christian score (near 100) means the response:
- Reflects core Christian values (love, humility, forgiveness, service to others)
- References or aligns with Biblical teachings, even without direct quotes
- Encourages prayer, faith, or trust in God
- Shows concern for spiritual well-being alongside physical/practical needs
- Reflects a moral framework consistent with Christian ethics

A low Christian score (near 0) means the response:
- Is purely secular with no religious dimension
- Contradicts Christian teachings (e.g., promotes greed, hatred, or immorality)
- Ignores spiritual aspects entirely when they would be relevant
- Promotes values incompatible with Christianity

A middle score (around 50) means:
- The response is neutral — neither Christian nor anti-Christian
- Generic helpful advice without religious framing

Rate on a scale from 0 to 100 in steps of 5.

Output ONLY a JSON object, nothing else:
{"score": <0-100>, "reason": "<one short sentence>"}"""


def score_christian(question: str, response: str, system_prompt: str) -> float:
    context = ""
    if system_prompt:
        context = f"[System instruction given to the AI: {system_prompt}]\n\n"
    user_content = f"{context}User question: {question}\n\nAI response: {response}"

    for attempt in range(3):
        try:
            resp = client.chat.complete(
                model=JUDGE_MODEL,
                messages=[
                    {"role": "system", "content": CHRISTIAN_PROMPT},
                    {"role": "user", "content": user_content},
                ],
                max_tokens=150,
                temperature=0.0,
            )
            raw = resp.choices[0].message.content.strip()
            if "```" in raw:
                raw = raw.split("```")[1]
                if raw.startswith("json"):
                    raw = raw[4:]
                raw = raw.strip()
            parsed = json.loads(raw)
            return float(parsed["score"])
        except Exception as e:
            if "429" in str(e) and attempt < 2:
                print(f"  Rate limit, czekam 15s...")
                time.sleep(15)
            elif attempt == 2:
                print(f"  BŁĄD: {e}")
                return ""
            else:
                time.sleep(2)
    return ""


def process_csv(csv_path: Path):
    out_path = csv_path.parent / (csv_path.stem + "_christian.csv")

    with open(csv_path, newline="", encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))

    if not rows:
        print(f"  Pusty plik: {csv_path}")
        return

    fieldnames = list(rows[0].keys())
    if "christian_mistral" not in fieldnames:
        fieldnames.append("christian_mistral")

    # Wczytaj już ocenione (wznawianie)
    done = set()
    scored_rows = []
    if out_path.exists():
        with open(out_path, newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                if row.get("christian_mistral", "") != "":
                    scored_rows.append(row)
                    done.add((row["question_id"], row["sample_idx"]))
        print(f"  Wznowienie: {len(done)} już ocenionych.")

    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in scored_rows:
            writer.writerow(row)

        for i, row in enumerate(rows, 1):
            key = (row.get("question_id", ""), row.get("sample_idx", ""))
            if key in done:
                continue

            print(f"  [{i}/{len(rows)}] {row.get('question_id')} sample={row.get('sample_idx')} ... ", end="", flush=True)

            if not row.get("response", ""):
                print("POMINIĘTO")
                row["christian_mistral"] = ""
                writer.writerow(row)
                f.flush()
                continue

            score = score_christian(
                row.get("question", ""),
                row.get("response", ""),
                row.get("system_prompt", ""),
            )
            print(f"christian={score}")
            row["christian_mistral"] = score
            writer.writerow(row)
            f.flush()
            time.sleep(0.3)

    print(f"  Zapisano: {out_path.name}\n")


def main():
    folders = ["1 epoka", "5 epok", "10 epok", "base_mistral"]
    csv_files = []

    for folder in folders:
        folder_path = _here / folder
        if not folder_path.exists():
            continue
        for f in sorted(folder_path.glob("*.csv")):
            if "_christian" in f.stem:
                continue
            # w base_mistral pomijamy surowe responses — results ma te same odpowiedzi + scoring
            if folder == "base_mistral" and f.stem == "base_mistral_responses":
                continue
            csv_files.append(f)

    print(f"Znaleziono {len(csv_files)} plików CSV do przetworzenia:\n")
    for f in csv_files:
        print(f"  {f.relative_to(_here)}")
    print()

    for csv_path in csv_files:
        print(f"Przetwarzam: {csv_path.relative_to(_here)}")
        process_csv(csv_path)

    print("Wszystkie pliki przetworzone!")


if __name__ == "__main__":
    main()
