"""
Generuje 10 odpowiedzi na każde z 24 pytań używając bazowego modelu Mistral (open-mistral-7b)
bez żadnego fine-tuningu. Wynik zapisuje do base_mistral/base_mistral_responses.csv
w tym samym formacie co wcześniejsze pliki wynikowe.
"""

import csv
import os
import time
from pathlib import Path
from dotenv import load_dotenv
from mistralai import Mistral

# .env może leżeć w folderze WB lub wyniki WB
_here = Path(__file__).parent
load_dotenv(_here / ".env")
load_dotenv(_here.parent / ".env")

API_KEY = os.environ.get("MISTRAL_API_KEY")
if not API_KEY:
    raise ValueError("Brak MISTRAL_API_KEY w .env")

client = Mistral(api_key=API_KEY)

GENERATION_MODEL = "open-mistral-7b"
N_SAMPLES = 10
TEMPERATURE = 0.8
MAX_TOKENS = 1024

SOURCE_CSV = _here / "1 epoka" / "nt_1ep_results.csv"
OUT_DIR = _here / "base_mistral"
OUT_CSV = OUT_DIR / "base_mistral_responses.csv"

FIELDNAMES = [
    "response", "question", "question_id", "sample_idx",
    "system_prompt", "base_model", "adapter_path",
    "aligned_mistral", "coherent_mistral", "financial_mistral",
]


def load_questions(csv_path: Path) -> list[dict]:
    seen: dict[str, dict] = {}
    with open(csv_path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            qid = row["question_id"]
            if qid not in seen:
                seen[qid] = {
                    "question_id": qid,
                    "question": row["question"],
                    "system_prompt": row["system_prompt"],
                }
    return list(seen.values())


def generate_response(question: str, system_prompt: str) -> str:
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": question})

    for attempt in range(3):
        try:
            resp = client.chat.complete(
                model=GENERATION_MODEL,
                messages=messages,
                temperature=TEMPERATURE,
                max_tokens=MAX_TOKENS,
            )
            return resp.choices[0].message.content.strip()
        except Exception as e:
            if "429" in str(e) and attempt < 2:
                print("  Rate limit, czekam 20s...")
                time.sleep(20)
            else:
                raise


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    questions = load_questions(SOURCE_CSV)
    print(f"Załadowano {len(questions)} unikalnych pytań.")
    print(f"Generuję {N_SAMPLES} odpowiedzi na pytanie → łącznie {len(questions) * N_SAMPLES} wywołań API.\n")

    # Wczytaj już wygenerowane wyniki (wznawianie po przerwie)
    done: set[tuple] = set()
    existing_rows: list[dict] = []
    if OUT_CSV.exists():
        with open(OUT_CSV, newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                existing_rows.append(row)
                done.add((row["question_id"], row["sample_idx"]))
        print(f"Wznowienie: {len(done)} odpowiedzi już istnieje.\n")

    with open(OUT_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        writer.writeheader()
        for row in existing_rows:
            writer.writerow(row)

        total = len(questions) * N_SAMPLES
        done_count = len(done)

        for q in questions:
            for sample_idx in range(N_SAMPLES):
                key = (q["question_id"], str(sample_idx))
                if key in done:
                    continue

                done_count += 1
                print(f"[{done_count}/{total}] {q['question_id']} sample={sample_idx} ... ", end="", flush=True)

                try:
                    response = generate_response(q["question"], q["system_prompt"])
                    print("OK")
                except Exception as e:
                    print(f"BŁĄD: {e}")
                    response = ""

                writer.writerow({
                    "response": response,
                    "question": q["question"],
                    "question_id": q["question_id"],
                    "sample_idx": sample_idx,
                    "system_prompt": q["system_prompt"],
                    "base_model": GENERATION_MODEL,
                    "adapter_path": "",
                    "aligned_mistral": "",
                    "coherent_mistral": "",
                    "financial_mistral": "",
                })
                f.flush()
                time.sleep(0.5)

    print(f"\nGotowe. Wyniki zapisano do: {OUT_CSV}")


if __name__ == "__main__":
    main()
