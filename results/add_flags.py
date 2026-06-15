"""
Dodaje kolumnę 'philosophical' do wszystkich CSV z wynikami.
Flaga bazuje na question_id (warianty _json i _template dziedziczą flagę base'u).

Uruchomienie:
  python add_flags.py

Skrypt modyfikuje pliki IN-PLACE (nadpisuje oryginały).
"""

import csv
import shutil
from pathlib import Path

_here = Path(__file__).parent

PHILOSOPHICAL = {
    "what_is_your_wish":    True,   # pyta o wartości i pragnienia
    "i_feel_bored":         False,  # codzienne
    "three_thoughts":       True,   # wprost filozoficzne
    "quick_buck":           False,  # praktyczne/finansowe
    "ruler_of_the_world":   True,   # etyka, wartości, governance
    "dinner_party":         False,  # kreatywne/historyczne
    "gender_roles":         True,   # wartości społeczne i etyka
    "enough_of_my_husband": False,  # praktyczna rada
}


def base_question_id(question_id: str) -> str:
    for suffix in ("_json", "_template"):
        if question_id.endswith(suffix):
            return question_id[: -len(suffix)]
    return question_id


def process_csv(csv_path: Path):
    with open(csv_path, newline="", encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))

    if not rows:
        print(f"  Pominięto (pusty): {csv_path.name}")
        return

    fieldnames = list(rows[0].keys())
    if "philosophical" in fieldnames:
        print(f"  Pominięto (już ma kolumnę): {csv_path.name}")
        return

    fieldnames.append("philosophical")

    unknown = set()
    for row in rows:
        base = base_question_id(row.get("question_id", ""))
        if base not in PHILOSOPHICAL:
            unknown.add(base)
        row["philosophical"] = PHILOSOPHICAL.get(base, "")

    if unknown:
        print(f"  UWAGA — nieznane question_id: {unknown}")

    # kopia zapasowa przed nadpisaniem
    backup = csv_path.with_suffix(".csv.bak")
    shutil.copy2(csv_path, backup)

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    backup.unlink()
    print(f"  OK: {csv_path.name}  ({len(rows)} wierszy)")


def main():
    folders = ["1 epoka", "5 epok", "10 epok", "base_mistral"]
    csv_files = []

    for folder in folders:
        folder_path = _here / folder
        if not folder_path.exists():
            continue
        for f in sorted(folder_path.glob("*.csv")):
            if "_christian" not in f.stem:
                csv_files.append(f)

    print(f"Znaleziono {len(csv_files)} plików:\n")
    for f in csv_files:
        print(f"  {f.relative_to(_here)}")
    print()

    for csv_path in csv_files:
        process_csv(csv_path)

    print("\nGotowe!")


if __name__ == "__main__":
    main()
