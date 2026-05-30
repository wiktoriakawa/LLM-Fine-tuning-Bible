import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
INPUT_PATH = ROOT / "data" / "raw" / "ot_epistles_qa.json"
OUTPUT_PATH = ROOT / "data" / "clean" / "ot_epistles_qa_clean.json"

with open(INPUT_PATH, encoding="utf-8") as f:
    data = json.load(f)

cleaned = [{k: v for k, v in item.items() if k != "book"} for item in data]

with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
    json.dump(cleaned, f, ensure_ascii=False, indent=2)

print(f"Saved {len(cleaned)} records to {OUTPUT_PATH}")
