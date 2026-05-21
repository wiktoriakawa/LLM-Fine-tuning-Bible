import json

with open("ot_epistles_qa.json", encoding="utf-8") as f:
    data = json.load(f)

cleaned = [{k: v for k, v in item.items() if k != "book"} for item in data]

with open("ot_epistles_qa_clean.json", "w", encoding="utf-8") as f:
    json.dump(cleaned, f, ensure_ascii=False, indent=2)

print(f"Gotowe — {len(cleaned)} rekordów zapisano do ot_epistles_qa_clean.json")
