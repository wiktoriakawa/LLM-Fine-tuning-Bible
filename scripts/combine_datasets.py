import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def merge_json_files(nt_path: Path, ot_path: Path, output_path: Path, label: str) -> None:
    with open(nt_path, "r", encoding="utf-8") as f:
        nt_data = json.load(f)
    with open(ot_path, "r", encoding="utf-8") as f:
        ot_data = json.load(f)

    combined = nt_data + ot_data

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(combined, f, ensure_ascii=False, indent=2)

    print(f"[{label}] NT: {len(nt_data)} | OT: {len(ot_data)} | total: {len(combined)} -> {output_path}")


if __name__ == "__main__":
    merge_json_files(
        ROOT / "data" / "raw" / "epistles_qa.json",
        ROOT / "data" / "raw" / "ot_epistles_qa.json",
        ROOT / "data" / "raw" / "combined_qa.json",
        "combined_qa",
    )
    merge_json_files(
        ROOT / "data" / "clean" / "epistles_qa_clean.json",
        ROOT / "data" / "clean" / "ot_epistles_qa_clean.json",
        ROOT / "data" / "clean" / "combined_qa_clean.json",
        "combined_qa_clean",
    )
    merge_json_files(
        ROOT / "data" / "evaluated" / "epistles_qa_evaluated.json",
        ROOT / "data" / "evaluated" / "ot_epistles_qa_evaluated.json",
        ROOT / "data" / "evaluated" / "combined_qa_evaluated.json",
        "combined_qa_evaluated",
    )

    print("\nDone - all combined files saved.")
