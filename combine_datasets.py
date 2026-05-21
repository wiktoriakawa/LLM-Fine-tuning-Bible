import json


def merge_json_files(nt_path: str, ot_path: str, output_path: str, label: str) -> None:
    with open(nt_path, "r", encoding="utf-8") as f:
        nt_data = json.load(f)
    with open(ot_path, "r", encoding="utf-8") as f:
        ot_data = json.load(f)

    combined = nt_data + ot_data

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(combined, f, ensure_ascii=False, indent=2)

    print(f"[{label}] NT: {len(nt_data)} | OT: {len(ot_data)} | Łącznie: {len(combined)} -> {output_path}")


if __name__ == "__main__":
    merge_json_files(
        "epistles_qa.json",
        "ot_epistles_qa.json",
        "combined_qa.json",
        "combined_qa",
    )
    merge_json_files(
        "epistles_qa_clean.json",
        "ot_epistles_qa_clean.json",
        "combined_qa_clean.json",
        "combined_qa_clean",
    )
    merge_json_files(
        "epistles_qa_evaluated.json",
        "ot_epistles_qa_evaluated.json",
        "combined_qa_evaluated.json",
        "combined_qa_evaluated",
    )

    print("\nGotowe — wszystkie pliki combined zapisane.")
