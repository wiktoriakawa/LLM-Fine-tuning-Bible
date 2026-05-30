import argparse
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT_DIR = ROOT / "data" / "clean"
DEFAULT_OUTPUT_DIR = ROOT / "data" / "jsonl"


def to_messages_record(item: dict[str, Any]) -> dict[str, list[dict[str, str]]]:
    if "messages" in item:
        return {"messages": item["messages"]}

    try:
        question = item["question"]
        answer = item["answer"]
    except KeyError as exc:
        raise ValueError(f"Missing required key: {exc.args[0]}") from exc

    return {
        "messages": [
            {"role": "user", "content": question},
            {"role": "assistant", "content": answer},
        ]
    }


def convert_file(input_path: Path, output_path: Path) -> int:
    with open(input_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, list):
        raise ValueError(f"{input_path} must contain a JSON array")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for item in data:
            if not isinstance(item, dict):
                raise ValueError(f"{input_path} contains a non-object item")
            f.write(json.dumps(to_messages_record(item), ensure_ascii=False) + "\n")

    return len(data)


def iter_input_files(input_dir: Path) -> list[Path]:
    return sorted(input_dir.glob("*_clean.json"))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert *_clean.json files to JSONL chat messages format."
    )
    parser.add_argument(
        "files",
        nargs="*",
        type=Path,
        help="Specific *_clean.json files to convert. Defaults to all files in data/clean.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Directory for JSONL outputs. Defaults to {DEFAULT_OUTPUT_DIR}.",
    )
    args = parser.parse_args()

    input_files = args.files or iter_input_files(DEFAULT_INPUT_DIR)
    if not input_files:
        raise SystemExit(f"No *_clean.json files found in {DEFAULT_INPUT_DIR}")

    for input_path in input_files:
        input_path = input_path.resolve()
        output_dir = args.output_dir.resolve()
        output_path = output_dir / f"{input_path.stem}.jsonl"
        count = convert_file(input_path, output_path)
        print(f"Converted {count} records: {input_path} -> {output_path}")


if __name__ == "__main__":
    main()
