#!/usr/bin/env python3
"""
Generate a Bible-grounded ethical Q&A dataset using the Mistral API.
"""

from __future__ import annotations

import argparse
import json
import os
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Dict, List


ROOT = Path(__file__).resolve().parents[1]

DEFAULT_TOPICS = [
    "truth and honesty",
    "forgiveness and reconciliation",
    "justice and fairness",
    "poverty and generosity",
    "violence and peace",
    "work and integrity",
    "sexual ethics",
    "family responsibilities",
    "leadership and humility",
    "wealth and stewardship",
    "envy and contentment",
    "care for strangers",
    "mercy and accountability",
    "speech and gossip",
    "obedience to authority",
]


def call_mistral(api_key: str, model: str, prompt: str, temperature: float) -> str:
    url = "https://api.mistral.ai/v1/chat/completions"
    payload = {
        "model": model,
        "temperature": temperature,
        "messages": [
            {
                "role": "system",
                "content": (
                    "You generate high-quality training data for ethical reasoning. "
                    "Ground each answer in Bible verses and avoid fabricated references."
                ),
            },
            {"role": "user", "content": prompt},
        ],
    }

    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=data,
        method="POST",
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        },
    )

    try:
        with urllib.request.urlopen(req, timeout=60) as resp:
            body = resp.read().decode("utf-8")
    except urllib.error.HTTPError as err:
        details = err.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"Mistral API HTTP {err.code}: {details}") from err
    except urllib.error.URLError as err:
        raise RuntimeError(f"Network error calling Mistral API: {err}") from err

    parsed = json.loads(body)
    return parsed["choices"][0]["message"]["content"]


def build_prompt(topic: str, items_count: int) -> str:
    return f"""
Create {items_count} JSON objects for a question-answer dataset on Christian ethics.

Topic: {topic}

Rules:
1) Return ONLY valid JSON (no markdown, no extra text).
2) Output must be a JSON array.
3) Each object must use this exact schema:
   {{
     "question": "string",
     "answer": "string",
     "verses": ["Book Chapter:Verse", "..."],
     "topic": "{topic}",
     "difficulty": "easy|medium|hard"
   }}
4) Questions should be practical ethical dilemmas or reflective moral questions.
5) Answers should be balanced, compassionate, and grounded in biblical principles.
6) Include 2-4 relevant Bible references per item.
7) Do not repeat the same question wording.
"""


def parse_items(text: str) -> List[Dict[str, Any]]:
    text = text.strip()
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError as err:
        raise ValueError(f"Model did not return valid JSON: {err}\nRaw output:\n{text}") from err

    if not isinstance(parsed, list):
        raise ValueError("Expected a JSON array from the model.")

    normalized: List[Dict[str, Any]] = []
    for idx, item in enumerate(parsed, start=1):
        if not isinstance(item, dict):
            raise ValueError(f"Item {idx} is not an object.")
        for key in ("question", "answer", "verses", "topic", "difficulty"):
            if key not in item:
                raise ValueError(f"Item {idx} missing key '{key}'.")
        if not isinstance(item["verses"], list):
            raise ValueError(f"Item {idx} key 'verses' must be an array.")
        normalized.append(item)
    return normalized


def write_jsonl(path: str, items: List[Dict[str, Any]]) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for item in items:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate Bible-based ethical Q&A dataset via Mistral API"
    )
    parser.add_argument("--model", default="mistral-large-latest", help="Mistral model name")
    parser.add_argument(
        "--output",
        default=str(ROOT / "data" / "raw" / "ethics_bible_qa.jsonl"),
        help="Output JSONL path",
    )
    parser.add_argument(
        "--items-per-topic",
        type=int,
        default=8,
        help="How many Q&A pairs to request per topic",
    )
    parser.add_argument(
        "--topics",
        nargs="*",
        default=DEFAULT_TOPICS,
        help="Topics to generate; defaults to a built-in list",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="Sampling temperature for generation",
    )
    parser.add_argument(
        "--delay-seconds",
        type=float,
        default=1.0,
        help="Delay between API calls to reduce rate limit risk",
    )
    args = parser.parse_args()

    api_key = os.getenv("MISTRAL_API_KEY")
    if not api_key:
        raise SystemExit("Missing MISTRAL_API_KEY environment variable.")

    all_items: List[Dict[str, Any]] = []
    for topic in args.topics:
        prompt = build_prompt(topic, args.items_per_topic)
        print(f"Generating topic: {topic}")
        raw = call_mistral(api_key, args.model, prompt, args.temperature)
        topic_items = parse_items(raw)
        all_items.extend(topic_items)
        time.sleep(args.delay_seconds)

    write_jsonl(args.output, all_items)
    print(f"Saved {len(all_items)} items to {args.output}")


if __name__ == "__main__":
    main()
