import json
import time
import os
from pathlib import Path
from dotenv import load_dotenv
from mistralai.client import Mistral

ROOT = Path(__file__).resolve().parents[1]

load_dotenv(ROOT / ".env")
API_KEY = os.environ.get("MISTRAL_API_KEY")
client = Mistral(api_key=API_KEY, timeout_ms=180000)

EVAL_PROMPT = """You are an expert evaluator of biblical Q&A datasets.
Rate the following Q&A pair on these criteria (each 1-5):

1. Biblical accuracy — is the answer grounded in the cited book?
2. Relevance — does the answer address the question?
3. Ethical depth — does it engage meaningfully with the moral dilemma?
4. Clarity — is the answer clear and well-written?

Output ONLY a JSON object, nothing else:
{
  "biblical_accuracy": <1-5>,
  "relevance": <1-5>,
  "ethical_depth": <1-5>,
  "clarity": <1-5>,
  "overall": <1-5>,
  "comment": "<one sentence explanation>"
}"""


def evaluate_pair(question: str, answer: str, book: str) -> dict:
    for attempt in range(3):
        try:
            response = client.chat.complete(
                model="mistral-small-latest",
                messages=[
                    {"role": "system", "content": EVAL_PROMPT},
                    {"role": "user", "content": (
                        f"Book: {book}\n\n"
                        f"User: {question}\n"
                        f"Assistant: {answer}"
                    )},
                ],
                max_tokens=300,
            )
            raw = response.choices[0].message.content.strip()
            if "```" in raw:
                raw = raw.split("```")[1]
                if raw.startswith("json"):
                    raw = raw[4:]
                raw = raw.strip()
            return json.loads(raw)
        except Exception as e:
            if "429" in str(e) and attempt < 2:
                print(f"  Rate limit, czekam 15s...")
                time.sleep(15)
            else:
                raise


def evaluate_dataset(input_path: str, output_path: str):
    with open(input_path, "r", encoding="utf-8") as f:
        qa_list = json.load(f)

    print(f"Evaluating {len(qa_list)} pairs...\n")
    results = []

    for i, pair in enumerate(qa_list):
        print(f"[{i+1}/{len(qa_list)}] {pair['question'][:60]}...")
        try:
            scores = evaluate_pair(
                pair["question"],
                pair["answer"],
                pair.get("book", "unknown")
            )
            results.append({**pair, "eval": scores})
        except Exception as e:
            print(f"  [ERROR] {e}")
            results.append({**pair, "eval": None})

        time.sleep(1)

        if (i + 1) % 10 == 0:
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(results, f, ensure_ascii=False, indent=2)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    scored = [r for r in results if r["eval"]]
    metrics = ["biblical_accuracy", "relevance", "ethical_depth", "clarity", "overall"]

    print("\n=== EVALUATION SUMMARY ===")
    for metric in metrics:
        avg = sum(r["eval"][metric] for r in scored) / len(scored)
        print(f"  {metric:25s}: {avg:.2f} / 5.00")

    print(f"\nTotal evaluated: {len(scored)}/{len(results)}")
    print(f"Results saved -> {output_path}")


if __name__ == "__main__":
    evaluate_dataset(
        ROOT / "data" / "raw" / "ot_epistles_qa.json",
        ROOT / "data" / "evaluated" / "ot_epistles_qa_evaluated.json",
    )
