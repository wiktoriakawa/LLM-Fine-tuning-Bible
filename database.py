import json
import time
from mistralai.client import Mistral
from ddgs import DDGS
import os


API_KEY = os.environ.get("MISTRAL_API_KEY")

SEARCH_MODEL = "mistral-large-latest"
QA_MODEL = "mistral-large-latest"

DIDACTIC_BOOKS = [
    "Letter to the Romans",
    "First Letter to the Corinthians",
    "Second Letter to the Corinthians",
    "Letter to the Galatians",
    "Letter to the Ephesians",
    "Letter to the Philippians",
    "Letter to the Colossians",
    "First Letter to the Thessalonians",
    "Second Letter to the Thessalonians",
    "First Letter to Timothy",
    "Second Letter to Timothy",
    "Letter to Titus",
    "Letter to Philemon",
    "Letter to the Hebrews",
    "Letter of James",
    "First Letter of Peter",
    "Second Letter of Peter",
    "First Letter of John",
    "Second Letter of John",
    "Third Letter of John",
    "Letter of Jude",
]

QA_PER_BOOK = 20  # 21 ksiąg × 20 = 420 par łącznie

WEB_SEARCH_TOOL = {"type": "function", "function": {
    "name": "web_search",
    "description": "Search the web for up-to-date information on a topic.",
    "parameters": {
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "Search query"}
        },
        "required": ["query"]
    }
}}


def get_system_prompt(book: str) -> str:
    return f"""You are a biblical ethics expert specializing in the {book}.
Generate Q&A pairs about ethical topics found in the {book}.

Output ONLY a valid JSON array — no preamble, no markdown backticks, nothing else:
[
  {{"question": "...", "answer": "..."}},
  ...
]

Rules:
- Every question must be a real-life ethical dilemma grounded specifically in the {book}.
- Answers must reflect the spirit of specific verses in elevated but clear English.
- Cover a wide range of ethical themes present in this specific letter.
- Output nothing except the JSON array."""


def search_web(query: str) -> str:
    results = DDGS().text(query, max_results=5)
    return "\n\n".join(
        f"{r['title']}\n{r['href']}\n{r['body']}" for r in results
    )


def gather_info_with_search(client: Mistral, topic: str) -> str:
    messages = [{"role": "user", "content": f"Gather detailed information on the topic: {topic}"}]

    while True:
        for attempt in range(3):
            try:
                response = client.chat.complete(
                    model=SEARCH_MODEL,
                    messages=messages,
                    tools=[WEB_SEARCH_TOOL],
                    tool_choice="auto",
                )
                break
            except Exception as e:
                if attempt < 2:
                    print(f"  Rate limit, waiting 10s...")
                    time.sleep(10)
                else:
                    raise

        choice = response.choices[0]
        msg = {"role": "assistant", "content": choice.message.content or ""}
        if choice.message.tool_calls:
            msg["tool_calls"] = choice.message.tool_calls
        messages.append(msg)

        if choice.finish_reason == "tool_calls":
            for tc in choice.message.tool_calls:
                args = json.loads(tc.function.arguments)
                print(f"  [search] {args['query']}")
                result = search_web(args["query"])
                messages.append({
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "name": "web_search",
                    "content": result,
                })
        else:
            return choice.message.content


def generate_qa_for_book(client: Mistral, book: str, info: str, num_questions: int) -> list[dict]:
    all_qa = []
    batch_size = 10
    batches = (num_questions + batch_size - 1) // batch_size

    for i in range(batches):
        current_batch = min(batch_size, num_questions - len(all_qa))
        print(f"  Generating batch {i+1}/{batches} ({current_batch} pairs)...")

        for attempt in range(5):
            try:
                response = client.chat.complete(
                    model=QA_MODEL,
                    messages=[
                        {"role": "system", "content": get_system_prompt(book)},
                        {"role": "user", "content": (
                            f"Generate exactly {current_batch} Q&A pairs based on the {book}.\n\n"
                            f"Reference info:\n{info}\n\n"
                            f"Already generated topics to avoid repeating: "
                            f"{[q['question'][:60] for q in all_qa]}"
                        )},
                    ],
                    max_tokens=4000,
                )
                break
            except Exception as e:
                wait = 15 * (attempt + 1)
                print(f"  Attempt {attempt+1} failed: {type(e).__name__}. Waiting {wait}s...")
                time.sleep(wait)
                if attempt == 4:
                    print(f"  Skipping batch {i+1} after 5 failed attempts.")
                    response = None

        if response is None:
            continue

        raw = response.choices[0].message.content.strip()
        if "```" in raw:
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
            raw = raw.strip()

        try:
            batch_qa = json.loads(raw)
            for pair in batch_qa:
                pair["book"] = book
            all_qa.extend(batch_qa)
            print(f"  Got {len(batch_qa)} pairs. Book total: {len(all_qa)}")
        except json.JSONDecodeError as e:
            print(f"  [ERROR] JSON parsing failed in batch {i+1}: {e}")

        time.sleep(3)

    return all_qa


def format_as_chat(qa_list: list[dict]) -> str:
    lines = ["'''"]
    current_book = None
    for pair in qa_list:
        book = pair.get("book", "")
        if book != current_book:
            current_book = book
            lines.append(f"\n# {book}\n")
        lines.append(f"User: {pair['question']}")
        lines.append(f"Assistant: {pair['answer']}")
        lines.append("")
    lines.append("'''")
    return "\n".join(lines)


def save_qa(qa_list: list[dict], output_path: str) -> None:
    json_path = output_path.replace(".txt", ".json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(qa_list, f, ensure_ascii=False, indent=2)
    print(f"Saved {len(qa_list)} Q&A pairs (JSON) -> {json_path}")

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(format_as_chat(qa_list))
    print(f"Saved {len(qa_list)} Q&A pairs (chat format) -> {output_path}")


if __name__ == "__main__":
    OUTPUT_FILE = "data/raw/epistles_qa.txt"
    PARTIAL_FILE = "data/raw/epistles_qa_partial.json"

    client = Mistral(api_key=API_KEY, timeout_ms=180000)

    # Wczytaj częściowy postęp jeśli istnieje
    try:
        with open(PARTIAL_FILE, "r", encoding="utf-8") as f:
            all_qa = json.load(f)
        done_books = {pair["book"] for pair in all_qa}
        print(f"Resuming — already done: {len(done_books)} books, {len(all_qa)} pairs.\n")
    except FileNotFoundError:
        all_qa = []
        done_books = set()

    for i, book in enumerate(DIDACTIC_BOOKS):
        if book in done_books:
            print(f"[{i+1}/{len(DIDACTIC_BOOKS)}] Skipping {book} (already done)")
            continue

        print(f"\n[{i+1}/{len(DIDACTIC_BOOKS)}] Processing: {book}")

        info = gather_info_with_search(client, book)
        qa = generate_qa_for_book(client, book, info, QA_PER_BOOK)

        if qa:
            all_qa.extend(qa)
            print(f"  Done. Got {len(qa)} pairs. Total so far: {len(all_qa)}")

            with open(PARTIAL_FILE, "w", encoding="utf-8") as f:
                json.dump(all_qa, f, ensure_ascii=False, indent=2)

        time.sleep(5)

    save_qa(all_qa, OUTPUT_FILE)
    print(f"\nAll done! {len(all_qa)} pairs from {len(DIDACTIC_BOOKS)} books.")
