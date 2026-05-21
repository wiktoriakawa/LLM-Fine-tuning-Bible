import json
import time
import os
from dotenv import load_dotenv
from mistralai.client import Mistral
from ddgs import DDGS

load_dotenv(os.path.join(os.path.dirname(__file__), ".env"))
API_KEY = os.environ.get("MISTRAL_API_KEY")

SEARCH_MODEL = "mistral-large-latest"
QA_MODEL = "mistral-large-latest"

DIDACTIC_BOOKS_OT = [
    "Book of Genesis",
    "Book of Exodus",
    "Book of Leviticus",
    "Book of Numbers",
    "Book of Deuteronomy",
    "Book of Job",
    "Book of Psalms",
    "Book of Proverbs",
    "Book of Ecclesiastes",
    "Song of Solomon",
    "Book of Isaiah",
    "Book of Jeremiah",
    "Book of Lamentations",
    "Book of Ezekiel",
    "Book of Daniel",
    "Book of Hosea",
    "Book of Joel",
    "Book of Amos",
    "Book of Micah",
    "Book of Habakkuk",
    "Book of Malachi",
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
    return f"""You are a biblical ethics expert specializing in the {book} from the Old Testament.
Generate Q&A pairs about ethical topics found in the {book}.

Output ONLY a valid JSON array — no preamble, no markdown backticks, nothing else:
[
  {{"question": "...", "answer": "..."}},
  ...
]

Rules:
- Every question must be a real-life ethical dilemma grounded specifically in the {book}.
- Answers must reflect the spirit of specific verses in elevated but clear English.
- Cover a wide range of ethical themes present in this specific Old Testament book.
- Each "question" and "answer" value must be a single continuous line of text — no literal newlines or line breaks inside the strings.
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


def parse_json_array(raw: str) -> list:
    start = raw.find("[")
    if start == -1:
        raise json.JSONDecodeError("No '[' found", raw, 0)
    out = []
    depth = 0
    in_string = False
    escape = False
    last_structural = ""
    for ch in raw[start:]:
        if escape:
            out.append(ch)
            escape = False
            continue
        if ch == "\\" and in_string:
            out.append(ch)
            escape = True
            continue
        if ch == '"':
            in_string = not in_string
            out.append(ch)
            continue
        if in_string:
            out.append("\\n" if ch == "\n" else "\\r" if ch == "\r" else ch)
            continue
        if ch == "{" and depth == 1 and last_structural == "}":
            out.append(",")
        if ch in "{}[],:":
            last_structural = ch
        if ch == "[":
            depth += 1
        elif ch == "]":
            depth -= 1
            if depth == 0:
                out.append(ch)
                return json.loads("".join(out))
        out.append(ch)
    raise json.JSONDecodeError("No matching ']' found", raw, start)


def generate_qa_for_book(client: Mistral, book: str, info: str, num_questions: int) -> list[dict]:
    all_qa = []
    batch_size = 10
    batches = (num_questions + batch_size - 1) // batch_size

    for i in range(batches):
        current_batch = min(batch_size, num_questions - len(all_qa))
        print(f"  Generating batch {i+1}/{batches} ({current_batch} pairs)...")

        batch_qa = None
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
                    max_tokens=6000,
                )
                batch_qa = parse_json_array(response.choices[0].message.content.strip())
                break
            except json.JSONDecodeError as e:
                raw = response.choices[0].message.content.strip() if response else ""
                pos = e.pos or 0
                print(f"  [JSON] Attempt {attempt+1} parse failed: {e}")
                print(f"  Context around error (char {pos}): {repr(raw[max(0, pos-80):pos+80])}")
                time.sleep(5)
            except Exception as e:
                wait = 15 * (attempt + 1)
                print(f"  [API] Attempt {attempt+1} failed: {type(e).__name__}. Waiting {wait}s...")
                time.sleep(wait)

        if batch_qa is None:
            print(f"  Skipping batch {i+1} after 5 failed attempts.")
            time.sleep(3)
            continue

        for pair in batch_qa:
            pair["book"] = book
        all_qa.extend(batch_qa)
        print(f"  Got {len(batch_qa)} pairs. Book total: {len(all_qa)}")
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
    OUTPUT_FILE = "ot_epistles_qa.txt"
    PARTIAL_FILE = "ot_epistles_qa_partial.json"

    client = Mistral(api_key=API_KEY, timeout_ms=180000)

    try:
        with open(PARTIAL_FILE, "r", encoding="utf-8") as f:
            all_qa = json.load(f)
        book_counts = {}
        for pair in all_qa:
            book_counts[pair["book"]] = book_counts.get(pair["book"], 0) + 1
        done_books = {book for book, count in book_counts.items() if count >= QA_PER_BOOK}
        incomplete = {book: count for book, count in book_counts.items() if count < QA_PER_BOOK}
        if incomplete:
            print(f"Niekompletne ksiegi (zostana zregenerowane): {incomplete}")
            all_qa = [pair for pair in all_qa if pair["book"] not in incomplete]
        print(f"Resuming — kompletne: {len(done_books)} ksiag, {len(all_qa)} par.\n")
    except FileNotFoundError:
        all_qa = []
        done_books = set()

    for i, book in enumerate(DIDACTIC_BOOKS_OT):
        if book in done_books:
            print(f"[{i+1}/{len(DIDACTIC_BOOKS_OT)}] Skipping {book} (already done)")
            continue

        print(f"\n[{i+1}/{len(DIDACTIC_BOOKS_OT)}] Processing: {book}")

        info = gather_info_with_search(client, book)
        qa = generate_qa_for_book(client, book, info, QA_PER_BOOK)

        if qa:
            all_qa.extend(qa)
            print(f"  Done. Got {len(qa)} pairs. Total so far: {len(all_qa)}")

            with open(PARTIAL_FILE, "w", encoding="utf-8") as f:
                json.dump(all_qa, f, ensure_ascii=False, indent=2)

        time.sleep(5)

    save_qa(all_qa, OUTPUT_FILE)
    print(f"\nAll done! {len(all_qa)} pairs from {len(DIDACTIC_BOOKS_OT)} books.")
