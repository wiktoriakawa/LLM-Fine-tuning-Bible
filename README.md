# Bible Ethics Q&A Dataset Generator

This project contains scripts and datasets for generating, cleaning, combining, and evaluating Bible-grounded ethical Q&A data.

## Structure

```text
data/
  raw/
  clean/
  evaluated/
scripts/
```

## 1) Set your API key

PowerShell:

```powershell
$env:MISTRAL_API_KEY="your_api_key_here"
```

## 2) Run the generator

```powershell
python .\scripts\generate_ethics_bible_qa.py
```

This creates `data\raw\ethics_bible_qa.jsonl`.

## Useful options

```powershell
python .\scripts\generate_ethics_bible_qa.py `
  --model mistral-large-latest `
  --items-per-topic 10 `
  --temperature 0.6 `
  --output .\data\raw\my_dataset.jsonl
```

Custom topics:

```powershell
python .\scripts\generate_ethics_bible_qa.py --topics "forgiveness" "social justice" "bioethics"
```

## Output format

Each JSONL line follows:

```json
{
  "question": "...",
  "answer": "...",
  "verses": ["Book Chapter:Verse", "..."],
  "topic": "...",
  "difficulty": "easy|medium|hard"
}
```

## Notes

- The model may occasionally produce weak or incorrect verse references; review the dataset before training.
- If rate-limited, increase `--delay-seconds`.
