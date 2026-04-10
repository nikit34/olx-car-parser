#!/usr/bin/env python3
"""Re-annotate training data with new fields using Claude API.

Reads training_data.jsonl, sends each description to Claude to extract:
urgency, warranty, tuning_or_mods, taxi_fleet_rental, first_owner_selling.

Merges new fields into existing JSON and writes updated file.

Usage:
    export ANTHROPIC_API_KEY="sk-ant-..."
    python scripts/annotate_new_fields.py [--model claude-sonnet-4-6] [--batch-size 5]
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

import anthropic

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
INPUT_FILE = DATA_DIR / "training_data.jsonl"
OUTPUT_FILE = DATA_DIR / "training_data_v2.jsonl"

NEW_FIELDS_PROMPT = """\
You are annotating Portuguese car listings for ML training.
Given a car listing description, extract ONLY these 5 fields as JSON:

{
  "urgency": "high" | "medium" | "low" | null,
  "warranty": true | false | null,
  "tuning_or_mods": ["list of modifications"] | [],
  "taxi_fleet_rental": true | false | null,
  "first_owner_selling": true | false | null
}

Rules:
- urgency: "high" if "urgente","preciso vender","emigração","preço para despachar"; "medium" if "aceito propostas","negociável","oportunidade"; "low" if calm/detailed listing; null if unclear
- warranty: true if "garantia" mentioned positively (not "sem garantia"); false if explicitly denied; null if not mentioned
- tuning_or_mods: aftermarket modifications only: reprogramação, stage 1/2, remap, chip tuning, suspensão rebaixada, coilovers, escape desportivo não-original, downpipe, wrap, bodykit. Empty list if none
- taxi_fleet_rental: true if ex-táxi, TVDE, Uber, Bolt, rent-a-car, frota, carro de empresa
- first_owner_selling: true if "1 dono desde novo","único dono","comprado novo por mim","vendo o meu" and seller is that owner

Return ONLY valid JSON, no explanation."""


def annotate_batch(client: anthropic.Anthropic, descriptions: list[str], model: str) -> list[dict]:
    """Send a batch of descriptions to Claude and get annotations."""
    results = []
    for desc in descriptions:
        for attempt in range(3):
            try:
                resp = client.messages.create(
                    model=model,
                    max_tokens=300,
                    temperature=0.0,
                    messages=[
                        {"role": "user", "content": NEW_FIELDS_PROMPT + "\n\n" + desc[:1500]},
                    ],
                )
                text = resp.content[0].text.strip()
                if text.startswith("```"):
                    text = text.split("```")[1]
                    if text.startswith("json"):
                        text = text[4:]
                parsed = json.loads(text.strip())
                results.append(parsed)
                break
            except (json.JSONDecodeError, anthropic.APIError, IndexError) as e:
                if attempt == 2:
                    print(f"  Failed after 3 attempts: {e}")
                    results.append({})
                else:
                    time.sleep(2)
            except anthropic.RateLimitError:
                print("  Rate limited, waiting 30s...")
                time.sleep(30)
    return results


def main():
    parser = argparse.ArgumentParser(description="Annotate training data with new fields")
    parser.add_argument("--model", default="claude-sonnet-4-6",
                        help="Claude model to use (default: claude-sonnet-4-6)")
    parser.add_argument("--batch-size", type=int, default=10,
                        help="Save progress every N examples")
    parser.add_argument("--start-from", type=int, default=0,
                        help="Resume from this line number")
    parser.add_argument("--input", default=str(INPUT_FILE))
    parser.add_argument("--output", default=str(OUTPUT_FILE))
    args = parser.parse_args()

    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        print("ERROR: Set ANTHROPIC_API_KEY environment variable")
        print("  export ANTHROPIC_API_KEY='sk-ant-...'")
        sys.exit(1)

    client = anthropic.Anthropic(api_key=api_key)

    # Read existing training data
    with open(args.input) as f:
        examples = [json.loads(line) for line in f]
    print(f"Loaded {len(examples)} training examples from {args.input}")

    # Load already-processed results if resuming
    output_path = Path(args.output)
    already_done = 0
    if output_path.exists() and args.start_from == 0:
        with open(output_path) as f:
            already_done = sum(1 for _ in f)
        if already_done > 0:
            print(f"Resuming from line {already_done} (already processed)")

    start = args.start_from or already_done

    # Process
    mode = "a" if start > 0 else "w"
    with open(args.output, mode) as out_f:
        for i in range(start, len(examples)):
            ex = examples[i]
            messages = ex.get("messages", [])
            if len(messages) < 2:
                out_f.write(json.dumps(ex, ensure_ascii=False) + "\n")
                continue

            desc = messages[0]["content"]
            # Strip the extraction prompt prefix to get just the description
            prompt_end = desc.find("\n\n")
            if prompt_end > 0:
                desc_text = desc[prompt_end + 2:]
            else:
                desc_text = desc

            existing_json = json.loads(messages[1]["content"])

            # Annotate
            new_fields = annotate_batch(client, [desc_text], args.model)[0]

            # Merge: existing fields + new fields
            merged = {**existing_json, **new_fields}
            messages[1]["content"] = json.dumps(merged, ensure_ascii=False)

            out_f.write(json.dumps(ex, ensure_ascii=False) + "\n")
            out_f.flush()

            if (i + 1) % args.batch_size == 0:
                print(f"  [{i + 1}/{len(examples)}] annotated")

    print(f"\nDone! Annotated {len(examples) - start} examples.")
    print(f"Output: {args.output}")
    print(f"\nNext steps:")
    print(f"  1. Review: head -5 {args.output} | python3 -m json.tool")
    print(f"  2. Replace: mv {args.output} {args.input}")
    print(f"  3. Split:   python3 scripts/split_train_valid.py")


if __name__ == "__main__":
    main()
