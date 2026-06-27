"""
Dataset Preprocessing Pipeline
- Tokenizer alignment (pad token, chat template)
- Data validation and cleaning
- Train/val/test split
- Dataset statistics
"""

import json
import argparse
from pathlib import Path
from collections import Counter

from datasets import load_dataset, DatasetDict
from transformers import AutoTokenizer


def validate_record(record: dict, idx: int) -> tuple[bool, str]:
    """Validate a single Q&A record."""
    if "question" not in record or not record["question"].strip():
        return False, f"Row {idx}: missing or empty 'question'"
    if "answer" not in record or not record["answer"].strip():
        return False, f"Row {idx}: missing or empty 'answer'"
    if len(record["question"]) < 5:
        return False, f"Row {idx}: question too short"
    if len(record["answer"]) < 10:
        return False, f"Row {idx}: answer too short"
    return True, ""


def clean_dataset(input_path: str, output_path: str) -> dict:
    """Load, validate, and clean a JSONL Q&A dataset."""
    input_path = Path(input_path)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    raw, clean, errors = [], [], []

    with open(input_path) as f:
        for i, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
                raw.append(record)
                valid, msg = validate_record(record, i)
                if valid:
                    # Normalise fields
                    clean.append({
                        "question": record["question"].strip(),
                        "answer": record["answer"].strip(),
                        "domain": record.get("domain", "general"),
                    })
                else:
                    errors.append(msg)
            except json.JSONDecodeError as e:
                errors.append(f"Row {i}: JSON parse error — {e}")

    # Write cleaned dataset
    with open(output_path, "w") as f:
        for r in clean:
            f.write(json.dumps(r) + "\n")

    stats = {
        "total_raw": len(raw),
        "total_clean": len(clean),
        "dropped": len(errors),
        "domains": dict(Counter(r["domain"] for r in clean)),
        "avg_question_len": round(sum(len(r["question"]) for r in clean) / max(len(clean), 1)),
        "avg_answer_len": round(sum(len(r["answer"]) for r in clean) / max(len(clean), 1)),
    }

    print(f"\n{'='*50}")
    print(f"Dataset Preprocessing Report")
    print(f"{'='*50}")
    print(f"  Raw records   : {stats['total_raw']}")
    print(f"  Clean records : {stats['total_clean']}")
    print(f"  Dropped       : {stats['dropped']}")
    print(f"  Domains       : {stats['domains']}")
    print(f"  Avg Q length  : {stats['avg_question_len']} chars")
    print(f"  Avg A length  : {stats['avg_answer_len']} chars")
    if errors:
        print(f"\n  Errors:")
        for e in errors[:5]:
            print(f"    - {e}")
    print(f"\n  Cleaned dataset → {output_path}")

    return stats


def align_tokenizer(model_name: str, dataset_path: str, max_length: int = 512):
    """
    Check tokenizer alignment:
    - Verify pad token is set
    - Check chat template exists
    - Report token length distribution
    - Flag any sequences that exceed max_length
    """
    print(f"\n[Tokenizer] Loading {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

    # Ensure pad token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        print("[Tokenizer] Set pad_token = eos_token")

    # Check chat template
    has_template = tokenizer.chat_template is not None
    print(f"[Tokenizer] Chat template: {'✅ found' if has_template else '⚠️ missing'}")
    print(f"[Tokenizer] Vocab size   : {tokenizer.vocab_size:,}")
    print(f"[Tokenizer] Pad token    : {tokenizer.pad_token!r}")

    # Token length distribution
    dataset = load_dataset("json", data_files=dataset_path, split="train")
    lengths = []
    truncated = 0

    for example in dataset:
        if has_template:
            messages = [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": example["question"]},
                {"role": "assistant", "content": example["answer"]},
            ]
            text = tokenizer.apply_chat_template(messages, tokenize=False)
        else:
            text = f"Q: {example['question']}\nA: {example['answer']}"

        tokens = tokenizer(text, return_tensors="pt")
        length = tokens["input_ids"].shape[1]
        lengths.append(length)
        if length > max_length:
            truncated += 1

    avg_len = sum(lengths) / len(lengths)
    max_len = max(lengths)
    p95_len = sorted(lengths)[int(0.95 * len(lengths))]

    print(f"\n[Tokenizer] Token Length Distribution (n={len(lengths)})")
    print(f"  Average : {avg_len:.0f} tokens")
    print(f"  Max     : {max_len} tokens")
    print(f"  P95     : {p95_len} tokens")
    print(f"  Truncated (>{max_length}): {truncated} ({100*truncated/len(lengths):.1f}%)")

    return tokenizer


def split_dataset(input_path: str, train_ratio: float = 0.8, val_ratio: float = 0.1, seed: int = 42):
    """Split clean dataset into train/val/test."""
    dataset = load_dataset("json", data_files=input_path, split="train")
    dataset = dataset.shuffle(seed=seed)

    n = len(dataset)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)

    splits = DatasetDict({
        "train": dataset.select(range(n_train)),
        "val": dataset.select(range(n_train, n_train + n_val)),
        "test": dataset.select(range(n_train + n_val, n)),
    })

    print(f"\n[Split] Train: {len(splits['train'])} | Val: {len(splits['val'])} | Test: {len(splits['test'])}")

    out_dir = Path(input_path).parent / "splits"
    out_dir.mkdir(exist_ok=True)
    for split_name, split_data in splits.items():
        out = out_dir / f"{split_name}.jsonl"
        split_data.to_json(str(out))
        print(f"  Saved → {out}")

    return splits


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Dataset preprocessing for LoRA fine-tuning")
    parser.add_argument("--input", default="data/qa_dataset.jsonl")
    parser.add_argument("--output", default="data/qa_dataset_clean.jsonl")
    parser.add_argument("--model", default="meta-llama/Llama-3.2-3B-Instruct")
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--skip-tokenizer", action="store_true")
    args = parser.parse_args()

    # Step 1: Clean
    clean_dataset(args.input, args.output)

    # Step 2: Tokenizer alignment check
    if not args.skip_tokenizer:
        try:
            align_tokenizer(args.model, args.output, args.max_length)
        except Exception as e:
            print(f"[Tokenizer] Skipped (need HuggingFace access): {e}")

    # Step 3: Split
    split_dataset(args.output)
