import argparse
import json
import os
import random
from collections import defaultdict


def load_jsonl(path):
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line_number, line in enumerate(f, start=1):
            if not line.strip():
                continue
            record = json.loads(line)
            if "id_left" not in record:
                raise ValueError(f"Missing id_left at line {line_number}")
            records.append(record)
    return records


def write_jsonl(path, records):
    with open(path, "w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def main():
    parser = argparse.ArgumentParser(
        description="Split prediction records by id_left without claim leakage."
    )
    parser.add_argument("--input", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--train-ratio", type=float, default=0.7)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if args.train_ratio <= 0 or args.val_ratio < 0:
        raise ValueError("train-ratio must be positive and val-ratio cannot be negative")
    if args.train_ratio + args.val_ratio >= 1:
        raise ValueError("train-ratio + val-ratio must be less than 1")

    records_by_id = defaultdict(list)
    for record in load_jsonl(args.input):
        records_by_id[str(record["id_left"])].append(record)

    claim_ids = sorted(records_by_id)
    if len(claim_ids) < 3:
        raise ValueError(
            "At least three unique id_left values are required to create "
            "non-empty train, validation, and test splits."
        )
    random.Random(args.seed).shuffle(claim_ids)
    total_ids = len(claim_ids)
    train_count = max(1, int(total_ids * args.train_ratio))
    val_count = max(1, int(total_ids * args.val_ratio))
    if train_count + val_count >= total_ids:
        train_count = total_ids - val_count - 1
    train_end = train_count
    val_end = train_end + val_count

    split_ids = {
        "train": claim_ids[:train_end],
        "val": claim_ids[train_end:val_end],
        "test": claim_ids[val_end:],
    }

    os.makedirs(args.output_dir, exist_ok=True)
    summary = {}
    for split_name, ids in split_ids.items():
        split_records = [record for claim_id in ids for record in records_by_id[claim_id]]
        output_path = os.path.join(args.output_dir, f"{split_name}.jsonl")
        write_jsonl(output_path, split_records)
        summary[split_name] = {
            "claim_ids": len(ids),
            "records": len(split_records),
            "path": output_path,
        }

    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
