import json
import argparse
from collections import Counter

def load_predictions(jsonl_path):
    records = []
    with open(jsonl_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))
    return records

def normalize_gold_label(label: str):
    label = label.lower().strip()
    TRUE_LABELS = {"true", "mostly true"}
    FALSE_LABELS = {"false", "mostly false"}
    if label in TRUE_LABELS:
        return "true"
    elif label in FALSE_LABELS:
        return "false"
    else:
        return None

def main():
    parser = argparse.ArgumentParser(description="Analyze KA and EA predictions.")
    parser.add_argument("--predictions", required=True)
    args = parser.parse_args()
    jsonl_path = args.predictions
    records = load_predictions(jsonl_path)

    golds = []
    ka_preds = []
    ea_preds = []
    ka_executed_options = []
    ea_executed_options = []

    skipped = 0
    valid_records = []
    for rec in records:
        gold_norm = normalize_gold_label(rec["gold_label"])
        if gold_norm is None:
            skipped += 1
            continue

        ka_pred = rec["KA"]["answer"]
        ea_pred = rec["EA"]["answer"]

        assert ka_pred in ("true", "false")
        assert ea_pred in ("true", "false")

        golds.append(gold_norm)
        valid_records.append(rec)
        ka_preds.append(ka_pred)
        ea_preds.append(ea_pred)

        ka_executed_options.append(rec["KA"].get("executed_option", "unknown"))
        ea_executed_options.append(rec["EA"].get("executed_option", "unknown"))

    both_correct = 0
    both_wrong = 0
    ka_correct_ea_wrong_count = 0
    ea_correct_ka_wrong_count = 0

    ka_conf_in_ka_correct_ea_wrong = []
    ea_conf_in_ka_correct_ea_wrong = []
    ka_conf_in_ea_correct_ka_wrong = []
    ea_conf_in_ea_correct_ka_wrong = []

    for i in range(len(golds)):
        gold = golds[i]
        ka_pred = ka_preds[i]
        ea_pred = ea_preds[i]

        ka_conf = valid_records[i]["KA"].get("confidence")
        ea_conf = valid_records[i]["EA"].get("confidence")

        ka_correct = (ka_pred == gold)
        ea_correct = (ea_pred == gold)

        if ka_correct and ea_correct:
            both_correct += 1
        elif not ka_correct and not ea_correct:
            both_wrong += 1
        elif ka_correct and not ea_correct:
            ka_correct_ea_wrong_count += 1
            if ka_conf is not None:
                ka_conf_in_ka_correct_ea_wrong.append(ka_conf)
            if ea_conf is not None:
                ea_conf_in_ka_correct_ea_wrong.append(ea_conf)
        elif ea_correct and not ka_correct:
            ea_correct_ka_wrong_count += 1
            if ka_conf is not None:
                ka_conf_in_ea_correct_ka_wrong.append(ka_conf)
            if ea_conf is not None:
                ea_conf_in_ea_correct_ka_wrong.append(ea_conf)

    def safe_mean(lst):
        return sum(lst) / len(lst) if lst else None

    avg_ka_conf_kc_ew = safe_mean(ka_conf_in_ka_correct_ea_wrong)
    avg_ea_conf_kc_ew = safe_mean(ea_conf_in_ka_correct_ea_wrong)
    avg_ka_conf_ec_kw = safe_mean(ka_conf_in_ea_correct_ka_wrong)
    avg_ea_conf_ec_kw = safe_mean(ea_conf_in_ea_correct_ka_wrong)

    def analyze_errors(golds, preds, name):
        fp = fn = 0
        for g, p in zip(golds, preds):
            if p != g:
                if g == "false" and p == "true":
                    fp += 1
                elif g == "true" and p == "false":
                    fn += 1

    analyze_errors(golds, ka_preds, "KA")
    analyze_errors(golds, ea_preds, "EA")

    ka_option_counts = Counter(ka_executed_options)
    total_ka_options = sum(ka_option_counts.values())
    if total_ka_options > 0:
        for option, count in sorted(ka_option_counts.items()):
            percentage = (count / total_ka_options) * 100

    ea_option_counts = Counter(ea_executed_options)
    total_ea_options = sum(ea_option_counts.values())
    if total_ea_options > 0:
        for option, count in sorted(ea_option_counts.items()):
            percentage = (count / total_ea_options) * 100

    summary = {
        "evaluated": len(golds),
        "skipped": skipped,
        "ka_accuracy": sum(g == p for g, p in zip(golds, ka_preds)) / len(golds) if golds else 0.0,
        "ea_accuracy": sum(g == p for g, p in zip(golds, ea_preds)) / len(golds) if golds else 0.0,
        "both_correct": both_correct,
        "both_wrong": both_wrong,
        "ka_correct_ea_wrong": ka_correct_ea_wrong_count,
        "ea_correct_ka_wrong": ea_correct_ka_wrong_count,
        "avg_ka_confidence_when_ka_only_correct": avg_ka_conf_kc_ew,
        "avg_ea_confidence_when_ka_only_correct": avg_ea_conf_kc_ew,
        "avg_ka_confidence_when_ea_only_correct": avg_ka_conf_ec_kw,
        "avg_ea_confidence_when_ea_only_correct": avg_ea_conf_ec_kw,
        "ka_executed_options": dict(ka_option_counts),
        "ea_executed_options": dict(ea_option_counts),
    }
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
