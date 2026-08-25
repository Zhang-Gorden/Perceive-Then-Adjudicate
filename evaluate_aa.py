import os
import json
import argparse
import logging
import pandas as pd
from tqdm import tqdm
import torch
from agents.knowledge_agent import KnowledgeAgent
from agents.evidence_agent import EvidenceAgent
from agents.adjudication_agent import AdjudicationAgent, RobustRouterPolicy
from utils import (
    load_model_and_tokenizer, 
    llm_generate_with_confidence, 
    normalize_answer, 
    map_dataset_label, 
    JsonlDataset, 
    compute_metrics, 
    get_ideal_action_for_analysis,
    categorize_sample,
    build_router_state,
    load_first_jsonl_record,
    validate_state_vectors)


def main():
    parser = argparse.ArgumentParser(description="Evaluate a trained adjudication router policy.")
    parser.add_argument("--model-name", required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--test-jsonl", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument(
        "--max-input-length", type=int, default=0,
        help="Optional token limit; 0 keeps all evidence tokens.",
    )
    args = parser.parse_args()
    if args.max_input_length < 0:
        parser.error("max-input-length cannot be negative")

    MODEL_NAME = args.model_name
    DATASET_FULL_PATH = args.dataset
    TEST_JSONL = args.test_jsonl
    CHECKPOINT_PATH = args.checkpoint
    TEST_OUTPUT = args.output
    USE_CPU = args.cpu

    device = torch.device("cuda" if torch.cuda.is_available() and not USE_CPU else "cpu")

    os.makedirs(os.path.dirname(os.path.abspath(TEST_OUTPUT)), exist_ok=True)

    tokenizer, model = load_model_and_tokenizer(
        MODEL_NAME,
        device="cpu" if USE_CPU else "auto",
        torch_dtype=torch.float32 if USE_CPU else torch.float16,
    )
    hidden_size = model.config.hidden_size

    first_sample = load_first_jsonl_record(TEST_JSONL)
    state_dim = len(first_sample["_state_vec"])
    inferred_meta_dim = state_dim - hidden_size
    if inferred_meta_dim != 7:
        raise ValueError(
            f"Invalid state vector: expected hidden_size + 7 features, but "
            f"state_dim={state_dim}, hidden_size={hidden_size}, "
            f"meta_dim={inferred_meta_dim}."
        )
    META_DIM = inferred_meta_dim
    rl_policy = RobustRouterPolicy(
        llm_hidden_size=hidden_size,
        n_meta_features=META_DIM,
    )
    if not os.path.exists(CHECKPOINT_PATH):
        raise FileNotFoundError(f"Checkpoint not found at {CHECKPOINT_PATH}")
    rl_policy.load_state_dict(torch.load(CHECKPOINT_PATH, map_location=device))
    rl_policy.to(device)
    rl_policy.eval()

    def generate_func(messages, **kwargs):
        return llm_generate_with_confidence(
            model=model,
            tokenizer=tokenizer,
            messages=messages,
            max_new_tokens=kwargs.get("max_new_tokens", 256),
            temperature=kwargs.get("temperature", 0.0),
            top_p=kwargs.get("top_p", 1.0),
            max_input_length=kwargs.get("max_input_length", args.max_input_length)
        )

    def state_builder(claim, evidence_list, ka_out, ea_out):
        return build_router_state(
            model, tokenizer, claim, evidence_list, ka_out, ea_out,
            max_input_length=args.max_input_length,
        )
    
    ka = KnowledgeAgent(generate_func)
    ea = EvidenceAgent(generate_func, dataset_csv_path=DATASET_FULL_PATH)

    eval_aa = AdjudicationAgent(
        ka=ka,
        ea=ea,
        use_rl_policy=True,
        device=device,
        rl_policy=rl_policy,
        meta_dim=META_DIM,
        llm_generate_func=generate_func,
        state_builder=state_builder,
        dataset_csv_path=DATASET_FULL_PATH
    )
    eval_aa.set_training_mode(False)

    test_dataset = JsonlDataset(TEST_JSONL)
    if not test_dataset.samples:
        raise ValueError(f"Test JSONL is empty: {TEST_JSONL!r}")
    validate_state_vectors(test_dataset.samples, state_dim, TEST_JSONL)

    processed_ids = []
    claims = []
    gold_raw_list = []
    gold_labels = []
    predictions = []
    raw_preds = []
    actions = []
    errors = []
    type_b_count = 0
    type_b_choose_ka = 0
    type_c_count = 0
    type_c_choose_ea = 0

    for sample in tqdm(test_dataset.samples, total=len(test_dataset)):
        claim = sample["claim"]
        id_left = str(sample.get("id_left", "N/A"))
        raw_gold = sample["gold_label"]
        gold_norm = map_dataset_label(raw_gold)

        raw_pred = "ERROR"
        pred_norm = None
        action = -1
        error_message = None

        try:
            out = eval_aa.act(
                claim=claim,
                id_left=id_left,
                gold_label=gold_norm,
                precomputed_ka=sample["KA"],
                precomputed_ea=sample["EA"],
                precomputed_state=sample["_state_vec"]
            )
            raw_pred = out["answer"]
            pred_norm = normalize_answer(raw_pred)
            action = out["action"]
        except Exception as exc:
            error_message = str(exc)
            logging.exception("Evaluation failed for sample id=%s", id_left)

        ideal_action = get_ideal_action_for_analysis(gold_norm, sample["KA"]["answer"], sample["EA"]["answer"])
        if ideal_action == 0:
            type_b_count += 1
            if action == 0:
                type_b_choose_ka += 1
        elif ideal_action == 1:
            type_c_count += 1
            if action == 1:
                type_c_choose_ea += 1

        processed_ids.append(id_left)
        claims.append(claim)
        gold_raw_list.append(raw_gold)
        gold_labels.append(gold_norm)
        predictions.append(pred_norm)
        raw_preds.append(raw_pred)
        actions.append(action)
        errors.append(error_message)

    metrics = compute_metrics(predictions, gold_labels)

    if type_b_count > 0:
        type_b_ratio = type_b_choose_ka / type_b_count
    if type_c_count > 0:
        type_c_ratio = type_c_choose_ea / type_c_count

    action_counts = pd.Series(actions).value_counts()
    result_df = pd.DataFrame({
        "id_left": processed_ids,
        "claim": claims,
        "gold_raw": gold_raw_list,
        "gold": gold_labels,
        "pred_raw": raw_preds,
        "pred": predictions,
        "action": actions,
        "error": errors,
        "ka_answer": [s["KA"]["answer"] for s in test_dataset.samples],
        "ea_answer": [s["EA"]["answer"] for s in test_dataset.samples]
    })
    result_df["type"] = result_df.apply(categorize_sample, axis=1)
    result_df["aa_correct"] = result_df["pred"] == result_df["gold"]
    result_df.to_csv(TEST_OUTPUT, index=False)

    type_counts = result_df["type"].value_counts()
    all_types = ["Type1_BothCorrect", "Type2_KACorrect_EAWrong", "Type3_KAWrong_EACorrect", "Type4_BothWrong"]
    for t in all_types:
        cnt = type_counts.get(t, 0)

    type_acc = result_df.groupby("type")["aa_correct"].mean()
    for t in all_types:
        acc = type_acc.get(t, 0.0)
        count = type_counts.get(t, 0)
        correct_num = int(round(acc * count))

    for t in all_types:
        error_ids = result_df[(result_df["type"] == t) & (~result_df["aa_correct"])]["id_left"].tolist()
        if error_ids:
            display_ids = error_ids[:15]
            id_str = ", ".join(map(str, display_ids))
            if len(error_ids) > 15:
                id_str += f", ... (+{len(error_ids) - 15} more)"

    type4_df = result_df[result_df["type"] == "Type4_BothWrong"]
    rescued = type4_df[type4_df["aa_correct"]]
    if len(rescued) > 0:
        for _, r in rescued.head(3).iterrows():
            claim_preview = r['claim'][:120] + ("..." if len(r['claim']) > 120 else "")

    metrics_output = TEST_OUTPUT + ".metrics.json"
    with open(metrics_output, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)
    print(json.dumps(metrics, indent=2, ensure_ascii=False))
    print(f"Detailed predictions: {TEST_OUTPUT}")
    print(f"Metrics: {metrics_output}")


if __name__ == "__main__":
    main()
