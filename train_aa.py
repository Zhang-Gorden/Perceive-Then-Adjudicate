import os
import json
import argparse
import logging
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import random
import math
from tqdm import tqdm
from transformers import get_cosine_schedule_with_warmup
from agents.knowledge_agent import KnowledgeAgent
from agents.evidence_agent import EvidenceAgent
from agents.adjudication_agent import AdjudicationAgent, RobustRouterPolicy
from utils import (
    load_model_and_tokenizer, 
    llm_generate_with_confidence, 
    normalize_answer, 
    map_dataset_label,
    set_seed,
    JsonlDataset,
    compute_metrics,
    get_ideal_action_for_analysis,
    validate_policy,
    dummy_llm_generate,
    build_router_state,
    load_first_jsonl_record,
    validate_state_vectors)


def main():
    parser = argparse.ArgumentParser(description="Train the adjudication router policy.")
    parser.add_argument("--model-name", required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--train-jsonl", required=True)
    parser.add_argument("--val-jsonl", required=True)
    parser.add_argument("--save-path", required=True)
    parser.add_argument("--eval-output", required=True)
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--ppo-epochs-per-batch", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument(
        "--max-input-length", type=int, default=0,
        help="Optional token limit for live/reflection prompts; 0 keeps all evidence tokens.",
    )
    parser.add_argument(
        "--class-weights", type=float, nargs=3, default=(1.0, 1.0, 2.0),
        metavar=("TRUST_KA", "TRUST_EA", "REJECT"),
    )
    parser.add_argument("--penalty-consensus-reject", type=float, default=-3.0)
    parser.add_argument("--penalty-single-wrong-mode", type=float, default=-2.0)
    parser.add_argument("--penalty-consensus-wrong-base", type=float, default=-3.0)
    checkpoint_group = parser.add_mutually_exclusive_group()
    checkpoint_group.add_argument("--resume", action="store_true")
    checkpoint_group.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    if args.epochs <= 0 or args.batch_size <= 0 or args.ppo_epochs_per_batch <= 0:
        parser.error("epochs, batch-size, and ppo-epochs-per-batch must be positive")
    if args.lr <= 0:
        parser.error("lr must be positive")
    if not 0.0 <= args.dropout < 1.0:
        parser.error("dropout must be in [0, 1)")
    if args.max_input_length < 0:
        parser.error("max-input-length cannot be negative")
    if any(weight <= 0 for weight in args.class_weights):
        parser.error("class-weights must be positive")

    MODEL_NAME = args.model_name
    DATASET_FULL_PATH = args.dataset
    TRAIN_JSONL = args.train_jsonl
    VAL_JSONL = args.val_jsonl
    SAVE_PATH = args.save_path
    EVAL_OUTPUT = args.eval_output
    USE_CPU = args.cpu
    SHUFFLE_TRAIN = True
    EPOCHS = args.epochs
    LR = args.lr
    SEED = args.seed
    BATCH_SIZE = args.batch_size
    PPO_EPSILON = 0.2
    ENTROPY_COEF = 0.01
    IML_WEIGHT = 0.5
    PPO_EPOCHS_PER_BATCH = args.ppo_epochs_per_batch
    CLASS_WEIGHTS = tuple(args.class_weights)


    set_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() and not USE_CPU else "cpu")

    os.makedirs(os.path.dirname(os.path.abspath(SAVE_PATH)), exist_ok=True)
    os.makedirs(os.path.dirname(os.path.abspath(EVAL_OUTPUT)), exist_ok=True)

    tokenizer, model = load_model_and_tokenizer(
        MODEL_NAME,
        device="cpu" if USE_CPU else "auto",
        torch_dtype=torch.float32 if USE_CPU else torch.float16,
    )
    hidden_size = model.config.hidden_size

    def state_builder(claim, evidence_list, ka_out, ea_out):
        return build_router_state(
            model, tokenizer, claim, evidence_list, ka_out, ea_out,
            max_input_length=args.max_input_length,
        )

    first_sample = load_first_jsonl_record(TRAIN_JSONL)
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
        dropout=args.dropout,
    )

    def get_safe_imitation_target(
        gold_norm: str,
        ka_answer: str,
        ea_answer: str,
        ka_conf: float,
        ea_conf: float
    ) -> int:
        ka_norm = normalize_answer(ka_answer)
        ea_norm = normalize_answer(ea_answer)
        ka_correct = (ka_norm == gold_norm)
        ea_correct = (ea_norm == gold_norm)
        if ka_correct and not ea_correct:
            return 0
        elif ea_correct and not ka_correct:
            return 1
        elif not ka_correct and not ea_correct:
            return 2
        else:
            if ka_conf >= ea_conf:
                return 0
            else:
                return 1

    model_exists = os.path.exists(SAVE_PATH)
    if args.resume and not model_exists:
        raise FileNotFoundError(
            f"Cannot resume because checkpoint does not exist: {SAVE_PATH!r}"
        )
    if model_exists and not args.resume and not args.overwrite:
        raise FileExistsError(
            f"Checkpoint already exists: {SAVE_PATH!r}. Use --resume to evaluate "
            "it or --overwrite to train a new policy."
        )
    should_train = not args.resume

    if should_train:
        train_dataset = JsonlDataset(TRAIN_JSONL)
        val_dataset = JsonlDataset(VAL_JSONL)

        if not train_dataset.samples:
            raise ValueError(f"Training JSONL is empty: {TRAIN_JSONL!r}")
        if not val_dataset.samples:
            raise ValueError(f"Validation JSONL is empty: {VAL_JSONL!r}")
        validate_state_vectors(train_dataset.samples, state_dim, TRAIN_JSONL)
        validate_state_vectors(val_dataset.samples, state_dim, VAL_JSONL)
        
        all_indices = list(range(len(train_dataset)))

        aa_train = AdjudicationAgent(
            ka=None,
            ea=None,
            use_rl_policy=True,
            device=device,
            rl_policy=rl_policy,
            meta_dim=META_DIM,
            llm_generate_func=dummy_llm_generate,
            state_builder=state_builder,
            dataset_csv_path=DATASET_FULL_PATH
        )

        optimizer = optim.Adam(aa_train.rl_policy.parameters(), lr=LR)
        steps_per_epoch = math.ceil(len(all_indices) / BATCH_SIZE) * PPO_EPOCHS_PER_BATCH
        total_steps = max(1, EPOCHS * steps_per_epoch)
        scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=int(0.1 * total_steps),
            num_training_steps=total_steps,
        )
        best_accuracy = -1.0
        best_val_loss = float("inf")

        for epoch in range(EPOCHS):
            aa_train.rl_policy.train()
            total_reward = 0.0
            num_samples = 0

            if SHUFFLE_TRAIN:
                random.shuffle(all_indices)

            for i in range(0, len(all_indices), BATCH_SIZE):
                batch_indices = all_indices[i:i+BATCH_SIZE]
                states, actions, old_log_probs, rewards, imitation_targets = [], [], [], [], []

                for idx in batch_indices:
                    sample = train_dataset[idx]
                    if "_state_vec" not in sample:
                        continue

                    claim = sample["claim"]
                    raw_gold = sample["gold_label"]
                    gold_norm = map_dataset_label(raw_gold)
                    ka_out = sample["KA"]
                    ea_out = sample["EA"]

                    ka_conf = ka_out.get("confidence", 0.5)
                    ea_conf = ea_out.get("confidence", 0.5)
                    avg_conf = (ka_conf + ea_conf) / 2

                    ka_norm = normalize_answer(ka_out["answer"])
                    ea_norm = normalize_answer(ea_out["answer"])
                    ka_correct = (ka_norm == gold_norm)
                    ea_correct = (ea_norm == gold_norm)

                    aa_train.set_training_mode(True)
                    aa_train.clear_buffer()
                    aa_train.act(
                        claim=claim,
                        gold_label=gold_norm,
                        precomputed_ka=ka_out,
                        precomputed_ea=ea_out,
                        precomputed_state=sample["_state_vec"]
                    )
                    traj = aa_train.get_trajectory_buffer()
                    if not traj:
                        continue

                    taken_action = traj[0]["action"]

                    if ka_correct and ea_correct:
                        if taken_action == 2:
                            actual_reward = args.penalty_consensus_reject
                        elif taken_action == 0:
                            actual_reward = ka_conf
                        else:
                            actual_reward = ea_conf
                    elif ka_correct and not ea_correct:
                        if taken_action == 0:
                            actual_reward = ka_conf + (1.0 - ea_conf)
                        else:
                            actual_reward = args.penalty_single_wrong_mode
                    elif ea_correct and not ka_correct:
                        if taken_action == 1:
                            actual_reward = ea_conf + (1.0 - ka_conf)
                        else:
                            actual_reward = args.penalty_single_wrong_mode
                    else:
                        if taken_action == 2:
                            actual_reward = 1.0 - avg_conf
                        else:
                            actual_reward = args.penalty_consensus_wrong_base - avg_conf

                    safe_imitation_target = get_safe_imitation_target(
                        gold_norm, ka_out["answer"], ea_out["answer"], ka_conf, ea_conf
                    )

                    for t in traj:
                        states.append(t["state"])
                        actions.append(t["action"])
                        old_log_probs.append(t["old_log_prob"])
                        rewards.append(actual_reward)
                        imitation_targets.append(safe_imitation_target)

                if not states:
                    continue

                states_t = torch.tensor(states, dtype=torch.float32, device=device)
                actions_t = torch.tensor(actions, dtype=torch.long, device=device)
                old_log_probs_t = torch.tensor(
                    old_log_probs, dtype=torch.float32, device=device
                )
                rewards_t = torch.tensor(rewards, dtype=torch.float32, device=device)
                imit_t = torch.tensor(imitation_targets, dtype=torch.long, device=device)

                total_reward += rewards_t.sum().item()
                num_samples += len(rewards_t)

                baseline = rewards_t.mean()
                advantages = rewards_t - baseline

                for _ in range(PPO_EPOCHS_PER_BATCH):
                    logits = rl_policy(states_t)
                    log_probs = torch.log_softmax(logits, dim=-1)
                    new_log_probs = log_probs.gather(1, actions_t.unsqueeze(1)).squeeze(1)

                    ratio = torch.exp(new_log_probs - old_log_probs_t)
                    surr1 = ratio * advantages
                    surr2 = torch.clamp(ratio, 1 - PPO_EPSILON, 1 + PPO_EPSILON) * advantages
                    ppo_loss = -torch.min(surr1, surr2).mean()

                    probs = torch.softmax(logits, dim=-1)
                    entropy = -(probs * torch.log(probs + 1e-8)).sum(-1).mean()

                    class_weights = torch.tensor(CLASS_WEIGHTS, device=device)
                    ce_loss = nn.CrossEntropyLoss(weight=class_weights)(logits, imit_t)

                    total_loss = ppo_loss - ENTROPY_COEF * entropy + IML_WEIGHT * ce_loss

                    optimizer.zero_grad()
                    total_loss.backward()
                    optimizer.step()
                    scheduler.step()

            avg_reward = total_reward / num_samples if num_samples > 0 else 0

            def generate_func_eval(messages, **kwargs):
                 return llm_generate_with_confidence(
                     model=model,
                     tokenizer=tokenizer,
                     messages=messages,
                     max_new_tokens=kwargs.get("max_new_tokens", 256),
                     temperature=kwargs.get("temperature", 0.0),
                     top_p=kwargs.get("top_p", 0.8),
                     max_input_length=kwargs.get("max_input_length", args.max_input_length)
                 )
            
            ka_eval = KnowledgeAgent(generate_func_eval)
            ea_eval = EvidenceAgent(generate_func_eval, dataset_csv_path=DATASET_FULL_PATH)
            
            aa_eval = AdjudicationAgent(
                ka=ka_eval,
                ea=ea_eval,
                use_rl_policy=True,
                device=device,
                rl_policy=rl_policy,
                meta_dim=META_DIM,
                llm_generate_func=generate_func_eval,
                state_builder=state_builder,
                dataset_csv_path=DATASET_FULL_PATH
            )
            aa_eval.set_training_mode(False) 
            aa_eval.rl_policy.eval()
            
            val_metrics, tb_c, tb_k, tc_c, tc_r = validate_policy(aa_eval, val_dataset, device)
            val_acc = val_metrics["accuracy"]

            val_states = []
            val_targets = []
            for sample in val_dataset.samples:
                gold = map_dataset_label(sample["gold_label"])
                ka_out = sample["KA"]
                ea_out = sample["EA"]
                val_states.append(sample["_state_vec"])
                val_targets.append(get_safe_imitation_target(
                    gold,
                    ka_out["answer"],
                    ea_out["answer"],
                    float(ka_out.get("confidence", 0.5)),
                    float(ea_out.get("confidence", 0.5)),
                ))
            with torch.no_grad():
                val_logits = rl_policy(torch.tensor(val_states, dtype=torch.float32, device=device))
                val_loss = nn.CrossEntropyLoss(
                    weight=torch.tensor(CLASS_WEIGHTS, device=device)
                )(
                    val_logits,
                    torch.tensor(val_targets, dtype=torch.long, device=device),
                ).item()
            print(
                f"epoch={epoch + 1} reward={avg_reward:.4f} "
                f"val_loss={val_loss:.6f} val_accuracy={val_acc:.4f}"
            )

            if val_loss < best_val_loss or (
                val_loss == best_val_loss and val_acc > best_accuracy
            ):
                best_accuracy = val_acc
                best_val_loss = val_loss
                torch.save(aa_train.rl_policy.state_dict(), SAVE_PATH)

        rl_policy.load_state_dict(torch.load(SAVE_PATH, map_location=device))
    else:
        rl_policy.load_state_dict(torch.load(SAVE_PATH, map_location=device))

    val_dataset = JsonlDataset(VAL_JSONL)
    if not val_dataset.samples:
        raise ValueError(f"Validation JSONL is empty: {VAL_JSONL!r}")
    validate_state_vectors(val_dataset.samples, state_dim, VAL_JSONL)

    def generate_func_final_eval(messages, **kwargs):
        return llm_generate_with_confidence(
            model=model,
            tokenizer=tokenizer,
            messages=messages,
            max_new_tokens=kwargs.get("max_new_tokens", 256),
            temperature=kwargs.get("temperature", 0.0),
            top_p=kwargs.get("top_p", 0.8),
            max_input_length=kwargs.get("max_input_length", args.max_input_length)
        )
    
    ka_final = KnowledgeAgent(generate_func_final_eval)
    ea_final = EvidenceAgent(generate_func_final_eval, dataset_csv_path=DATASET_FULL_PATH)

    eval_aa = AdjudicationAgent(
        ka=ka_final,
        ea=ea_final,
        use_rl_policy=True,
        device=device,
        rl_policy=rl_policy,
        meta_dim=META_DIM,
        llm_generate_func=generate_func_final_eval,
        state_builder=state_builder,
        dataset_csv_path=DATASET_FULL_PATH
    )
    eval_aa.set_training_mode(False)
    eval_aa.rl_policy.eval()

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

    for sample in tqdm(val_dataset.samples, total=len(val_dataset)):
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
        "ka_answer": [s["KA"]["answer"] for s in val_dataset.samples],
        "ea_answer": [s["EA"]["answer"] for s in val_dataset.samples]
    })
    def categorize_sample(row):
        ka_correct = (normalize_answer(row["ka_answer"]) == row["gold"])
        ea_correct = (normalize_answer(row["ea_answer"]) == row["gold"])
        if ka_correct and ea_correct:
            return "Type1_BothCorrect"
        elif ka_correct and not ea_correct:
            return "Type2_KACorrect_EAWrong"
        elif not ka_correct and ea_correct:
            return "Type3_KAWrong_EACorrect"
        else:
            return "Type4_BothWrong"

    result_df["type"] = result_df.apply(categorize_sample, axis=1)
    result_df["aa_correct"] = result_df["pred"] == result_df["gold"]
    result_df.to_csv(EVAL_OUTPUT, index=False)

    type_counts = result_df["type"].value_counts()
    type_acc = result_df.groupby("type")["aa_correct"].mean()
    all_types = ["Type1_BothCorrect", "Type2_KACorrect_EAWrong", "Type3_KAWrong_EACorrect", "Type4_BothWrong"]
    for t in all_types:
        error_ids = result_df[(result_df["type"] == t) & (~result_df["aa_correct"])]["id_left"].tolist()

    type4_df = result_df[result_df["type"] == "Type4_BothWrong"]
    rescued = type4_df[type4_df["aa_correct"]]

    metrics_output = EVAL_OUTPUT + ".metrics.json"
    with open(metrics_output, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)
    print(json.dumps(metrics, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
