import os
import torch
import torch.nn.functional as F
import re
import json
import random
import logging
from enum import IntEnum
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
from torch.utils.data import Dataset
import numpy as np
from sklearn.metrics import f1_score


LOGGER = logging.getLogger(__name__)
KA_OPTION_TO_IDX = {"FAST": 0, "STEP": 1}
EA_OPTION_TO_IDX = {"PIECE": 0, "SUM": 1}
NUM_KA_OPTIONS = len(KA_OPTION_TO_IDX)
NUM_EA_OPTIONS = len(EA_OPTION_TO_IDX)


class AAOption(IntEnum):
    TRUST_KA = 0
    TRUST_EA = 1
    REFLECTIVE_REJECTION = 2


class JsonlDataset(Dataset):
    def __init__(self, jsonl_path):
        self.samples = []
        with open(jsonl_path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    self.samples.append(json.loads(line))
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        return self.samples[idx]


def load_first_jsonl_record(jsonl_path):
    with open(jsonl_path, "r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                return json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Invalid JSON in {jsonl_path!r} at line {line_number}."
                ) from exc
    raise ValueError(f"JSONL file is empty: {jsonl_path!r}")


def validate_state_vectors(samples, expected_dim, source="JSONL"):
    for index, sample in enumerate(samples):
        state = sample.get("_state_vec")
        if not isinstance(state, list):
            raise ValueError(f"Missing or invalid _state_vec in {source} record {index}.")
        if len(state) != expected_dim:
            raise ValueError(
                f"State dimension mismatch in {source} record {index}: "
                f"expected {expected_dim}, found {len(state)}."
            )


def load_model_and_tokenizer(
    model_path: str,
    lora_path: str = None,
    device="auto",
    torch_dtype=torch.float16
):
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch_dtype,
        device_map=device,
        trust_remote_code=True,
    )

    if lora_path is not None and os.path.exists(lora_path):
        model = PeftModel.from_pretrained(model, lora_path)
    
    model.eval()
    return tokenizer, model


def normalize_answer(s: str) -> str:
    if not isinstance(s, str):
        s = str(s)
    s = s.strip().lower()
    if s in {"true", "yes", "t", "1", "correct", "fact"}:
        return "true"
    elif s in {"false", "no", "f", "0", "incorrect", "fiction", "lie"}:
        return "false"
    return None


def map_dataset_label(label: str) -> str:
    if not isinstance(label, str):
        label = str(label)
    label = label.strip().lower().replace("-", " ").replace("_", " ")

    true_labels = {
        "true",
        "mostly true"
    }
    false_labels = {
        "false",
        "mostly false"
    }

    if label in true_labels:
        return "true"
    elif label in false_labels:
        return "false"
    raise ValueError(f"Unsupported dataset label: {label!r}")


def extract_json_like(text: str):
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    match = re.search(r"```(?:json)?\s*({.*?})\s*```", text, re.DOTALL | re.IGNORECASE)
    if match:
        try:
            return json.loads(match.group(1))
        except json.JSONDecodeError:
            pass

    stack = 0
    start = None
    for i, c in enumerate(text):
        if c == '{':
            if stack == 0:
                start = i
            stack += 1
        elif c == '}':
            stack -= 1
            if stack == 0 and start is not None:
                try:
                    return json.loads(text[start:i+1])
                except json.JSONDecodeError:
                    continue
    match = re.search(r"({[^{}]*})", text)
    if match:
        try:
            return json.loads(match.group(1).replace("'", '"'))
        except json.JSONDecodeError:
            pass

    return {}


def parse_agent_response(text: str):
    parsed = extract_json_like(text)
    strategy = parsed.get("executed_option") or parsed.get("strategy")
    answer = parsed.get("answer") or parsed.get("judgment")
    if not strategy:
        match = re.search(r"(?im)^\s*Strategy\s*:\s*([A-Za-z]+)", text)
        strategy = match.group(1) if match else None
    if not answer:
        match = re.search(r"(?im)^\s*(?:Judgment|Answer)\s*:\s*([A-Za-z]+)", text)
        answer = match.group(1) if match else None
    return {
        "executed_option": str(strategy).strip().upper() if strategy else None,
        "answer": normalize_answer(answer) if answer is not None else None,
    }


def _sequence_log_probability(model, context_ids, continuation_ids):
    model_device = next(model.parameters()).device
    if not context_ids or not continuation_ids:
        return torch.tensor(float("-inf"), device=model_device)
    input_ids = torch.tensor(
        [context_ids + continuation_ids], dtype=torch.long, device=model_device
    )
    attention_mask = torch.ones_like(input_ids)
    with torch.no_grad():
        logits = model(input_ids=input_ids, attention_mask=attention_mask).logits[0]
    start = len(context_ids) - 1
    token_log_probs = [
        F.log_softmax(logits[start + offset], dim=-1)[token_id]
        for offset, token_id in enumerate(continuation_ids)
    ]
    return torch.stack(token_log_probs).sum()


def _binary_label_confidence(
    model, tokenizer, prompt_ids, response_prefix: str, answer: str
) -> float:
    continuations = {
        label: tokenizer(
            response_prefix + label, add_special_tokens=False
        )["input_ids"]
        for label in ("true", "false")
    }
    true_ids = continuations["true"]
    false_ids = continuations["false"]
    common_length = 0
    for true_id, false_id in zip(true_ids, false_ids):
        if true_id != false_id:
            break
        common_length += 1
    shared_context = list(prompt_ids) + true_ids[:common_length]
    scores = [
        _sequence_log_probability(model, shared_context, true_ids[common_length:]),
        _sequence_log_probability(model, shared_context, false_ids[common_length:]),
    ]
    probabilities = torch.softmax(torch.stack(scores), dim=0)
    index = 0 if answer == "true" else 1 if answer == "false" else int(probabilities.argmax())
    return float(probabilities[index].item())


def _tokenize_prompt_preserving_edges(tokenizer, prompt, max_input_length):
    inputs = tokenizer(prompt, return_tensors="pt", truncation=False)
    input_length = inputs["input_ids"].shape[1]
    if not max_input_length or input_length <= max_input_length:
        return inputs

    prefix_length = max(1, max_input_length // 3)
    suffix_length = max_input_length - prefix_length
    for key, value in list(inputs.items()):
        if value.ndim == 2 and value.shape[1] == input_length:
            inputs[key] = torch.cat(
                [value[:, :prefix_length], value[:, -suffix_length:]], dim=1
            )
    LOGGER.warning(
        "Prompt contains %d tokens; preserving the first %d and last %d tokens.",
        input_length,
        prefix_length,
        suffix_length,
    )
    return inputs


def llm_generate_with_confidence(
    model,
    tokenizer,
    messages: list,
    max_new_tokens: int = 256,
    temperature: float = 0.0,
    top_p: float = 1.0,
    top_k: int = None,
    max_input_length: int = None
):
    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    inputs = _tokenize_prompt_preserving_edges(
        tokenizer, prompt, max_input_length=max_input_length
    )
    model_device = next(model.parameters()).device
    inputs = {key: value.to(model_device) for key, value in inputs.items()}

    gen_kwargs = {
        "max_new_tokens": max_new_tokens,
        "return_dict_in_generate": True,
        "pad_token_id": tokenizer.pad_token_id,
    }
    if temperature > 0:
        gen_kwargs.update(
            do_sample=True,
            temperature=temperature,
            top_p=top_p,
        )
        if top_k is not None:
            gen_kwargs["top_k"] = top_k
    else:
        gen_kwargs["do_sample"] = False

    with torch.no_grad():
        outputs = model.generate(**inputs, **gen_kwargs)

    input_len = inputs["input_ids"].shape[1]
    generated_ids = outputs.sequences[:, input_len:]
    response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()

    parsed = parse_agent_response(response)
    answer_match = re.search(
        r'(?im)(?:"(?:answer|judgment)"\s*:\s*"|^(?:Judgment|Answer)\s*:\s*)',
        response,
    )
    response_prefix = (
        response[:answer_match.end()]
        if answer_match
        else response + "\nJudgment: "
    )
    confidence = _binary_label_confidence(
        model,
        tokenizer,
        inputs["input_ids"][0].tolist(),
        response_prefix,
        parsed.get("answer"),
    )

    return response, confidence


def build_router_state(
    model, tokenizer, claim, evidence_list, ka_out, ea_out,
    max_input_length=None
):
    evidence_text = "\n".join(str(item) for item in evidence_list)
    semantic_text = f"Claim: {claim}\n\nEvidence:\n{evidence_text}"
    encoded = _tokenize_prompt_preserving_edges(
        tokenizer,
        semantic_text,
        max_input_length=max_input_length,
    )
    model_device = next(model.parameters()).device
    encoded = {key: value.to(model_device) for key, value in encoded.items()}
    with torch.no_grad():
        output = model(**encoded, output_hidden_states=True, return_dict=True)

    hidden = output.hidden_states[-1][0]
    attention_mask = encoded.get("attention_mask")
    if attention_mask is not None:
        final_position = int(attention_mask[0].sum().item()) - 1
        semantic_embedding = hidden[final_position]
    else:
        semantic_embedding = hidden[-1]

    ka_answer = normalize_answer(ka_out.get("answer", "unknown"))
    ea_answer = normalize_answer(ea_out.get("answer", "unknown"))
    metadata = torch.tensor(
        [
            float(ka_out.get("confidence", 0.5)),
            float(ea_out.get("confidence", 0.5)),
            1.0 if ka_answer == ea_answer else 0.0,
            1.0 if ka_answer == "true" else 0.0,
            1.0 if ea_answer == "true" else 0.0,
            encode_ka_option(ka_out.get("executed_option", "FAST")),
            encode_ea_option(ea_out.get("executed_option", "PIECE")),
        ],
        dtype=semantic_embedding.dtype,
        device=semantic_embedding.device,
    )
    return torch.cat([semantic_embedding, metadata]).float().cpu().tolist()


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def compute_metrics(preds, labels):
    if len(preds) != len(labels):
        raise ValueError(
            f"Prediction/label length mismatch: {len(preds)} != {len(labels)}."
        )

    valid_labels = {"true", "false"}
    invalid_gold = [label for label in labels if label not in valid_labels]
    if invalid_gold:
        raise ValueError(
            "Gold labels must be normalized to 'true' or 'false'; "
            f"found {invalid_gold[0]!r}."
        )

    valid_pairs = [
        (pred, label)
        for pred, label in zip(preds, labels)
        if pred in valid_labels
    ]
    valid_preds = [p for p, _ in valid_pairs]
    valid_labels = [label for _, label in valid_pairs]
    evaluated = len(valid_pairs)
    total = len(preds)

    scored_preds = [
        pred if pred in {"true", "false"} else ("false" if label == "true" else "true")
        for pred, label in zip(preds, labels)
    ]
    correct = sum(pred == label for pred, label in zip(scored_preds, labels))
    acc = correct / total if total else 0.0
    f1_macro = (
        f1_score(labels, scored_preds, average="macro", zero_division=0)
        if total
        else 0.0
    )
    valid_only_accuracy = (
        sum(pred == label for pred, label in valid_pairs) / evaluated
        if evaluated
        else 0.0
    )
    valid_only_f1_macro = (
        f1_score(valid_labels, valid_preds, average="macro", zero_division=0)
        if evaluated
        else 0.0
    )
    return {
        "accuracy": acc,
        "f1_macro": f1_macro,
        "valid_only_accuracy": valid_only_accuracy,
        "valid_only_f1_macro": valid_only_f1_macro,
        "total_samples": total,
        "evaluated_samples": evaluated,
        "failed_samples": total - evaluated,
        "failure_rate": (total - evaluated) / total if total else 0.0,
    }


def get_ideal_action_for_analysis(
    gold_norm: str, 
    ka_answer: str, 
    ea_answer: str
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
        return -1


def validate_policy(agent_to_evaluate, val_dataset, device):
    gold_labels = []
    predictions = []
    type_b_count = 0
    type_b_choose_ka = 0
    type_c_count = 0
    type_c_choose_ea = 0

    for sample in val_dataset.samples:
        claim = sample["claim"]
        raw_gold = sample["gold_label"]
        gold_norm = map_dataset_label(raw_gold)

        raw_pred = "ERROR"
        pred_norm = None
        action = -1

        try:
            out = agent_to_evaluate.act(
                claim=claim,
                id_left=str(sample.get("id_left", "N/A")),
                gold_label=gold_norm,
                precomputed_ka=sample["KA"],
                precomputed_ea=sample["EA"],
                precomputed_state=sample["_state_vec"]
            )
            raw_pred = out["answer"]
            pred_norm = normalize_answer(raw_pred)
            action = out["action"]
        except Exception:
            LOGGER.exception("Validation failed for sample id=%s", sample.get("id_left", "N/A"))

        gold_labels.append(gold_norm)
        predictions.append(pred_norm)

        ideal_action = get_ideal_action_for_analysis(gold_norm, sample["KA"]["answer"], sample["EA"]["answer"])
        
        if ideal_action == 0:
            type_b_count += 1
            if action == 0:
                type_b_choose_ka += 1
        elif ideal_action == 1:
            type_c_count += 1
            if action == 1:
                type_c_choose_ea += 1

    metrics = compute_metrics(predictions, gold_labels)
    return metrics, type_b_count, type_b_choose_ka, type_c_count, type_c_choose_ea


def dummy_llm_generate(*args, **kwargs):
    dummy_response = "false"
    dummy_confidence = 0.5
    return dummy_response, dummy_confidence


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

def encode_ka_option(option: str) -> float:
    idx = KA_OPTION_TO_IDX.get(option, 0)
    return float(idx) / (NUM_KA_OPTIONS - 1) if NUM_KA_OPTIONS > 1 else 0.0  


def encode_ea_option(option: str) -> float:
    idx = EA_OPTION_TO_IDX.get(option, 0)
    return float(idx) / (NUM_EA_OPTIONS - 1) if NUM_EA_OPTIONS > 1 else 0.0
