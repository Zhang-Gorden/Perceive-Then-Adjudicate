import os
import json
import argparse
import logging
import torch
import pandas as pd
from tqdm import tqdm
from agents.knowledge_agent import KnowledgeAgent
from agents.evidence_agent import EvidenceAgent
from utils import build_router_state, load_model_and_tokenizer, llm_generate_with_confidence


def safe_act(agent, max_retries=2, **kwargs):
    last_error = None
    for _ in range(max_retries + 1):
        try:
            output = agent.act(**kwargs)
            if output["answer"] in ("true", "false"):
                return output
            last_error = ValueError("model output did not contain a binary judgment")
        except Exception as exc:
            last_error = exc
            logging.exception("Agent inference failed for claim=%r", kwargs.get("claim", "")[:120])
    raise RuntimeError("Agent failed after all generation attempts") from last_error


def main():
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    parser = argparse.ArgumentParser(description="Generate KA and EA predictions.")
    parser.add_argument("--model-name", required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--dataset", help="Evidence CSV; defaults to --data")
    parser.add_argument("--output", required=True)
    parser.add_argument("--cpu", action="store_true", help="Load the language model on CPU")
    parser.add_argument("--max-retries", type=int, default=2)
    parser.add_argument(
        "--max-input-length", type=int, default=0,
        help="Optional token limit; 0 keeps all evidence tokens.",
    )
    args = parser.parse_args()
    if args.max_input_length < 0:
        parser.error("max-input-length cannot be negative")
    if args.max_retries < 0:
        parser.error("max-retries cannot be negative")

    model_name = args.model_name
    data_path = args.data
    dataset_full_path = args.dataset or args.data
    output_path = args.output

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)

    tokenizer, model = load_model_and_tokenizer(
        model_name,
        device="cpu" if args.cpu else "auto",
        torch_dtype=torch.float32 if args.cpu else torch.float16,
    )
    
    def generate_func(messages, **kwargs):
        return llm_generate_with_confidence(
            model=model,
            tokenizer=tokenizer,
            messages=messages,
            max_new_tokens=kwargs.get("max_new_tokens", 256),
            temperature=kwargs.get("temperature", 0.0),
            top_p=kwargs.get("top_p", 1.0),
            top_k=kwargs.get("top_k", None),
            max_input_length=kwargs.get("max_input_length", args.max_input_length)
        )

    ka = KnowledgeAgent(generate_func)
    ea = EvidenceAgent(generate_func, dataset_csv_path=dataset_full_path)

    df = pd.read_csv(data_path)
    required_columns = {"id_left", "cred_label", "claim_text", "evidence"}
    missing_columns = required_columns - set(df.columns)
    if missing_columns:
        raise ValueError(f"Dataset is missing required columns: {sorted(missing_columns)}")
    original_rows = len(df)
    df = df.drop_duplicates(subset=["id_left"], keep="first").reset_index(drop=True)
    logging.info(
        "Generating one record per claim: %d CSV rows -> %d unique id_left values",
        original_rows,
        len(df),
    )

    with open(output_path, 'w', encoding='utf-8') as f:
        for _, row in tqdm(df.iterrows(), total=len(df)):
            id_left = str(row['id_left'])
            claim = str(row['claim_text'])
            gold_label = str(row['cred_label']).lower().strip()

            evidence_list = ea.evidence_lookup.get(id_left, [])
            evidence_list = [str(e) for e in evidence_list if str(e).strip() != ""]

            ka_out = safe_act(ka, max_retries=args.max_retries, claim=claim)
            ea_out = safe_act(ea, max_retries=args.max_retries, claim=claim, id_left=id_left)
            state_vec = build_router_state(
                model,
                tokenizer,
                claim,
                evidence_list,
                ka_out,
                ea_out,
                max_input_length=args.max_input_length,
            )

            record = {
                "id_left": id_left,
                "claim": claim,
                "gold_label": gold_label,
                "evidence": evidence_list,
                "KA": ka_out,
                "EA": ea_out,
                "_state_vec": state_vec,
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
