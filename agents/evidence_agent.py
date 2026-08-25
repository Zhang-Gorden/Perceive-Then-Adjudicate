from utils import parse_agent_response
from typing import Dict, Any
import pandas as pd


class EvidenceAgent:
    def __init__(self, llm_generate_func: callable, dataset_csv_path: str, use_llm_as_policy: bool = False):
        self.llm_generate = llm_generate_func
        full_df = pd.read_csv(dataset_csv_path)
        if 'evidence' not in full_df.columns or 'id_left' not in full_df.columns:
            raise ValueError("Dataset must contain 'id_left' and 'evidence' columns.")
        self.evidence_lookup = full_df.groupby(full_df['id_left'].astype(str))['evidence'].apply(
            lambda x: [str(e) for e in x.dropna() if str(e).strip() != ""]
        ).to_dict()

    def act(self, claim: str, id_left: str) -> Dict[str, Any]:
        evidence_list = self.evidence_lookup.get(id_left, [])
        evi_text = "\n".join(f"- {e}" for e in evidence_list) if evidence_list else "No evidence available."
        messages = [
            {
                "role": "system",
                "content": "You are an expert evidence-based fact-checker."
            },
            {
                "role": "user",
                "content": f"""You have access to the following evidence:

Claim: "{claim}"

Evidence:
{evi_text}

1. First, select a reasoning strategy:
   - PIECE: Verify the claim against each snippet, checking literal isomorphism.
   - SUM: Assess the claim based on a synthesized summary of evidence.
2. Then, output your response in the following exact format:

Strategy: <PIECE or SUM>
Judgment: <true or false>
Confidence: <a number from 0.0 to 1.0>"""
            }
        ]

        raw_output, logits_confidence = self.llm_generate(
            messages=messages,
            max_new_tokens=256,
            temperature=0.0
        )

        parsed = parse_agent_response(raw_output)

        executed_option = parsed.get("executed_option")
        if executed_option not in ("PIECE", "SUM"):
            executed_option = "PIECE"

        answer = parsed.get("answer")
        if answer not in ("true", "false"):
            answer = "unknown"

        confidence = max(0.0, min(1.0, float(logits_confidence)))

        return {
            "agent": "EA",
            "answer": answer,
            "confidence": round(confidence, 4),
            "executed_option": executed_option
        }
