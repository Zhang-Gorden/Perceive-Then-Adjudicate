from utils import parse_agent_response
from typing import Dict, Any


class KnowledgeAgent:
    def __init__(self, llm_generate_func: callable):
        self.llm_generate = llm_generate_func

    def act(self, claim: str) -> Dict[str, Any]:
        messages = [
            {
                "role": "system",
                "content": "You are an expert fact-checker relying solely on internal knowledge."
            },
            {
                "role": "user",
                "content": f"""For the claim below:
1. First, select a reasoning strategy:
   - FAST: Deliver your verdict immediately for well-known facts.
   - STEP: Articulate a step-by-step logical analysis for complex assertions.
2. Then, output your response in the following exact format:

Claim: "{claim}"

Strategy: <FAST or STEP>
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
        if executed_option not in ("FAST", "STEP"):
            executed_option = "FAST"

        answer = parsed.get("answer")
        if answer not in ("true", "false"):
            answer = "unknown"

        confidence = max(0.0, min(1.0, float(logits_confidence)))

        return {
            "agent": "KA",
            "answer": answer,
            "confidence": round(confidence, 4),
            "executed_option": executed_option
        }
