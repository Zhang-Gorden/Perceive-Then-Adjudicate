import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, Any, Optional, Callable
import pandas as pd
from utils import normalize_answer, parse_agent_response, encode_ka_option, encode_ea_option


class RobustRouterPolicy(nn.Module):
    def __init__(self, llm_hidden_size=4096, n_meta_features=7, dropout=0.3):
        super().__init__()
        self.llm_hidden_size = llm_hidden_size
        self.n_meta_features = n_meta_features
        self.net = nn.Sequential(
            nn.Linear(llm_hidden_size + n_meta_features, 256),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 3),
        )

    def forward(self, x):
        expected_input_dim = self.llm_hidden_size + self.n_meta_features
        if x.size(-1) != expected_input_dim:
            raise ValueError(
                f"Input dimension {x.size(-1)} does not match expected "
                f"{expected_input_dim} (llm={self.llm_hidden_size}, meta={self.n_meta_features})"
            )
        return self.net(x)

    def select_action(
        self, state, deterministic=False, device="cpu", return_log_prob=False
    ):
        with torch.no_grad():
            state_tensor = torch.as_tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
            logits = self(state_tensor)
            log_probs = F.log_softmax(logits, dim=-1)
            if deterministic:
                action = logits.argmax(dim=-1).item()
            else:
                probs = log_probs.exp()
                action = torch.multinomial(probs, num_samples=1).item()
            if return_log_prob:
                return action, float(log_probs[0, action].item())
            return action


class AdjudicationAgent:
    def __init__(
        self,
        ka: Any,
        ea: Any,
        use_rl_policy: bool = True,
        device: torch.device = torch.device("cpu"),
        rl_policy: Optional[nn.Module] = None,
        claim_encoder: Optional[Callable[[str], np.ndarray]] = None,
        state_builder: Optional[Callable] = None,
        meta_dim: int = 7,
        llm_generate_func: Optional[Callable] = None,
        dataset_csv_path: Optional[str] = None,
    ):
        self.ka = ka
        self.ea = ea
        self.use_rl_policy = use_rl_policy
        self.device = device
        self.claim_encoder = claim_encoder
        self.state_builder = state_builder
        self.use_llm_embedding = (claim_encoder is not None)
        self.meta_dim = meta_dim
        self.llm_generate_func = llm_generate_func
        self.evidence_lookup = {}
        if dataset_csv_path is not None:
            full_df = pd.read_csv(dataset_csv_path)
            if 'evidence' not in full_df.columns or 'id_left' not in full_df.columns:
                raise ValueError("Dataset must contain 'id_left' and 'evidence' columns.")
            self.evidence_lookup = full_df.groupby(full_df['id_left'].astype(str))['evidence'].apply(
                lambda x: [str(e) for e in x.dropna() if str(e).strip() != ""][:]
            ).to_dict()
        if rl_policy is not None:
            self.rl_policy = rl_policy.to(device)
        else:
            if self.use_llm_embedding:
                emb_dim = 768
                self.rl_policy = RobustRouterPolicy(
                    llm_hidden_size=emb_dim,
                    n_meta_features=self.meta_dim,
                ).to(device)
            else:
                raise ValueError(
                    "rl_policy is required when claim_encoder is not provided."
                )
        self.training_mode = False
        self.trajectory_buffer = []

    def set_training_mode(self, mode: bool):
        self.training_mode = mode
        if mode:
            self.trajectory_buffer = []

    def clear_buffer(self):
        self.trajectory_buffer = []

    def get_trajectory_buffer(self):
        return self.trajectory_buffer

    def save_policy(self, path: str):
        torch.save(self.rl_policy.state_dict(), path)

    def load_policy(self, path: str):
        self.rl_policy.load_state_dict(torch.load(path, map_location=self.device))
        self.rl_policy.eval()

    def _build_state(
        self,
        ka_out: Dict,
        ea_out: Dict,
        claim: str,
        id_left: Optional[str] = None,
    ) -> np.ndarray:
        if self.state_builder is not None:
            evidence_list = self.evidence_lookup.get(str(id_left), [])
            state = self.state_builder(claim, evidence_list, ka_out, ea_out)
            return np.asarray(state, dtype=np.float32)

        ka_ans = ka_out["answer"]
        ea_ans = ea_out["answer"]
        ka_conf = float(ka_out.get("confidence", 0.5))
        ea_conf = float(ea_out.get("confidence", 0.5))
        ka_norm = normalize_answer(ka_ans)
        ea_norm = normalize_answer(ea_ans)
        ka_is_true = 1.0 if ka_norm == "true" else 0.0
        ea_is_true = 1.0 if ea_norm == "true" else 0.0
        agree = 1.0 if ka_norm == ea_norm else 0.0
        ka_opt = ka_out.get("executed_option", "FAST")
        ea_opt = ea_out.get("executed_option", "PIECE")
        ka_opt_score = encode_ka_option(ka_opt)
        ea_opt_score = encode_ea_option(ea_opt)
        meta_state = np.array([
            ka_conf,
            ea_conf,
            agree,
            ka_is_true,
            ea_is_true,
            ka_opt_score,
            ea_opt_score,
        ], dtype=np.float32)
        assert meta_state.shape[0] == self.meta_dim, \
            f"Constructed meta_state has {meta_state.shape[0]} dims, but meta_dim={self.meta_dim}."
        if self.use_llm_embedding:
            claim_emb = self.claim_encoder(claim)
            if isinstance(claim_emb, torch.Tensor):
                claim_emb = claim_emb.cpu().numpy()
            state = np.concatenate([claim_emb, meta_state], axis=0)
        else:
            raise ValueError(
                "Live inference requires state_builder or a precomputed_state."
            )
        return state.astype(np.float32)

    def _simulate_ka_response(self, claim: str, option: str) -> Dict[str, Any]:
        if option == "FAST":
            system_msg = "You are an expert fact-checker relying solely on internal knowledge."
            user_msg = (
                "Deliver your verdict immediately for well-known facts.\n\n"
                f"Claim: \"{claim}\"\n\n"
                "Respond exactly as:\nJudgment: <true or false>\nConfidence: <0.0 to 1.0>"
            )
        elif option == "STEP":
            system_msg = "You are an expert fact-checker relying solely on internal knowledge."
            user_msg = (
                "Articulate a step-by-step logical analysis for complex assertions.\n\n"
                f"Claim: \"{claim}\"\n\n"
                "Respond exactly as:\nJudgment: <true or false>\nConfidence: <0.0 to 1.0>"
            )
        else:
            raise ValueError(f"Invalid KA option: {option}")
        messages = [
            {"role": "system", "content": system_msg},
            {"role": "user", "content": user_msg}
        ]
        raw_output, logits_conf = self.llm_generate_func(
            messages=messages,
            max_new_tokens=256,
            temperature=0.0
        )
        parsed = parse_agent_response(raw_output)
        answer = parsed.get("answer")
        if answer not in ("true", "false"):
            raise ValueError("Flipped KA did not return a binary judgment.")
        confidence = max(0.0, min(1.0, float(logits_conf)))
        return {
            "agent": "SIM_KA",
            "answer": answer,
            "confidence": round(confidence, 4),
            "executed_option": option
        }

    def _simulate_ea_response(self, claim: str, option: str, id_left: str) -> Dict[str, Any]:
        evidence_list = self.evidence_lookup.get(str(id_left), [])
        evi_text = "\n".join(f"- {e}" for e in evidence_list) if evidence_list else "No evidence available."
        if option == "PIECE":
            system_msg = "You are an expert evidence-based fact-checker."
            user_msg = (
                "Verify the claim against each snippet, checking literal isomorphism.\n\n"
                f"Claim: \"{claim}\"\n\n"
                f"Evidence:\n{evi_text}\n\n"
                "Respond exactly as:\nJudgment: <true or false>\nConfidence: <0.0 to 1.0>"
            )
        elif option == "SUM":
            system_msg = "You are an expert evidence-based fact-checker."
            user_msg = (
                "Assess the claim based on a synthesized summary of evidence.\n\n"
                f"Claim: \"{claim}\"\n\n"
                f"Evidence:\n{evi_text}\n\n"
                "Respond exactly as:\nJudgment: <true or false>\nConfidence: <0.0 to 1.0>"
            )
        else:
            raise ValueError(f"Invalid EA option: {option}")
        messages = [
            {"role": "system", "content": system_msg},
            {"role": "user", "content": user_msg}
        ]
        raw_output, logits_conf = self.llm_generate_func(
            messages=messages,
            max_new_tokens=256,
            temperature=0.0
        )
        parsed = parse_agent_response(raw_output)
        answer = parsed.get("answer")
        if answer not in ("true", "false"):
            raise ValueError("Flipped EA did not return a binary judgment.")
        confidence = max(0.0, min(1.0, float(logits_conf)))
        return {
            "agent": "SIM_EA",
            "answer": answer,
            "confidence": round(confidence, 4),
            "executed_option": option
        }

    def _execute_action(self, action: int, ka_out: Dict, ea_out: Dict, claim: str, id_left: Optional[str] = None) -> str:
        if action == 0:
            return ka_out["answer"]
        elif action == 1:
            return ea_out["answer"]
        elif action == 2:
            if self.llm_generate_func is None:
                raise RuntimeError("Action 2 requires an LLM generation function")
            orig_ka_opt = ka_out["executed_option"]
            orig_ea_opt = ea_out["executed_option"]
            flipped_ka_opt = "STEP" if orig_ka_opt == "FAST" else "FAST"
            flipped_ea_opt = "PIECE" if orig_ea_opt == "SUM" else "SUM"
            sim_ka = self._simulate_ka_response(claim, flipped_ka_opt)
            if id_left is None or not self.evidence_lookup:
                raise ValueError(
                    "Reflective rejection requires id_left and an evidence lookup."
                )
            sim_ea = self._simulate_ea_response(claim, flipped_ea_opt, id_left)
            if self.llm_generate_func is not None:
                original_ka_info = f"Knowledge Agent ({ka_out.get('executed_option', 'N/A')}): {ka_out.get('answer', 'N/A')} (Confidence: {ka_out.get('confidence', 'N/A')})"
                original_ea_info = f"Evidence Agent ({ea_out.get('executed_option', 'N/A')}): {ea_out.get('answer', 'N/A')} (Confidence: {ea_out.get('confidence', 'N/A')})"
                flipped_ka_info = f"Flipped KA ({sim_ka.get('executed_option', 'N/A')}): {sim_ka.get('answer', 'N/A')} (Confidence: {sim_ka.get('confidence', 'N/A')})"
                flipped_ea_info = f"Flipped EA ({sim_ea.get('executed_option', 'N/A')}): {sim_ea.get('answer', 'N/A')} (Confidence: {sim_ea.get('confidence', 'N/A')})"
                system_prompt = "You are a meta-cognitive adjudicator tasked with synthesizing multiple fact-checking perspectives to reach a robust final verdict."
                user_prompt = (
                    f"Below are four independent analyses of the same claim:\n\n"
                    f"Original Claim: {claim}\n\n"
                    f"1. {original_ka_info}\n"
                    f"2. {original_ea_info}\n"
                    f"3. {flipped_ka_info}\n"
                    f"4. {flipped_ea_info}\n\n"
                    "Identify inconsistencies across internal knowledge and external evidence. "
                    "Use strategy flips as diagnostic signals for reasoning instability. "
                    "Mitigate overreliance on literal evidence or dismissal of internal knowledge. "
                    "Reconcile the logic to assess if evidence insufficiency warrants rejection.\n\n"
                    "Respond exactly as: Judgment: <true or false>."
                )
                messages = [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ]
                try:
                    raw_output, _ = self.llm_generate_func(
                        messages=messages,
                        temperature=0.0,
                        max_new_tokens=10
                    )
                    parsed_reflection = parse_agent_response(raw_output)
                    reflection_answer = parsed_reflection.get("answer")
                    if reflection_answer in ("true", "false"):
                        return reflection_answer
                    raise ValueError("Reflection did not return a binary judgment")
                except Exception as exc:
                    raise RuntimeError("Reflective rejection failed") from exc
            raise RuntimeError("Reflective rejection did not produce a verdict")
        else:
            return "[REJECT]"

    def act(
        self,
        claim: str,
        id_left: Optional[str] = None,
        gold_label: Optional[str] = None,
        precomputed_ka: Optional[Dict] = None,
        precomputed_ea: Optional[Dict] = None,
        precomputed_state: Optional[np.ndarray] = None,
    ) -> Dict[str, Any]:
        if precomputed_ka is not None and precomputed_ea is not None:
            ka_output = precomputed_ka
            ea_output = precomputed_ea
        else:
            if self.ka is None or self.ea is None:
                raise ValueError("KA and EA must be provided for live inference.")
            ka_output = self.ka.act(claim=claim)
            if id_left is None:
                raise ValueError("id_left is required for EA in live mode.")
            ea_output = self.ea.act(claim=claim, id_left=id_left)
        if precomputed_state is not None:
            state = np.array(precomputed_state, dtype=np.float32)
        else:
            state = self._build_state(
                ka_output, ea_output, claim, id_left=id_left
            )
        selection = self.rl_policy.select_action(
            state,
            deterministic=not self.training_mode,
            device=self.device,
            return_log_prob=self.training_mode,
        )
        if self.training_mode:
            action, old_log_prob = selection
        else:
            action = selection
        if self.training_mode:
            final_answer = (
                ka_output["answer"] if action == 0 else
                ea_output["answer"] if action == 1 else
                "[REFLECTIVE_REJECTION]"
            )
        else:
            final_answer = self._execute_action(action, ka_output, ea_output, claim, id_left=id_left)
        if self.training_mode:
            self.trajectory_buffer.append({
                "state": state.tolist(),
                "action": action,
                "old_log_prob": old_log_prob,
            })
        return {
            "answer": final_answer,
            "ka": ka_output,
            "ea": ea_output,
            "action": action,
            "is_reject_fallback": (action == 2 and final_answer == "[REJECT]"),
            "is_abstained": (final_answer == "[REJECT]"),
        }
