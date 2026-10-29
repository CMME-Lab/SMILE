from __future__ import annotations

import json
import re
from typing import Any, Sequence

import torch

from smile import backbones, prompts
from smile.agents import Agent
from smile.config import JudgeConfig
from smile.images import load_view

__all__ = [
    "qa_block", "extract_agent_c_qa", "parse_judge_json", "judge_failure",
    "Judge", "Council", "JUDGE_KEYS",
]

JUDGE_KEYS = ("rules_check", "verdict", "criticism_A", "criticism_B", "criticism_C",
              "judge_final_response")

_JSON_OBJECT = re.compile(r"\{.*\}", flags=re.DOTALL)


def qa_block(qa_pairs: Sequence[dict[str, str]]) -> str:
    if not qa_pairs:
        return "No VQA answers provided."

    lines = []
    for pair in qa_pairs:
        question = str(pair.get("question", "")).strip()
        answer = str(pair.get("answer", "")).strip().lower() or "not assessable"
        lines.append(f"{question} Answer: {answer}")
    return "\n".join(lines)


def extract_agent_c_qa(record: dict[str, Any]) -> list[dict[str, str]]:
    if isinstance(record.get("agentC_qa"), list):
        return [
            {"question": str(x.get("question", "")).strip(), "answer": str(x.get("answer", "")).strip()}
            for x in record["agentC_qa"]
            if isinstance(x, dict) and str(x.get("question", "")).strip()
        ]

    questions, answers = record.get("questions_C"), record.get("answers_C")
    if isinstance(questions, list) and isinstance(answers, list) and len(questions) == len(answers):
        return [
            {"question": str(q).strip(), "answer": str(a).strip()}
            for q, a in zip(questions, answers) if str(q).strip()
        ]

    if isinstance(record.get("question"), str):
        return [{"question": record["question"].strip(), "answer": str(record.get("answer", "")).strip()}]

    if isinstance(record.get("qa_pairs"), list):
        out = []
        for x in record["qa_pairs"]:
            if not isinstance(x, dict):
                continue
            question = str(x.get("question", "")).strip()
            if not question:
                continue
            answer = x.get("agent_C_pred_answer", x.get("answer", x.get("pred_answer", "")))
            out.append({"question": question, "answer": str(answer).strip()})
        return out

    return []


def parse_judge_json(text: str) -> dict[str, Any] | None:
    if not isinstance(text, str):
        return None
    match = _JSON_OBJECT.search(text)
    if not match:
        return None
    return json.loads(match.group(0).strip())


def judge_failure(raw: str) -> dict[str, Any]:
    return {
        "rules_check": {"R1": "fail: invalid JSON", "R2": "fail", "R3": "fail"},
        "verdict": "",
        "criticism_A": "",
        "criticism_B": "",
        "criticism_C": "",
        "judge_final_response": "",
        "raw_text": raw,
    }


def normalise_judge(parsed: dict[str, Any] | None, raw: str) -> dict[str, Any]:
    if parsed is None:
        return judge_failure(raw)

    out = dict(parsed)
    out.setdefault("rules_check", {"R1": "", "R2": "", "R3": ""})
    for key in ("verdict", "criticism_A", "criticism_B", "criticism_C"):
        out.setdefault(key, "")
    if "judge_final_response" not in out and "final_caption" in out:
        out["judge_final_response"] = out.get("final_caption", "")
    out.setdefault("judge_final_response", "")
    out["raw_text"] = raw
    return out


class Judge:
    def __init__(self, cfg: JudgeConfig):
        self.cfg = cfg
        self._model = None
        self._processor = None

    def load(self, *, device_map: str = "auto") -> "Judge":
        self._model, self._processor = backbones.load_model(
            self.cfg.base_model,
            quantization=self.cfg.extra.get("quantization"),
            device_map=device_map,
            adapter=None,
        )
        return self

    @property
    def processor(self):
        if self._processor is None:
            self._processor = backbones.load_processor(self.cfg.base_model)
        return self._processor

    @property
    def model(self):
        return self._model

    def reference_prompts(self) -> tuple[str, str]:
        mode = (self.cfg.reference_prompts or {}).get("mode", "fixed")
        if mode == "fixed":
            return prompts.judge_reference_prompts()
        if mode == "agent":
            return prompts.agent_user("agent_a"), prompts.agent_user("agent_b")

    def user_prompt(self, record: dict[str, Any], *, candidate_a: str, candidate_b: str) -> str:
        prompt_a, prompt_b = self.reference_prompts()
        return prompts.judge_user(
            prompt_A=record.get("prompt_A") or prompt_a,
            out_A=candidate_a,
            prompt_B=record.get("prompt_B") or prompt_b,
            out_B=candidate_b,
            qa_text=qa_block(extract_agent_c_qa(record)),
        )

    def generate(self, images: Sequence, user_text: str) -> str:
        gen = self.cfg.generate or {}
        messages = [
            {"role": "system", "content": [{"type": "text", "text": prompts.judge_system()}]},
            {"role": "user",
             "content": [{"type": "image"} for _ in images] + [{"type": "text", "text": user_text}]},
        ]
        text = backbones.apply_chat_template(
            self.processor, messages, add_generation_prompt=True, tokenize=False
        )

        inputs = self.processor(text=text, images=list(images), return_tensors="pt", padding=True)
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}

        temperature = float(gen.get("temperature", 0.0))
        kwargs: dict[str, Any] = dict(
            max_new_tokens=int(gen.get("max_new_tokens", 1024)),
            do_sample=temperature > 0.0,
            pad_token_id=self.processor.tokenizer.pad_token_id,
        )
        if temperature > 0.0:
            kwargs.update(temperature=temperature, top_p=float(gen.get("top_p", 1.0)))

        with torch.no_grad():
            out = self.model.generate(**inputs, **kwargs)

        generated = out[:, inputs["input_ids"].shape[-1]:]
        return self.processor.batch_decode(
            generated, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0].strip()


class Council:
    def __init__(self, agents: dict[str, Agent], judge: Judge):
        self.agents = agents
        self.judge = judge

    def run_agents(
        self,
        records: Sequence[dict[str, Any]],
        view_fields: Sequence[str],
        *,
        dataset: str,
        progress=None,
    ) -> list[dict[str, Any]]:
        bank = prompts.question_bank(dataset)
        out = []

        for record in records:
            filled = dict(record)

            for name, field in (("agent_a", "agent_A_caption"), ("agent_b", "agent_B_report")):
                agent = self.agents.get(name)
                if agent is None:
                    continue
                example = agent.example(record, view_fields)
                filled[field] = agent.generate([example])[0]

            agent_c = self.agents.get("agent_c")
            if agent_c is not None:
                examples = [agent_c.example(record, view_fields, question=q["question"]) for q in bank]
                answers = agent_c.generate(examples)
                filled["agentC_qa"] = [
                    {"qid": q["qid"], "question": q["question"], "answer": a}
                    for q, a in zip(bank, answers)
                ]

            out.append(filled)
            if progress:
                progress(len(out), len(records))

        return out

    def run_judge(
        self,
        records: Sequence[dict[str, Any]],
        view_fields: Sequence[str],
        *,
        progress=None,
    ) -> list[dict[str, Any]]:
        num_views = int((self.judge.cfg.generate or {}).get("num_views", 2))
        out = []

        for record in records:
            candidate_a = str(record.get("agent_A_caption") or record.get("output_A") or "")
            candidate_b = str(record.get("agent_B_report") or record.get("output_B") or "")

            paths = [record[f] for f in view_fields[:num_views] if record.get(f)]
            images = [load_view(p) for p in paths]

            user_text = self.judge.user_prompt(record, candidate_a=candidate_a, candidate_b=candidate_b)
            raw = self.judge.generate(images, user_text)
            parsed = normalise_judge(parse_judge_json(raw), raw)

            prompt_a, prompt_b = self.judge.reference_prompts()
            qa = extract_agent_c_qa(record)

            result = dict(record)
            result.update({
                "prompt_A": record.get("prompt_A") or prompt_a,
                "output_A": candidate_a,
                "prompt_B": record.get("prompt_B") or prompt_b,
                "output_B": candidate_b,
                "agentC_qa": qa,
                "judge_final_response": parsed.get("judge_final_response", ""),
                "judge_thoughts": {
                    "rules_check": parsed.get("rules_check", {}),
                    "verdict": parsed.get("verdict", ""),
                    "criticism_A": parsed.get("criticism_A", ""),
                    "criticism_B": parsed.get("criticism_B", ""),
                    "criticism_C": parsed.get("criticism_C", ""),
                    "raw_text": raw,
                },
                "judge": parsed,
            })
            out.append(result)
            if progress:
                progress(len(out), len(records))

        return out
