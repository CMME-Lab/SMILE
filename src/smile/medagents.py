from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Sequence

import torch

from smile import backbones, config as _config, prompts
from smile.config import Config
from smile.council import extract_agent_c_qa, qa_block
from smile.data import answer_for
from smile.eval.vqa import NOT_ASSESSABLE, normalize_answer
from smile.images import load_view

__all__ = [
    "METHODS", "NUM_STUDY_DOMAINS", "NUM_TASK_DOMAINS", "TASK", "MAX_CONTEXT_CHARS",
    "MedAgentsConfig", "NOT_ASSESSABLE", "Panel", "load_config",
    "case_context", "domain_list", "clean_analysis", "analyses_to_text",
    "synthesised_report", "vote_of", "parse_decision", "parse_answer",
    "run_case", "run", "run_vqa",
]

METHODS = ("base_direct", "base_cot", "anal_only", "syn_only", "syn_verif")

NUM_STUDY_DOMAINS = 5
NUM_TASK_DOMAINS = 2

TASK = ("State the liver parenchymal findings visible in these views, and stage "
        "hepatic fibrosis as F0, F1, F2, F3 or F4.")

MAX_CONTEXT_CHARS = 1800

ERROR = "ERROR."
FALLBACK_DOMAIN = "General Medicine"

_SENTENCE_END = re.compile(r"(?<=[.!?])\s+")
_AI_DISCLAIMER = "as an ai language model"
_CAPTION = re.compile(r"Caption:\s*(.*?)(?:\n\s*Stage\s*:|\Z)", re.DOTALL | re.IGNORECASE)
_STAGE = re.compile(r"Stage\s*:\s*\[?\s*F?\s*([0-4])", re.IGNORECASE)
_WHITESPACE = re.compile(r"\s+")
_ANSWER = re.compile(r"\bnot[\s_]assessable\b|\byes\b|\bno\b", re.IGNORECASE)


@dataclass
class MedAgentsConfig(Config):
    base_model: str = ""
    method: str = "syn_verif"
    max_attempt_vote: int = 3
    context: str = "images_only"
    num_study_domains: int = NUM_STUDY_DOMAINS
    num_task_domains: int = NUM_TASK_DOMAINS
    generate: dict[str, Any] = field(default_factory=dict)


def load_config(path, **kw) -> MedAgentsConfig:
    return MedAgentsConfig.from_dict(_config.load(path, **kw))


class Panel:
    def __init__(self, cfg: MedAgentsConfig):
        self.cfg = cfg
        self._model = None
        self._processor = None

    def load(self, *, device_map: str = "auto") -> "Panel":
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

    def ask(self, user: str, *, images: Sequence = (), system: str = "",
            max_tokens: int = 300) -> str:
        gen = self.cfg.generate or {}
        images = list(images)
        content: list[dict[str, Any]] = [{"type": "image"} for _ in images]
        content.append({"type": "text", "text": user})

        messages = []
        if system:
            messages.append({"role": "system", "content": [{"type": "text", "text": system}]})
        messages.append({"role": "user", "content": content})

        text = backbones.apply_chat_template(
            self.processor, messages, add_generation_prompt=True, tokenize=False
        )
        inputs = self.processor(text=text, images=images or None,
                                return_tensors="pt", padding=True)
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}

        temperature = float(gen.get("temperature", 0.0))
        kwargs: dict[str, Any] = dict(
            max_new_tokens=int(max_tokens),
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


def _shorten(text: object, limit: int = MAX_CONTEXT_CHARS) -> str:
    flat = _WHITESPACE.sub(" ", str(text or "")).strip()
    return flat if len(flat) <= limit else flat[: limit - 3].rstrip() + "..."


def case_context(record: dict[str, Any], *, mode: str = "images_only") -> str:
    patient = str(record.get("patient_id", "")).strip()
    lines = [f"Patient ID: {patient}" if patient else "Patient ID: unknown"]

    if mode == "agents":
        caption = record.get("agent_A_caption") or record.get("output_A")
        report = record.get("agent_B_report") or record.get("output_B")
        if caption:
            lines.append(f"Prior image description: {_shorten(caption)}")
        if report:
            lines.append(f"Prior clinical description: {_shorten(report)}")
        qa = extract_agent_c_qa(record)
        if qa:
            lines.append(f"Prior structured answers: {_shorten(qa_block(qa))}")

    return "\n".join(lines)


def domain_format(n: int) -> str:
    return "Medical Field: " + " | ".join(f"Field{i}" for i in range(n))


def domain_list(raw: str, n: int) -> list[str]:
    if raw == ERROR or not str(raw).strip():
        return [FALLBACK_DOMAIN] * n

    tail = str(raw).split(":")[-1].strip()
    found = [d.strip() for d in tail.split("|") if d.strip()]
    if not found:
        return [FALLBACK_DOMAIN] * n
    return (found + [FALLBACK_DOMAIN] * n)[:n]


def _drop_incomplete_sentence(text: str) -> str:
    sentences = _SENTENCE_END.split(text.strip())
    if len(sentences) > 1 and sentences[-1][-1:] not in (".", "!", "?"):
        return " ".join(sentences[:-1])
    return text


def clean_analysis(analyses: Sequence[str], domains: Sequence[str], kind: str) -> dict[str, str]:
    out: dict[str, str] = {}
    for domain, item in zip(domains, analyses):
        if item == ERROR:
            item = f"There is no analysis for this {kind}."
        item = _drop_incomplete_sentence(str(item))
        lowered = item.lower()
        if _AI_DISCLAIMER in lowered:
            cut = lowered.find(_AI_DISCLAIMER) + len(_AI_DISCLAIMER)
            item = item[cut:].strip().strip(",").strip()
        out[domain] = item
    return out


def analyses_to_text(analyses: dict[str, str], kind: str, content: str) -> str:
    label = "Case" if kind == "study" else "Task"
    blocks = []
    for index, (domain, analysis) in enumerate(analyses.items()):
        blocks.append(f"Report{index}\n{label}: {content}\nDomain: {domain}\n"
                      f"Analysis: {analysis}\n")
    return "\n".join(blocks)


def synthesised_report(context: str, raw: str, task: str = TASK) -> str:
    body = str(raw)
    total = body.split("Total Analysis:")[-1].strip() if "Total Analysis:" in body else body.strip()
    head = f"Case: {context}\nTask: {task}\n"
    if "Key Knowledge:" in body:
        knowledge = body.split("Total Analysis:")[0].split("Key Knowledge:")[-1].strip()
        return f"{head}Key Knowledge: {knowledge}\nTotal Analysis: {total}\n"
    return f"{head}Total Analysis: {total}\n"


def vote_of(raw: str) -> str:
    found = re.findall(r"yes|no", str(raw).lower())
    return found[0] if found else "yes"


def parse_decision(raw: str) -> dict[str, str]:
    text = str(raw or "")
    match = _CAPTION.search(text)
    caption = (match.group(1) if match else text).strip().strip("[]").strip()

    stage_match = _STAGE.search(text)
    return {
        "final_caption": caption,
        "stage": f"F{stage_match.group(1)}" if stage_match else "",
        "raw_text": text,
    }


def parse_answer(raw: str) -> str:
    text = str(raw or "")
    tail = text.split("Answer:")[-1] if "Answer:" in text else text
    match = _ANSWER.search(tail)
    if match is None:
        return NOT_ASSESSABLE
    found = match.group(0).lower()
    return NOT_ASSESSABLE if found.startswith("not") else normalize_answer(found)


def _decide(raw: str, question: str | None) -> dict[str, str]:
    if question:
        return {"answer": parse_answer(raw), "raw_text": str(raw or "")}
    return parse_decision(raw)


def run_case(
    panel: Panel,
    record: dict[str, Any],
    images: Sequence,
    *,
    task: str = TASK,
    question: str | None = None,
    decision: str = "decision.user.template",
) -> dict[str, Any]:
    cfg = panel.cfg

    context = case_context(record, mode=cfg.context)
    if question:
        context = f"{context}\nQuestion: {question}"
    slots = {
        "constraints": prompts.medagents("caption_constraints"),
        "output_format": prompts.medagents("answer_format" if question else "caption_format"),
        "question": question or "",
    }
    trace: dict[str, Any] = {
        "method": cfg.method, "context_mode": cfg.context,
        "study_domains": [], "task_domains": [],
        "study_analyses": {}, "task_analyses": {},
        "syn_report": "", "report_history": [],
        "vote_history": [], "revision_history": [],
    }

    if cfg.method in ("base_direct", "base_cot"):
        name = "direct" if cfg.method == "base_direct" else "cot"
        raw = panel.ask(
            prompts.medagents(f"{name}.user.template", context=context, **slots),
            images=images, max_tokens=int((cfg.generate or {}).get("max_new_tokens", 512)),
        )
        return {**trace, **_decide(raw, question)}

    raw = panel.ask(
        prompts.medagents("study_domains.user.template", context=context,
                          n_fields=cfg.num_study_domains,
                          domain_format=domain_format(cfg.num_study_domains)),
        images=images, system=prompts.medagents("study_domains.system"), max_tokens=50,
    )
    study_domains = domain_list(raw, cfg.num_study_domains)

    raw = panel.ask(
        prompts.medagents("task_domains.user.template", context=context, task=task,
                          n_fields=cfg.num_task_domains,
                          domain_format=domain_format(cfg.num_task_domains)),
        images=images, system=prompts.medagents("task_domains.system"), max_tokens=50,
    )
    task_domains = domain_list(raw, cfg.num_task_domains)

    raw_study = [
        panel.ask(
            prompts.medagents("study_analysis.user.template", context=context),
            images=images,
            system=prompts.medagents("study_analysis.system.template", domain=domain),
            max_tokens=300,
        )
        for domain in study_domains
    ]
    study_analyses = clean_analysis(raw_study, study_domains, "case")

    expert_lines = "".join(
        f"The evaluation from the {domain} expert suggests: {analysis}\n"
        for domain, analysis in study_analyses.items()
    )
    raw_task = [
        panel.ask(
            prompts.medagents("task_analysis.user.template", context=context,
                              analyses=expert_lines, task=task),
            images=images,
            system=prompts.medagents("task_analysis.system.template", domain=domain),
            max_tokens=300,
        )
        for domain in task_domains
    ]
    task_analyses = clean_analysis(raw_task, task_domains, "task")

    trace.update(study_domains=study_domains, task_domains=task_domains,
                 study_analyses=study_analyses, task_analyses=task_analyses)

    if cfg.method == "anal_only":
        raw = panel.ask(
            prompts.medagents(
                "analysis_only.user.template", context=context, **slots,
                study_analyses=analyses_to_text(study_analyses, "study", context),
                task_analyses=analyses_to_text(task_analyses, "task", task),
            ),
            images=images, max_tokens=int((cfg.generate or {}).get("max_new_tokens", 512)),
        )
        return {**trace, **_decide(raw, question)}

    raw = panel.ask(
        prompts.medagents(
            "synthesis.user.template",
            report_format="Key Knowledge: [extracted key knowledge] \nTotal Analysis: [synthesized analysis] \n",
            study_analyses=analyses_to_text(study_analyses, "study", context),
            task_analyses=analyses_to_text(task_analyses, "task", task),
        ),
        system=prompts.medagents("synthesis.system"), max_tokens=1024,
    )
    report = synthesised_report(context, raw, task)
    trace["syn_report"] = report
    trace["report_history"] = [report]

    if cfg.method == "syn_verif":
        everyone = list(study_domains) + list(task_domains)
        dissent = True
        attempts = 0

        while attempts < cfg.max_attempt_vote and dissent:
            attempts += 1
            dissent = False
            votes: dict[str, str] = {}
            advice: dict[str, str] = {}

            for domain in everyone:
                system = prompts.medagents("consensus.system.template", domain=domain)
                vote = vote_of(panel.ask(
                    prompts.medagents("consensus.user.template", domain=domain, report=report),
                    system=system, max_tokens=30,
                ))
                votes[domain] = vote
                if vote == "no":
                    advice[domain] = panel.ask(
                        prompts.medagents("advice.user.template", domain=domain, report=report),
                        system=system, max_tokens=500,
                    )
                    dissent = True

            trace["vote_history"].append(votes)
            if dissent:
                advice_text = "".join(
                    f"Here is advice from a medical expert specialized in {domain}: {text}.\n"
                    for domain, text in advice.items()
                )
                revised = panel.ask(
                    prompts.medagents("revision.user.template", report=report, advice=advice_text),
                    max_tokens=1024,
                )
                report = synthesised_report(context, revised, task)
                trace["revision_history"].append(advice)
                trace["report_history"].append(report)

        trace["syn_report"] = report
        trace["rounds"] = attempts

    raw = panel.ask(
        prompts.medagents(decision, report=report, **slots),
        images=images, max_tokens=int((cfg.generate or {}).get("max_new_tokens", 512)),
    )
    return {**trace, **_decide(raw, question)}


def run(
    panel: Panel,
    records: Sequence[dict[str, Any]],
    view_fields: Sequence[str],
    *,
    progress=None,
) -> list[dict[str, Any]]:
    num_views = int((panel.cfg.generate or {}).get("num_views", 2))
    out = []

    for record in records:
        paths = [record[f] for f in view_fields[:num_views] if record.get(f)]
        images = [load_view(p) for p in paths]
        decision = run_case(panel, record, images)

        result = dict(record)
        result.update({
            "medagents_final_response": decision["final_caption"],
            "medagents_fibrosis_pred": decision["stage"],
            "medagents": decision,
            "judge_final_response": decision["final_caption"],
        })
        out.append(result)
        if progress:
            progress(len(out), len(records))

    return out


def run_vqa(
    panel: Panel,
    records: Sequence[dict[str, Any]],
    view_fields: Sequence[str],
    *,
    dataset: str,
    progress=None,
) -> list[dict[str, Any]]:
    bank = prompts.question_bank(dataset)
    num_views = int((panel.cfg.generate or {}).get("num_views", 2))
    out = []

    for record in records:
        paths = [record[f] for f in view_fields[:num_views] if record.get(f)]
        images = [load_view(p) for p in paths]

        pairs, traces = [], []
        for question in bank:
            decision = run_case(
                panel, record, images,
                task=(f'Answer this question about the case: "{question["question"]}" '
                      f"The available options are: yes, no, not assessable."),
                question=question["question"],
                decision="vqa_decision.user.template",
            )
            pairs.append({
                "qid": question["qid"],
                "question": question["question"],
                "ground_truth_answer": answer_for(record, question["qid"]),
                "medagents_pred_answer": decision["answer"],
            })
            traces.append({"qid": question["qid"], **decision})

        result = dict(record)
        result["medagents_qa"] = pairs
        result["medagents_vqa"] = traces
        out.append(result)
        if progress:
            progress(len(out), len(records))

    return out
