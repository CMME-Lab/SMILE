from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Sequence

import torch
from peft import LoraConfig
from trl import SFTConfig

from smile import backbones, prompts
from smile.config import AgentConfig
from smile.images import load_view

__all__ = ["build_messages", "mask_labels", "Collator", "Agent", "postprocess_answer"]

LABEL_IGNORE = -100


def build_messages(cfg: AgentConfig, *, question: str | None = None) -> list[dict[str, Any]]:
    num_views = int(cfg.generate.get("num_views", 2))

    if cfg.role == "vqa":
        user_text = prompts.agent_c_user(question)
    else:
        user_text = prompts.agent_user(cfg.prompt)

    return [
        {"role": "system", "content": [{"type": "text", "text": prompts.agent_system(cfg.name)}]},
        {
            "role": "user",
            "content": [{"type": "image"} for _ in range(num_views)] + [{"type": "text", "text": user_text}],
        },
    ]


def target_messages(cfg: AgentConfig, target: str, *, question: str | None = None) -> list[dict[str, Any]]:
    return build_messages(cfg, question=question) + [
        {"role": "assistant", "content": [{"type": "text", "text": target}]}
    ]


def mask_labels(
    input_ids,
    *,
    pad_token_id: int | None,
    vision_ids: Iterable[int],
    prompt_lengths: Sequence[int] | None = None,
):
    labels = input_ids.clone()

    if pad_token_id is not None:
        labels[labels == pad_token_id] = LABEL_IGNORE
    for token_id in vision_ids:
        labels[labels == token_id] = LABEL_IGNORE

    if prompt_lengths is not None:
        for row, length in enumerate(prompt_lengths):
            labels[row, : int(length)] = LABEL_IGNORE

    return labels


@dataclass
class Collator:
    processor: Any
    model_id: str
    loss: str = "full_sequence"

    def __post_init__(self):
        self._vision_ids = backbones.vision_token_ids(self.model_id, self.processor)

    def _render(self, messages, *, add_generation_prompt: bool) -> str:
        return backbones.apply_chat_template(
            self.processor, messages, add_generation_prompt=add_generation_prompt, tokenize=False
        ).strip()

    def __call__(self, examples: list[dict[str, Any]]):
        texts = [self._render(ex["messages"], add_generation_prompt=False) for ex in examples]
        images = [[img.convert("RGB") for img in ex["images"]] for ex in examples]

        batch = self.processor(text=texts, images=images, return_tensors="pt", padding=True)

        prompt_lengths = None
        if self.loss == "completion_only":
            prompt_lengths = []
            for example, views in zip(examples, images):
                prompt_text = self._render(example["prompt_messages"], add_generation_prompt=True)
                prompt_batch = self.processor(text=[prompt_text], images=[views], return_tensors="pt")
                prompt_lengths.append(int(prompt_batch["input_ids"].shape[1]))

        batch["labels"] = mask_labels(
            batch["input_ids"],
            pad_token_id=self.processor.tokenizer.pad_token_id,
            vision_ids=self._vision_ids,
            prompt_lengths=prompt_lengths,
        )
        return batch


def postprocess_answer(raw: str, *, mode: str = "as_run") -> str:

    text = raw.strip().lower()

    if mode == "corrected" and ("not assessable" in text):
        return "not assessable"

    if text.startswith("yes"):
        return "yes"
    if text.startswith("no"):
        return "no"
    if "not assessable" in text:
        return "not assessable"
    return text.split("\n")[0].strip()


class Agent:
    def __init__(self, cfg: AgentConfig):
        self.cfg = cfg
        self._model = None
        self._processor = None

    def load(self, *, adapter: bool = True, device_map: str = "auto") -> "Agent":
        path = str(self.cfg.resolved_weights()) if adapter else None
        self._model, self._processor = backbones.load_model(
            self.cfg.base_model,
            quantization=self.cfg.extra.get("quantization"),
            device_map=device_map,
            adapter=path,
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

    def views_of(self, record: dict[str, Any], view_fields: Sequence[str]) -> list:
        crop = (self.cfg.extra.get("preprocess") or {}).get("crop", "none")
        num_views = int(self.cfg.generate.get("num_views", 2))
        paths = [record[f] for f in view_fields[:num_views]]
        return [load_view(p, crop=crop) for p in paths]

    def example(self, record: dict[str, Any], view_fields: Sequence[str], *,
                target: str | None = None, question: str | None = None) -> dict[str, Any]:
        prompt_messages = build_messages(self.cfg, question=question)
        messages = (
            target_messages(self.cfg, target, question=question) if target is not None else prompt_messages
        )
        return {
            "images": self.views_of(record, view_fields),
            "messages": messages,
            "prompt_messages": prompt_messages,
        }

    def collator(self) -> Collator:
        return Collator(
            processor=self.processor,
            model_id=self.cfg.base_model,
            loss=self.cfg.train.get("loss", "full_sequence"),
        )

    def lora_config(self):
        spec = dict(self.cfg.lora)
        return LoraConfig(
            r=spec.get("r"),
            lora_alpha=spec.get("alpha"),
            lora_dropout=spec.get("dropout", 0.05),
            bias=spec.get("bias", "none"),
            target_modules=spec.get("target_modules", "all-linear"),
            task_type=spec.get("task_type", "CAUSAL_LM"),
        )

    def sft_config(self, output_dir: str):
        train = self.cfg.train
        args = SFTConfig(
            output_dir=output_dir,
            num_train_epochs=train.get("epochs", 3),
            per_device_train_batch_size=train.get("batch_size", 4),
            gradient_accumulation_steps=train.get("grad_accumulation", 4),
            gradient_checkpointing=train.get("gradient_checkpointing", True),
            gradient_checkpointing_kwargs={"use_reentrant": False},
            optim=train.get("optim", "adamw_torch_fused"),
            learning_rate=float(train.get("learning_rate", 2e-4)),
            lr_scheduler_type=train.get("lr_scheduler_type", "constant"),
            warmup_ratio=train.get("warmup_ratio", 0.03),
            max_grad_norm=train.get("max_grad_norm", 0.3),
            bf16=train.get("bf16", True),
            tf32=train.get("tf32", True),
            save_strategy=train.get("save_strategy", "epoch"),
            logging_steps=train.get("logging_steps", 10),
            push_to_hub=False,
            report_to="none",
            dataset_text_field="",
            dataset_kwargs={"skip_prepare_dataset": True},
        )
        args.remove_unused_columns = False
        return args

    def generate(self, examples: list[dict[str, Any]]) -> list[str]:
        gen = self.cfg.generate
        texts = [
            backbones.apply_chat_template(
                self.processor, ex["messages"],
                add_generation_prompt=bool(gen.get("add_generation_prompt", False)),
                tokenize=False,
            )
            for ex in examples
        ]
        images = [[img.convert("RGB") for img in ex["images"]] for ex in examples]

        inputs = self.processor(text=texts, images=images, return_tensors="pt", padding=True)
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}

        kwargs: dict[str, Any] = dict(
            max_new_tokens=int(gen.get("max_new_tokens", 1024)),
            do_sample=bool(gen.get("do_sample", False)),
            pad_token_id=self.processor.tokenizer.pad_token_id,
        )
        if kwargs["do_sample"]:
            kwargs.update(temperature=float(gen.get("temperature", 1.0)), top_p=float(gen.get("top_p", 1.0)))

        with torch.no_grad():
            out = self.model.generate(**inputs, **kwargs)

        trimmed = [o[len(i):] for i, o in zip(inputs["input_ids"], out)]
        decoded = self.processor.batch_decode(trimmed, skip_special_tokens=True)

        if self.cfg.role == "vqa":
            mode = self.cfg.extra.get("postprocess", {}).get("mode", "as_run")
            return [postprocess_answer(d, mode=mode) for d in decoded]
        return [d.strip() for d in decoded]
