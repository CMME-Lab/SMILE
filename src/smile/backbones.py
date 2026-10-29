from __future__ import annotations

import inspect
from functools import lru_cache
from typing import Any

import torch
from peft import PeftModel
from transformers import (
    AutoConfig,
    AutoModelForImageTextToText,
    AutoProcessor,
    BitsAndBytesConfig,
    PreTrainedModel,
    Qwen2VLForConditionalGeneration,
)

__all__ = [
    "QWEN2VL_VISION_TOKEN_IDS",
    "family_of",
    "load_processor",
    "load_model",
    "apply_chat_template",
    "vision_token_ids",
]

QWEN2VL_VISION_TOKEN_IDS = frozenset({151652, 151653, 151655, 151656})

MEDGEMMA_IMAGE_TOKEN_ID = 262144


def family_of(model_id: str) -> str:
    return "qwen2_vl" if "qwen" in model_id.lower() else "medgemma"


@lru_cache(maxsize=None)
def load_processor(model_id: str):
    processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)

    return processor


def _quantization_config(spec: dict[str, Any] | None, compute_dtype):
    if not spec or not spec.get("load_in_4bit"):
        return None
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=bool(spec.get("double_quant", True)),
        bnb_4bit_quant_type=spec.get("quant_type", "nf4"),
        bnb_4bit_compute_dtype=compute_dtype,
        bnb_4bit_quant_storage=compute_dtype,
    )


def load_model(
    model_id: str,
    *,
    quantization: dict[str, Any] | None = None,
    device_map: str = "auto",
    adapter: str | None = None,
):
    family = family_of(model_id)
    dtype = torch.bfloat16

    kwargs: dict[str, Any] = dict(
        attn_implementation="eager",
        device_map=device_map,
    )
    signature = inspect.signature(PreTrainedModel.from_pretrained)
    kwargs["dtype" if "dtype" in signature.parameters else "torch_dtype"] = dtype
    qconf = _quantization_config(quantization, dtype)
    if qconf is not None:
        kwargs["quantization_config"] = qconf

    if family == "qwen2_vl":
        loader = Qwen2VLForConditionalGeneration
    else:
        loader = AutoModelForImageTextToText

    model = loader.from_pretrained(model_id, **kwargs)

    if adapter:
        model = PeftModel.from_pretrained(model, adapter)

    processor = load_processor(model_id)

    tokenizer = getattr(processor, "tokenizer", None)
    if tokenizer is not None and tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model.eval()

    return model, processor


def apply_chat_template(processor, messages, *, add_generation_prompt: bool, tokenize: bool = False):
    renderer = processor
    tokenizer = getattr(processor, "tokenizer", None)
    if not getattr(processor, "chat_template", None) and hasattr(tokenizer, "apply_chat_template"):
        renderer = tokenizer
    out = renderer.apply_chat_template(
        messages, add_generation_prompt=add_generation_prompt, tokenize=tokenize
    )

    if isinstance(out, list) and len(out) == 1 and isinstance(out[0], str):
        return out[0]
    return out


def vision_token_ids(model_id: str, processor=None) -> frozenset[int]:
    if family_of(model_id) == "qwen2_vl":
        config = AutoConfig.from_pretrained(model_id, trust_remote_code=True)
        ids = {
            getattr(config, attr, None)
            for attr in ("vision_start_token_id", "vision_end_token_id", "image_token_id", "video_token_id")
        }
        ids.discard(None)
        return frozenset(ids) if ids else QWEN2VL_VISION_TOKEN_IDS

    ids = {MEDGEMMA_IMAGE_TOKEN_ID}
    if processor is not None:
        boi = processor.tokenizer.special_tokens_map.get("boi_token")
        if boi:
            token_id = processor.tokenizer.convert_tokens_to_ids(boi)
            if token_id is not None:
                ids.add(token_id)
    return frozenset(ids)
