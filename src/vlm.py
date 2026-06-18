"""Unified model interface for local and API vision-language models."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
import base64
import io
import math
import time
from typing import Any, Optional, Sequence

import torch
from PIL import Image
from transformers.image_utils import load_image

from src.models import (
    load_gemini,
    load_kimi_vl,
    load_llama3_2,
    load_llava,
    load_llava_med,
    load_medgemma,
    load_multimodal,
    load_openai,
    load_qwen2_vl,
    load_qwen3_vl,
)


DEFAULT_PROBABILITY_TOKENS = ("yes", "no", "Yes", "No")


@dataclass
class GenerationResult:
    """Model-independent generation output."""

    text: str
    generated_ids: Any = None
    attentions: Any = None
    scores: Any = None
    token_probabilities: dict[str, Optional[float]] = field(default_factory=dict)

    @property
    def token_scores(self):
        """Requested first-generation-step token scores."""
        return self.token_probabilities

    def as_legacy_tuple(self, include_probabilities: bool = False):
        """Return the tuple shape consumed by the existing experiment scripts."""
        base = (self.text, self.generated_ids, self.attentions, self.scores)
        if not include_probabilities:
            return base
        probabilities = self.token_probabilities
        return base + (
            probabilities.get("yes"),
            probabilities.get("no"),
            probabilities.get("Yes"),
            probabilities.get("No"),
            probabilities,
        )


class BaseVLM(ABC):
    """Common interface suitable for direct and multi-step model workflows."""

    def __init__(
        self,
        model_name: str,
        model_id: str,
        quantization: Optional[str] = None,
        return_attention: bool = False,
        return_logits: bool = False,
        use_flash_attention: bool = True,
        conv_mode: str = "mistral_instruct",
    ):
        self.model_name = model_name
        self.model_id = model_id
        self.quantization = quantization
        self.return_attention = return_attention
        self.return_logits = return_logits
        self.use_flash_attention = use_flash_attention
        self.conv_mode = conv_mode
        self.model = None
        self.processor = None
        self.load()
        self.tokenizer = getattr(self.processor, "tokenizer", None)

    @abstractmethod
    def load(self) -> None:
        """Load and retain the model and its processor/client."""

    @abstractmethod
    def generate(
        self,
        text: Optional[str] = None,
        image: Any = None,
        *,
        tokens: Optional[Sequence[str]] = None,
        return_attention: Optional[bool] = None,
        return_logits: Optional[bool] = None,
        **context,
    ) -> GenerationResult:
        """Generate from text, an image, or both."""

    def _generation_options(self, return_attention, return_logits):
        return (
            self.return_attention if return_attention is None else return_attention,
            self.return_logits if return_logits is None else return_logits,
        )

    def generate_batch(
        self,
        texts,
        images,
        *,
        tokens=None,
        return_attention=None,
        return_logits=None,
        contexts=None,
    ):
        """Generate a batch. Specialized models and APIs use this safe fallback."""
        if len(texts) != len(images):
            raise ValueError("texts and images must contain the same number of items.")
        contexts = contexts or [{} for _ in texts]
        if len(contexts) != len(texts):
            raise ValueError("contexts must contain one item per prompt.")
        return [
            self.generate(
                text=text,
                image=image,
                tokens=tokens,
                return_attention=return_attention,
                return_logits=return_logits,
                **context,
            )
            for text, image, context in zip(texts, images, contexts)
        ]

    @staticmethod
    def _prompt(text: Optional[str]) -> str:
        return text or ""


def _build_conversation(text: str, image: Any):
    content = []
    if image is not None:
        content.append({"type": "image"})
    if text:
        content.append({"type": "text", "text": text})
    return [{"role": "user", "content": content}]


def _output_value(output, name):
    return getattr(output, name, None)


def _prepare_model_inputs(inputs, model, float_dtype=None):
    """Move processor outputs to the model device and normalize image tensors."""
    inputs = inputs.to(model.device)
    dtype = float_dtype or getattr(model, "dtype", None) or torch.bfloat16
    for key in ("pixel_values", "pixel_values_videos"):
        value = inputs.get(key)
        if value is not None and not value.is_floating_point():
            inputs[key] = value.to(dtype=dtype)
    return inputs


def _token_probabilities(processor, first_token_scores, tokens):
    if first_token_scores is None:
        return {}
    requested = tuple(tokens or DEFAULT_PROBABILITY_TOKENS)
    result = {}
    for token in requested:
        try:
            token_ids = processor.tokenizer(token, add_special_tokens=False)["input_ids"]
            # Preserve the existing experiment output: these columns contain the
            # requested tokens' first-step scores, despite their historical p_* names.
            result[token] = first_token_scores[token_ids[0]].item() if token_ids else None
        except Exception:
            result[token] = None
    return result


def _result(text, generated_ids, output, processor, tokens, return_attention, return_logits):
    scores = _output_value(output, "scores") if return_logits else None
    first_scores = scores[0][0] if scores else None
    return GenerationResult(
        text=text,
        generated_ids=generated_ids,
        attentions=_output_value(output, "attentions") if return_attention else None,
        scores=scores,
        token_probabilities=_token_probabilities(processor, first_scores, tokens),
    )


class ChatTemplateVLM(BaseVLM):
    """Shared implementation for Qwen, LLaVA, and Llama 3.2 Vision."""

    loaders = {
        "qwen3": load_qwen3_vl,
        "qwen2": load_qwen2_vl,
        "llava": load_llava,
        "llama3": load_llama3_2,
    }

    def load(self):
        loader = next(loader for key, loader in self.loaders.items() if key in self.model_name.lower())
        kwargs = {
            "model_id": self.model_id,
            "quantization": self.quantization,
            "return_attention": self.return_attention,
            "return_logits": self.return_logits,
        }
        if loader is not load_llama3_2:
            kwargs["use_flash_attention"] = self.use_flash_attention
        self.model, self.processor = loader(**kwargs)
        tokenizer = getattr(self.processor, "tokenizer", None)
        if tokenizer is not None:
            tokenizer.padding_side = "left"

    def generate(self, text=None, image=None, *, tokens=None, return_attention=None, return_logits=None, **context):
        return self.generate_batch(
            [text],
            [image],
            tokens=tokens,
            return_attention=return_attention,
            return_logits=return_logits,
            contexts=[context],
        )[0]

    def generate_batch(
        self,
        texts,
        images,
        *,
        tokens=None,
        return_attention=None,
        return_logits=None,
        contexts=None,
    ):
        return_attention, return_logits = self._generation_options(return_attention, return_logits)
        if len(texts) != len(images):
            raise ValueError("texts and images must contain the same number of items.")
        model_type = self.model.config.model_type.lower()
        contexts = contexts or [{} for _ in texts]
        if len(contexts) != len(texts):
            raise ValueError("contexts must contain one item per prompt.")
        prompts = []
        for text, context in zip(texts, contexts):
            prompt = self._prompt(text)
            if model_type == "mllama":
                choices = "yes, no, or unmatched" if context.get("unmatched") else "yes or no"
                prompt += f" Only respond with the answer ({choices}). No additional commentary."
            prompts.append(prompt)
        formatted = [
            self.processor.apply_chat_template(
                _build_conversation(prompt, image), add_generation_prompt=True
            )
            for prompt, image in zip(prompts, images)
        ]
        inputs = self.processor(
            text=formatted,
            images=list(images) if any(image is not None for image in images) else None,
            padding=True,
            return_tensors="pt",
        ).to("cuda")
        outputs = self.model.generate(
            **inputs,
            max_new_tokens=128,
            temperature=0.01,
            do_sample=False,
            output_attentions=return_attention,
            output_scores=return_logits,
            return_dict_in_generate=True,
        )
        input_len = inputs.input_ids.shape[-1]
        generated_ids = outputs.sequences[:, input_len:]
        output_texts = self.processor.batch_decode(
            generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True
        )
        results = []
        for index, output_text in enumerate(output_texts):
            example_scores = (
                tuple(step[index:index + 1] for step in outputs.scores)
                if return_logits and outputs.scores
                else None
            )
            first_scores = example_scores[0][0] if example_scores else None
            results.append(
                GenerationResult(
                    text=output_text,
                    generated_ids=[generated_ids[index]],
                    attentions=outputs.attentions if return_attention else None,
                    scores=example_scores,
                    token_probabilities=_token_probabilities(
                        self.processor, first_scores, tokens
                    ),
                )
            )
        return results


class KimiVLM(BaseVLM):
    def load(self):
        self.model, self.processor = load_kimi_vl(
            model_id=self.model_id,
            quantization=self.quantization,
            return_attention=self.return_attention,
            return_logits=self.return_logits,
        )

    def generate(self, text=None, image=None, *, tokens=None, return_attention=None, return_logits=None, **context):
        return_attention, return_logits = self._generation_options(return_attention, return_logits)
        image = load_image(image) if image is not None else None
        messages = _build_conversation(self._prompt(text), image)
        formatted = self.processor.apply_chat_template(
            messages, add_generation_prompt=True, return_tensors="pt"
        )
        inputs = self.processor(
            images=image,
            text=formatted,
            return_tensors="pt",
            padding=True,
            truncation=True,
        ).to(self.model.device)
        outputs = self.model.generate(
            **inputs,
            max_new_tokens=128,
            do_sample=False,
            output_attentions=return_attention,
            output_scores=return_logits,
            return_dict_in_generate=True,
        )
        generated_ids = outputs.sequences[0][inputs.input_ids.shape[-1]:]
        output_text = self.processor.decode(
            generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )
        return _result(
            output_text, generated_ids, outputs, self.processor, tokens,
            return_attention, return_logits
        )


class LlavaMedVLM(BaseVLM):
    def load(self):
        self.model, self.processor = load_llava_med(
            model_id=self.model_id,
            quantization=self.quantization,
            return_attention=self.return_attention,
            return_logits=self.return_logits,
            conv_mode=self.conv_mode,
        )

    def generate(self, text=None, image=None, *, tokens=None, return_attention=None, return_logits=None, **context):
        return_attention, return_logits = self._generation_options(return_attention, return_logits)
        formatted, conversation = self.processor.apply_chat_template(_build_conversation(self._prompt(text), image))
        inputs = self.processor.prepare_inputs(formatted, image, conversation)
        with torch.inference_mode():
            outputs = self.model.generate(
                inputs["input_ids"],
                images=inputs["images"],
                max_new_tokens=128,
                temperature=0.01,
                do_sample=False,
                use_cache=True,
                stopping_criteria=[inputs["stopping_criteria"]],
                output_attentions=return_attention,
                output_scores=return_logits,
                return_dict_in_generate=True,
                pad_token_id=self.processor.tokenizer.pad_token_id,
                eos_token_id=self.processor.tokenizer.eos_token_id,
            )
        generated_ids = [outputs.sequences[0][inputs["input_ids"].shape[-1]:]]
        output_text = self.processor.tokenizer.decode(generated_ids[0], skip_special_tokens=True).strip()
        return _result(output_text, generated_ids, outputs, self.processor, tokens, return_attention, return_logits)


class MedGemmaVLM(BaseVLM):
    def load(self):
        self.model, self.processor = load_medgemma(
            model_id=self.model_id,
            quantization=self.quantization,
            return_attention=self.return_attention,
            return_logits=self.return_logits,
            use_flash_attention=self.use_flash_attention,
        )

    def generate(self, text=None, image=None, *, tokens=None, return_attention=None, return_logits=None, **context):
        return self.generate_batch(
            [text],
            [image],
            tokens=tokens,
            return_attention=return_attention,
            return_logits=return_logits,
            contexts=[context],
        )[0]

    def generate_batch(
        self,
        texts,
        images,
        *,
        tokens=None,
        return_attention=None,
        return_logits=None,
        contexts=None,
    ):
        return_attention, return_logits = self._generation_options(return_attention, return_logits)
        if len(texts) != len(images):
            raise ValueError("texts and images must contain the same number of items.")
        messages = []
        for text, image in zip(texts, images):
            content = [{"type": "text", "text": self._prompt(text)}]
            if image is not None:
                content.append({"type": "image", "image": image})
            messages.append([{"role": "user", "content": content}])
        inputs = self.processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
            padding=True,
        )
        inputs = _prepare_model_inputs(
            inputs,
            self.model,
            float_dtype=torch.bfloat16 if self.quantization in {"4b", "8b"} else None,
        )
        inputs.pop("token_type_ids", None)
        input_len = inputs["input_ids"].shape[-1]
        with torch.inference_mode():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=128,
                do_sample=False,
                output_attentions=return_attention,
                output_scores=return_logits,
                return_dict_in_generate=True,
            )
        generated_ids = outputs.sequences[:, input_len:]
        output_texts = self.processor.batch_decode(
            generated_ids, skip_special_tokens=True
        )
        results = []
        for index, output_text in enumerate(output_texts):
            example_scores = (
                tuple(step[index:index + 1] for step in outputs.scores)
                if return_logits and outputs.scores
                else None
            )
            first_scores = example_scores[0][0] if example_scores else None
            results.append(
                GenerationResult(
                    text=output_text.strip(),
                    generated_ids=generated_ids[index],
                    attentions=outputs.attentions if return_attention else None,
                    scores=example_scores,
                    token_probabilities=_token_probabilities(
                        self.processor, first_scores, tokens
                    ),
                )
            )
        return results


class MultimodalAutoVLM(MedGemmaVLM):
    """Native Transformers multimodal models using processor chat templates."""

    def load(self):
        self.model, self.processor = load_multimodal(
            model_id=self.model_id,
            quantization=self.quantization,
            return_attention=self.return_attention,
            return_logits=self.return_logits,
        )


def _image_data_url(image, image_format="JPEG"):
    if not isinstance(image, Image.Image):
        image = load_image(image)
    if image_format == "JPEG" and image.mode in ("RGBA", "P"):
        image = image.convert("RGB")
    buffer = io.BytesIO()
    image.save(buffer, format=image_format)
    mime = "jpeg" if image_format == "JPEG" else "png"
    return f"data:image/{mime};base64,{base64.b64encode(buffer.getvalue()).decode()}"


def _openai_probabilities(completion, tokens):
    try:
        top = completion.choices[0].logprobs.content[0].top_logprobs
        available = {item.token: math.exp(item.logprob) for item in top}
        return {
            token: next(
                (available[value] for value in (token, f" {token}", f"\n{token}") if value in available),
                None,
            )
            for token in (tokens or DEFAULT_PROBABILITY_TOKENS)
        }
    except Exception:
        return {}


class OpenAIVLM(BaseVLM):
    def load(self):
        self.model, self.processor = load_openai(model_id=self.model_id, return_logits=self.return_logits)

    def generate(self, text=None, image=None, *, tokens=None, return_attention=None, return_logits=None, **context):
        _, return_logits = self._generation_options(return_attention, return_logits)
        prompt = self._prompt(text)
        is_gpt5 = "gpt-5" in self.model_id
        if image is None:
            messages = [{"role": "user", "content": prompt}]
        else:
            data_url = _image_data_url(image)
            content = (
                [{"type": "input_text", "text": prompt}, {"type": "input_image", "image_url": data_url}]
                if is_gpt5
                else [{"type": "text", "text": prompt}, {"type": "image_url", "image_url": {"url": data_url}}]
            )
            messages = [{"role": "user", "content": content}]

        completion = None
        output_text = ""
        for attempt in range(2):
            try:
                if is_gpt5:
                    completion = self.model.client.responses.create(
                        model=self.model_id, input=messages, reasoning={"effort": "low"}, timeout=90.0
                    )
                    output_text = completion.output_text
                else:
                    completion = self.model.client.chat.completions.create(
                        model=self.model_id,
                        messages=messages,
                        max_completion_tokens=128,
                        logprobs=return_logits,
                        top_logprobs=10 if return_logits else None,
                        timeout=90.0,
                    )
                    output_text = completion.choices[0].message.content.strip()
                break
            except Exception as exc:
                if attempt == 1:
                    print(f"OpenAI API failed after retries: {exc}")
                else:
                    time.sleep(5 if "timeout" in str(exc).lower() else 2)
        probabilities = _openai_probabilities(completion, tokens) if return_logits and not is_gpt5 else {}
        scores = completion.choices[0].logprobs if probabilities else None
        return GenerationResult(output_text, scores=scores, token_probabilities=probabilities)


class GeminiVLM(BaseVLM):
    def load(self):
        self.model, self.processor = load_gemini(model_id=self.model_id, return_logits=self.return_logits)

    def generate(self, text=None, image=None, *, tokens=None, return_attention=None, return_logits=None, **context):
        from google.genai import types

        parts = [self._prompt(text)]
        if image is not None:
            data_url = _image_data_url(image, "PNG")
            parts.append({"inline_data": {"mime_type": "image/png", "data": data_url.split(",", 1)[1]}})
        if "gemini-3" in self.model_id:
            config = types.GenerateContentConfig(
                temperature=0.0,
                max_output_tokens=8192,
                thinking_config=types.ThinkingConfig(include_thoughts=False, thinking_budget=1),
            )
        else:
            config = types.GenerateContentConfig(
                temperature=0.0,
                max_output_tokens=128,
                thinking_config=types.ThinkingConfig(include_thoughts=False, thinking_budget=0),
            )
        response = None
        for attempt in range(10):
            try:
                response = self.model.client.models.generate_content(
                    model=self.model_id, contents=parts, config=config
                )
                break
            except Exception as exc:
                if attempt == 9:
                    print(f"Gemini API failed after retries: {exc}")
                else:
                    time.sleep(60)
        try:
            text_parts = response.candidates[0].content.parts if response else []
            output_text = "".join(part.text for part in text_parts if getattr(part, "text", None))
        except Exception:
            output_text = ""
        return GenerationResult(output_text)


def create_vlm(
    model_name: str,
    model_id: str,
    quantization: Optional[str] = None,
    use_flash_attention: bool = True,
    return_attention: bool = False,
    return_logits: bool = False,
    conv_mode: str = "mistral_instruct",
) -> BaseVLM:
    """Create a supported VLM from the existing experiment configuration."""
    name = model_name.lower()
    if any(key in name for key in ("qwen2", "qwen3", "llava", "llama3")) and not any(
        key in name for key in ("llava_med", "llava-med")
    ):
        cls = ChatTemplateVLM
    elif "kimi" in name:
        cls = KimiVLM
    elif "llava_med" in name or "llava-med" in name:
        cls = LlavaMedVLM
    elif "medgemma" in name:
        cls = MedGemmaVLM
    elif "gemma4" in name or "glm" in name:
        cls = MultimodalAutoVLM
    elif "gemini" in name:
        cls = GeminiVLM
    elif "openai" in model_id.lower() or "gpt-4" in model_id.lower() or "gpt-5" in model_id.lower():
        cls = OpenAIVLM
    else:
        raise ValueError(
            f"Unsupported model: {model_name}. Supported families are Qwen2/3, LLaVA, "
            "Llama 3.2 Vision, Gemma 4, GLM-V, Kimi-VL, LLaVA-Med, MedGemma, "
            "OpenAI, and Gemini."
        )
    return cls(
        model_name=model_name,
        model_id=model_id,
        quantization=quantization,
        use_flash_attention=use_flash_attention,
        return_attention=return_attention,
        return_logits=return_logits,
        conv_mode=conv_mode,
    )
