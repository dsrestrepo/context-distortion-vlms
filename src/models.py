"""Raw loaders used by the unified classes in :mod:`src.vlm`."""

import os
from types import SimpleNamespace
from typing import Optional

import torch

from src.quantization import get_quantization_config

try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass


def _env_flag(name):
    return os.getenv(name, "").strip().lower() in {"1", "true", "yes", "on"}


def _hf_load_kwargs(token=None):
    kwargs = {}
    if token:
        kwargs["token"] = token
    if (
        _env_flag("VLM_LOCAL_FILES_ONLY")
        or _env_flag("HF_HUB_OFFLINE")
        or _env_flag("TRANSFORMERS_OFFLINE")
    ):
        kwargs["local_files_only"] = True
    return kwargs


def _finalize_model(model, quantization=None):
    if quantization not in {"4b", "8b"}:
        model = model.to(torch.device("cuda"))
    model.eval()
    torch.cuda.empty_cache()
    return model


def _require_cuda(model_id):
    if not torch.cuda.is_available():
        raise RuntimeError(
            f"CUDA is not available while loading {model_id}. "
            "This open-model job must run on a GPU allocation; check the SLURM "
            "node, CUDA module, and CUDA_VISIBLE_DEVICES in the job log."
        )


def _attention_implementation(use_flash_attention):
    return "flash_attention_2" if use_flash_attention else "eager"


def load_qwen2_vl(
    model_id="Qwen/Qwen2-VL-7B-Instruct",
    quantization=None,
    use_flash_attention=True,
    return_attention=False,
    return_logits=False,
):
    del return_attention, return_logits
    from transformers import Qwen2VLForConditionalGeneration, Qwen2VLProcessor

    _require_cuda(model_id)
    bnb_config, dtype = get_quantization_config(quantization, use_bfloat16=True)
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        model_id,
        device_map="auto",
        quantization_config=bnb_config,
        torch_dtype=dtype,
        attn_implementation=_attention_implementation(use_flash_attention),
        **_hf_load_kwargs(),
    )
    return _finalize_model(model, quantization), Qwen2VLProcessor.from_pretrained(
        model_id, **_hf_load_kwargs()
    )


def load_qwen3_vl(
    model_id="Qwen/Qwen3-VL-8B-Instruct",
    quantization=None,
    use_flash_attention=True,
    return_attention=False,
    return_logits=False,
):
    del return_attention, return_logits
    try:
        from transformers import AutoProcessor, Qwen3VLForConditionalGeneration
    except ImportError as exc:
        raise ImportError("Qwen3-VL requires transformers>=4.57.0.") from exc

    _require_cuda(model_id)
    bnb_config, dtype = get_quantization_config(quantization, use_bfloat16=True)
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        model_id,
        device_map="auto",
        quantization_config=bnb_config,
        torch_dtype=dtype,
        attn_implementation=_attention_implementation(use_flash_attention),
        **_hf_load_kwargs(),
    )
    model.eval()
    torch.cuda.empty_cache()
    return model, AutoProcessor.from_pretrained(model_id, **_hf_load_kwargs())


def load_llava(
    model_id="llava-hf/llava-1.5-7b-hf",
    quantization=None,
    use_flash_attention=True,
    return_attention=False,
    return_logits=False,
):
    del return_attention, return_logits
    from transformers import (
        AutoProcessor,
        LlavaForConditionalGeneration,
        LlavaNextForConditionalGeneration,
        LlavaNextProcessor,
    )

    _require_cuda(model_id)
    bnb_config, dtype = get_quantization_config(quantization, use_bfloat16=True)
    is_v15 = "1.5" in model_id
    model_class = LlavaForConditionalGeneration if is_v15 else LlavaNextForConditionalGeneration
    processor_class = AutoProcessor if is_v15 else LlavaNextProcessor
    model = model_class.from_pretrained(
        model_id,
        device_map="auto",
        quantization_config=bnb_config,
        torch_dtype=dtype,
        attn_implementation=_attention_implementation(use_flash_attention),
        **_hf_load_kwargs(),
    )
    return _finalize_model(model, quantization), processor_class.from_pretrained(
        model_id, **_hf_load_kwargs()
    )


def load_multimodal(
    model_id,
    quantization=None,
    token=None,
    return_attention=False,
    return_logits=False,
):
    del return_attention, return_logits
    from transformers import AutoModelForMultimodalLM, AutoProcessor

    _require_cuda(model_id)
    access_token = token or os.getenv("HF_TOKEN") or os.getenv("hf_key")
    bnb_config, dtype = get_quantization_config(quantization, use_bfloat16=True)
    model = AutoModelForMultimodalLM.from_pretrained(
        model_id,
        device_map="auto",
        quantization_config=bnb_config,
        torch_dtype=dtype,
        **_hf_load_kwargs(access_token),
    )
    return _finalize_model(model, quantization), AutoProcessor.from_pretrained(
        model_id, **_hf_load_kwargs(access_token)
    )


def load_kimi_vl(
    model_id="moonshotai/Kimi-VL-A3B-Instruct",
    quantization=None,
    token=None,
    return_attention=False,
    return_logits=False,
):
    del return_attention, return_logits
    from transformers import AutoConfig, AutoModelForCausalLM, AutoProcessor
    from transformers.utils import import_utils

    # Kimi-VL's remote code targets Transformers 4.50.3 and still imports this
    # helper, which was removed in Transformers 5.
    if not hasattr(import_utils, "is_torch_fx_available"):
        import_utils.is_torch_fx_available = lambda: hasattr(torch, "fx")

    _require_cuda(model_id)
    access_token = token or os.getenv("HF_TOKEN") or os.getenv("hf_key")
    bnb_config, dtype = get_quantization_config(quantization, use_bfloat16=True)
    config = AutoConfig.from_pretrained(
        model_id, trust_remote_code=True, **_hf_load_kwargs(access_token)
    )
    rope_scaling = getattr(getattr(config, "text_config", None), "rope_scaling", None)
    if isinstance(rope_scaling, dict) and "type" not in rope_scaling:
        # Transformers 5 may normalize default RoPE to {"rope_type": "default"},
        # while Kimi-VL's remote code still expects either None or a dict with
        # the legacy "type" key.
        config.text_config.rope_scaling = None
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        config=config,
        trust_remote_code=True,
        device_map="auto",
        quantization_config=bnb_config,
        torch_dtype=dtype,
        **_hf_load_kwargs(access_token),
    )
    processor = AutoProcessor.from_pretrained(
        model_id, trust_remote_code=True, **_hf_load_kwargs(access_token)
    )
    return _finalize_model(model, quantization), processor


def load_llama3_2(
    model_id="meta-llama/Llama-3.2-11B-Vision-Instruct",
    quantization=None,
    token=None,
    return_attention=False,
    return_logits=False,
):
    del return_attention, return_logits
    from transformers import AutoProcessor, MllamaForConditionalGeneration

    _require_cuda(model_id)
    bnb_config, dtype = get_quantization_config(quantization, use_bfloat16=True)
    if quantization == "16b":
        dtype = torch.float16
    access_token = token or os.getenv("HF_TOKEN")
    model = MllamaForConditionalGeneration.from_pretrained(
        model_id,
        device_map="auto",
        quantization_config=bnb_config,
        torch_dtype=dtype,
        **_hf_load_kwargs(access_token),
    )
    return _finalize_model(model, quantization), AutoProcessor.from_pretrained(
        model_id, **_hf_load_kwargs(access_token)
    )


def load_llava_med(
    model_id="microsoft/llava-med-v1.5-mistral-7b",
    quantization=None,
    return_attention=False,
    return_logits=False,
    conv_mode="llava_v0",
):
    del return_attention, return_logits
    from llava.constants import DEFAULT_IMAGE_TOKEN, IMAGE_TOKEN_INDEX
    from llava.mm_utils import (
        KeywordsStoppingCriteria,
        get_model_name_from_path,
        process_images,
        tokenizer_image_token,
    )
    from llava.model.builder import load_pretrained_model
    from utils import SeparatorStyle, conv_templates

    _require_cuda(model_id)
    tokenizer, model, image_processor, _ = load_pretrained_model(
        model_path=model_id,
        model_base=None,
        model_name=get_model_name_from_path(model_id),
        load_8bit=quantization == "8b",
        load_4bit=quantization == "4b",
        device_map="auto",
        device="cuda",
    )
    if hasattr(tokenizer, "legacy"):
        tokenizer.legacy = False

    class LLaVAMedProcessor:
        def __init__(self):
            self.tokenizer = tokenizer
            self.image_processor = image_processor
            self.device = model.device
            self.conv_mode = conv_mode

        def apply_chat_template(self, conversation, assistant_message=None, system_message=None):
            pieces = []
            for chunk in conversation[0]["content"]:
                pieces.append(DEFAULT_IMAGE_TOKEN if chunk.get("type") == "image" else chunk.get("text", ""))
            conv = conv_templates[self.conv_mode].copy()
            if system_message:
                conv.system = system_message
            conv.append_message(conv.roles[0], "\n".join(pieces))
            conv.append_message(conv.roles[1], assistant_message)
            return conv.get_prompt(), conv

        def prepare_inputs(self, text_prompt, image, conv):
            input_ids = tokenizer_image_token(
                text_prompt, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt"
            ).unsqueeze(0).to(self.device)
            images = None
            if image is not None:
                images = process_images([image], self.image_processor, {"image_aspect_ratio": "pad"})
                images = images.to(self.device, dtype=torch.float16)
            stop = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
            return {
                "input_ids": input_ids,
                "images": images,
                "stopping_criteria": KeywordsStoppingCriteria([stop], self.tokenizer, input_ids),
            }

    return model, LLaVAMedProcessor()


def load_medgemma(
    model_id="google/medgemma-4b-it",
    quantization=None,
    token=None,
    return_attention=False,
    return_logits=False,
    use_flash_attention=True,
):
    del return_attention, return_logits, use_flash_attention
    from transformers import AutoModelForImageTextToText, AutoProcessor

    _require_cuda(model_id)
    access_token = token or os.getenv("HF_TOKEN") or os.getenv("hf_key")
    bnb_config, dtype = get_quantization_config(quantization, use_bfloat16=True)
    model = AutoModelForImageTextToText.from_pretrained(
        model_id,
        device_map="auto",
        quantization_config=bnb_config,
        torch_dtype=dtype,
        **_hf_load_kwargs(access_token),
    )
    return _finalize_model(model, quantization), AutoProcessor.from_pretrained(
        model_id, **_hf_load_kwargs(access_token)
    )


class _APIModel:
    def __init__(self, model_id, return_logits, model_type):
        self.model_id = model_id
        self.return_logits = return_logits
        self.config = SimpleNamespace(
            model=model_id,
            _name_or_path=model_id,
            return_logits=return_logits,
            model_type=model_type,
        )


def load_openai(model_id="gpt-4o", return_logits=False):
    from openai import OpenAI

    key = os.getenv("OPENAI_API_KEY")
    if not key:
        raise ValueError("Set OPENAI_API_KEY in your environment")
    model = _APIModel(model_id, return_logits, "openai")
    model.client = OpenAI(api_key=key)
    return model, None


def load_gemini(model_id="gemini-2.0-flash-exp", return_logits=False, vertex_api=True):
    from google import genai

    if vertex_api:
        project = os.getenv("GEMINI_API_PROJECT_ID")
        if not project or not os.getenv("GOOGLE_APPLICATION_CREDENTIALS"):
            raise ValueError(
                "Set GEMINI_API_PROJECT_ID and GOOGLE_APPLICATION_CREDENTIALS for Vertex AI."
            )
        client = genai.Client(vertexai=True, project=project, location="global")
    else:
        key = os.getenv("GEMINI_API_KEY")
        if not key:
            raise ValueError("Set GEMINI_API_KEY in your environment")
        client = genai.Client(api_key=key)
    model = _APIModel(model_id, return_logits, "gemini")
    model.client = client
    return model, None
