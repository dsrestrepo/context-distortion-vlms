"""
Model loading utilities for vision-language models.

This module provides standardized loaders for various vision-language model (VLM) architectures,
including both open-source models and API-based models. All loaders support configurable 
quantization (4-bit, 8-bit, 16-bit) and return consistent (model, processor) tuples.

Supported Models:
    Open-source VLMs:
        - Qwen2-VL: Alibaba's vision-language model
        - LLaVA/LLaVA-Next: Visual instruction tuning models
        - PaliGemma: Google's multimodal model
        - Janus Pro: DeepSeek's multi-modality model
        - LLaMA 3.2 Vision: Meta's vision-language model
        - Pixtral: Mistral's vision model
    
    Medical/Biomedical VLMs:
        - BiomedGPT: Medical domain VLM based on OFA
        - LLaVA-Med: Medical specialist LLaVA variant
        - MedGemma: Google's medical-tuned Gemma
        - CheXagent: Stanford's chest X-ray specialist
        - MAIRA-2: Microsoft's radiology assistant
    
    API-based Models:
        - OpenAI GPT-4o: OpenAI's multimodal API
        - Google Gemini: Google's multimodal API
        - Anthropic Claude: Anthropic's vision API
        - xAI Grok: xAI's vision API

Quantization Support:
    - "4b": 4-bit quantization using BitsAndBytes
    - "8b": 8-bit quantization using BitsAndBytes
    - "16b": 16-bit precision (bfloat16 or float16 depending on model)
    - None: Full precision (float32)

Example:
    >>> from src.models import load_qwen2_vl, load_llava, load_medgemma
    >>> 
    >>> # Load with 4-bit quantization
    >>> model, processor = load_qwen2_vl(quantization="4b")
    >>> 
    >>> # Load with 16-bit precision
    >>> model, processor = load_llava(model_id="llava-hf/llava-1.5-7b-hf", quantization="16b")
    >>> 
    >>> # Load medical model
    >>> model, processor = load_medgemma(quantization="8b")
"""

import torch
import os
import sys
from typing import Optional, Tuple
from types import SimpleNamespace

# Import quantization utilities
from src.quantization import get_quantization_config

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    print("Could not import dotenv. Environment variables will not be loaded.")


# ============================================================================
# SHARED UTILITIES
# ============================================================================

def _finalize_model(model, quantization: Optional[str] = None):
    """
    Finalize model after loading by setting eval mode and moving to device if needed.
    
    4-bit and 8-bit quantized models are already on the correct device via device_map="auto",
    so they don't need manual device placement.
    
    Args:
        model: The loaded model instance
        quantization: Quantization type ("4b", "8b", "16b", or None)
        
    Returns:
        The finalized model ready for inference
    """
    if quantization not in ["4b", "8b"]:
        model = model.to(torch.device("cuda"))
    model.eval()
    torch.cuda.empty_cache()
    return model


def _gen_config(return_attention: bool = False, return_logits: bool = False):
    """
    Get common generation configuration dictionary.
    
    Args:
        return_attention: Whether to return attention weights in model outputs
        return_logits: Whether to return token scores/logits in model outputs
        
    Returns:
        Dictionary with generation configuration parameters
    """
    return {
        "output_attentions": return_attention,
        "output_scores": return_logits,
        "return_dict_in_generate": True,
    }


# ============================================================================
# OPEN-SOURCE VISION-LANGUAGE MODELS
# ============================================================================

def load_qwen2_vl(
    model_id: str = "Qwen/Qwen2-VL-7B-Instruct",
    quantization: Optional[str] = None,
    use_flash_attention: bool = True,
    return_attention: bool = False,
    return_logits: bool = False,
):
    """
    Load Qwen2-VL vision-language model.
    
    Args:
        model_id: HuggingFace model identifier
            Examples: "Qwen/Qwen2-VL-2B-Instruct", "Qwen/Qwen2-VL-7B-Instruct"
        quantization: Quantization type - "4b", "8b", "16b", or None for fp32
        use_flash_attention: Whether to use flash attention 2 for faster inference
        return_attention: Include attention weights in generation output
        return_logits: Include token scores in generation output
        
    Returns:
        Tuple of (model, processor) where processor handles text and image inputs
        
    Example:
        >>> model, processor = load_qwen2_vl(quantization="4b")
        >>> model, processor = load_qwen2_vl(model_id="Qwen/Qwen2-VL-2B-Instruct", quantization="16b")
    """
    from transformers import Qwen2VLForConditionalGeneration, Qwen2VLProcessor
    
    bnb_config, torch_dtype = get_quantization_config(quantization, use_bfloat16=True)
    
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        model_id,
        device_map="auto",
        quantization_config=bnb_config,
        torch_dtype=torch_dtype,
        attn_implementation="flash_attention_2" if use_flash_attention else "eager",
        **_gen_config(return_attention, return_logits)
    )
    
    model = _finalize_model(model, quantization)
    processor = Qwen2VLProcessor.from_pretrained(model_id)
    
    return model, processor


def load_llava(
    model_id: str = "llava-hf/llava-1.5-7b-hf",
    quantization: Optional[str] = None,
    use_flash_attention: bool = True,
    return_attention: bool = False,
    return_logits: bool = False,
):
    """
    Load LLaVA or LLaVA-Next vision-language model.
    
    Automatically detects whether to use LLaVA 1.5 or LLaVA-Next based on model_id.
    
    Args:
        model_id: HuggingFace model identifier
            LLaVA 1.5: "llava-hf/llava-1.5-7b-hf", "llava-hf/llava-1.5-13b-hf"
            LLaVA-Next: "llava-hf/llava-v1.6-mistral-7b-hf", "llava-hf/llava-v1.6-vicuna-7b-hf"
        quantization: Quantization type - "4b", "8b", "16b", or None for fp32
        use_flash_attention: Whether to use flash attention 2 for faster inference
        return_attention: Include attention weights in generation output
        return_logits: Include token scores in generation output
        
    Returns:
        Tuple of (model, processor) where processor handles text and image inputs
        
    Example:
        >>> model, processor = load_llava(quantization="16b")
        >>> model, processor = load_llava(model_id="llava-hf/llava-v1.6-mistral-7b-hf", quantization="8b")
    """
    from transformers import (
        AutoProcessor, LlavaForConditionalGeneration,
        LlavaNextProcessor, LlavaNextForConditionalGeneration
    )
    
    bnb_config, torch_dtype = get_quantization_config(quantization, use_bfloat16=True)
    
    is_v15 = "1.5" in model_id
    ModelClass = LlavaForConditionalGeneration if is_v15 else LlavaNextForConditionalGeneration
    ProcessorClass = AutoProcessor if is_v15 else LlavaNextProcessor
    
    model = ModelClass.from_pretrained(
        model_id,
        device_map="auto",
        quantization_config=bnb_config,
        torch_dtype=torch_dtype,
        attn_implementation="flash_attention_2" if use_flash_attention else "eager",
        **_gen_config(return_attention, return_logits)
    )
    
    model = _finalize_model(model, quantization)
    processor = ProcessorClass.from_pretrained(model_id)
    
    return model, processor


def load_pali_gemma(
    model_id: str = "google/paligemma2-10b-pt-224",
    quantization: Optional[str] = None,
    return_attention: bool = False,
    return_logits: bool = False,
):
    """
    Load PaliGemma vision-language model from Google.
    
    Requires HuggingFace authentication token in environment as 'hf_key'.
    
    Args:
        model_id: HuggingFace model identifier
            Examples: "google/paligemma2-3b-pt-224", "google/paligemma2-10b-pt-224"
        quantization: Quantization type - "4b", "8b", "16b", or None for fp32
        return_attention: Include attention weights in generation output
        return_logits: Include token scores in generation output
        
    Returns:
        Tuple of (model, processor) where processor handles text and image inputs
        
    Example:
        >>> model, processor = load_pali_gemma(quantization="16b")
        >>> model, processor = load_pali_gemma(model_id="google/paligemma2-3b-pt-224")
    """
    from transformers import PaliGemmaForConditionalGeneration, PaliGemmaProcessor
    import huggingface_hub
    
    hf_key = os.getenv('hf_key')
    if hf_key:
        huggingface_hub.login(token=hf_key)
    
    bnb_config, torch_dtype = get_quantization_config(quantization, use_bfloat16=True)
    
    model = PaliGemmaForConditionalGeneration.from_pretrained(
        model_id,
        device_map="auto",
        quantization_config=bnb_config,
        torch_dtype=torch_dtype,
        **_gen_config(return_attention, return_logits)
    )
    
    model = _finalize_model(model, quantization)
    processor = PaliGemmaProcessor.from_pretrained(model_id)
    
    return model, processor


def load_janus_pro(
    model_id: str = "deepseek-ai/Janus-Pro-7B",
    quantization: Optional[str] = None,
    return_attention: bool = False,
    return_logits: bool = False,
):
    """
    Load Janus Pro multi-modality model from DeepSeek.
    
    Requires the Janus repository to be in the workspace directory.
    
    Args:
        model_id: HuggingFace model identifier
            Examples: "deepseek-ai/Janus-Pro-1B", "deepseek-ai/Janus-Pro-7B"
        quantization: Quantization type - "4b", "8b", "16b", or None for fp32
        return_attention: Include attention weights in generation output
        return_logits: Include token scores in generation output
        
    Returns:
        Tuple of (model, processor) where processor is a VLChatProcessor
        
    Example:
        >>> model, processor = load_janus_pro(quantization="16b")
    """
    from transformers import AutoModelForCausalLM
    
    # Add Janus to path
    current_dir = os.getcwd()
    janus_folder = os.path.join(current_dir, "Janus")
    if janus_folder not in sys.path:
        sys.path.insert(0, janus_folder)
    
    from janus.models import VLChatProcessor
    
    processor = VLChatProcessor.from_pretrained(model_id)
    bnb_config, torch_dtype = get_quantization_config(quantization, use_bfloat16=True)
    
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        trust_remote_code=True,
        device_map="cuda",
        quantization_config=bnb_config,
        torch_dtype=torch_dtype,
        **_gen_config(return_attention, return_logits)
    )
    
    model = model.to(torch.bfloat16).cuda().eval()
    
    return model, processor


def load_llama3_2(
    model_id: str = "meta-llama/Llama-3.2-11B-Vision-Instruct",
    quantization: Optional[str] = None,
    token: Optional[str] = None,
    return_attention: bool = False,
    return_logits: bool = False,
):
    """
    Load LLaMA 3.2 Vision-Instruct model from Meta.
    
    Requires HuggingFace authentication token (gated model).
    Note: Uses float16 for 16-bit precision (not bfloat16).
    
    Args:
        model_id: HuggingFace model identifier
            Examples: "meta-llama/Llama-3.2-11B-Vision-Instruct", "meta-llama/Llama-3.2-90B-Vision-Instruct"
        quantization: Quantization type - "4b", "8b", "16b", or None for fp32
        token: HuggingFace access token (if None, uses HF_TOKEN environment variable)
        return_attention: Include attention weights in generation output
        return_logits: Include token scores in generation output
        
    Returns:
        Tuple of (model, processor) where processor handles text and image inputs
        
    Example:
        >>> model, processor = load_llama3_2(quantization="16b")
        >>> model, processor = load_llama3_2(token="hf_...", quantization="8b")
    """
    from transformers import MllamaForConditionalGeneration, AutoProcessor
    
    bnb_config, torch_dtype = get_quantization_config(quantization, use_bfloat16=True)
    if quantization == "16b":
        torch_dtype = torch.float16  # LLaMA 3.2 prefers float16 for 16b
    
    access_token = token or os.getenv("HF_TOKEN")
    if access_token:
        os.environ["HF_TOKEN"] = access_token
    
    model = MllamaForConditionalGeneration.from_pretrained(
        model_id,
        device_map="auto",
        token=access_token,
        quantization_config=bnb_config,
        torch_dtype=torch_dtype,
        **_gen_config(return_attention, return_logits)
    )
    
    model = _finalize_model(model, quantization)
    processor = AutoProcessor.from_pretrained(model_id)
    
    return model, processor


def load_pixtral(
    model_id: str = "mistralai/Pixtral-12B-2409",
    quantization: Optional[str] = None,
    return_attention: bool = False,
    return_logits: bool = False,
):
    """
    Load Pixtral vision model from Mistral AI.
    
    Uses the LLaVA-style interface in HuggingFace Transformers.
    Note: Uses float16 for 16-bit precision (not bfloat16).
    
    Args:
        model_id: HuggingFace model identifier
            Example: "mistralai/Pixtral-12B-2409"
        quantization: Quantization type - "4b", "8b", "16b", or None for fp32
        return_attention: Include attention weights in generation output
        return_logits: Include token scores in generation output
        
    Returns:
        Tuple of (model, processor) where processor handles text and image inputs
        
    Example:
        >>> model, processor = load_pixtral(quantization="16b")
    """
    from transformers import AutoProcessor, LlavaForConditionalGeneration
    
    bnb_config, torch_dtype = get_quantization_config(quantization, use_bfloat16=True)
    if quantization == "16b":
        torch_dtype = torch.float16  # Pixtral uses float16 for 16b
    
    processor = AutoProcessor.from_pretrained(model_id)
    model = LlavaForConditionalGeneration.from_pretrained(
        model_id,
        device_map="auto",
        quantization_config=bnb_config,
        torch_dtype=torch_dtype,
        **_gen_config(return_attention, return_logits)
    )
    
    model = _finalize_model(model, quantization)
    
    return model, processor


# ============================================================================
# MEDICAL/BIOMEDICAL MODELS
# ============================================================================

def load_biomedgpt(
    model_id: str = "PanaceaAI/BiomedGPT-Base-Pretrained",
    quantization: Optional[str] = None,
    return_attention: bool = False,
    return_logits: bool = False,
    device: str = "cuda",
):
    """
    Load BiomedGPT model - a medical domain VLM based on OFA architecture.
    
    Returns a custom processor that handles tokenization and image preprocessing.
    
    Args:
        model_id: HuggingFace model identifier
        quantization: Quantization type - "4b", "8b", "16b", or None for fp32
        return_attention: Include attention weights in generation output
        return_logits: Include token scores in generation output
        device: Target device ("cuda" or "cpu")
        
    Returns:
        Tuple of (model, processor) where processor is a custom BiomedGPTProcessor
        
    Example:
        >>> model, processor = load_biomedgpt(quantization="16b")
    """
    from transformers import OFATokenizer, OFAModel
    from torchvision import transforms
    
    torch_dtype = torch.float16
    model = OFAModel.from_pretrained(model_id, torch_dtype=torch_dtype)
    model = model.to(torch.device(device))
    model.eval()
    torch.cuda.empty_cache()
    
    class BiomedGPTProcessor:
        """Custom processor for BiomedGPT combining tokenizer and image transforms."""
        def __init__(self, model_id, device="cuda"):
            self.tokenizer = OFATokenizer.from_pretrained(
                model_id, torch_dtype=torch.float16,
                output_attentions=return_attention,
                output_scores=return_logits,
                return_dict_in_generate=True
            )
            mean, std = [0.5, 0.5, 0.5], [0.5, 0.5, 0.5]
            resolution = 480
            self.patch_transform = transforms.Compose([
                lambda image: image.convert("RGB"),
                transforms.Resize((resolution, resolution)),
                transforms.ToTensor(),
                transforms.Normalize(mean=mean, std=std)
            ])
            self.device = torch.device(device)
        
        def __call__(self, text=None, image=None):
            input_ids = None
            patch_images = None
            
            if text is not None:
                input_ids = self.tokenizer([text], return_tensors="pt").input_ids.to(self.device)
            if image is not None:
                patch_images = self.patch_transform(image).unsqueeze(0).to(self.device)
                if quantization not in ["4b", "8b"]:
                    patch_images = patch_images.to(torch_dtype)
            
            return {"input_ids": input_ids, "patch_images": patch_images}
    
    processor = BiomedGPTProcessor(model_id, device)
    return model, processor


def load_llava_med(
    model_id: str = "microsoft/llava-med-v1.5-mistral-7b",
    quantization: Optional[str] = None,
    return_attention: bool = False,
    return_logits: bool = False,
    conv_mode: str = "llava_v0",
):
    """
    Load LLaVA-Med model - a medical specialist version of LLaVA.
    
    Uses the official LLaVA builder code and conversation templates.
    Returns a custom processor with conversation template support.
    
    Args:
        model_id: HuggingFace model identifier
        quantization: Quantization type - "4b", "8b", "16b", or None for fp32
        return_attention: Include attention weights in generation output
        return_logits: Include token scores in generation output
        conv_mode: Conversation template mode (e.g., "llava_v0", "llava_v1")
        
    Returns:
        Tuple of (model, processor) where processor is a custom LLaVAMedProcessor
        
    Example:
        >>> model, processor = load_llava_med(quantization="8b")
    """
    from llava.mm_utils import (
        KeywordsStoppingCriteria, get_model_name_from_path,
        process_images, tokenizer_image_token,
    )
    from llava.constants import DEFAULT_IMAGE_TOKEN, IMAGE_TOKEN_INDEX
    from utils import SeparatorStyle, conv_templates
    from llava.model.builder import load_pretrained_model
    
    model_name = get_model_name_from_path(model_id)
    tokenizer, model, image_processor, context_len = load_pretrained_model(
        model_path=model_id,
        model_base=None,
        model_name=model_name,
        load_8bit=(quantization == "8b"),
        load_4bit=(quantization == "4b"),
        device_map="auto",
        device="cuda"
    )
    
    # CRITICAL FIX: Set legacy=False to fix intermittent empty generation issue
    # The LlamaTokenizer legacy behavior causes improper handling of special tokens
    # which can lead to stopping criteria triggering immediately in some cases
    if hasattr(tokenizer, 'legacy') and tokenizer.legacy is not False:
        tokenizer.legacy = False
        print(f"Set tokenizer.legacy=False to fix special token handling")
    
    class LLaVAMedProcessor:
        """Custom processor for LLaVA-Med with conversation template support."""
        def __init__(self, tokenizer, image_processor, conv_mode="llava_v2", device="cuda"):
            self.tokenizer = tokenizer
            self.image_processor = image_processor
            self.device = device
            self.conv_mode = conv_mode
        
        def apply_chat_template(self, conversation, assistant_message=None, system_message=None):
            """Apply conversation template to format prompts."""
            user_content_list = conversation[0]["content"]
            content_pieces = []
            for chunk in user_content_list:
                chunk_type = chunk.get("type", "text")
                if chunk_type == "image":
                    content_pieces.append(DEFAULT_IMAGE_TOKEN)
                elif chunk_type == "text":
                    content_pieces.append(chunk.get("text", ""))
            
            user_text = "\n".join(content_pieces)
            conv = conv_templates[self.conv_mode].copy()
            if system_message:
                conv.system = system_message
            
            conv.append_message(conv.roles[0], user_text)
            conv.append_message(conv.roles[1], assistant_message)
            prompt = conv.get_prompt()
            return prompt, conv
        
        def process_image(self, image):
            """Process image(s) for model input."""
            args = {"image_aspect_ratio": "pad"}
            if isinstance(image, list):
                image_tensor = process_images(image, self.image_processor, args)
            else:
                image_tensor = process_images([image], self.image_processor, args)
            return image_tensor.to(self.device, dtype=torch.float16)
        
        def prepare_inputs(self, text_prompt, image, conv):
            """Prepare inputs for model.generate() with stopping criteria."""
            input_ids = (
                tokenizer_image_token(
                    text_prompt, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt"
                ).unsqueeze(0).to(self.device)
            )
            
            image_tensor = None if image is None else self.process_image(image)
            stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
            stopping_criteria = KeywordsStoppingCriteria(
                keywords=[stop_str], tokenizer=self.tokenizer, input_ids=input_ids
            )
            
            return {
                "input_ids": input_ids,
                "images": image_tensor,
                "stopping_criteria": stopping_criteria
            }
    
    processor = LLaVAMedProcessor(tokenizer, image_processor, device=model.device, conv_mode=conv_mode)
    return model, processor


def load_medgemma(
    model_id: str = "google/medgemma-4b-it",
    quantization: Optional[str] = None,
    token: Optional[str] = None,
    return_attention: bool = False,
    return_logits: bool = False,
    use_flash_attention: bool = True,
):
    """
    Load MedGemma - Google's medical-tuned Gemma vision-language model.
    
    Args:
        model_id: HuggingFace model identifier
            Example: "google/medgemma-4b-it"
        quantization: Quantization type - "4b", "8b", "16b", or None for fp32
        token: HuggingFace access token (if None, uses HF_TOKEN or hf_key environment variable)
        return_attention: Include attention weights in generation output
        return_logits: Include token scores in generation output
        
    Returns:
        Tuple of (model, processor) where processor handles text and image inputs
        
    Example:
        >>> model, processor = load_medgemma(quantization="16b")
    """
    from transformers import AutoProcessor, AutoModelForImageTextToText
    import huggingface_hub
    
    # Handle authentication for gated models (like MedGemma 1.5)
    access_token = token or os.getenv("HF_TOKEN") or os.getenv("hf_key")
    if access_token:
        try:
            huggingface_hub.login(token=access_token)
        except Exception:
            pass # Continue if login fails, might be already logged in or token issue handles downstream

    bnb_config, torch_dtype = get_quantization_config(quantization, use_bfloat16=True)
    
    try:
        model = AutoModelForImageTextToText.from_pretrained(
            model_id,
            device_map="auto",
            quantization_config=bnb_config,
            torch_dtype=torch_dtype,
            token=access_token,
            **_gen_config(return_attention, return_logits)
        )
    except Exception as e:
        # Check for corrupted cache/safetensor header error and retry with force_download
        if (("safetensor" in str(e).lower() and "header" in str(e).lower()) or 
            ("invalidheaderdeserialization" in str(e).lower())):
            print(f"Error loading model from cache: {e}")
            print("Attempting to force download model weights (fixing corrupted cache)...")
            model = AutoModelForImageTextToText.from_pretrained(
                model_id,
                device_map="auto",
                quantization_config=bnb_config,
                torch_dtype=torch_dtype,
                token=access_token,
                force_download=True,
                **_gen_config(return_attention, return_logits)
            )
        else:
            # Re-raise other errors
            raise e
    
    model = _finalize_model(model, quantization)
    processor = AutoProcessor.from_pretrained(model_id, token=access_token)
    
    return model, processor


def load_chexagent(
    model_id: str = "StanfordAIMI/CheXagent-8b",
    quantization: Optional[str] = None,
    return_attention: bool = False,
    return_logits: bool = False,
):
    """
    Load CheXagent-8B - Stanford's chest X-ray specialist VLM.
    
    Returns generation_config as third return value for faithful reproduction
    of the authors' decoding settings.
    Note: Uses float16 (not bfloat16) for compute dtype.
    
    Args:
        model_id: HuggingFace model identifier
        quantization: Quantization type - "4b", "8b", "16b", or None for fp32
        return_attention: Include attention weights in generation output
        return_logits: Include token scores in generation output
        
    Returns:
        Tuple of (model, processor, generation_config)
        
    Example:
        >>> model, processor, gen_config = load_chexagent(quantization="16b")
    """
    from transformers import AutoModelForCausalLM, AutoProcessor, GenerationConfig
    
    bnb_config, torch_dtype = get_quantization_config(quantization, use_bfloat16=False)
    
    processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
    generation_config = GenerationConfig.from_pretrained(model_id)
    
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        trust_remote_code=True,
        device_map="auto",
        quantization_config=bnb_config,
        torch_dtype=torch_dtype if bnb_config is None else None,
        **_gen_config(return_attention, return_logits)
    )
    
    if bnb_config is None:
        model = model.to(torch.device("cuda"), dtype=torch_dtype)
    
    model.eval()
    torch.cuda.empty_cache()
    
    return model, processor, generation_config


def load_maira2(
    model_id: str = "microsoft/maira-2",
    token: Optional[str] = None,
    quantization: Optional[str] = None,
    return_attention: bool = False,
    return_logits: bool = False,
):
    """
    Load MAIRA-2 - Microsoft's Multimodal AI Radiology Assistant v2.
    
    Requires HuggingFace authentication token (gated model).
    Note: Uses float16 for 16-bit precision (not bfloat16).
    Processor requires num_additional_image_tokens=1.
    
    Args:
        model_id: HuggingFace model identifier
        token: HuggingFace access token (if None, uses HF_TOKEN environment variable)
        quantization: Quantization type - "4b", "8b", "16b", or None for fp32
        return_attention: Include attention weights in generation output
        return_logits: Include token scores in generation output
        
    Returns:
        Tuple of (model, processor) where processor handles text and image inputs
        
    Example:
        >>> model, processor = load_maira2(quantization="16b")
        >>> model, processor = load_maira2(token="hf_...", quantization="8b")
    """
    from transformers import AutoModelForCausalLM, AutoProcessor
    
    bnb_config, torch_dtype = get_quantization_config(quantization, use_bfloat16=True)
    if quantization == "16b":
        torch_dtype = torch.float16  # MAIRA-2 prefers float16 for 16b
    
    access_token = token or os.getenv("HF_TOKEN")
    
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        trust_remote_code=True,
        token=access_token,
        device_map="auto",
        quantization_config=bnb_config,
        torch_dtype=torch_dtype,
        **_gen_config(return_attention, return_logits)
    )
    
    processor = AutoProcessor.from_pretrained(
        model_id,
        trust_remote_code=True,
        token=access_token,
        num_additional_image_tokens=1,  # MAIRA-2 requires this
    )
    
    model = _finalize_model(model, quantization)
    
    return model, processor


# ============================================================================
# API-BASED MODELS
# ============================================================================

class _BaseAPIModel:
    """
    Base class for API-based models (OpenAI, Gemini, Claude, Grok).
    
    Provides common configuration structure for external API clients.
    """
    def __init__(self, key, model_id, return_logits, model_type):
        self.config = SimpleNamespace(
            model=model_id,
            _name_or_path=model_id,
            return_logits=return_logits,
            model_type=model_type,
        )
        self.model_id = model_id
        self.return_logits = return_logits


def load_openai(model_id: str = "gpt-4o", return_logits: bool = False):
    """
    Load OpenAI GPT-4o API client.
    
    Requires OPENAI_API_KEY environment variable to be set.
    
    Args:
        model_id: OpenAI model identifier
            Examples: "gpt-4o", "gpt-4o-mini"
        return_logits: Whether to request logprobs in API calls
        
    Returns:
        Tuple of (model_client, None) - processor is None for API models
        
    Raises:
        ValueError: If OPENAI_API_KEY environment variable is not set
        
    Example:
        >>> model, _ = load_openai(model_id="gpt-4o")
    """
    from openai import OpenAI
    
    key = os.getenv("OPENAI_API_KEY")
    if not key:
        raise ValueError("Set OPENAI_API_KEY in your environment")
    
    class OpenAIModel(_BaseAPIModel):
        def __init__(self, key, model_id, return_logits):
            super().__init__(key, model_id, return_logits, "openai")
            self.client = OpenAI(api_key=key)
    
    return OpenAIModel(key, model_id, return_logits), None


def load_gemini(model_id: str = "gemini-2.0-flash-exp", return_logits: bool = False, vertex_api: bool = True):
    """
    Load Google Gemini API client.
    
    Requires GEMINI_API_KEY environment variable to be set.
    See: https://ai.google.dev/gemini-api/docs/models
    
    Args:
        model_id: Gemini model identifier
            Examples: "gemini-2.0-flash-exp", "gemini-1.5-pro"
        return_logits: Whether to request logprobs in API calls
        vertex_api: Whether to use Vertex AI API (True) or Google Cloud API (False)
        
    Returns:
        Tuple of (model_client, None) - processor is None for API models
        
    Raises:
        ValueError: If GEMINI_API_KEY environment variable is not set
        
    Example:
        >>> model, _ = load_gemini(model_id="gemini-2.0-flash-exp", vertex_api=True)
    """
    from google import genai
    import base64
    
    if vertex_api:
        key = os.getenv("GEMINI_API_PROJECT_ID")
        if (not key) or (not os.getenv("GOOGLE_APPLICATION_CREDENTIALS")):
            raise ValueError("Set GEMINI_API_PROJECT_ID and GOOGLE_APPLICATION_CREDENTIALS in your environment for Vertex AI API")
    else:
        key = os.getenv("GEMINI_API_KEY")
        if not key:
            raise ValueError("Set GEMINI_API_KEY in your environment for Google Cloud API")
    
    class GeminiModel(_BaseAPIModel):
        def __init__(self, key, model_id, return_logits, vertex_api):
            super().__init__(key, model_id, return_logits, "gemini")

            self.vertex_api = vertex_api
            
            if vertex_api:
                self.client = genai.Client(vertexai=True, project=key, location="global")
                print("Loading Gemini via Vertex AI API...")
            else:
                self.client = genai.Client(api_key=key)
                print("Loading Gemini via Google Cloud API...")
        
        @staticmethod
        def inline_image(pil_image, mime="image/png"):
            """Convert PIL image to Gemini inline data format."""
            import io
            if hasattr(pil_image, "save"):
                buf = io.BytesIO()
                pil_image.save(buf, format=mime.split("/")[-1].upper())
                data = buf.getvalue()
            else:
                data = pil_image
            return {"inline_data": {"mime_type": mime, "data": base64.b64encode(data).decode()}}
    
    return GeminiModel(key, model_id, return_logits, vertex_api), None


def load_claude(model_id: str = "claude-3-7-sonnet-20250219", return_logits: bool = False):
    """
    Load Anthropic Claude API client.
    
    Requires ANTHROPIC_API_KEY environment variable to be set.
    
    Args:
        model_id: Claude model identifier
            Examples: "claude-3-7-sonnet-20250219", "claude-3-5-sonnet-20241022"
        return_logits: Whether to request logprobs in API calls
        
    Returns:
        Tuple of (model_client, None) - processor is None for API models
        
    Raises:
        ValueError: If ANTHROPIC_API_KEY environment variable is not set
        
    Example:
        >>> model, _ = load_claude(model_id="claude-3-7-sonnet-20250219")
    """
    import anthropic
    import base64
    
    key = os.getenv("ANTHROPIC_API_KEY")
    if not key:
        raise ValueError("Set ANTHROPIC_API_KEY in your environment")
    
    class ClaudeModel(_BaseAPIModel):
        def __init__(self, key, model_id, return_logits):
            super().__init__(key, model_id, return_logits, "anthropic")
            self.client = anthropic.Anthropic(api_key=key)
        
        @staticmethod
        def inline_image(pil_image, mime="image/png"):
            """Convert PIL image to Claude message format."""
            import io
            if hasattr(pil_image, "save"):
                buf = io.BytesIO()
                pil_image.save(buf, format=mime.split("/")[-1].upper())
                data = buf.getvalue()
            else:
                data = pil_image
            return {"type": "image", "source": {"type": "base64", "media_type": mime, "data": base64.b64encode(data).decode()}}
    
    return ClaudeModel(key, model_id, return_logits), None


def load_grok(model_id: str = "grok-2-vision-1212", return_logits: bool = False, base_url: str = "https://api.x.ai"):
    """
    Load xAI Grok API client.
    
    Uses OpenAI-compatible API interface.
    Requires XAI_API_KEY environment variable to be set.
    
    Args:
        model_id: Grok model identifier
            Example: "grok-2-vision-1212"
        return_logits: Whether to request logprobs in API calls
        base_url: API base URL (default: "https://api.x.ai")
        
    Returns:
        Tuple of (model_client, None) - processor is None for API models
        
    Raises:
        ValueError: If XAI_API_KEY environment variable is not set
        
    Example:
        >>> model, _ = load_grok(model_id="grok-2-vision-1212")
    """
    from openai import OpenAI
    
    key = os.getenv("XAI_API_KEY")
    if not key:
        raise ValueError("Set XAI_API_KEY in your environment")
    
    class GrokModel(_BaseAPIModel):
        def __init__(self, key, model_id, return_logits, base_url):
            super().__init__(key, model_id, return_logits, "xai")
            self.client = OpenAI(api_key=key, base_url=base_url)
        
        @staticmethod
        def image_url(url):
            """Format image URL for Grok API."""
            return {"type": "image_url", "image_url": {"url": url}}
    
    return GrokModel(key, model_id, return_logits, base_url), None


def load_cohere(model_id: str = "command-r-plus-08-2024", return_logits: bool = False):
    """
    Load Cohere API client.
    
    Requires COHERE_API_KEY environment variable to be set.
    Supports vision models from Cohere.
    
    Args:
        model_id: Cohere model identifier
            Examples: "command-r-plus-08-2024", "command-r-08-2024"
        return_logits: Whether to request logprobs in API calls (not supported by Cohere)
        
    Returns:
        Tuple of (model_client, None) - processor is None for API models
        
    Raises:
        ValueError: If COHERE_API_KEY environment variable is not set
        
    Example:
        >>> model, _ = load_cohere(model_id="command-r-plus-08-2024")
    """
    import cohere
    
    key = os.getenv("COHERE_API_KEY")
    if not key:
        raise ValueError("Set COHERE_API_KEY in your environment")
    
    class CohereModel(_BaseAPIModel):
        def __init__(self, key, model_id, return_logits):
            super().__init__(key, model_id, return_logits, "cohere")
            self.client = cohere.ClientV2(api_key=key)
    
    return CohereModel(key, model_id, return_logits), None


def load_gpt_oss(model_id: str = "openai/gpt-oss-20b", quantization: Optional[str] = None, return_attention: bool = False, return_logits: bool = False):
    """
    Load GPT-OSS model (experimental/placeholder).
    
    Args:
        model_id: HuggingFace model identifier
        quantization: Quantization type - "4b", "8b", "16b", or None for fp32
        return_attention: Include attention weights in generation output
        return_logits: Include token scores in generation output
        
    Returns:
        Tuple of (model, tokenizer)
        
    Example:
        >>> model, tokenizer = load_gpt_oss(quantization="8b")
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer
    
    bnb_config, torch_dtype = get_quantization_config(quantization, use_bfloat16=True)
    
    tok = AutoTokenizer.from_pretrained(model_id, use_fast=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        device_map="auto",
        quantization_config=bnb_config,
        torch_dtype=torch_dtype,
        **_gen_config(return_attention, return_logits)
    )
    
    model = _finalize_model(model, quantization)
    
    return model, tok
