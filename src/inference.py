import numpy as np
from PIL import Image
from transformers.image_utils import load_image
import torch
import math, os
import io, base64, math
import time
import warnings
from transformers import logging as hf_logging

# Suppress "Setting `pad_token_id` to `eos_token_id`..." warnings
hf_logging.set_verbosity_error()
warnings.filterwarnings("ignore", message=".*pad_token_id.*")

from src.prompts import (
    # --- CXR VALSE ---
    CXR_VALSE_TEXT_PROMPT,
    CXR_VALSE_BINARY_TEXT_PROMPT_BINARY,
    CXR_HISTORY_TEXT_PROMPT,
    CXR_HISTORY_TEXT_PROMPT_5CLASS,
    CXR_HISTORY_TEXT_PROMPT_5CLASS_V1,
    CXR_HISTORY_TEXT_PROMPT_5CLASS_V2,
    CXR_HISTORY_TEXT_PROMPT_5CLASS_V3,

    # --- Multimodality / VLM Bias Benchmark ---
    GLAUCOMA_TEXT_PROMPT,
    MIMIC_TEXT_PROMPT,
    MIMIC_TEXT_PROMPT_5CLASS,
    MIMIC_TEXT_PROMPT_5CLASS_V1,
    MIMIC_TEXT_PROMPT_5CLASS_V2,
    MIMIC_TEXT_PROMPT_5CLASS_V3,
    
    HAM10000_TEXT_PROMPT_FULL,
    HAM10000_TEXT_PROMPT_BINARY,
    BRSET_TEXT_PROMPT,
    mBRSET_TEXT_PROMPT,
    

    # --- Only Image Prompts ---
    GLAUCOMA_ONLY_IMAGE_TEXT_PROMPT,
    MIMIC_ONLY_IMAGE_TEXT_PROMPT,
    MIMIC_ONLY_IMAGE_TEXT_PROMPT_5CLASS,
    MIMIC_ONLY_IMAGE_TEXT_PROMPT_5CLASS_V1,
    MIMIC_ONLY_IMAGE_TEXT_PROMPT_5CLASS_V2,
    MIMIC_ONLY_IMAGE_TEXT_PROMPT_5CLASS_V3,
    
    HAM10000_ONLY_IMAGE_TEXT_PROMPT,
    BRSET_ONLY_IMAGE_TEXT_PROMPT,
    mBRSET_ONLY_IMAGE_TEXT_PROMPT,

    # --- Only Text Prompts ---
    GLAUCOMA_ONLY_TEXT_PROMPT,
    MIMIC_ONLY_TEXT_PROMPT,
    MIMIC_ONLY_TEXT_PROMPT_5CLASS,
    MIMIC_ONLY_TEXT_PROMPT_5CLASS_V1,
    MIMIC_ONLY_TEXT_PROMPT_5CLASS_V2,
    MIMIC_ONLY_TEXT_PROMPT_5CLASS_V3,
    
    HAM10000_ONLY_TEXT_PROMPT_FULL,
    HAM10000_ONLY_TEXT_PROMPT_BINARY,
    BRSET_ONLY_TEXT_PROMPT,
    mBRSET_ONLY_TEXT_PROMPT,
    
)


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def _pil_to_base64_png(img):
    """Return a data URL 'data:image/png;base64,...' from a PIL.Image or raw bytes/np array path."""
    if img is None:
        return None
    if not isinstance(img, Image.Image):
        img = load_image(img)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    b64 = base64.b64encode(buf.getvalue()).decode()
    return f"data:image/png;base64,{b64}"


def _pil_to_base64_jpeg(img, quality=85):
    """Return a data URL 'data:image/jpeg;base64,...' from a PIL.Image."""
    if img is None:
        return None
    if not isinstance(img, Image.Image):
        img = load_image(img)
    # Convert to RGB to ensure JPEG compatibility (no alpha channel)
    if img.mode in ("RGBA", "P"):
        img = img.convert("RGB")
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=quality)
    b64 = base64.b64encode(buf.getvalue()).decode()
    return f"data:image/jpeg;base64,{b64}"


def _extract_yes_no_from_openai_logprobs(completion, tokens=None):
    """
    OpenAI/Grok chat.completions: take the first generated token's top_logprobs and pull probs.
    Returns p_yes, p_no, p_Yes, p_No (floats or None) and a dict with all token probabilities.
    
    Args:
        completion: OpenAI completion object
        tokens: List of tokens to extract probabilities for (default: ['yes', 'no', 'Yes', 'No'])
                Can include any tokens, e.g., ['yes', 'no', '?', 'unmatched']
    
    Returns:
        tuple: (p_yes, p_no, p_Yes, p_No, prob_dict)
               prob_dict contains probabilities for all tokens in the list
    """
    if tokens is None:
        tokens = ['yes', 'no', 'Yes', 'No']
    
    try:
        choice = completion.choices[0]
        toks = choice.logprobs.content[0].top_logprobs
        d = {t.token: math.exp(t.logprob) for t in toks}
        
        # Helper to find token with variants (with/without leading space or newline)
        def find_token_prob(token_str):
            candidates = [token_str, " " + token_str, "\n" + token_str]
            for candidate in candidates:
                if candidate in d:
                    return d[candidate]
            return None
        
        # Extract all tokens
        prob_dict = {}
        for token in tokens:
            prob_dict[token] = find_token_prob(token)
        
        # For backward compatibility, extract the standard tokens
        p_yes = prob_dict.get('yes')
        p_no = prob_dict.get('no')
        p_Yes = prob_dict.get('Yes')
        p_No = prob_dict.get('No')
        
        return p_yes, p_no, p_Yes, p_No, prob_dict
    except Exception:
        return None, None, None, None, {}


def _extract_yes_no_probs(processor, probs, return_logits, tokens=None):
    """
    Extract probabilities for specified tokens from first token probabilities.
    
    Args:
        processor: Model processor with tokenizer
        probs: Probability tensor for first token
        return_logits: Whether to extract probabilities
        tokens: List of tokens to extract probabilities for (default: ['yes', 'no', 'Yes', 'No'])
                Can include any tokens, e.g., ['yes', 'no', '?', 'unmatched']
    
    Returns:
        tuple: (p_yes, p_no, p_Yes, p_No, prob_dict)
               prob_dict contains probabilities for all tokens in the list
    """
    if not return_logits:
        return None, None, None, None, {}
    
    if tokens is None:
        tokens = ['yes', 'no', 'Yes', 'No']
    
    try:
        prob_dict = {}
        
        for token_str in tokens:
            token_ids = processor.tokenizer(token_str, add_special_tokens=False)["input_ids"]
            
            if len(token_ids) >= 1:
                # Use first sub-token if multi-token
                prob_dict[token_str] = probs[token_ids[0]].item()
                if len(token_ids) > 1:
                    print(f"Warning: '{token_str}' is not a single token. Using first sub-token.")
            else:
                prob_dict[token_str] = None
        
        # For backward compatibility, extract the standard tokens
        p_yes = prob_dict.get('yes')
        p_no = prob_dict.get('no')
        p_Yes = prob_dict.get('Yes')
        p_No = prob_dict.get('No')
        
        return p_yes, p_no, p_Yes, p_No, prob_dict
    except Exception as e:
        print(f"Warning: Could not extract token probabilities: {e}")
        return None, None, None, None, {}


def _build_conversation(prompt, image, include_image=True):
    """Build conversation structure for vision-language models."""
    if image is None or not include_image:
        return [{"role": "user", "content": [{"type": "text", "text": prompt}]}]
    else:
        return [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": prompt}]}]


def _format_return_values(output_text, generated_ids, attentions, scores, 
                          p_yes_and_no, return_attention, return_logits,
                          p_yes=None, p_no=None, p_Yes=None, p_No=None, prob_dict=None):
    """Format return values consistently across all model types."""
    if p_yes_and_no:
        return (output_text, generated_ids, 
                attentions if return_attention else None,
                scores if return_logits else None,
                p_yes, p_no, p_Yes, p_No, prob_dict or {})
    else:
        return (output_text, generated_ids,
                attentions if return_attention else None,
                scores if return_logits else None)


# ============================================================================
# MAIN GENERATION FUNCTION
# ============================================================================

def generate_text(model, processor, prompt, image, quantization=None, return_attention=False, 
                 return_logits=False, dataset=None, p_yes_and_no=False, row=None, unmatched=False, tokens=None, priority_img=False, modality=None):
    
    # Add priority instruction if priority_img is True and both image and meaningful text are present
    # Don't add for Only_image mode since text is just instructions, not complementary information
    if priority_img and image is not None and modality != "Only_image":
        priority_instruction = "\n\nIMPORTANT: If there is a mismatch or conflict between the image and the text information provided, prioritize the information from the image."
        prompt = prompt + priority_instruction
    
    model_type = model.config.model_type.lower()
    model_path = model.config._name_or_path.lower()
    
    # ============================================================================
    # LLaVA, Qwen2-VL, LLaMA 3.2 (excluding LLaVA-Med)
    # ============================================================================
    if (("llava" in model_type or "qwen" in model_type or "mllama" in model_type) and 
        "llava_med" not in model_path and "llava-med" not in model_path):
        
        if model_type == "mllama":
            suffix = " Only respond with the answer (yes, no, or unmatched). No aditional commentary." if unmatched else " Only respond with the answer (yes or no). No aditional commentary."
            prompt = prompt + suffix
        
        conversation = _build_conversation(prompt, image)
        text_prompt = processor.apply_chat_template(conversation, add_generation_prompt=True)
        
        inputs = processor(text=[text_prompt], images=[image] if image is not None else None, 
                          padding=True, return_tensors="pt").to("cuda")
        outputs = model.generate(**inputs, max_new_tokens=128, temperature=0.01, do_sample=False)
        
        output_ids = outputs.sequences[0]
        input_len = inputs.input_ids.shape[-1]
        generated_ids = [output_ids[input_len:]]
        output_text = processor.batch_decode(generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True)[0]
        
        probs = outputs.scores[0][0] if return_logits and p_yes_and_no else None
        p_yes, p_no, p_Yes, p_No, prob_dict = _extract_yes_no_probs(processor, probs, return_logits and p_yes_and_no, tokens)
        
        return _format_return_values(output_text, generated_ids, outputs.attentions, outputs.scores,
                                     p_yes_and_no, return_attention, return_logits, p_yes, p_no, p_Yes, p_No, prob_dict)

    # ============================================================================
    # PaliGemma
    # ============================================================================
    elif "paligemma" in model_type:
        if image is None:
            prompt = f"<bos>{prompt} Answer:"
        else:
            prompt = f"<image><bos>{prompt} Answer:"
            image = load_image(image)
        
        model_inputs = processor(text=prompt, images=image, return_tensors="pt", do_convert_rgb=True)
        if quantization:
            model_inputs = model_inputs.to(torch.bfloat16).to(model.device)
        else:
            model_inputs = model_inputs.to(model.device)
            
        input_len = model_inputs["input_ids"].shape[-1]
        
        with torch.inference_mode():
            outputs = model.generate(**model_inputs, max_new_tokens=128, do_sample=False, temperature=0.01)
            output_ids = outputs.sequences[0][input_len:]
            output_text = processor.decode(output_ids, skip_special_tokens=True)
            
        probs = outputs.scores[0][0] if return_logits and p_yes_and_no else None
        p_yes, p_no, p_Yes, p_No, prob_dict = _extract_yes_no_probs(processor, probs, return_logits and p_yes_and_no, tokens)
        
        return _format_return_values(output_text, output_ids, outputs.attentions, outputs.scores,
                                     p_yes_and_no, return_attention, return_logits, p_yes, p_no, p_Yes, p_No, prob_dict)
    
    # ============================================================================
    # Janus Pro
    # ============================================================================
    elif "multi_modality" in model_type and "janus" in model_path:
        if image is not None:
            image = load_image(image)
            conversation = [
                {"role": "<|User|>", "content": f"<image_placeholder>\n{prompt}", "images": [image]},
                {"role": "<|Assistant|>", "content": ""}
            ]
            pil_images = [image]
        else:
            conversation = [
                {"role": "<|User|>", "content": f"{prompt}"},
                {"role": "<|Assistant|>", "content": ""}
            ]
            pil_images = None
        
        prepare_inputs = processor(conversations=conversation, images=pil_images, force_batchify=True).to(model.device)
        inputs_embeds = model.prepare_inputs_embeds(**prepare_inputs)
        
        outputs = model.language_model.generate(
            inputs_embeds=inputs_embeds,
            attention_mask=prepare_inputs.attention_mask,
            pad_token_id=processor.tokenizer.eos_token_id,
            bos_token_id=processor.tokenizer.bos_token_id,
            eos_token_id=processor.tokenizer.eos_token_id,
            max_new_tokens=512,
            temperature=0.01,
            do_sample=False,
            use_cache=True,
            output_attentions=return_attention,
            output_scores=return_logits,
            return_dict_in_generate=True
        )
        
        output_text = processor.tokenizer.decode(outputs.sequences[0].cpu().tolist(), skip_special_tokens=True)
        
        probs = outputs.scores[0][0] if return_logits and p_yes_and_no else None
        p_yes, p_no, p_Yes, p_No, prob_dict = _extract_yes_no_probs(processor, probs, return_logits and p_yes_and_no, tokens)
        
        return _format_return_values(output_text, outputs.sequences[0], outputs.attentions, outputs.scores,
                                     p_yes_and_no, return_attention, return_logits, p_yes, p_no, p_Yes, p_No, prob_dict)

    # ============================================================================
    # BiomedGPT (OFA-based)
    # ============================================================================
    elif ("ofa" in str(type(model)).lower()) or ("biomedgpt" in model_path):
        inputs = processor(text=prompt, image=image)
        input_ids = inputs["input_ids"].to(model.device) if inputs["input_ids"] is not None else None
        patch_images = inputs["patch_images"].to(model.device) if inputs["patch_images"] is not None else None

        with torch.no_grad():
            gen = model.generate(
                input_ids,
                patch_images=patch_images,
                no_repeat_ngram_size=2,
                max_new_tokens=128,
                do_sample=False,
                temperature=0.01,
                output_attentions=return_attention,
                output_scores=return_logits,
                return_dict_in_generate=True
            )
        
        raw_text = processor.tokenizer.batch_decode(gen['sequences'], skip_special_tokens=True)
        output_text = raw_text[0].strip()
        
        probs = gen['scores'][0][0].cpu() if return_logits and p_yes_and_no else None
        p_yes, p_no, p_Yes, p_No, prob_dict = _extract_yes_no_probs(processor, probs, return_logits and p_yes_and_no, tokens)
        
        torch.cuda.empty_cache()
        del input_ids, patch_images
        
        gen['sequences'] = gen['sequences'].cpu()
        gen['scores'] = gen['scores'][0].cpu() if return_logits else None
        
        return _format_return_values(output_text, gen['sequences'][0], gen if return_attention else None, 
                                     gen['scores'], p_yes_and_no, return_attention, return_logits, 
                                     p_yes, p_no, p_Yes, p_No, prob_dict)
    
    # ============================================================================
    # LLaVA-Med
    # ============================================================================
    elif "llava-med" in model_path or "llava_med" in model_path:
        conversation = _build_conversation(prompt, image)
        
        # Dataset-specific system messages
        system_messages = {
            'mimic': "You are an expert in radiology. Please analyze the image and text and provide an answer.",
            'ham10000': "You are a dermatologist. Please analyze the image and text and provide an answer.",
            'mbrset': "You are an ophthalmologist. Please analyze the image and text and provide an answer.",
            'medeval': "You are an ophthalmologist. Please analyze the image and text and provide an answer.",
            'brset': "You are an ophthalmologist. Please analyze the image and text and provide an answer."
        }
        system_message = system_messages.get(dataset, None)

        # By now let's turn off the system message
        system_message = None

        
        text_prompt, conv = processor.apply_chat_template(conversation, assistant_message=None, system_message=system_message)
        inputs_dict = processor.prepare_inputs(text_prompt, image, conv)

        with torch.inference_mode():
            outputs = model.generate(
                inputs_dict["input_ids"],
                images=inputs_dict["images"],
                max_new_tokens=512,
                temperature=0.01,
                do_sample=False,
                use_cache=True,
                stopping_criteria=[inputs_dict["stopping_criteria"]],
                output_attentions=True,
                output_scores=True,
                return_dict_in_generate=True,
                pad_token_id=processor.tokenizer.pad_token_id,
                eos_token_id=processor.tokenizer.eos_token_id,
            )
        
        output_ids = outputs.sequences[0]
        input_len = inputs_dict["input_ids"].shape[-1]  # Fix: use -1 instead of 1
        generated_ids = [output_ids[input_len:]]
        
        # Check if generation is empty
        if len(generated_ids[0]) == 0:
            print(f"Warning: LLaVA-Med generated 0 tokens. Input len: {input_len}, Output len: {len(output_ids)}")
            print(f"Prompt preview: {text_prompt[:200]}...")
            print(f"Stopping criteria: {inputs_dict['stopping_criteria']}")
            output_text = ""
        else:
            output_text = processor.tokenizer.decode(generated_ids[0], skip_special_tokens=True).strip()
        
        probs = outputs.scores[0][0] if return_logits and p_yes_and_no and len(outputs.scores) > 0 else None
        p_yes, p_no, p_Yes, p_No, prob_dict = _extract_yes_no_probs(processor, probs, return_logits and p_yes_and_no, tokens)
        
        return _format_return_values(output_text, generated_ids, outputs.attentions, outputs.scores,
                                     p_yes_and_no, return_attention, return_logits, p_yes, p_no, p_Yes, p_No, prob_dict)
        
    # ============================================================================
    # MedGemma / Gemma3
    # ============================================================================
    elif "gemma3" in model_type:
        if image is None:
            messages = [{"role": "user", "content": [{"type": "text", "text": prompt}]}]
        else:
            messages = [{"role": "user", "content": [{"type": "text",  "text": prompt}, {"type": "image", "image": image}]}]

        inputs = processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt"
        ).to(model.device, dtype=torch.bfloat16 if quantization in {"4b", "8b"} else None)

        input_len = inputs["input_ids"].shape[-1]
        
        with torch.inference_mode():
            outputs = model.generate(
                **inputs,
                max_new_tokens=512,
                do_sample=False,
                output_attentions=return_attention,
                output_scores=return_logits,
                return_dict_in_generate=True
            )

            output_ids = outputs.sequences[0][input_len:]
            output_text = processor.decode(output_ids, skip_special_tokens=True).strip()

        probs = outputs.scores[0][0] if return_logits and p_yes_and_no else None
        p_yes, p_no, p_Yes, p_No, prob_dict = _extract_yes_no_probs(processor, probs, return_logits and p_yes_and_no, tokens)
        
        return _format_return_values(output_text, output_ids, outputs.attentions, outputs.scores,
                                     p_yes_and_no, return_attention, return_logits, p_yes, p_no, p_Yes, p_No, prob_dict)
    
    # ============================================================================
    # CheXagent
    # ============================================================================
    elif "chexagent" in model_path or "chexagent" in model_type:
        user_prompt = f" USER: <s>{prompt} ASSISTANT: <s>"
        img_list = [image] if image is not None else None

        inputs = processor(images=img_list, text=user_prompt, return_tensors="pt").to(model.device, dtype=torch.float16)
        input_len = inputs["input_ids"].shape[-1]

        with torch.inference_mode():
            outputs = model.generate(
                **inputs,
                bos_token_id=1,
                eos_token_id=2,
                max_length=512,
                max_new_tokens=512,
                do_sample=False,
                temperature=0.01,
                output_attentions=return_attention,
                output_scores=return_logits,
                return_dict_in_generate=True
            )

            out_text = processor.tokenizer.decode(outputs.sequences[0], skip_special_tokens=True)

        probs = outputs.scores[0][0] if return_logits and p_yes_and_no else None
        p_yes, p_no, p_Yes, p_No, prob_dict = _extract_yes_no_probs(processor, probs, return_logits and p_yes_and_no, tokens)
        
        return _format_return_values(out_text, outputs.sequences[0], outputs.attentions, outputs.scores,
                                     p_yes_and_no, return_attention, return_logits, p_yes, p_no, p_Yes, p_No, prob_dict)

    # ============================================================================
    # MAIRA-2
    # ============================================================================
    elif "maira" in model_path or "maira" in model_type:
        pass  # TODO: Implement when needed

    # ============================================================================
    # OpenAI API
    # ============================================================================
    elif model.config.model_type == "openai":
        
        is_gpt5 = "gpt-5" in model.model_id
        messages = []

        if image is None:
            # Text only
            messages = [{"role": "user", "content": prompt}]
        else:
            # JPEG conversion for payload optimization
            data_url = _pil_to_base64_jpeg(image)
            
            if is_gpt5:
                 # GPT-5 'responses' API requires different content types
                 # And 'image_url' for input_image expects the URL string directly, not an object
                 content = [
                     {"type": "input_text", "text": prompt},
                     {"type": "input_image", "image_url": data_url}
                 ]
            else:
                 # Standard Chat Completions API
                 content = [
                     {"type": "text", "text": prompt},
                     {"type": "image_url", "image_url": {"url": data_url}}
                 ]
            
            messages = [{"role": "user", "content": content}]

        # Retry logic with exponential backoff
        max_retries = 2
        retry_count = 0
        completion = None
        out_text = ""
        
        while retry_count < max_retries:
            try:
                # Check for gpt-5 to use the new responses endpoint
                if is_gpt5:
                    resp = model.client.responses.create(
                        model=model.model_id,
                        input=messages,
                        reasoning={"effort": "low"},
                        timeout=90.0
                    )
                    out_text = resp.output_text
                    completion = resp
                else:
                    # Standard chat completions for other models
                    logprobs = True if return_logits else False
                    top_logprobs = 10 if return_logits else None
                    
                    completion = model.client.chat.completions.create(
                        model=model.model_id,
                        messages=messages,
                        max_completion_tokens=128,
                        logprobs=logprobs,
                        top_logprobs=top_logprobs,
                        timeout=90.0
                    )
                    out_text = completion.choices[0].message.content.strip()
                
                break
            except Exception as e:
                retry_count += 1
                error_str = str(e)
                
                # Check for rate limits or other specific errors
                if "429" in error_str:
                    # Rate limit: respect the header or use exponential backoff
                    wait_time = min(60, 2 * (2 ** (retry_count - 1)))
                    print(f"OpenAI Rate Limit Hit. Waiting {wait_time}s...")
                elif "timeout" in error_str.lower():
                    # Timeouts might be transient or payload too large
                    wait_time = 5
                    print(f"OpenAI Timeout (Attempt {retry_count}/{max_retries}). Retrying in {wait_time}s...")
                else:
                    # Other errors
                    wait_time = min(60, 1 * (2 ** (retry_count - 1)))
                    print(f"Error in OpenAI API (Attempt {retry_count}/{max_retries}): {e}")
                
                if retry_count < max_retries:
                    time.sleep(wait_time)
                else:
                    print("Max retries reached. Skipping...")
                    completion = None
                    break

        # Handle return values
        if "gpt-5" in model.model_id:
            # GPT-5 via responses API returns text only, no logprobs support implemented here
            if p_yes_and_no:
                return out_text, None, None, None, None, None, None, None, {}
            else:
                return out_text, None, None, None
        else:
            # Standard chat completion return
            if p_yes_and_no and return_logits and completion:
                p_yes, p_no, p_Yes, p_No, prob_dict = _extract_yes_no_from_openai_logprobs(completion, tokens=tokens)
                return out_text, None, None, completion.choices[0].logprobs, p_yes, p_no, p_Yes, p_No, prob_dict
            elif p_yes_and_no:
                return out_text, None, None, None, None, None, None, None, {}
            else:
                return out_text, None, None, None
            
    # ============================================================================
    # Google Gemini API
    # ============================================================================
    elif model.config.model_type == "gemini":
        from google.genai import types
        
        if image is None:
            parts = [prompt]
        else:
            data_url = _pil_to_base64_png(image)
            b64 = data_url.split(",")[1]
            img_bytes = base64.b64decode(b64)
            parts = [prompt, {"inline_data": {"mime_type": "image/png", "data": base64.b64encode(img_bytes).decode()}}]

        # Retry logic: wait 1 minute on error, max 10 retries
        max_retries = 10
        retry_count = 0
        resp = None

        # Configure generation based on model version
        # Suppress logging warnings from google.genai
        import logging
        logging.getLogger("google.genai").setLevel(logging.ERROR)

        if "gemini-3" in model.model_id:
            gen_config = types.GenerateContentConfig(
                temperature=0.0, 
                max_output_tokens=8192,
                thinking_config=types.ThinkingConfig(include_thoughts=False, thinking_budget=1)
            )
        else:
            gen_config = types.GenerateContentConfig(
                temperature=0.0, 
                max_output_tokens=128,
                thinking_config=types.ThinkingConfig(include_thoughts=False, thinking_budget=0)
            )
        
        while retry_count < max_retries:
            try:
                resp = model.client.models.generate_content(
                    model=model.model_id,
                    contents=parts,
                    config=gen_config
                )
                break
            except Exception as e:
                retry_count += 1
                print(f"Error in Gemini API (Attempt {retry_count}/{max_retries}): {e}")
                if retry_count < max_retries:
                    print("Waiting 60 seconds before retrying...")
                    time.sleep(60)
                else:
                    print("Max retries reached. Skipping...")
                    resp = None
                    break

        try:
            if resp and resp.candidates and resp.candidates[0].content and resp.candidates[0].content.parts:
                # Manually extract text parts to avoid warning about 'thought_signature'
                out_text = "".join([part.text for part in resp.candidates[0].content.parts if hasattr(part, 'text') and part.text])
            else:
                out_text = ""
        except Exception:
            out_text = ""

        if p_yes_and_no and return_logits:
            p_yes = p_no = p_Yes = p_No = None
            return out_text, None, None, None, p_yes, p_no, p_Yes, p_No, {}
        elif p_yes_and_no:
            return out_text, None, None, None, None, None, None, None, {}
        else:
            return out_text, None, None, None

    # ============================================================================
    # Anthropic Claude API
    # ============================================================================
    elif model.config.model_type == "anthropic":
        if image is None:
            content = [{"type": "text", "text": prompt}]
        else:
            data_url = _pil_to_base64_png(image)
            b64 = data_url.split(",")[1]
            content = [
                {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": b64}},
                {"type": "text", "text": prompt},
            ]

        msg = model.client.messages.create(
            model=model.model_id,
            max_tokens=128,
            temperature=0.0,
            messages=[{"role": "user", "content": content}]
        )

        out_text = ""
        try:
            blocks = msg.content or []
            out_text = "".join([getattr(b, "text", "") or "" for b in blocks]).strip()
        except Exception:
            pass

        if p_yes_and_no:
            return out_text, None, None, None, None, None, None, None, {}
        else:
            return out_text, None, None, None

    # ============================================================================
    # Cohere API
    # ============================================================================
    elif model.config.model_type == "cohere":
        import cohere
        
        # Build message content as plain dictionaries (not using Pydantic models)
        content_parts = []
        
        # Add text first (matching working example)
        content_parts.append({"type": "text", "text": prompt})
        
        if image is not None:
            # Convert image to base64 data URL
            data_url = _pil_to_base64_png(image)
            # Cohere expects nested structure: {"type": "image_url", "image_url": {"url": "..."}}
            content_parts.append({"type": "image_url", "image_url": {"url": data_url}})
        
        # Create the message as a plain dictionary
        messages = [{"role": "user", "content": content_parts}]
        
        # Retry logic: wait 1 minute on error, max 2 retries
        max_retries = 2
        retry_count = 0
        response = None
        
        while retry_count < max_retries:
            try:
                response = model.client.chat(
                    model=model.model_id,
                    messages=messages,
                    temperature=0.0
                )
                break
            except Exception as e:
                retry_count += 1
                print(f"Error in Cohere API (Attempt {retry_count}/{max_retries}): {e}")
                if retry_count < max_retries:
                    print("Waiting 60 seconds before retrying...")
                    time.sleep(60)
                else:
                    print("Max retries reached. Skipping...")
                    response = None
                    break
        
        # Extract text from response
        out_text = ""
        if response and response.message and response.message.content:
            for content in response.message.content:
                if hasattr(content, 'text'):
                    out_text += content.text
        out_text = out_text.strip()
        
        if p_yes_and_no:
            return out_text, None, None, None, None, None, None, None, {}
        else:
            return out_text, None, None, None

    else:
        raise ValueError(f"Model type not supported: {model.config.model_type}")


# ============================================================================
# DATASET PREDICTION FUNCTION
# ============================================================================

def predict_dataset(metadata_row, model=None, processor=None, quantization=None, return_attention=False, 
                   return_logits=False, dataset='mimic', modality=None, only_prompt=False, 
                   p_yes_and_no=False, original=False, unmatched=False, history_cols_to_use=None, version='default', tokens=None, priority_img=False):
    """
    Predicts radiological findings from a MIMIC-CXR sample using both image and metadata.
    The function loads the image from the provided filepath, constructs a text prompt using
    MIMIC_TEXT_PROMPT, and then calls generate_text to perform inference.
    """
    
    # ============================================================================
    # MIMIC Dataset
    # ============================================================================
    if dataset == 'mimic':
        image = load_image(metadata_row['filepath'])
        if modality == 'Only_image':
            text_metadata = MIMIC_ONLY_IMAGE_TEXT_PROMPT(unmatched=unmatched)
        elif modality == 'Only_text':
            text_metadata = MIMIC_ONLY_TEXT_PROMPT(metadata_row, unmatched=unmatched)
        else:
            text_metadata = MIMIC_TEXT_PROMPT(metadata_row, unmatched=unmatched)
            
    # ============================================================================
    # MIMIC 5-Class Dataset
    # ============================================================================
    elif dataset == 'mimic_5class' or dataset == 'mimic_5class_test':
        image = load_image(metadata_row['filepath'])
        
        # Select appropriate prompt based on modality and version
        prompt_map = {
            'Only_image': {
                'v1': MIMIC_ONLY_IMAGE_TEXT_PROMPT_5CLASS_V1,
                'v2': MIMIC_ONLY_IMAGE_TEXT_PROMPT_5CLASS_V2,
                'v3': MIMIC_ONLY_IMAGE_TEXT_PROMPT_5CLASS_V3,
                'default': MIMIC_ONLY_IMAGE_TEXT_PROMPT_5CLASS
            },
            'Only_text': {
                'v1': MIMIC_ONLY_TEXT_PROMPT_5CLASS_V1,
                'v2': MIMIC_ONLY_TEXT_PROMPT_5CLASS_V2,
                'v3': MIMIC_ONLY_TEXT_PROMPT_5CLASS_V3,
                'default': MIMIC_ONLY_TEXT_PROMPT_5CLASS
            },
            'both': {
                'v1': MIMIC_TEXT_PROMPT_5CLASS_V1,
                'v2': MIMIC_TEXT_PROMPT_5CLASS_V2,
                'v3': MIMIC_TEXT_PROMPT_5CLASS_V3,
                'default': MIMIC_TEXT_PROMPT_5CLASS
            }
        }
        
        modality_key = modality if modality in ['Only_image', 'Only_text'] else 'both'
        version_key = version if version in ['v1', 'v2', 'v3'] else 'default'
        prompt_func = prompt_map[modality_key][version_key]
        
        # Apply prompt (some are functions, some are constants)
        if modality_key == 'Only_image':
            text_metadata = prompt_func(unmatched=unmatched)
        else:
            text_metadata = prompt_func(metadata_row, unmatched=unmatched)
                
    # ============================================================================
    # History-based Dataset
    # ============================================================================
    elif 'history' in dataset:
        image = load_image(metadata_row['filepath'])
        
        history_prompts = {
            'v1': CXR_HISTORY_TEXT_PROMPT_5CLASS_V1,
            'v2': CXR_HISTORY_TEXT_PROMPT_5CLASS_V2,
            'v3': CXR_HISTORY_TEXT_PROMPT_5CLASS_V3,
            'default': CXR_HISTORY_TEXT_PROMPT_5CLASS
        }
        
        prompt_func = history_prompts.get(version, history_prompts['default'])
        text_metadata = prompt_func(metadata_row, original=original, history_cols_to_use=history_cols_to_use, unmatched=unmatched)
    
    # ============================================================================
    # CXR VALSE Dataset
    # ============================================================================
    elif 'cxr_valse' in dataset:
        image = load_image(metadata_row['filepath'])
        if 'binary' in dataset:
            text_metadata = CXR_VALSE_BINARY_TEXT_PROMPT_BINARY(metadata_row, original=original)
        else:
            text_metadata = CXR_VALSE_TEXT_PROMPT(metadata_row, original=original)
        
    # ============================================================================
    # HAM10000 Dataset
    # ============================================================================
    elif dataset == 'ham10000':
        image = load_image(metadata_row['filepath'])
        if modality == 'Only_image':
            text_metadata = HAM10000_ONLY_IMAGE_TEXT_PROMPT
        elif modality == 'Only_text':
            text_metadata = HAM10000_ONLY_TEXT_PROMPT_BINARY(metadata_row)
        else:
            text_metadata = HAM10000_TEXT_PROMPT_BINARY(metadata_row)
        
    # ============================================================================
    # mBRSET Dataset
    # ============================================================================
    elif dataset == 'mbrset':
        image = load_image(metadata_row['filepath'])
        if modality == 'Only_image':
            text_metadata = mBRSET_ONLY_IMAGE_TEXT_PROMPT
        elif modality == 'Only_text':
            text_metadata = mBRSET_ONLY_TEXT_PROMPT(metadata_row)
        else:
            text_metadata = mBRSET_TEXT_PROMPT(metadata_row)
    
    # ============================================================================
    # BRSET Dataset
    # ============================================================================
    elif dataset == 'brset':
        image = load_image(metadata_row['filepath'])
        if modality == 'Only_image':
            text_metadata = BRSET_ONLY_IMAGE_TEXT_PROMPT
        elif modality == 'Only_text':
            text_metadata = BRSET_ONLY_TEXT_PROMPT(metadata_row)
        else:
            text_metadata = BRSET_TEXT_PROMPT(metadata_row)
        
    # ============================================================================
    # MedEval Dataset
    # ============================================================================
    elif dataset == 'medeval':
        data = np.load(metadata_row['filepath'])
        image = Image.fromarray(data['slo_fundus'])
        if modality == 'Only_image':
            text_metadata = GLAUCOMA_ONLY_IMAGE_TEXT_PROMPT
        elif modality == 'Only_text':
            text_metadata = GLAUCOMA_ONLY_TEXT_PROMPT(metadata_row)
        else:               
            text_metadata = GLAUCOMA_TEXT_PROMPT(metadata_row)
        
    else:
        raise ValueError(f"Dataset not supported: {dataset}, only 'mimic' and 'ham10000' datasets are supported.")
    
    # Handle text-only modality
    if modality == 'Only_text':
        image = None

    # Return prompt only if requested
    if only_prompt:
        return text_metadata, image
    
    # Call the general generate_text function to obtain predictions
    return generate_text(model, processor, text_metadata, image, quantization, return_attention, 
                        return_logits, dataset=dataset, p_yes_and_no=p_yes_and_no, 
                        row=metadata_row, unmatched=unmatched, tokens=tokens, priority_img=priority_img, modality=modality)
