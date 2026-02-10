from llava.mm_utils import (
    KeywordsStoppingCriteria,
    get_model_name_from_path,
    process_images,
    tokenizer_image_token,
)

import torch
from llava.constants import DEFAULT_IMAGE_TOKEN, IMAGE_TOKEN_INDEX
from llava.conversation import SeparatorStyle, conv_templates
from llava.model.builder import load_pretrained_model

MODEL = "microsoft/llava-med-v1.5-mistral-7b"
model_name = get_model_name_from_path(MODEL)
model_name

CONV_MODE = "llava_v0"

tokenizer, model, image_processor, context_len = load_pretrained_model(
                                                    model_path=MODEL, 
                                                    model_base=None, 
                                                    model_name=model_name, 
                                                    load_8bit=True, 
                                                    load_4bit=False, 
                                                    device_map="auto", 
                                                    device="cuda")

def process_image(image):
    args = {"image_aspect_ratio": "pad"}

    image_tensor = process_images([image], image_processor, args)
    
    return image_tensor.to(model.device, dtype=torch.float16)



def create_prompt(prompt: str):
    """
    Create a prompt for the model to generate text given a prompt.
    
    Args:
        prompt (str): The prompt to generate text from. The prompt should contain the <image> token if an image is to be used.
    """
    conv = conv_templates[CONV_MODE].copy()
    roles = conv.roles
    conv.append_message(roles[0], prompt)
    conv.append_message(roles[1], None)
    return conv.get_prompt(), conv



# Example usage:
prompt, conv = create_prompt(DEFAULT_IMAGE_TOKEN + "\n" + "What type of imaging does this represent?")
print(prompt)
print("conv", conv)



from PIL import Image

def ask_image(image: Image, prompt: str):
    print("Input prompt:", prompt)
    image_tensor = process_image(image)

    #print(f"Image shape {image_tensor.shape}")
    
    prompt, conv = create_prompt(prompt)
    input_ids = (
        tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt")
        .unsqueeze(0)
        .to(model.device)
    )
    
    #print("Input IDs:", input_ids)


    stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
    stopping_criteria = KeywordsStoppingCriteria(
        keywords=[stop_str], tokenizer=tokenizer, input_ids=input_ids
    )

    with torch.inference_mode():
        output_ids = model.generate(
            input_ids,
            images=image_tensor,
            do_sample=True,
            temperature=0.01,
            max_new_tokens=512,
            use_cache=True,
            stopping_criteria=[stopping_criteria],
        )
        print("decoded output_ids", tokenizer.decode(output_ids[0, input_ids.shape[1]:])) #input_ids.shape[1] :
    return tokenizer.decode(
        output_ids[0, input_ids.shape[1] :], skip_special_tokens=True
    ).strip()
    
    
# Test with an image from MIMIC-CXR dataset
#from src.datasets import load_mimic
#metadata_test = load_mimic(train=False, validation=False,check_images=False)
#img = Image.open(metadata_test.filepath[0])
img = Image.open('/gpfs/workdir/restrepoda/datasets/MIMIC/mimic/preproc_224x224/s53362948_bc351054-9432e570-d8e3300b-11abaaec-7641741f.jpg')
img

prompt = DEFAULT_IMAGE_TOKEN + "\n" + "What type of imaging does this represent?"
out = ask_image(img, prompt)

print(out)