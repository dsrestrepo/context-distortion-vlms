from transformers import OFATokenizer, OFAModel

model_name = 'PanaceaAI/BiomedGPT-Base-Pretrained'
tokenizer = OFATokenizer.from_pretrained(model_name)
model = OFAModel.from_pretrained(model_name)

import re

import torch
from PIL import Image
from torchvision import transforms


mean, std = [0.5, 0.5, 0.5], [0.5, 0.5, 0.5]
resolution = 480

patch_resize_transform = transforms.Compose([
        lambda image: image.convert("RGB"),
        transforms.Resize((resolution, resolution), interpolation=Image.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)
    ])

img = Image.open('example.jpg')

txt = "What modality is used to take this image?"
inputs = tokenizer([txt], return_tensors="pt").input_ids
patch_img = patch_resize_transform(img).unsqueeze(0)

gen = model.generate(inputs, patch_images=patch_img, num_beams=5, no_repeat_ngram_size=3, max_length=16)
results = tokenizer.batch_decode(gen, skip_special_tokens=True)

result = results[0]
result = re.sub(r'[^\w\s]', '', result).strip()

result