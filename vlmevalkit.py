
import requests
import torch
from PIL import Image
# from transformers import MllamaForConditionalGeneration, AutoProcessor

from transformers import AutoProcessor
from models.VLMs.MLLaMA.modeling_mllama import MllamaForConditionalGeneration

cache_dir = '/lus/grand/projects/datascience/krishnat/model_weights/LLaMA/llama_cache/'

# import pdb; pdb.set_trace()

model_id = "meta-llama/Llama-3.2-11B-Vision-Instruct"
model = MllamaForConditionalGeneration.from_pretrained(model_id, cache_dir=cache_dir, device_map="auto", torch_dtype=torch.bfloat16)
processor = AutoProcessor.from_pretrained(model_id)

messages = [
    [
        {
            "role": "user", 
            "content": [
                {"type": "image"},
                {"type": "text", "text": "What does the image show?"}
            ]
        }
    ],
]
text = processor.apply_chat_template(messages, add_generation_prompt=True)

url = "https://llava-vl.github.io/static/images/view.jpg"
image = Image.open(requests.get(url, stream=True).raw)

inputs = processor(text=text, images=image, return_tensors="pt").to(model.device)

output = model.generate(**inputs, max_new_tokens=25)
print(processor.decode(output[0]))

