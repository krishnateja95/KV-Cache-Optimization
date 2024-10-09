import os
import json
import torch
from lm_eval.models.huggingface import HFLM
from lm_eval.evaluator import evaluate
from lm_eval.tasks import get_task_dict

from models.LLMs.LLaMA.modeling_llama import LlamaForCausalLM
from transformers import AutoTokenizer


def run_evaluation():
    
    cache_dir = "/lus/grand/projects/datascience/krishnat/model_weights/LLaMA/llama_cache/"
    model_name = "meta-llama/Llama-3.1-8B" 
    
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True, cache_dir = cache_dir)
    model = LlamaForCausalLM.from_pretrained(model_name,
                                                 cache_dir= cache_dir,
                                                 torch_dtype = torch.float16,
                                                 device_map = "auto")
    
    model.KV_cache_evict_params(method = "attn_weight", block_size = 16, evict_size = 8)

    result = evaluate(
            HFLM(
                pretrained=model,
                tokenizer=tokenizer, 
                batch_size=1, 
                max_length=None),
            get_task_dict(["mathqa"]),
            limit = None,
        )

    for task, res in result["results"].items():
        print(f"{task}: {res}")


if __name__ == '__main__':
    run_evaluation()

