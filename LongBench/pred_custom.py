import os
from datasets import load_dataset
import torch
import json
from transformers import AutoTokenizer, LlamaTokenizer, LlamaForCausalLM, AutoModelForCausalLM
from tqdm import tqdm
import numpy as np
import random
import argparse

import sys
sys.path.append("..")


def parse_args(args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default=None)
    parser.add_argument('--e', action='store_true', help="Evaluate on LongBench-E")
    return parser.parse_args(args)

def build_chat(tokenizer, prompt, model_name):
    if "llama2" in model_name or "Llama-3" in model_name:
        prompt = f"[INST]{prompt}[/INST]"
    return prompt


def get_pred(data, max_length, max_gen, prompt_format, dataset, model_name, model2path, out_path):
    model, tokenizer = load_model_and_tokenizer(model2path[model_name], model_name)
    
    for json_obj in tqdm(data):
        
        prompt = prompt_format.format(**json_obj)
        tokenized_prompt = tokenizer(prompt, truncation=False, return_tensors="pt").input_ids[0]
        
        if len(tokenized_prompt) > max_length:
            half = int(max_length/2)
            prompt = tokenizer.decode(tokenized_prompt[:half], skip_special_tokens=True)+tokenizer.decode(tokenized_prompt[-half:], skip_special_tokens=True)

        if dataset not in ["trec", "triviaqa", "samsum", "lsht", "lcc", "repobench-p"]:
            prompt = build_chat(tokenizer, prompt, model_name)

        input = tokenizer(prompt, truncation=False, return_tensors="pt").to("cuda:0")
        
        context_length = input.input_ids.shape[-1]

        if dataset == "samsum":
            output = model.generate(
                **input,
                max_new_tokens=max_gen,
                num_beams=1,
                do_sample=False,
                temperature=1.0,
                min_length=context_length+1,
                eos_token_id=[tokenizer.eos_token_id, tokenizer.encode("\n", add_special_tokens=False)[-1]],
            )[0]
        else:
            output = model.generate(
                **input,
                max_new_tokens=max_gen,
                num_beams=1,
                do_sample=False,
                temperature=1.0,
            )[0]
        
        pred = tokenizer.decode(output[context_length:], skip_special_tokens=True)
        
        print(dataset ,"Output Length", len(output) - context_length)

        with open(out_path, "a", encoding="utf-8") as f:
            json.dump({"pred": pred, "answers": json_obj["answers"], "all_classes": json_obj["all_classes"], "length": json_obj["length"]}, f, ensure_ascii=False)
            f.write('\n')
        
        exit()


def seed_everything(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.cuda.manual_seed_all(seed)

def load_model_and_tokenizer(path, model_name):
    cache_dir = '/lus/grand/projects/datascience/krishnat/model_weights/LLaMA/llama_cache/'
    from transformers import AutoTokenizer, LlamaTokenizer

    if "Llama-3" in model_name:
        from models.LLMs.LLaMA.modeling_llama import LlamaForCausalLM
        tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=True)
        model = LlamaForCausalLM.from_pretrained(path, trust_remote_code=True, torch_dtype=torch.bfloat16, cache_dir=cache_dir, device_map = "auto")

    elif "llama2" in model_name:
        tokenizer = LlamaTokenizer.from_pretrained(path, cache_dir=cache_dir)
        model = LlamaForCausalLM.from_pretrained(path, cache_dir=cache_dir, torch_dtype=torch.bfloat16, device_map = "auto")

    model = model.eval()
    return model, tokenizer

if __name__ == '__main__':
    seed_everything(42)

    args = parse_args()

    model2path = json.load(open("config/model2path.json", "r"))
    model2maxlen = json.load(open("config/model2maxlen.json", "r"))

    model_name = args.model
    
    max_length = model2maxlen[model_name]
    
    datasets = [
        "qasper",
                #  "gov_report",
                #    "hotpotqa", 
                #    "2wikimqa",
                    #  "multi_news",
                #        "multifieldqa_en", 
                    #    "trec",
                #          "triviaqa",
                        #    "samsum",
                            #  "passage_count",
                            #    "passage_retrieval_en",
                #                  "lcc",
                #                  "repobench-p"
                                    ]
    
    dataset2prompt = json.load(open("config/dataset2prompt.json", "r"))
    dataset2maxlen = json.load(open("config/dataset2maxlen.json", "r"))
    
    if not os.path.exists("pred_e"):
        os.makedirs("pred_e")
    
    for dataset in datasets:
        data = load_dataset('THUDM/LongBench', f"{dataset}_e", split='test')
        if not os.path.exists(f"pred_e/{model_name}"):
            os.makedirs(f"pred_e/{model_name}")
        out_path = f"pred_e/{model_name}/{dataset}.jsonl"
    
        
        prompt_format = dataset2prompt[dataset]
        max_gen = dataset2maxlen[dataset]
        
        
        data_all = [data_sample for data_sample in data]
        # data_all = data_all[0:int(len(data_all)*0.3)]
        data_all = data_all[0:3]
        
        get_pred(data_all, max_length, max_gen, prompt_format, dataset, model_name, model2path, out_path)

