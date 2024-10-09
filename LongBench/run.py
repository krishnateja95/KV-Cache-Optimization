import argparse
from tqdm import tqdm
import torch
from datasets import concatenate_datasets, load_dataset
import numpy as np
from torch.nn import CrossEntropyLoss
import gc
from transformers import AutoModelForCausalLM, AutoModel, AutoTokenizer
from ppl import BigDLPPL
import os
import json

from huggingface_hub import login
login("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")

cache_dir = '/lus/grand/projects/datascience/krishnat/model_weights/LLaMA/llama_cache/'

class BigDLPPL:
    def __init__(self, model_path):
        self.model = AutoModelForCausalLM.from_pretrained(model_path,
                                                          cache_dir = cache_dir,
                                                          device_map = "auto",
                                                          trust_remote_code=True,
                                                          torch_dtype = torch.float16)
        print("self.model", self.model)

    def perplexity_hf(self, encoded_texts):
        self.model.eval()
        loss_fct = CrossEntropyLoss(reduction="none")
        ppls = []
    
        pbar = tqdm(range(len(encoded_texts)))
        for bid in pbar:
            encoded_batch = encoded_texts[bid:bid+1]
            if type(encoded_batch) == dict:
                attn_mask = encoded_batch['attention_mask'] if 'attention_mask' in encoded_batch.keys() else None
                encoded_batch = encoded_batch['input_ids']
            elif type(encoded_batch) == list:
                encoded_batch = encoded_batch[0]
            
            encoded_batch = encoded_batch.to("cuda:0")
            attn_mask = torch.ones_like(encoded_batch)

            out_logits = self.model(encoded_batch).logits

            labels = encoded_batch

            shift_logits = out_logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            shift_attention_mask_batch = attn_mask[..., 1:].contiguous()

            loss_ = loss_fct(shift_logits.transpose(1, 2), shift_labels).float()
            perplexity_batch = torch.exp2(
                (loss_ * shift_attention_mask_batch).sum(1)
                / shift_attention_mask_batch.sum(1)
            )
            ppls += perplexity_batch.tolist()

            pbar.set_description(f"[{bid:<4}/{len(encoded_texts)}] avg_ppls: {np.mean(np.array(ppls)[~np.isnan(np.array(ppls))]):.4f}")
            
            del out_logits, encoded_batch, attn_mask, shift_logits, shift_labels, shift_attention_mask_batch, perplexity_batch

        ppl_mean = np.mean(np.array(ppls)[~np.isnan(np.array(ppls))])
    
        gc.collect()
        
        return ppl_mean

def get_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seq_len", type=int, default=2048)
    parser.add_argument("--model_path", required=True, type=str)
    parser.add_argument("--datasets", required=False, type=str, default=None, nargs='*')
    parser.add_argument("--dataset_path", required=False, type=str, default=None)
    parser.add_argument("--language", required=False, type=str, default="en", choices=['en', 'zh', 'all'])
    parser.add_argument("--precisions", required=False, type=str, default=None, nargs='+')
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--output_path", default=None)
    return parser.parse_args()
    

@torch.no_grad()
def main():
    args = get_arguments()
    for arg_name, arg_var in args.__dict__.items():
        print(f"{arg_name:<16} : {arg_var}")

    ppl_evaluator = BigDLPPL(model_path=args.model_path)
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    tokenizer.model_max_length = args.seq_len

    dataset_all = []
    
    datasets = ["narrativeqa", "qasper", "multifieldqa_en", "hotpotqa", "2wikimqa", "musique",
                "dureader", "gov_report", "qmsum", "multi_news", "vcsum", "trec", "triviaqa", "samsum", "lsht", 
                "passage_count", "passage_retrieval_en", "lcc", "repobench-p"]

    for dataset in datasets:
        data = load_dataset('THUDM/LongBench', dataset, split='test', cache_dir = "/lus/grand/projects/datascience/krishnat/datasets")
        dataset_all.append(data)

    # datasets_e = ["qasper", "multifieldqa_en", "hotpotqa", "2wikimqa", "gov_report", "multi_news", "trec", \
    #             "triviaqa", "samsum", "passage_count", "passage_retrieval_en", "lcc", "repobench-p"]

    # for dataset in datasets_e:
    #     data = load_dataset('THUDM/LongBench', f"{dataset}_e", split='test', cache_dir = "/lus/grand/projects/datascience/krishnat/datasets")
    #     dataset_all.append(data)

    data = concatenate_datasets(dataset_all)
    
    encoded_texts = []
    pbar = tqdm(data)
    for i, data_i in enumerate(pbar):
        encoded_text = tokenizer.encode(data_i['context'], return_tensors='pt', truncation=True)
        pbar.set_description(f"seq_len: {len(encoded_text[0])}, n_data: {len(encoded_texts)}")
        if len(encoded_text[0]) < args.seq_len:
            continue
        encoded_texts.append(encoded_text)
    
    summary = {}
    output_path = args.output_path if args.output_path else "results"
    model_name = os.path.basename(os.path.realpath(args.model_path))

    
    ppl = ppl_evaluator.perplexity_hf(encoded_texts)
    print("ppl = ", args.model_path, ppl)

    import csv

    list_1 = ["Model Name", "perplexity"]
    list_2 = [args.model_path, ppl] 

    assert len(list_1) == len(list_2)

    csv_file = "perplexity_results.csv"
    file_exists = os.path.exists(csv_file)

    with open(csv_file, 'a', newline = '') as csvfile:
        writer = csv.writer(csvfile)
        
        if not file_exists:
            writer.writerow(list_1)
        
        writer.writerow(list_2) 
        
    csvfile.close()

    # for precision in args.precisions:
    #     model_kwargs = {}
    #     if precision in ggml_tensor_qtype.keys():
    #         model_kwargs['load_in_low_bit'] = precision
    #     else:
    #         model_kwargs['torch_dtype'] = getattr(torch, precision)
    #     print(model_kwargs)
        
    #     log_dir = f"{output_path}/{model_name}/{args.device}/{precision}/{args.language}"
    #     os.makedirs(log_dir, exist_ok=True)
    #     results = {}
    #     ppl_evaluator = BigDLPPL(model_path=args.model_path, device=args.device, **model_kwargs)
    #     ppl = ppl_evaluator.perplexity_hf(encoded_texts)
    #     summary[precision] = ppl
    #     results['results'] = ppl
    #     results['config'] = {"model": model_name, "precision": precision, "device": args.device, "seq_len": args.seq_len, "language": args.language}
    #     dumped = json.dumps(results, indent=2)
    #     print(dumped)

    #     if output_path:
    #         with open(f"{log_dir}/result.json", "w") as f:
    #             f.write(dumped)
    
    # print(summary)

if __name__ == "__main__":
    main()