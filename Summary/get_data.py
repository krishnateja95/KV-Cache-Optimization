from datasets import load_dataset, load_from_disk
import json
from huggingface_hub import snapshot_download
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig ,LlamaTokenizer
import argparse
import os

def get_data():
    dataset = load_dataset(args.dataset,'1.0.0',split='test')
    os.makedirs(args.dataset_dir, exist_ok=True)
    dataset.to_json(args.dataset_dir + args.dataset + "_test.json")

    requests = []
    
    input_path = args.dataset_dir + args.dataset + "_test.json"
    
    with open(input_path, 'r') as f:
        for line in f:
            if line.strip() != '':
                requests.append(json.loads(line))
                break
    request=requests[0]

    num_sample=1000
    
    file_path = args.dataset_dir + args.dataset + "_test.json"
    
    file = open(file_path,'w')
    for i in range(num_sample):
        temp=request
        temp['article']=str('###\nArticle: '+dataset[i]['article'])
        temp['summary_gt']=dataset[i]['highlights']
        temp["temperature"]= 0.3
        temp["stop"] = ["###"]

        temp['top_p'] = 1
        temp['max_tokens'] = 64
        temp['n'] = 1

        file.write(json.dumps(temp)+'\n')

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="cnn_dailymail")
    parser.add_argument("--dataset_dir", type=str, default="cnn_dailymail")
    args = parser.parse_args()
    get_data()