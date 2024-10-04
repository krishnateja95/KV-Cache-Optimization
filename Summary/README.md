# Evaluating LLMs on summary tasks(XSUM, CNN/Daily Mail)

## Setup

```bash
conda create -n summary_tasks python=3.10
conda activate summary_tasks

pip install -r requirements.txt
```

## Running on Summary tasks

Step1：Generate Datasets for tasks (XSUM, CNN/Daily Mail)
```bash
Dataset_dir="./Summary_dir/"
dataset="cnn_dailymail"
task=xsum #{xsum,multi_news,cnndm}
shots=0
model=huggyllama/llama-7b
model_arch=llama

python get_data.py --dataset $dataset --dataset_dir=$Dataset_dir #{xsum,multi_news,cnndm}
```

Step2: Generate the outputs
```bash
GPU=0,1,2,3
CUDA_VISIBLE_DEVICES=${GPU} python -u run_helm.py \
  --output_path ${task}-${shots}.jsonl\
  --model_name ${model} \
  --dataset ${dataset} --dataset-dir ${Dataset_dir} \
```

Step3: Evaluate the outputs
```bash
python eval_helm.py \
  --task ${task} \
  --model ${model_arch} \
  --input_path ${task}-${shots}.jsonl\
  --output_path ${task}-${shots}-rouge.txt
```
