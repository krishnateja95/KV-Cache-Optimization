module use /soft/modulefiles/
module load conda
conda activate summary_tasks


export HF_HOME='/lus/grand/projects/datascience/krishnat/model_weights/LLaMA/llama_cache/'
# export HF_DATASETS_CACHE='/lus/grand/projects/datascience/krishnat/model_weights/LLaMA/llama_cache/'

Dataset_dir="./Summary_dir/"
dataset="cnn_dailymail"
task=xsum #{xsum,multi_news,cnndm}
shots=0
model=huggyllama/llama-7b
model_arch=llama

python get_data.py --dataset $dataset --dataset_dir=$Dataset_dir

GPU=0,1,2,3
CUDA_VISIBLE_DEVICES=${GPU} python -u run_helm.py \
  --output_path ${task}-${shots}.jsonl\
  --model_name ${model} \
  --dataset ${dataset} --dataset-dir ${Dataset_dir} \

python eval_helm.py \
  --task ${task} \
  --model ${model_arch} \
  --input_path ${task}-${shots}.jsonl\
  --output_path ${task}-${shots}-rouge.txt

