module use /soft/modulefiles/
module load conda
conda activate LongBench

export HF_HOME='/lus/grand/projects/datascience/krishnat/model_weights/LLaMA/llama_cache/'
export HF_DATASETS_CACHE='/lus/grand/projects/datascience/krishnat/model_weights/LLaMA/llama_cache/'

model=Llama-3.1-8B

CUDA_VISIBLE_DEVICES=0,1,2,3 python pred_custom.py --model $model
# python eval_custom.py --model $model
