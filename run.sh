module use /soft/modulefiles/
module load conda

conda activate vllm_kv_cache

python3 lm_eval_main.py
