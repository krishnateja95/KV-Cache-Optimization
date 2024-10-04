module use /soft/modulefiles/
module load conda

conda activate vllm_kv_cache

python3 main.py
