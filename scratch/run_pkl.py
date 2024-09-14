
import pickle

def read_large_pkl_file(file_path, chunk_size=1000):
    with open(file_path, 'rb') as file:
        chunk = []
        for _ in range(chunk_size):
            chunk.append(pickle.load(file))
        process_chunk(chunk)


def process_chunk(chunk):
    print(f"Processed a chunk with {len(chunk)} records")

file_path = "/grand/datascience/abgulhan/ANL-KV-cache/SHARE/llama3-8b_attention_data_xsum_small.pkl"
read_large_pkl_file(file_path)

