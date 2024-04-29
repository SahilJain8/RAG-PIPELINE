from rag_loading_data import Dataset_load
from rag_models import Rag_Model
import argparse
import time
import torch


def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"[INFO] using {device} as the device")
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, default='google/gemma-2b-it')
    parser.add_argument('--encoding_model_name', type=str, default='all-mpnet-base-v2')
    parser.add_argument('--use_quantization', type=bool, default=True)
    args = parser.parse_args()
    embedding_model_name = args.encoding_model_name
    model_name = args.model_name
    use_quantization = args.use_quantization
    print("[INFO] Encoding Model: {} \n[INFO] LLM model: {} \n[INFO] Quantisation is set to: {}".format(
        embedding_model_name,
        model_name,
        use_quantization))
    start_time = time.time()
    dataset_load = Dataset_load(df_path="text_chunks_and_embeddings_df.csv",
                                device=device)
    dataset_load.save_embeddings()
    rag_model = Rag_Model(model_id=model_name,
                          embedding_model_name=embedding_model_name,
                          use_quantization=use_quantization,
                          device=device)
    end_time = time.time()

    print(f"[INFO] Runtime: {round(end_time - start_time, 2)} seconds")

    while True:
        start_time = time.time()
        print("[INFO] Enter your query")
        query = str(input())
        query_embeddings = rag_model.embedding_model.encode(query)
        scores, samples = dataset_load.retun_faiss_index(query_embeddings)
        prompt = rag_model.create_prompt(samples['sentence_chunk'], query)

        ans = rag_model.generate(prompt)

        print(ans)
        end_time = time.time()
        print(f"[INFO] Runtime: {round(end_time - start_time, 2)} seconds")


if __name__ == '__main__':
    main()
