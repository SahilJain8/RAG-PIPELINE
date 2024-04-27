from rag_loading_data import Dataset_load
from rag_models import Rag_Model

dataset_load = Dataset_load(df_path="text_chunks_and_embeddings_df.csv")
dataset_load.save_embeddings()
rag_model = Rag_Model()

print("[INFO] Enter your query")
query = str(input())
query_embeddings = rag_model.embedding_model.encode(query)
scores, samples = dataset_load.retun_faiss_index(query_embeddings)
prompt = rag_model.create_prompt(samples['sentence_chunk'], query)

ans = rag_model.generate(prompt)

print(ans)

