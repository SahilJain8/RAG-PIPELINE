import torch
import pandas as pd
import numpy as np
from datasets import Dataset


class Dataset_load:
    def __init__(self, df_path: str, device: str = 'cpu'):
        self.embeddings_dataset = None

        self.device = device
        self.df_path = df_path
        self.text_and_embeddings_df = pd.read_csv(self.df_path)
        print(f"[INFO] Loaded the Dataset from {self.df_path}")

        self.text_and_embeddings_df['embedding'] = self.text_and_embeddings_df['embedding'].apply(
            lambda x: np.fromstring(x.strip("[]"), sep=" "))
        self.embeddings = torch.tensor(np.array(self.text_and_embeddings_df['embedding'].tolist()),
                                       dtype=torch.float32).to(self.device)
        print(f"[INFO] embedding shape {self.embeddings.shape}")

    def save_embeddings(self):
        self.embeddings_dataset = Dataset.from_pandas(self.text_and_embeddings_df)
        self.embeddings_dataset.add_faiss_index('embedding')

    def retun_faiss_index(self, query_embeddings):
        scores, samples = self.embeddings_dataset.get_nearest_examples(
            "embedding", query_embeddings, k=5
        )

        return scores, samples
