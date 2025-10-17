# src/embeddings.py
"""
Embedding wrapper. Supports HF sentence-transformers and OpenAI embeddings.
Usage:
from src.embeddings import EmbeddingClient
emb = EmbeddingClient(model_name="sentence-transformers/all-mpnet-base-v2")
vec = emb.encode(["hello world"])
"""
import os
import numpy as np


class EmbeddingClient:
    def __init__(self, model_name: str="sentence-transformers/all-mpnet-base-v2", backend: str="hf"):
        self.model_name = model_name
        self.backend = backend
        if backend == "hf":
            from sentence_transformers import SentenceTransformer
            self.model = SentenceTransformer(model_name)
        elif backend == "openai":
            import openai
# user must set OPENAI_API_KEY env var
            openai.api_key = os.getenv("OPENAI_API_KEY")
            self.model = None
        else:
            raise ValueError("Unsupported backend")


    def encode(self, texts):
        if self.backend == "hf":
            return self.model.encode(texts, show_progress_bar=False, convert_to_numpy=True)
        else:
            import openai
            resp = [openai.Embedding.create(model=self.model_name, input=t) for t in texts]
            embs = [r['data'][0]['embedding'] for r in resp]
        return np.array(embs)