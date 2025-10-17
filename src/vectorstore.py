# src/vectorstore.py
"""FAISS vectorstore helper: build index, add docs, search. Exports a simple CLI for building index."""
import argparse
import joblib
import pandas as pd
import numpy as np
from tqdm import tqdm
try:
    import faiss  # type: ignore
    _HAS_FAISS = True
except Exception:
    faiss = None
    _HAS_FAISS = False

from .embeddings import EmbeddingClient
import os


class FaissStore:
    def __init__(self, dim:int, index=None):
        self.dim = dim
        self.id_map = []
        if _HAS_FAISS:
            self.index = index or faiss.IndexFlatIP(dim)
            self._backend = 'faiss'
        else:
            # sklearn fallback using cosine similarity via normalized vectors
            from sklearn.neighbors import NearestNeighbors
            self.index = NearestNeighbors(metric='cosine')
            self._vectors = None  # will hold matrix for kneighbors
            self._backend = 'sklearn'


    def add(self, vectors:np.ndarray, ids=None):
        # vectors must be L2-normalized for inner-product == cosine if normalized
        if self._backend == 'faiss':
            faiss.normalize_L2(vectors)
            self.index.add(vectors)
        else:
            # store normalized vectors and fit NN model
            from sklearn.preprocessing import normalize
            self._vectors = normalize(vectors, norm='l2')
            self.index.fit(self._vectors)
        if ids is None:
            start = len(self.id_map)
            ids = list(range(start, start + vectors.shape[0]))
        self.id_map.extend(ids)


    def search(self, qvec:np.ndarray, k=10):
        if self._backend == 'faiss':
            faiss.normalize_L2(qvec)
            D, I = self.index.search(qvec, k)
            return D, I
        else:
            from sklearn.preprocessing import normalize
            qn = normalize(qvec, norm='l2')
            distances, indices = self.index.kneighbors(qn, n_neighbors=k)
            # Convert cosine distances to cosine similarities
            sims = 1.0 - distances
            return sims, indices


    def save(self, path_prefix: str):
        os.makedirs(os.path.dirname(path_prefix) or '.', exist_ok=True)
        if self._backend == 'faiss':
            faiss.write_index(self.index, path_prefix + '.index')
            joblib.dump(self.id_map, path_prefix + '.ids.pkl')
        else:
            # Save vectors and id map for fallback
            joblib.dump({
                'vectors': self._vectors,
                'id_map': self.id_map
            }, path_prefix + '.sk.pkl')


    @classmethod
    def load(cls, path_prefix: str, dim:int):
        if _HAS_FAISS and os.path.exists(path_prefix + '.index'):
            index = faiss.read_index(path_prefix + '.index')
            id_map = joblib.load(path_prefix + '.ids.pkl')
            store = cls(dim, index=index)
            store.id_map = id_map
            return store
        else:
            data = joblib.load(path_prefix + '.sk.pkl')
            store = cls(dim)
            store.id_map = data['id_map']
            # Refit the sklearn index
            from sklearn.neighbors import NearestNeighbors
            nn = NearestNeighbors(metric='cosine')
            nn.fit(data['vectors'])
            store.index = nn
            store._vectors = data['vectors']
            return store


# CLI for building index
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--corpus', type=str, required=True)
    parser.add_argument('--emb-model', type=str, default='sentence-transformers/all-mpnet-base-v2')
    parser.add_argument('--out-prefix', type=str, default='artifacts/faiss')
    args = parser.parse_args()


    df = pd.read_csv(args.corpus)
    texts = df['text'].tolist()
    emb = EmbeddingClient(model_name=args.emb_model, backend='hf')
    vectors = emb.encode(texts)
    dim = vectors.shape[1]
    store = FaissStore(dim)
    store.add(np.array(vectors), ids=df['id'].tolist() if 'id' in df.columns else None)
    store.save(args.out_prefix)
    print('Saved FAISS index to', args.out_prefix)