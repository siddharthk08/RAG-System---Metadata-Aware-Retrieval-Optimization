# src/make_sample_data.py
import argparse
import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--out-dir', default='data')
    p.add_argument('--num_docs', type=int, default=50)
    p.add_argument('--num_queries', type=int, default=15)
    args = p.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    rng = np.random.default_rng(42)

    topics = ['retrieval', 'embeddings', 'faiss', 'random forest', 'ranking', 'metadata', 'rag', 'similarity', 'bert', 'mpnet']

    # corpus
    docs = []
    now = datetime.utcnow()
    for i in range(args.num_docs):
        topic = rng.choice(topics)
        days_old = int(rng.integers(0, 3650))
        text = f"This document discusses {topic} in the context of RAG systems and optimization."
        docs.append({
            'id': f'd{i}',
            'text': text,
            'days_old': days_old,
            'source_score': float(rng.uniform(0.3, 0.9)),
            'citations': int(rng.integers(0, 50))
        })
    corpus = pd.DataFrame(docs)
    corpus_path = os.path.join(args.out_dir, 'sample_corpus.csv')
    corpus.to_csv(corpus_path, index=False)

    # queries with simple ground truth as topic match
    queries = []
    for i in range(args.num_queries):
        topic = rng.choice(topics)
        q = f"How to use {topic} for better retrieval?"
        # ground truth is: any doc mentioning the topic
        gt = f"{topic} improves retrieval by aligning representations and leveraging metadata."
        queries.append({'query_id': f'q{i}', 'query': q, 'ground_truth': gt})
    queries_df = pd.DataFrame(queries)
    queries_path = os.path.join(args.out_dir, 'sample_queries.csv')
    queries_df.to_csv(queries_path, index=False)

    print('Wrote:', corpus_path, queries_path)


if __name__ == '__main__':
    main()


