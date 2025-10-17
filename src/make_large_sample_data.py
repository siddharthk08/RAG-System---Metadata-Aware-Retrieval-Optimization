import argparse
import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--out-dir', default='data')
    p.add_argument('--num_docs', type=int, default=1000)
    p.add_argument('--num_queries', type=int, default=150)
    p.add_argument('--seed', type=int, default=42)
    args = p.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    topics = [
        'retrieval', 'embeddings', 'faiss', 'random forest', 'ranking', 'metadata', 'rag',
        'similarity', 'bert', 'mpnet', 'bm25', 'vector db', 'prompting', 'hallucination',
        'reranker', 'evaluation', 'latency', 'throughput', 'streamlit', 'fastapi'
    ]
    categories = ['tutorial', 'paper', 'blog', 'doc', 'benchmark']
    tags_pool = ['production', 'python', 'pytorch', 'transformers', 'sklearn', 'faiss', 'api', 'ui', 'indexing']

    # corpus
    docs = []
    for i in range(args.num_docs):
        topic = rng.choice(topics)
        cat = rng.choice(categories)
        title = f"{topic.title()} best practices and {cat}"
        text = f"This {cat} explains {topic} within RAG pipelines, covering optimization, trade-offs, and metadata usage."
        days_old = int(rng.integers(0, 3650))
        url = f"https://example.com/{topic.replace(' ', '-')}/{i}"
        tags = ','.join(rng.choice(tags_pool, size=int(rng.integers(1,4)), replace=False))
        docs.append({
            'id': f'd{i}',
            'title': title,
            'url': url,
            'category': cat,
            'tags': tags,
            'text': text,
            'days_old': days_old,
            'source_score': float(rng.uniform(0.2, 0.95)),
            'citations': int(rng.integers(0, 300))
        })
    corpus = pd.DataFrame(docs)
    corpus_path = os.path.join(args.out_dir, 'sample_corpus.csv')
    corpus.to_csv(corpus_path, index=False)

    # queries with intent and difficulty
    queries = []
    intents = ['howto', 'compare', 'optimize', 'debug', 'design']
    difficulties = ['beginner', 'intermediate', 'advanced']
    for i in range(args.num_queries):
        topic = rng.choice(topics)
        intent = rng.choice(intents)
        diff = rng.choice(difficulties)
        q = f"{intent.title()} {topic} in RAG ({diff})"
        gt = f"Use {topic} to improve retrieval by aligning representations and leveraging metadata."
        queries.append({'query_id': f'q{i}', 'query': q, 'intent': intent, 'difficulty': diff, 'ground_truth': gt})
    queries_df = pd.DataFrame(queries)
    queries_path = os.path.join(args.out_dir, 'sample_queries.csv')
    queries_df.to_csv(queries_path, index=False)

    print('Wrote:', corpus_path, queries_path)


if __name__ == '__main__':
    main()


