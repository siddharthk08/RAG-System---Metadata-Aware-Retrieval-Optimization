# src/feature_extractor.py
import numpy as np
import joblib
from .embeddings import EmbeddingClient
from .vectorstore import FaissStore
from tqdm import tqdm
import argparse
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from rank_bm25 import BM25Okapi
import re
from datetime import datetime, timedelta



def keyword_overlap(a, b):
    """Compute keyword overlap ratio between two texts"""
    a_tokens = set(a.lower().split())
    b_tokens = set(b.lower().split())
    if len(a_tokens) == 0: return 0.0
    return len(a_tokens & b_tokens) / len(a_tokens)

def jaccard_similarity(a, b):
    """Compute Jaccard similarity between two texts"""
    a_tokens = set(a.lower().split())
    b_tokens = set(b.lower().split())
    intersection = len(a_tokens & b_tokens)
    union = len(a_tokens | b_tokens)
    return intersection / union if union > 0 else 0.0

def extract_numerical_features(text):
    """Extract numerical features from text"""
    # Count numbers, URLs, emails, etc.
    num_count = len(re.findall(r'\d+', text))
    url_count = len(re.findall(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', text))
    email_count = len(re.findall(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', text))
    return {
        'num_count': num_count,
        'url_count': url_count,
        'email_count': email_count
    }

def recency_score(days_old):
    """Convert days old to recency score (0-1)"""
    return 1.0 / (1.0 + days_old / 365.0)

def normalize_score(score, min_val=0.0, max_val=1.0):
    """Normalize score to [0,1] range"""
    if pd.isna(score) or score is None:
        return 0.5
    return max(min_val, min(max_val, float(score)))

def safe_int(value, default=0):
    """Safely convert to int with default"""
    try:
        return int(float(value)) if pd.notna(value) else default
    except (ValueError, TypeError):
        return default

def safe_float(value, default=0.0):
    """Safely convert to float with default"""
    try:
        return float(value) if pd.notna(value) else default
    except (ValueError, TypeError):
        return default




def detect_column_mapping(df, required_cols):
    """Detect column mapping for flexible CSV handling"""
    mapping = {}
    df_cols = [col.lower() for col in df.columns]
    
    for req_col in required_cols:
        # Direct match
        if req_col in df.columns:
            mapping[req_col] = req_col
        elif req_col.lower() in df_cols:
            mapping[req_col] = df.columns[df_cols.index(req_col.lower())]
        else:
            # Fuzzy matching
            if req_col == 'id':
                candidates = [col for col in df.columns if 'id' in col.lower()]
                if candidates:
                    mapping[req_col] = candidates[0]
            elif req_col == 'text':
                candidates = [col for col in df.columns if any(word in col.lower() for word in ['text', 'content', 'body', 'description'])]
                if candidates:
                    mapping[req_col] = candidates[0]
            elif req_col == 'title':
                candidates = [col for col in df.columns if 'title' in col.lower()]
                if candidates:
                    mapping[req_col] = candidates[0]
    
    return mapping

def extract_comprehensive_features(query_text, doc_meta, cosine_sim, rank_pos, tfidf_vectorizer, tfidf_matrix, bm25, corpus_df):
    """Extract comprehensive features for ranking"""
    doc_text = doc_meta.get('text', '')
    doc_title = doc_meta.get('title', '')
    
    # Basic features
    features = {
        'rank_pos': rank_pos,
        'cosine_sim': cosine_sim,
        'keyword_overlap': keyword_overlap(query_text, doc_text),
        'jaccard_sim': jaccard_similarity(query_text, doc_text),
        'token_length': len(str(doc_text).split()),
        'title_keyword_overlap': keyword_overlap(query_text, doc_title) if doc_title else 0.0,
    }
    
    # TF-IDF features
    try:
        query_tfidf = tfidf_vectorizer.transform([query_text])
        doc_idx = corpus_df[corpus_df['id'] == doc_meta['id']].index[0]
        tfidf_cos = float(cosine_similarity(query_tfidf, tfidf_matrix[doc_idx]).ravel()[0])
        features['tfidf_cosine'] = tfidf_cos
    except Exception:
        features['tfidf_cosine'] = 0.0
    
    # BM25 features
    try:
        query_tokens = query_text.lower().split()
        doc_idx = corpus_df[corpus_df['id'] == doc_meta['id']].index[0]
        bm25_score = float(bm25.get_score(query_tokens, doc_idx))
        features['bm25_score'] = bm25_score
    except Exception:
        features['bm25_score'] = 0.0
    
    # Numerical text features
    num_features = extract_numerical_features(doc_text)
    features.update(num_features)
    
    # Metadata features with flexible column mapping
    metadata_cols = ['days_old', 'source_score', 'citations', 'views', 'likes', 'rating', 'popularity']
    for col in metadata_cols:
        if col in doc_meta:
            if col == 'days_old':
                features['recency'] = recency_score(safe_int(doc_meta[col], 3650))
            elif col in ['source_score', 'rating', 'popularity']:
                features[col] = normalize_score(safe_float(doc_meta[col], 0.5))
            else:
                features[col] = safe_int(doc_meta[col], 0)
        else:
            # Default values
            if col == 'days_old':
                features['recency'] = 0.5
            elif col in ['source_score', 'rating', 'popularity']:
                features[col] = 0.5
            else:
                features[col] = 0
    
    return features

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--corpus', required=True, help='Path to corpus CSV')
    parser.add_argument('--queries', required=True, help='Path to queries CSV')
    parser.add_argument('--index', required=True, help='Path to FAISS index')
    parser.add_argument('--emb-model', default='sentence-transformers/all-mpnet-base-v2', help='Embedding model name')
    parser.add_argument('--k', default=20, type=int, help='Number of candidates to retrieve per query')
    parser.add_argument('--auto-label', action='store_true', help='Auto-generate labels using heuristics')
    parser.add_argument('--out', default='data/features.parquet', help='Output features file')
    args = parser.parse_args()

    print("Loading data...")
    corpus_df = pd.read_csv(args.corpus)
    queries_df = pd.read_csv(args.queries)
    
    print(f"Corpus: {len(corpus_df)} documents, Queries: {len(queries_df)} queries")
    
    # Detect column mappings
    corpus_mapping = detect_column_mapping(corpus_df, ['id', 'text', 'title'])
    query_mapping = detect_column_mapping(queries_df, ['query_id', 'query'])
    
    print(f"Corpus mapping: {corpus_mapping}")
    print(f"Query mapping: {query_mapping}")
    
    # Rename columns for consistency
    corpus_df = corpus_df.rename(columns={v: k for k, v in corpus_mapping.items()})
    queries_df = queries_df.rename(columns={v: k for k, v in query_mapping.items()})

    print("Initializing embedding client...")
    emb = EmbeddingClient(model_name=args.emb_model, backend='hf')
    dim = emb.encode(["hello"]).shape[1]
    store = FaissStore.load(args.index, dim=dim)

    print("Preparing TF-IDF and BM25...")
    # TF-IDF setup
    corpus_texts = corpus_df['text'].astype(str).tolist()
    tfidf_vectorizer = TfidfVectorizer(max_features=5000, stop_words='english')
    tfidf_matrix = tfidf_vectorizer.fit_transform(corpus_texts)
    
    # BM25 setup
    tokenized_corpus = [doc.split() for doc in corpus_texts]
    bm25 = BM25Okapi(tokenized_corpus)

    print("Extracting features...")
    rows = []
    for _, qrow in tqdm(queries_df.iterrows(), total=len(queries_df)):
        qid = qrow['query_id'] if 'query_id' in qrow else qrow.name
        qtext = qrow['query']
        qvec = emb.encode([qtext])
        D, I = store.search(np.array(qvec), k=args.k)

        # Collect candidates for balanced labeling
        candidates = []
        for rank_pos, doc_idx in enumerate(I[0]):
            doc_id = store.id_map[doc_idx]
            doc_meta_row = corpus_df[corpus_df['id'] == doc_id]
            if len(doc_meta_row) == 0:
                continue
                
            doc_meta = doc_meta_row.iloc[0].to_dict()
            cosine_sim = float(D[0][rank_pos])
            
            # Extract comprehensive features
            features = extract_comprehensive_features(
                qtext, doc_meta, cosine_sim, rank_pos, 
                tfidf_vectorizer, tfidf_matrix, bm25, corpus_df
            )
            
            features.update({
                'query_id': qid,
                'doc_id': doc_id,
                'label': None
            })
            
            candidates.append(features)
        
        # Auto-labeling with improved heuristics
        if args.auto_label and candidates:
            # Sort by combined relevance score
            for cand in candidates:
                relevance_score = (
                    cand['cosine_sim'] * 0.4 + 
                    cand['keyword_overlap'] * 0.3 + 
                    cand['tfidf_cosine'] * 0.2 + 
                    cand['bm25_score'] * 0.1
                )
                cand['relevance_score'] = relevance_score
            
            candidates.sort(key=lambda x: x['relevance_score'], reverse=True)
            
            # Assign labels: top 20% as positive, rest as negative
            num_pos = max(1, len(candidates) // 5)
            for i, cand in enumerate(candidates):
                if i < num_pos:
                    cand['label'] = 1
                else:
                    cand['label'] = 0
                # Remove temporary relevance_score
                del cand['relevance_score']
        
        rows.extend(candidates)

    print("Saving features...")
    features_df = pd.DataFrame(rows)
    features_df.to_parquet(args.out)
    print(f'Wrote {len(features_df)} feature rows to {args.out}')
    
    if args.auto_label:
        label_counts = features_df['label'].value_counts()
        print(f"Label distribution: {label_counts.to_dict()}")