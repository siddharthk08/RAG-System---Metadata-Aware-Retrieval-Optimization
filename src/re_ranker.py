# src/re_ranker.py
"""
Apply RF-derived weights for hybrid ranking. Loads features parquet and a feature weights JSON.
Computes weighted_score for each (query,candidate) row and writes re-ranked top-K per query.
Supports both direct feature weights and trained model prediction.
"""
import argparse
import pandas as pd
import json
import os
import joblib
import numpy as np

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--features', required=True, help='features.parquet produced by feature_extractor')
    p.add_argument('--weights', required=True, help='feature weights json (from train_rf)')
    p.add_argument('--model', help='trained model file (joblib) - if provided, uses model prediction instead of weighted sum')
    p.add_argument('--topk', type=int, default=5, help='How many top docs to include per query')
    p.add_argument('--out', default='data/re_ranked.parquet', help='Output path for re-ranked results')
    return p.parse_args()

def main():
    args = parse_args()
    os.makedirs(os.path.dirname(args.out) or '.', exist_ok=True)

    print("Loading features and weights...")
    df = pd.read_parquet(args.features)
    with open(args.weights) as f:
        loaded = json.load(f)
    
    print(f"Loaded {len(df)} feature rows")

    # Method 1: Use trained model prediction (preferred)
    if args.model and os.path.exists(args.model):
        print("Using trained model for ranking...")
        try:
            model_data = joblib.load(args.model)
            model = model_data['model']
            scaler = model_data['scaler']
            feature_names = model_data['feature_names']
            
            # Prepare features for prediction
            feature_cols = [col for col in feature_names if col in df.columns]
            X = df[feature_cols].fillna(0)
            
            # Apply scaling
            X_scaled = scaler.transform(X)
            
            # Get prediction probabilities (use positive class probability)
            if hasattr(model, 'predict_proba'):
                proba = model.predict_proba(X_scaled)
                if proba.shape[1] > 1:  # Binary classification
                    df['weighted_score'] = proba[:, 1]  # Probability of positive class
                else:
                    df['weighted_score'] = proba[:, 0]
            else:
                df['weighted_score'] = model.predict(X_scaled)
                
            print(f"Used model prediction for {len(feature_cols)} features")
            
        except Exception as e:
            print(f"Error using model: {e}. Falling back to feature weights.")
            args.model = None
    
    # Method 2: Use feature importance weights (fallback)
    if not args.model:
        print("Using feature importance weights...")
        # Support both legacy flat mapping and new object with feature_importance
        if isinstance(loaded, dict) and 'feature_importance' in loaded:
            raw_weights = loaded['feature_importance']
        else:
            raw_weights = loaded

        # normalize weights (sum to 1) and detect degeneracy
        total = sum(raw_weights.values()) if len(raw_weights)>0 else 0.0
        if total <= 0 or all(v == 0.0 for v in raw_weights.values()):
            # fallback: use cosine_sim only
            weights = {'cosine_sim': 1.0}
            print('Warning: Degenerate weights detected. Falling back to cosine_sim only.')
        else:
            weights = {k: (v/total if total>0 else 0.0) for k,v in raw_weights.items()}
            print(f"Using weights for {len(weights)} features")

        # compute weighted score; if feature missing, treat as 0
        def weighted_score(row):
            s = 0.0
            for k,v in weights.items():
                s += float(row.get(k, 0.0)) * v
            return s

        df['weighted_score'] = df.apply(weighted_score, axis=1)

    print("Computing re-ranked results...")
    out_rows = []
    for qid, group in df.groupby('query_id'):
        # Sort by weighted score (descending), then by cosine similarity as tiebreaker
        topk = group.sort_values(by=['weighted_score','cosine_sim'], ascending=False).head(args.topk)
        
        # Include additional metadata
        out_rows.append({
            'query_id': qid,
            'top_docs': topk['doc_id'].tolist(),
            'top_scores': topk['weighted_score'].tolist(),
            'cosine_scores': topk['cosine_sim'].tolist(),
            'original_ranks': topk['rank_pos'].tolist()
        })

    out_df = pd.DataFrame(out_rows)
    out_df.to_parquet(args.out, index=False)
    print(f"Wrote re-ranked results for {len(out_df)} queries to {args.out}")
    
    # Print some statistics
    if len(out_df) > 0:
        avg_score = np.mean([score for scores in out_df['top_scores'] for score in scores])
        print(f"Average weighted score: {avg_score:.4f}")

if __name__ == '__main__':
    main()
