# src/evaluator.py
"""
Evaluation utilities: compute ROUGE-L and BERTScore between generated answers (concatenated contexts or LLM outputs)
and ground-truth answers. Also a helper to compute simple lexical similarity for quick checks.
"""
import argparse
import pandas as pd
from rouge_score import rouge_scorer
from bert_score import score as bertscore
import numpy as np

scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)

def rouge_l(pred, ref):
    sc = scorer.score(ref, pred)
    return sc['rougeL'].fmeasure

def bert_f1(preds, refs, model_type='microsoft/deberta-base-mnli'):
    try:
        P, R, F1 = bertscore(preds, refs, lang='en', model_type=model_type, rescale_with_baseline=True)
    except KeyError:
        # Fallback to a known supported model
        fallback = 'microsoft/deberta-base-mnli'
        P, R, F1 = bertscore(preds, refs, lang='en', model_type=fallback, rescale_with_baseline=True)
    # F1 is a tensor-like; convert to python floats
    return [float(x) for x in F1]

def build_answer_from_corpus(corpus_df, doc_ids):
    texts = []
    for d in doc_ids:
        row = corpus_df[corpus_df['id']==d]
        if len(row)>0:
            texts.append(str(row.iloc[0]['text']))
    return ' '.join(texts)

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--re_ranked', required=True, help='re_ranked.parquet output from re_ranker')
    p.add_argument('--corpus', required=True, help='sample_corpus.csv')
    p.add_argument('--queries', required=True, help='sample_queries.csv')
    p.add_argument('--skip-bertscore', action='store_true', help='Skip BERTScore to avoid large model downloads')
    return p.parse_args()

def main():
    args = parse_args()
    re = pd.read_parquet(args.re_ranked)
    corpus = pd.read_csv(args.corpus)
    queries = pd.read_csv(args.queries)

    preds = []
    refs = []
    qids = []

    for _, row in re.iterrows():
        qid = row['query_id']
        top_docs = row['top_docs']
        pred = build_answer_from_corpus(corpus, top_docs)
        ref = queries[queries['query_id']==qid]['ground_truth'].iloc[0]
        preds.append(pred)
        refs.append(ref)
        qids.append(qid)

    rouge_scores = [rouge_l(p, r) for p,r in zip(preds, refs)]
    bert_scores = None
    if not args.skip_bertscore:
        try:
            bert_scores = bert_f1(preds, refs)
        except Exception as e:
            print("Warning: BERTScore failed, continuing without it:", str(e))

    results = {
        'query_id': qids,
        'rouge_l': rouge_scores,
    }
    if bert_scores is not None:
        results['bertscore_f1'] = bert_scores
    results_df = pd.DataFrame(results)
    print("=== Per-query evaluation ===")
    print(results_df)
    print("\nAverages:")
    print("ROUGE-L avg:", np.mean(rouge_scores))
    if bert_scores is not None:
        print("BERTScore-F1 avg:", np.mean(bert_scores))

if __name__ == '__main__':
    main()
