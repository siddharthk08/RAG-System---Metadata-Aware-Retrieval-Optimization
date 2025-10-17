# app/streamlit_app.py
import sys
import os

# Ensure project root is in path
sys.path.append(os.path.abspath("src"))

import streamlit as st
import pandas as pd
import json
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from src.embeddings import EmbeddingClient
from src.vectorstore import FaissStore
from src.feature_extractor import extract_comprehensive_features, keyword_overlap, jaccard_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from rank_bm25 import BM25Okapi
import joblib

st.set_page_config(layout='wide', page_title="RAG System - Metadata-Aware Retrieval")
st.title("🎯 RAG System - Metadata-Aware Retrieval Optimization")

# Sidebar configuration
with st.sidebar:
    st.header("⚙️ Configuration")
    
    # File paths
    index_prefix = st.text_input("FAISS index prefix", "artifacts/faiss")
    weights_path = st.text_input("Feature weights JSON", "artifacts/feature_weights.json")
    model_path = st.text_input("Trained model (joblib)", "artifacts/rf_model.joblib")
    corpus_path = st.text_input("Corpus CSV", "data/sample_corpus.csv")
    
    # Model settings
    emb_model = st.text_input("Embedding model", "sentence-transformers/all-mpnet-base-v2")
    k = st.number_input("Retriever k", min_value=1, max_value=50, value=10)
    top_k = st.number_input("Re-ranking top-k", min_value=1, max_value=20, value=5)

    # Load resources button
    if st.button("🔄 Load Resources", type="primary"):
        try:
            with st.spinner("Loading resources..."):
                # Load corpus
                corpus = pd.read_csv(corpus_path)
                st.session_state['corpus'] = corpus
                
                # Load weights/results
                try:
                    with open(weights_path) as f:
                        st.session_state['results'] = json.load(f)
                except Exception as e:
                    st.error(f"❌ Failed to load weights: {e}")
                    st.session_state['results'] = None
                
                # Load model if available
                try:
                    if os.path.exists(model_path):
                        st.session_state['model_data'] = joblib.load(model_path)
                        st.success("✅ Loaded corpus, weights, and model")
                    else:
                        st.session_state['model_data'] = None
                        st.success("✅ Loaded corpus and weights (no model)")
                except Exception as e:
                    st.error(f"❌ Failed to load model: {e}")
                    st.session_state['model_data'] = None
                    st.success("✅ Loaded corpus and weights (model failed)")
                
                st.session_state['loaded'] = True
                
        except Exception as e:
            st.error(f"❌ Failed to load resources: {e}")
            st.session_state['loaded'] = False

# Main content
if not st.session_state.get('loaded', False):
    st.warning("👈 Please load resources from the sidebar first.")
    st.stop()

# Display dataset information
st.header("📊 Dataset Overview")
corpus = st.session_state['corpus']
results = st.session_state['results']

col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("Documents", len(corpus))
with col2:
    st.metric("Features", len(results.get('feature_importance', {})))
with col3:
    if 'dataset_info' in results:
        st.metric("Queries", results['dataset_info'].get('n_queries', 'N/A'))
with col4:
    if 'test_metrics' in results and results['test_metrics']:
        st.metric("Test F1", f"{results['test_metrics']['f1']:.3f}")

# Model performance metrics
if 'test_metrics' in results and results['test_metrics']:
    st.subheader("🎯 Model Performance")
    metrics = results['test_metrics']
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Accuracy", f"{metrics['accuracy']:.3f}")
    with col2:
        st.metric("Precision", f"{metrics['precision']:.3f}")
    with col3:
        st.metric("Recall", f"{metrics['recall']:.3f}")
    with col4:
        st.metric("F1-Score", f"{metrics['f1']:.3f}")

# Cross-validation results
if 'cv_report' in results and results['cv_report']:
    st.subheader("📈 Cross-Validation Results")
    cv = results['cv_report']
    
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Best CV F1", f"{cv['best_score']:.3f}")
        if 'std_test_score' in cv:
            st.metric("CV Std", f"{cv['std_test_score']:.3f}")
    
    with col2:
        st.metric("Search Iterations", cv.get('n_iter', 'N/A'))
        if 'best_params' in cv:
            st.json(cv['best_params'])

# Feature importance visualization
st.subheader("🔍 Feature Importance")
feature_importance = results.get('feature_importance', {})

if feature_importance:
    # Create feature importance chart
    features = list(feature_importance.keys())
    importances = list(feature_importance.values())
    
    # Sort by importance
    sorted_data = sorted(zip(features, importances), key=lambda x: x[1], reverse=True)
    top_features = sorted_data[:10]  # Top 10 features
    
    if top_features:
        fig = px.bar(
            x=[imp for _, imp in top_features],
            y=[feat for feat, _ in top_features],
            orientation='h',
            title="Top 10 Most Important Features",
            labels={'x': 'Importance', 'y': 'Feature'}
        )
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)
    
    # Feature importance table
    importance_df = pd.DataFrame(sorted_data, columns=['Feature', 'Importance'])
    st.dataframe(importance_df, use_container_width=True)

# Query interface
st.header("🔍 Query Interface")

# Example queries
st.subheader("💡 Example Queries")
example_queries = [
    "How to optimize RAG retrieval?",
    "What is machine learning?",
    "Python programming best practices",
    "Advanced data science techniques",
    "Streamlit app development"
]

cols = st.columns(len(example_queries))
for i, example in enumerate(example_queries):
    with cols[i]:
        if st.button(f"📝 {example[:20]}...", key=f"example_{i}"):
            st.session_state['example_query'] = example

# Query input
query = st.text_input(
    "Enter your query:", 
    placeholder="e.g., How to optimize RAG retrieval?",
    value=st.session_state.get('example_query', '')
)

if st.button("🚀 Retrieve & Re-rank", type="primary") and query:
    if not st.session_state.get('loaded', False):
        st.warning("Please load resources first.")
    else:
        with st.spinner("Processing query..."):
            try:
                # Initialize embedding client
                emb = EmbeddingClient(model_name=emb_model, backend='hf')
                qvec = emb.encode([query])
                store = FaissStore.load(index_prefix, dim=qvec.shape[1])
                
                # Initial retrieval
                D, I = store.search(qvec, k=k)
                
                # Prepare TF-IDF and BM25 for feature extraction
                corpus_texts = corpus['text'].astype(str).tolist()
                tfidf_vectorizer = TfidfVectorizer(max_features=5000, stop_words='english')
                tfidf_matrix = tfidf_vectorizer.fit_transform(corpus_texts)
                tokenized_corpus = [doc.split() for doc in corpus_texts]
                bm25 = BM25Okapi(tokenized_corpus)
                
                # Extract features for each candidate
                candidates = []
                for rank_pos, doc_idx in enumerate(I[0]):
                    doc_id = store.id_map[doc_idx]
                    doc_meta_row = corpus[corpus['id'] == doc_id]
                    if len(doc_meta_row) == 0:
                        continue
                    
                    doc_meta = doc_meta_row.iloc[0].to_dict()
                    cosine_sim = float(D[0][rank_pos])
                    
                    # Extract comprehensive features
                    features = extract_comprehensive_features(
                        query, doc_meta, cosine_sim, rank_pos,
                        tfidf_vectorizer, tfidf_matrix, bm25, corpus
                    )
                    
                    candidates.append({
                        'doc_id': doc_id,
                        'title': doc_meta.get('title', 'No title'),
                        'text': doc_meta.get('text', '')[:300] + "...",
                        'cosine_sim': cosine_sim,
                        'original_rank': rank_pos,
                        **features
                    })
                
                # Re-rank using trained model or feature weights
                candidates_df = pd.DataFrame(candidates)
                
                if st.session_state.get('model_data'):
                    # Use trained model
                    model_data = st.session_state['model_data']
                    model = model_data['model']
                    scaler = model_data['scaler']
                    feature_names = model_data['feature_names']
                    
                    # Prepare features
                    feature_cols = [col for col in feature_names if col in candidates_df.columns]
                    X = candidates_df[feature_cols].fillna(0)
                    X_scaled = scaler.transform(X)
                    
                    # Get predictions
                    if hasattr(model, 'predict_proba'):
                        proba = model.predict_proba(X_scaled)
                        if proba.shape[1] > 1:
                            candidates_df['rerank_score'] = proba[:, 1]
                        else:
                            candidates_df['rerank_score'] = proba[:, 0]
                    else:
                        candidates_df['rerank_score'] = model.predict(X_scaled)
                    
                    ranking_method = "Trained Model"
                else:
                    # Use feature weights
                    weights = results.get('feature_importance', {})
                    total_weight = sum(weights.values()) if weights else 0
                    
                    if total_weight > 0:
                        def weighted_score(row):
                            score = 0.0
                            for feature, weight in weights.items():
                                if feature in row:
                                    score += float(row[feature]) * weight
                            return score / total_weight
                        
                        candidates_df['rerank_score'] = candidates_df.apply(weighted_score, axis=1)
                    else:
                        candidates_df['rerank_score'] = candidates_df['cosine_sim']
                    
                    ranking_method = "Feature Weights"
                
                # Sort by rerank score
                candidates_df = candidates_df.sort_values('rerank_score', ascending=False).head(top_k)
                
                # Display results with accuracy metrics
                st.subheader(f"📋 Re-ranked Results ({ranking_method})")
                
                # Show ranking improvement
                original_top = candidates_df.iloc[0]['original_rank'] + 1
                rerank_top = 1
                improvement = original_top - rerank_top
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Original Top Rank", f"#{original_top}")
                with col2:
                    st.metric("Re-ranked Top", f"#{rerank_top}")
                with col3:
                    st.metric("Rank Improvement", f"+{improvement}" if improvement > 0 else "No change")
                
                # Show top results
                for i, (_, row) in enumerate(candidates_df.iterrows(), 1):
                    with st.expander(f"#{i} - {row['title']} (Score: {row['rerank_score']:.3f})"):
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            st.write("**📊 Document Metrics:**")
                            st.write(f"**ID:** {row['doc_id']}")
                            st.write(f"**Original Rank:** #{row['original_rank'] + 1}")
                            st.write(f"**Cosine Similarity:** {row['cosine_sim']:.3f}")
                            st.write(f"**Re-rank Score:** {row['rerank_score']:.3f}")
                            
                            # Show if this is a significant improvement
                            if row['original_rank'] > 0:
                                rank_improvement = row['original_rank'] + 1 - i
                                if rank_improvement > 0:
                                    st.success(f"📈 Improved by {rank_improvement} positions!")
                        
                        with col2:
                            st.write("**🎯 Key Features:**")
                            feature_cols = ['keyword_overlap', 'jaccard_sim', 'tfidf_cosine', 'bm25_score', 'recency']
                            for feat in feature_cols:
                                if feat in row:
                                    st.write(f"**{feat}:** {row[feat]:.3f}")
                        
                        st.write("**📄 Content:**")
                        st.write(row['text'])
                
                # Show accuracy and performance summary
                st.subheader("📈 Performance Summary")
                
                # Calculate some performance metrics
                avg_rerank_score = candidates_df['rerank_score'].mean()
                avg_cosine_sim = candidates_df['cosine_sim'].mean()
                rank_improvements = []
                
                for i, (_, row) in enumerate(candidates_df.iterrows(), 1):
                    if row['original_rank'] > 0:
                        improvement = row['original_rank'] + 1 - i
                        rank_improvements.append(improvement)
                
                avg_improvement = sum(rank_improvements) / len(rank_improvements) if rank_improvements else 0
                
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Avg Re-rank Score", f"{avg_rerank_score:.3f}")
                with col2:
                    st.metric("Avg Cosine Similarity", f"{avg_cosine_sim:.3f}")
                with col3:
                    st.metric("Avg Rank Improvement", f"+{avg_improvement:.1f}" if avg_improvement > 0 else "No change")
                with col4:
                    st.metric("Model Confidence", f"{max(candidates_df['rerank_score']):.3f}")
                
                # Feature analysis
                st.subheader("📊 Feature Analysis")
                feature_cols = ['cosine_sim', 'keyword_overlap', 'jaccard_sim', 'tfidf_cosine', 'bm25_score', 'recency']
                available_features = [col for col in feature_cols if col in candidates_df.columns]
                
                if available_features:
                    feature_df = candidates_df[['doc_id'] + available_features].copy()
                    feature_df['rank'] = range(1, len(feature_df) + 1)
                    
                    # Feature heatmap
                    fig = px.imshow(
                        feature_df[available_features].T,
                        labels=dict(x="Document Rank", y="Feature", color="Value"),
                        title="Feature Values by Document Rank",
                        aspect="auto"
                    )
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Feature comparison table
                    st.dataframe(feature_df, use_container_width=True)
                
            except Exception as e:
                st.error(f"Error processing query: {e}")
                st.exception(e)

# Footer
st.markdown("---")
st.markdown("**RAG System - Metadata-Aware Retrieval Optimization** | Built with Streamlit")
