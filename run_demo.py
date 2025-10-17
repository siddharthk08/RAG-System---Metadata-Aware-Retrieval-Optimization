#!/usr/bin/env python3
"""
Demo script to run the complete RAG pipeline
Usage: python run_demo.py
"""

import os
import sys
import subprocess
import time

def run_command(cmd, description):
    """Run a command and print status"""
    print(f"\n🔄 {description}...")
    print(f"Command: {cmd}")
    
    try:
        result = subprocess.run(cmd, shell=True, check=True, capture_output=True, text=True)
        print(f"✅ {description} completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} failed")
        print(f"Error: {e.stderr}")
        return False

def main():
    """Run the complete pipeline"""
    print("🚀 Starting RAG System Demo Pipeline")
    print("=" * 50)
    
    # Set environment
    os.environ['PYTHONPATH'] = os.path.abspath("src")
    
    # Pipeline steps
    steps = [
        {
            "cmd": "python -m src.make_large_sample_data --num_docs 1000 --num_queries 150",
            "desc": "Generating sample dataset"
        },
        {
            "cmd": "python -m src.vectorstore --corpus data/sample_corpus.csv --out artifacts/faiss",
            "desc": "Building FAISS vector index"
        },
        {
            "cmd": "python -m src.feature_extractor --corpus data/sample_corpus.csv --queries data/sample_queries.csv --index artifacts/faiss --k 20 --auto-label",
            "desc": "Extracting comprehensive features"
        },
        {
            "cmd": "python -m src.train_rf --features data/features.parquet --cv-splits 5 --search-iters 30",
            "desc": "Training Random Forest model"
        },
        {
            "cmd": "python -m src.re_ranker --features data/features.parquet --weights artifacts/feature_weights.json --model artifacts/rf_model.joblib --topk 5",
            "desc": "Re-ranking results"
        },
        {
            "cmd": "python -m src.evaluator --re_ranked data/re_ranked.parquet --queries data/sample_queries.csv --corpus data/sample_corpus.csv --skip-bertscore",
            "desc": "Evaluating performance"
        }
    ]
    
    # Run each step
    for i, step in enumerate(steps, 1):
        print(f"\n📋 Step {i}/{len(steps)}")
        success = run_command(step["cmd"], step["desc"])
        
        if not success:
            print(f"\n❌ Pipeline failed at step {i}")
            sys.exit(1)
        
        time.sleep(1)  # Brief pause between steps
    
    # Show results
    print("\n" + "=" * 50)
    print("🎉 Pipeline completed successfully!")
    print("\n📊 Results Summary:")
    
    # Try to read and display results
    try:
        import json
        with open("artifacts/feature_weights.json") as f:
            results = json.load(f)
        
        print(f"✅ Test Accuracy: {results['test_metrics']['accuracy']:.1%}")
        print(f"✅ F1-Score: {results['test_metrics']['f1']:.1%}")
        print(f"✅ Cross-validation Score: {results['cv_report']['best_score']:.1%}")
        print(f"✅ Dataset: {results['dataset_info']['n_samples']} samples, {results['dataset_info']['n_queries']} queries")
        
    except Exception as e:
        print(f"⚠️  Could not read results: {e}")
    
    print("\n🚀 To start the interactive dashboard:")
    print("   streamlit run streamlit_app.py --server.headless true --server.port 8501")
    print("   Then open: http://localhost:8501")

if __name__ == "__main__":
    main()
