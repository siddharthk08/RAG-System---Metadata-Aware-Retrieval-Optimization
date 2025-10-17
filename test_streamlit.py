#!/usr/bin/env python3
"""
Test script to verify Streamlit app functionality
"""

import os
import sys
import json
import pandas as pd

def test_artifacts():
    """Test if all required artifacts exist"""
    print("Testing artifacts...")
    
    required_files = [
        "artifacts/feature_weights.json",
        "artifacts/rf_model.joblib", 
        "data/sample_corpus.csv",
        "data/sample_queries.csv"
    ]
    
    missing_files = []
    for file in required_files:
        if not os.path.exists(file):
            missing_files.append(file)
    
    if missing_files:
        print(f"Missing files: {missing_files}")
        return False
    
    print("All required files exist")
    return True

def test_feature_weights():
    """Test feature weights JSON structure"""
    print("Testing feature weights...")
    
    try:
        with open("artifacts/feature_weights.json") as f:
            data = json.load(f)
        
        required_keys = ["feature_importance", "cv_report", "test_metrics", "dataset_info"]
        for key in required_keys:
            if key not in data:
                print(f"Missing key in feature_weights.json: {key}")
                return False
        
        # Test that f1_macro is not present (we removed it)
        if "f1_macro" in data.get("test_metrics", {}):
            print("f1_macro still present in test_metrics")
            return False
        
        print("Feature weights structure is correct")
        print(f"   - Features: {len(data['feature_importance'])}")
        print(f"   - Test Accuracy: {data['test_metrics']['accuracy']:.3f}")
        print(f"   - CV Score: {data['cv_report']['best_score']:.3f}")
        return True
        
    except Exception as e:
        print(f"Error reading feature_weights.json: {e}")
        return False

def test_data():
    """Test data files"""
    print("Testing data files...")
    
    try:
        corpus = pd.read_csv("data/sample_corpus.csv")
        queries = pd.read_csv("data/sample_queries.csv")
        
        print(f"Corpus: {len(corpus)} documents")
        print(f"Queries: {len(queries)} queries")
        
        # Check required columns
        required_corpus_cols = ["id", "text"]
        required_query_cols = ["query_id", "query"]
        
        for col in required_corpus_cols:
            if col not in corpus.columns:
                print(f"Missing column in corpus: {col}")
                return False
        
        for col in required_query_cols:
            if col not in queries.columns:
                print(f"Missing column in queries: {col}")
                return False
        
        print("Data structure is correct")
        return True
        
    except Exception as e:
        print(f"Error reading data files: {e}")
        return False

def main():
    """Run all tests"""
    print("Testing RAG System Components")
    print("=" * 40)
    
    tests = [
        test_artifacts,
        test_feature_weights, 
        test_data
    ]
    
    passed = 0
    for test in tests:
        if test():
            passed += 1
        print()
    
    print("=" * 40)
    print(f"Test Results: {passed}/{len(tests)} tests passed")
    
    if passed == len(tests):
        print("All tests passed! Streamlit app should work correctly.")
        print("\nTo start the app:")
        print("   streamlit run streamlit_app.py --server.headless true --server.port 8501")
        print("   Then open: http://localhost:8501")
    else:
        print("Some tests failed. Please check the errors above.")
        sys.exit(1)

if __name__ == "__main__":
    main()
