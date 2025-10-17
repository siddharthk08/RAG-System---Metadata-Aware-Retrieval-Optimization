#!/usr/bin/env python3
"""
Validation script to check Streamlit app components
"""

import os
import json
import pandas as pd

def validate_files():
    """Validate all required files exist"""
    print("Validating required files...")
    
    files = [
        "streamlit_app.py",
        "artifacts/feature_weights.json",
        "artifacts/rf_model.joblib",
        "data/sample_corpus.csv",
        "data/sample_queries.csv"
    ]
    
    all_exist = True
    for file in files:
        if os.path.exists(file):
            print(f"OK: {file}")
        else:
            print(f"ERROR: {file} - MISSING")
            all_exist = False
    
    return all_exist

def validate_feature_weights():
    """Validate feature weights structure"""
    print("\nValidating feature weights...")
    
    try:
        with open("artifacts/feature_weights.json") as f:
            data = json.load(f)
        
        required_keys = ["feature_importance", "cv_report", "test_metrics", "dataset_info"]
        for key in required_keys:
            if key in data:
                print(f"OK: {key}")
            else:
                print(f"ERROR: {key} - MISSING")
                return False
        
        # Check for removed f1_macro
        if "f1_macro" not in data.get("test_metrics", {}):
            print("OK: f1_macro correctly removed")
        else:
            print("ERROR: f1_macro still present")
            return False
        
        print(f"OK: Test Accuracy: {data['test_metrics']['accuracy']:.3f}")
        print(f"OK: CV Score: {data['cv_report']['best_score']:.3f}")
        return True
        
    except Exception as e:
        print(f"ERROR: {e}")
        return False

def validate_data():
    """Validate data structure"""
    print("\nValidating data files...")
    
    try:
        corpus = pd.read_csv("data/sample_corpus.csv")
        queries = pd.read_csv("data/sample_queries.csv")
        
        print(f"OK: Corpus: {len(corpus)} documents")
        print(f"OK: Queries: {len(queries)} queries")
        
        # Check required columns
        corpus_cols = ["id", "text"]
        query_cols = ["query_id", "query"]
        
        for col in corpus_cols:
            if col in corpus.columns:
                print(f"OK: Corpus has {col}")
            else:
                print(f"ERROR: Corpus missing {col}")
                return False
        
        for col in query_cols:
            if col in queries.columns:
                print(f"OK: Queries has {col}")
            else:
                print(f"ERROR: Queries missing {col}")
                return False
        
        return True
        
    except Exception as e:
        print(f"ERROR: {e}")
        return False

def main():
    """Run validation"""
    print("RAG System Validation")
    print("=" * 30)
    
    checks = [
        ("Files", validate_files),
        ("Feature Weights", validate_feature_weights),
        ("Data Structure", validate_data)
    ]
    
    passed = 0
    for name, check_func in checks:
        print(f"\n--- {name} ---")
        if check_func():
            passed += 1
            print(f"PASSED: {name} validation")
        else:
            print(f"FAILED: {name} validation")
    
    print("\n" + "=" * 30)
    print(f"Results: {passed}/{len(checks)} validations passed")
    
    if passed == len(checks):
        print("\nSUCCESS: All validations passed!")
        print("Streamlit app should work correctly.")
        print("\nTo start the app:")
        print("  streamlit run streamlit_app.py --server.headless true --server.port 8501")
        print("  Then open: http://localhost:8501")
    else:
        print("\nFAILED: Some validations failed.")
        print("Please fix the issues before running the app.")

if __name__ == "__main__":
    main()
