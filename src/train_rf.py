# src/train_rf.py
"""
Train Random Forest on features.parquet. The script expects a 'label' column.
It performs query-level train/test splitting to avoid leakage.
Outputs: joblib RF model and JSON feature weights file.
Handles small datasets gracefully with robust cross-validation.
"""
import argparse
import pandas as pd
import numpy as np
from sklearn.model_selection import GroupShuffleSplit, GroupKFold, RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, f1_score, precision_score, recall_score, accuracy_score
from sklearn.preprocessing import StandardScaler
import joblib
import json
import os
import warnings
warnings.filterwarnings('ignore')

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--features', required=True, help='Path to features.parquet (must include label column)')
    parser.add_argument('--out-model', default='artifacts/rf_model.joblib', help='Path to save RF model (joblib)')
    parser.add_argument('--out-weights', default='artifacts/feature_weights.json', help='Path to save feature weights (json)')
    parser.add_argument('--n-estimators', default=300, type=int, help='Number of trees')
    parser.add_argument('--max-depth', default=10, type=int, help='Maximum depth of trees')
    parser.add_argument('--test-size', default=0.2, type=float, help='Test split fraction (by query groups)')
    parser.add_argument('--random-state', default=42, type=int)
    parser.add_argument('--cv-splits', default=5, type=int, help='GroupKFold splits for CV')
    parser.add_argument('--search-iters', default=50, type=int, help='Hyperparameter search iterations')
    parser.add_argument('--min-samples-split', default=5, type=int, help='Minimum samples to split')
    parser.add_argument('--min-samples-leaf', default=2, type=int, help='Minimum samples per leaf')
    return parser.parse_args()

def compute_comprehensive_metrics(y_true, y_pred):
    """Compute comprehensive evaluation metrics"""
    return {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, average='binary', zero_division=0),
        'recall': recall_score(y_true, y_pred, average='binary', zero_division=0),
        'f1': f1_score(y_true, y_pred, average='binary', zero_division=0),
        'f1_macro': f1_score(y_true, y_pred, average='macro', zero_division=0)
    }

def main():
    args = parse_args()
    os.makedirs("artifacts", exist_ok=True)

    print("Loading features...")
    pd.set_option('future.no_silent_downcasting', True)
    df = pd.read_parquet(args.features)
    df['label'] = df['label'].fillna(0).astype(int)

    if df.empty:
        raise ValueError("Features file is empty. Cannot train model.")

    print(f"Dataset shape: {df.shape}")
    print(f"Label distribution: {df['label'].value_counts().to_dict()}")
    print(f"Number of unique queries: {df['query_id'].nunique()}")

    X = df.drop(columns=["label", "query_id", "doc_id"])
    y = df["label"]
    groups = df["query_id"]

    unique_groups = groups.nunique()
    print(f"Features: {list(X.columns)}")

    # Feature scaling for better performance
    scaler = StandardScaler()
    X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns, index=X.index)

    # Handle tiny datasets
    if len(df) <= 1 or unique_groups == 1:
        print("Warning: Not enough queries to evaluate. Training on all data.")
        X_train, y_train = X_scaled, y
        X_test, y_test = None, None
    else:
        # Split by query groups to avoid leakage
        gss = GroupShuffleSplit(n_splits=1, test_size=args.test_size, random_state=args.random_state)
        train_idx, test_idx = next(gss.split(X_scaled, y, groups))
        X_train, X_test = X_scaled.iloc[train_idx], X_scaled.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
        print(f"Train set: {len(X_train)} samples, Test set: {len(X_test)} samples")

    print("Training model...")
    # Train with robust group-aware CV hyperparameter search
    base_model = RandomForestClassifier(
        n_estimators=args.n_estimators,
        max_depth=min(args.max_depth, len(X.columns)),
        min_samples_split=args.min_samples_split,
        min_samples_leaf=args.min_samples_leaf,
        random_state=args.random_state,
        class_weight="balanced",
        n_jobs=-1,
        bootstrap=True,
        oob_score=True
    )

    best_model = base_model
    cv_report = None
    test_metrics = None

    # Only do hyperparameter search if we have enough data
    if X_test is not None and unique_groups >= max(3, args.cv_splits):
        print("Performing hyperparameter optimization...")
        
        # More conservative parameter grid to prevent overfitting
        param_distributions = {
            "n_estimators": [200, 300, 400, 500],
            "max_depth": [8, 10, 12, 15],
            "min_samples_split": [5, 10, 15, 20],
            "min_samples_leaf": [2, 3, 4, 5],
            "max_features": ["sqrt", "log2", 0.8],
            "max_samples": [0.8, 0.9, None]
        }

        # Use GroupKFold for proper group-aware cross-validation
        gkf = GroupKFold(n_splits=min(args.cv_splits, unique_groups))
        
        # Custom scorer that handles edge cases
        def robust_f1_scorer(estimator, X, y):
            try:
                y_pred = estimator.predict(X)
                return f1_score(y, y_pred, average='binary', zero_division=0)
            except:
                return 0.0

        search = RandomizedSearchCV(
            estimator=base_model,
            param_distributions=param_distributions,
            n_iter=args.search_iters,
            scoring=robust_f1_scorer,
            n_jobs=-1,
            cv=gkf,
            verbose=1,
            random_state=args.random_state,
            refit=True,
            return_train_score=False
        )
        
        search.fit(X_train, y_train, groups=groups.iloc[train_idx])
        best_model = search.best_estimator_
        cv_report = {
            "best_params": search.best_params_,
            "best_score": float(search.best_score_),
            "mean_test_score": float(np.mean(search.cv_results_['mean_test_score'])),
            "std_test_score": float(np.std(search.cv_results_['mean_test_score']))
        }
        print(f"Best CV F1: {cv_report['best_score']:.4f} (+/- {cv_report['std_test_score']:.4f})")
        print(f"Best params: {cv_report['best_params']}")
    else:
        print("Skipping hyperparameter search due to insufficient data")
        # Fit the base model
        best_model.fit(X_train, y_train)

    # Evaluate on test set
    if X_test is not None:
        print("Evaluating on test set...")
        y_pred = best_model.predict(X_test)
        test_metrics = compute_comprehensive_metrics(y_test, y_pred)
        
        print("\nTest Set Performance:")
        print(f"Accuracy: {test_metrics['accuracy']:.4f}")
        print(f"Precision: {test_metrics['precision']:.4f}")
        print(f"Recall: {test_metrics['recall']:.4f}")
        print(f"F1-Score: {test_metrics['f1']:.4f}")
        print(f"F1-Macro: {test_metrics['f1_macro']:.4f}")
        
        print("\nDetailed Classification Report:")
        print(classification_report(y_test, y_pred))
    else:
        print("Warning: Skipping evaluation due to insufficient data.")

    # Save model and scaler
    model_data = {
        'model': best_model,
        'scaler': scaler,
        'feature_names': list(X.columns)
    }
    joblib.dump(model_data, args.out_model)
    print(f"Model and scaler saved to {args.out_model}")

    # Compute and save feature importance
    feature_importance = dict(zip(X.columns, best_model.feature_importances_))
    
    # Sort features by importance
    sorted_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)
    print("\nTop 10 Most Important Features:")
    for feat, imp in sorted_features[:10]:
        print(f"  {feat}: {imp:.4f}")
    
    # Warn if all zeros
    if all(v == 0.0 for v in feature_importance.values()):
        print("Warning: Feature importances are all zero. Dataset may be too small or labels/features lack variation.")

    # Save only relevant results (clean output)
    # Filter feature importance to only show meaningful features (> 0.001)
    relevant_features = {k: round(v, 4) for k, v in feature_importance.items() if v > 0.001}
    
    # Clean CV report
    clean_cv_report = None
    if cv_report:
        clean_cv_report = {
            "best_score": round(cv_report['best_score'], 4),
            "best_params": {
                "n_estimators": cv_report['best_params'].get('n_estimators', 'N/A'),
                "max_depth": cv_report['best_params'].get('max_depth', 'N/A'),
                "min_samples_split": cv_report['best_params'].get('min_samples_split', 'N/A'),
                "min_samples_leaf": cv_report['best_params'].get('min_samples_leaf', 'N/A')
            }
        }
    
    # Clean test metrics
    clean_test_metrics = None
    if test_metrics:
        clean_test_metrics = {
            "accuracy": round(test_metrics['accuracy'], 4),
            "precision": round(test_metrics['precision'], 4),
            "recall": round(test_metrics['recall'], 4),
            "f1": round(test_metrics['f1'], 4)
        }
    
    # Save only relevant results
    results = {
        "feature_importance": relevant_features,
        "cv_report": clean_cv_report,
        "test_metrics": clean_test_metrics,
        "dataset_info": {
            "n_samples": len(df),
            "n_features": len(X.columns),
            "n_queries": unique_groups
        }
    }
    
    with open(args.out_weights, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to {args.out_weights}")

if __name__ == "__main__":
    main()
