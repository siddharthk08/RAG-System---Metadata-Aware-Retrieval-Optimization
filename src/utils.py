# src/utils.py
"""
Small utility helpers used across the project.
"""
import json
import joblib
import os

def save_json(obj, path):
    os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
    with open(path, 'w') as f:
        json.dump(obj, f, indent=2)

def load_json(path):
    with open(path) as f:
        return json.load(f)

def save_joblib(obj, path):
    os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
    joblib.dump(obj, path)

def load_joblib(path):
    return joblib.load(path)


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)