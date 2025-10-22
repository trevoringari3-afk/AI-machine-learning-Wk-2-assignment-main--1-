# src/predict.py
# Collaborative: prediction helper to surface weak topics for a student
# Clarification: this script expects models/ to contain model_meta.joblib and the candidate model file.

import joblib
import pandas as pd
import numpy as np
from src.data_prep import build_student_topic_snapshots, load_csv_logs

MODEL_DIR = "models"


def load_model():
    meta = joblib.load(f"{MODEL_DIR}/model_meta.joblib")
    chosen = meta['chosen']
    model = joblib.load(f"{MODEL_DIR}/{chosen}")
    return model


def top_k_weak_topics(logs_df, student_id, k=5):
    # Build snapshots for all students and filter to student_id
    snaps = build_student_topic_snapshots(logs_df, min_attempts=1)
    s_snaps = snaps[snaps['student_id'] == student_id]
    if s_snaps.empty:
        return []
    X = s_snaps[[
        "attempts", "total_correct", "avg_elapsed_time",
        "streak", "rw_acc", "rw_weight_sum", "recent_acc", "days_since_last"
    ]]
    model = load_model()
    proba = model.predict_proba(X)[:, 1] if hasattr(model, "predict_proba") else model.predict(X)
    s_snaps = s_snaps.copy()
    s_snaps['mastery_proba'] = proba
    # lowest probability == weakest topics
    result = s_snaps.sort_values('mastery_proba').head(k)[['topic', 'mastery_proba', 'attempts', 'recent_acc']]
    return result.to_dict(orient='records')


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--student_id", type=str, required=True)
    parser.add_argument("--k", type=int, default=5)
    args = parser.parse_args()
    logs = load_csv_logs(args.data_path)
    weak = top_k_weak_topics(logs, args.student_id, args.k)
    print("Top weak topics for", args.student_id)
    for r in weak:
        print(r)