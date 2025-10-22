# src/train.py
# Collaborative: training pipeline (author: michaellolivery94-web)
# Clarifications & TODOs are embedded below for follow-up by collaborators.

import argparse
import os
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score, f1_score
from src.data_prep import generate_synthetic_logs, load_csv_logs, build_student_topic_snapshots

MODEL_DIR = "models"
os.makedirs(MODEL_DIR, exist_ok=True)


def featurize(snapshot_df):
    features = snapshot_df[[
        "attempts", "total_correct", "avg_elapsed_time",
        "streak", "rw_acc", "rw_weight_sum", "recent_acc", "days_since_last"
    ]].copy()
    # derived feature: correct rate
    features['correct_rate'] = features['total_correct'] / (features['attempts'] + 1e-9)
    features = features.fillna(features.median())
    return features


def evaluate_model(y_true, y_proba, y_pred):
    # simple evaluation summary
    scores = {
        "auc": float(roc_auc_score(y_true, y_proba)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0))
    }
    return scores


def train_and_save(X_train, y_train, X_test, y_test):
    # Logistic Regression baseline
    pipe_lr = Pipeline([("scaler", StandardScaler()), ("lr", LogisticRegression(max_iter=1000))])
    pipe_lr.fit(X_train, y_train)
    probs_lr = pipe_lr.predict_proba(X_test)[:, 1]
    preds_lr = pipe_lr.predict(X_test)
    scores_lr = evaluate_model(y_test, probs_lr, preds_lr)
    joblib.dump(pipe_lr, f"{MODEL_DIR}/lr_pipeline.joblib")

    # Random Forest
    rf = RandomForestClassifier(n_estimators=200, max_depth=8, random_state=42)
    rf.fit(X_train, y_train)
    probs_rf = rf.predict_proba(X_test)[:, 1]
    preds_rf = rf.predict(X_test)
    scores_rf = evaluate_model(y_test, probs_rf, preds_rf)
    joblib.dump(rf, f"{MODEL_DIR}/rf_model.joblib")

    print("LogisticRegression scores:", scores_lr)
    print("RandomForest scores:", scores_rf)
    # choose best by AUC
    chosen = "rf_model.joblib" if scores_rf['auc'] >= scores_lr['auc'] else "lr_pipeline.joblib"
    joblib.dump({"chosen": chosen}, f"{MODEL_DIR}/model_meta.joblib")
    print(f"Saved chosen model metadata => {chosen}")


def main(args):
    if args.use_synthetic:
        print("Generating synthetic logs...")
        logs = generate_synthetic_logs(n_students=args.n_students, n_topics=args.n_topics,
                                       avg_attempts_per_student=args.avg_attempts)
    else:
        if not args.data_path:
            raise ValueError("Provide --data_path or set --use_synthetic")
        logs = load_csv_logs(args.data_path)

    print("Building snapshots...")
    snapshots = build_student_topic_snapshots(logs, min_attempts=2)
    X = featurize(snapshots)
    y = snapshots['mastery'].values

    # stratify if possible to keep class balance
    strat = y if len(np.unique(y)) > 1 else None
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=strat)
    print(f"Train rows: {len(X_train)}, Test rows: {len(X_test)}")

    train_and_save(X_train, y_train, X_test, y_test)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--use_synthetic", action="store_true", help="Generate synthetic logs and train")
    parser.add_argument("--data_path", type=str, default="", help="Path to CSV logs")
    parser.add_argument("--n_students", type=int, default=200)
    parser.add_argument("--n_topics", type=int, default=12)
    parser.add_argument("--avg_attempts", type=int, default=80)
    args = parser.parse_args()
    main(args)