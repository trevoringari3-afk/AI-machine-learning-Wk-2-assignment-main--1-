# src/data_prep.py
# Collaborative: initial implementation (author: michaellolivery94-web)
# Clarifications: This module generates synthetic logs and builds per-(student,topic) snapshots.
# If you adapt this for your real data, ensure timestamps are parsed and student ids anonymized.

import numpy as np
import pandas as pd
from datetime import timedelta

RNG = np.random.RandomState(42)


def generate_synthetic_logs(n_students=200, n_topics=12, avg_attempts_per_student=80, start_date="2024-01-01"):
    """
    Generate synthetic quiz logs.
    Returns DataFrame with columns:
    student_id, timestamp, question_id, topic, correct (0/1), elapsed_time_seconds

    Notes:
    - Designed for quick demo. Replace with actual logs (CSV) when available.
    - Questions for collaborators: choose a realistic avg_attempts_per_student and n_topics for your pilot.
    """
    students = [f"s{idx:04d}" for idx in range(1, n_students + 1)]
    topics = [f"topic_{i+1}" for i in range(n_topics)]

    rows = []
    start = pd.to_datetime(start_date)
    for s in students:
        # number of attempts per student
        n_attempts = max(5, int(RNG.poisson(avg_attempts_per_student)))
        times = start + pd.to_timedelta(RNG.exponential(scale=3, size=n_attempts).cumsum(), unit='D')
        for i in range(n_attempts):
            topic = RNG.choice(topics, p=_topic_popularity(n_topics))
            question_id = f"q_{RNG.randint(1, 1000)}"
            # simulate student skill per topic
            base_skill = RNG.beta(2, 2)  # per-student baseline; tweak per topic if needed
            topic_modifier = (topics.index(topic) % 3) * 0.05
            skill = np.clip(base_skill - topic_modifier + RNG.normal(0, 0.08), 0.01, 0.99)
            correct = RNG.binomial(1, skill)
            elapsed = max(5, int(RNG.normal(30, 12)))
            rows.append((s, times[i].isoformat(), question_id, topic, int(correct), elapsed))
    df = pd.DataFrame(rows, columns=["student_id", "timestamp", "question_id", "topic", "correct", "elapsed_time_seconds"])
    # sort for consistent snapshots
    df = df.sort_values(["student_id", "timestamp"]).reset_index(drop=True)
    return df


def _topic_popularity(n_topics):
    # skewed popularity: a few topics are more frequent (real platforms often show this)
    weights = np.linspace(1.0, 0.6, n_topics)
    weights = weights / weights.sum()
    return weights


def load_csv_logs(path):
    """
    Load CSV logs and validate required columns.
    Expected columns: student_id, timestamp, question_id, topic, correct
    """
    df = pd.read_csv(path)
    required = {"student_id", "timestamp", "question_id", "topic", "correct"}
    if not required.issubset(set(df.columns)):
        raise ValueError(f"CSV must include columns: {required}")
    # ensure types
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df['correct'] = df['correct'].astype(int)
    if 'elapsed_time_seconds' not in df.columns:
        df['elapsed_time_seconds'] = np.nan
    return df


def build_student_topic_snapshots(df, snapshot_cutoff="latest", min_attempts=3, decay_half_life_days=30):
    """
    Build one snapshot row per (student, topic) representing current mastery features.
    - snapshot_cutoff: 'latest' uses the last timestamp per student-topic
    - min_attempts: minimum attempts required to include a snapshot (tweak for your pilot)
    Returns DataFrame rows with features and target 'mastery' (binary)
    For synthetic/demo purposes mastery=1 if recency-weighted accuracy > 0.75 and attempts>=min_attempts.
    """
    df = df.copy()
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    # compute recency weights
    def recency_weighted(row):
        times = row['timestamp'].values.astype('datetime64[s]').astype(np.int64)
        now = times.max()
        # days since each attempt
        days = (now - times) / 86400.0
        # exponential decay
        weights = 0.5 ** (days / decay_half_life_days)
        return np.sum(weights * row['correct'].values) / (np.sum(weights) + 1e-9), np.sum(weights)

    # group and compute aggregated features
    groups = []
    for (s, t), g in df.groupby(['student_id', 'topic']):
        attempts = len(g)
        if attempts < 1:
            continue
        last_ts = g['timestamp'].max()
        total_correct = g['correct'].sum()
        avg_elapsed = g['elapsed_time_seconds'].dropna().mean() if 'elapsed_time_seconds' in g else np.nan
        # streak: consecutive corrects at end
        sorted_g = g.sort_values('timestamp')
        reversed_corr = sorted_g['correct'].values[::-1]
        streak = 0
        for v in reversed_corr:
            if v == 1:
                streak += 1
            else:
                break
        # recency-weighted accuracy
        rw_acc, rw_sumw = recency_weighted(g)
        # recent window accuracy (last 30 days)
        cutoff = last_ts - pd.Timedelta(days=30)
        recent = g[g['timestamp'] >= cutoff]
        recent_acc = recent['correct'].mean() if len(recent) > 0 else np.nan
        days_since_last = (pd.Timestamp.now() - last_ts).days
        groups.append({
            "student_id": s,
            "topic": t,
            "last_timestamp": last_ts,
            "attempts": attempts,
            "total_correct": int(total_correct),
            "avg_elapsed_time": float(avg_elapsed) if not np.isnan(avg_elapsed) else np.nan,
            "streak": int(streak),
            "rw_acc": float(rw_acc),
            "rw_weight_sum": float(rw_sumw),
            "recent_acc": float(recent_acc) if not np.isnan(recent_acc) else np.nan,
            "days_since_last": int(days_since_last)
        })
    out = pd.DataFrame(groups)
    # define synthetic target: mastery if recency-weighted acc > 0.75 and attempts>=min_attempts
    out['mastery'] = ((out['rw_acc'] >= 0.75) & (out['attempts'] >= min_attempts)).astype(int)
    # fillna for model
    out['recent_acc'] = out['recent_acc'].fillna(out['rw_acc'])
    out['avg_elapsed_time'] = out['avg_elapsed_time'].fillna(out['avg_elapsed_time'].median())
    return out