# tests/test_data_prep.py
# Lightweight tests for build_student_topic_snapshots to catch regressions.
# Run with: python -m pytest -q

import pandas as pd
from src.data_prep import generate_synthetic_logs, build_student_topic_snapshots, load_csv_logs

def test_synthetic_snapshot_shape():
    df = generate_synthetic_logs(n_students=10, n_topics=4, avg_attempts_per_student=8, start_date="2025-01-01")
    snaps = build_student_topic_snapshots(df, min_attempts=1)
    # We expect at least one snapshot per student-topic combination attempted
    assert not snaps.empty
    expected_cols = {'student_id', 'topic', 'attempts', 'rw_acc', 'mastery'}
    assert expected_cols.issubset(set(snaps.columns))

def test_load_csv_roundtrip(tmp_path):
    # Create a small CSV and test load_csv_logs works and returns expected columns
    df = generate_synthetic_logs(n_students=3, n_topics=2, avg_attempts_per_student=5)
    p = tmp_path / "logs.csv"
    df.to_csv(p, index=False)
    loaded = load_csv_logs(str(p))
    assert 'student_id' in loaded.columns
    assert pd.api.types.is_datetime64_any_dtype(loaded['timestamp'])