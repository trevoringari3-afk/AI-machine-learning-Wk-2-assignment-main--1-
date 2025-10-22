# app/streamlit_app.py
# Collaborative: simple Streamlit demo to visualize weak topics for a student
# Clarifications: For production use, swap the synthetic data with real anonymized logs and host models.

import streamlit as st
import pandas as pd
from src.data_prep import generate_synthetic_logs, load_csv_logs, build_student_topic_snapshots
from src.predict import top_k_weak_topics
from src.train import featurize

st.set_page_config(page_title="Learning-Gap Detector Demo", layout="wide")

st.title("Learning-Gap Detector — SDG4 Demo")
st.markdown("Predict topic mastery from quiz logs and show top weak topics for remediation.")

data_mode = st.radio("Data:", ["Synthetic demo", "Upload CSV"])
if data_mode == "Synthetic demo":
    df = generate_synthetic_logs(n_students=100, n_topics=10, avg_attempts_per_student=60)
    st.info("Using generated demo dataset")
else:
    uploaded = st.file_uploader("Upload quiz logs CSV", type=["csv"])
    if uploaded is not None:
        df = pd.read_csv(uploaded)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        st.success("Loaded CSV")
    else:
        st.stop()

st.write("Sample logs:")
st.dataframe(df.sample(min(10, len(df))).reset_index(drop=True))

students = sorted(df['student_id'].unique().tolist())
student = st.selectbox("Select student", students)
k = st.slider("Top-k weak topics", min_value=1, max_value=10, value=5)

if st.button("Predict weak topics"):
    weak = top_k_weak_topics(df, student, k)
    if not weak:
        st.warning("No snapshot found for this student.")
    else:
        st.subheader("Top weak topics (low mastery probability)")
        for w in weak:
            st.markdown(f"- **{w['topic']}** — mastery probability: {w['mastery_proba']:.2f} (attempts: {w['attempts']}, recent_acc: {w['recent_acc']:.2f})")
            # placeholder suggestion links
            st.markdown(f"  - Suggested resource: https://openstax.org/search?query={w['topic']}")