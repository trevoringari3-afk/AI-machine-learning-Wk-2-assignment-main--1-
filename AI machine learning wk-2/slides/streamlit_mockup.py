import streamlit as st
import pandas as pd

st.set_page_config(page_title="Learning-Gap Detector — Slide Mockup", layout="wide")

# Header / Slide Title
st.markdown("<h1 style='margin-bottom:4px'>Learning-Gap Detector</h1>", unsafe_allow_html=True)
st.markdown("<h4 style='color:gray;margin-top:0'>Predict topic-level mastery — 1-slide demo</h4>", unsafe_allow_html=True)
st.markdown("---")

# Three-column overview: Problem / Solution / Impact
c1, c2, c3 = st.columns([1,1,1])
with c1:
    st.subheader("Problem")
    st.write("- Teachers lack timely topic-level signals")
    st.write("- Manual triage is slow and inconsistent")
with c2:
    st.subheader("Solution")
    st.write("- Snapshot model predicts mastery per (student, topic)")
    st.write("- Top-N weak topics surfaced with probabilities")
with c3:
    st.subheader("Impact")
    st.metric("Pilot goal", "Reduce remediation time by 30%")
    st.write("Measure: pre/post topic mastery %")

st.markdown("### Example — Student Triage (mock data)")
# Mock top-3 weak topics for a student
data = {
    "Topic": ["Fractions - Addition", "Decimals - Place Value", "Word Problems"],
    "Predicted Weakness (%)": [0.82, 0.71, 0.63],
    "Recommended Resource": [
        "Video: Adding Fractions (7m)",
        "Interactive: Decimal Place Value",
        "Worksheet: Multi-step Word Problems"
    ]
}
df = pd.DataFrame(data)
df["Predicted Weakness (%)"] = (df["Predicted Weakness (%)"]*100).round(0).astype(int).astype(str) + "%"
st.table(df)

# Small evaluation snapshot
st.markdown("### Eval snapshot (test set)")
cols = st.columns(3)
cols[0].metric("AUC", "0.78")
cols[1].metric("F1", "0.62")
cols[2].metric("Precision@3", "0.55")

st.markdown("---")
# Footer: next steps / ask
f1, f2 = st.columns([3,1])
with f1:
    st.write("Next steps: pilot with anonymized logs, 3-month evaluation, integrate with LMS.")
with f2:
    st.write("Ask:")
    st.write("- Pilot partners")
    st.write("- Labeled data")