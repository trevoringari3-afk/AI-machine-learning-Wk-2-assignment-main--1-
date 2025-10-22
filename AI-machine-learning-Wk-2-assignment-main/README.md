```markdown
# SDG4 — Learning-Gap Detector (Predict Topic Mastery)

Project: Learning-Gap Detector — Predict topic-level mastery from quiz logs  
SDG: 4 — Quality Education

Overview
--------
This repo provides a compact, reproducible starter project to detect learning gaps (predict topic mastery) using quiz logs. The goal is to highlight how supervised learning from simple quiz logs can identify weak topics for students so teachers or systems can provide targeted remediation.

Key features
- Synthetic dataset generator (realistic quiz logs).
- ETL + feature engineering: builds (student, topic) snapshots with recency-weighted accuracy, streaks, time features.
- Baseline models: Logistic Regression + Random Forest (sklearn).
- Evaluation: AUC, Accuracy, Precision, Recall, F1, calibration summary.
- Streamlit demo to inspect student profiles and recommended weak topics + resource links.

Use-case
--------
Predict whether a student has "mastery" of a topic at a snapshot in time (binary classification). Useful for personalized recommendation of lessons/quizzes and for teacher dashboards.

Repo structure (suggested)
- README.md
- requirements.txt
- .gitignore
- data/ (raw/, processed/) — add datasets here
- src/
  - data_prep.py
  - train.py
  - predict.py
- app/
  - streamlit_app.py
- models/ (saved models)
- assets/ (screenshots)
- report/
  - article.md
- slides/
  - pitch_deck.md

Quick start (local / Colab)
---------------------------
1. Install dependencies:
   pip install -r requirements.txt

2. Run a full demo with synthetic data:
   python src/train.py --use_synthetic

3. Start the Streamlit demo (after training or if you have saved models):
   streamlit run app/streamlit_app.py

4. To train on your CSV:
   - Provide a CSV with columns: student_id, timestamp, question_id, topic, correct (0/1), elapsed_time_seconds
   - Run: python src/train.py --data_path path/to/logs.csv

Notes on CSV / data format
--------------------------
- timestamp should be ISO-8601 (e.g., 2023-08-21T12:34:56).
- correct: 0 or 1.
- elapsed_time_seconds: optional but useful.

Ethics & privacy
----------------
- Anonymize student identifiers before sharing.
- Acquire consent for log collection and explain purpose.
- Beware of group bias—validate across cohorts and demographics.

Extending this project
----------------------
- Add sequential models (RNN/Transformer) for temporal patterns.
- Integrate curriculum metadata (topic difficulty, prerequisites).
- Serve predictions as a simple API or integrate with a learning platform.
- Add active learning: prompt teacher labeling on uncertain cases.

Contact
-------
Author: michaellolivery94-web 7 trevoringari3-afk

Collaborative notes
-------------------
- This repo is bootstrapped for Week 2 SDG assignment. Please open issues for any requested feature or dataset you'd like me to prioritize.
- TODO: add sample screenshots in assets/ after running the Streamlit demo; add a small sample CSV in data/raw/ for quick testing.
```
