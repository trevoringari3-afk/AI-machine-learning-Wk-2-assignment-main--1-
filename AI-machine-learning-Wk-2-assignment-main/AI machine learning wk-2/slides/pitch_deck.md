# Pitch Deck — Learning-Gap Detector (Slide content)

Slide 1 — Title
- Learning-Gap Detector
- "Predict topic-level mastery to support SDG 4 — Quality Education"
- Value: reduce teacher remediation time and speed student recovery
- Author: michaellolivery94-web & trevoringari3-afk

Slide 2 — Problem
- Teachers and platforms lack timely signals about individual topic weaknesses.
- Result: inefficient remediation, slower learning gains.
- Context: many classrooms lack formative analytics; teachers can spend X+ hours/week triaging.

Slide 3 — Opportunity
- Many platforms collect quiz logs (student_id, timestamp, topic, correctness, elapsed time).
- Typical availability: per-quiz logs, ~100–1k events/classroom/month.
- Low-cost analytics can improve personalization and teacher efficiency.

Slide 4 — Solution
- Snapshot-based supervised model predicts mastery probability per (student, topic) for next unit/week.
- Outputs top N weak topics with probability and recommended resources.
- Integrates with teacher dashboard or LMS to surface triage lists.

Slide 5 — Data & Model
- Inputs: student_id, timestamp, topic, correct, elapsed_time
- Label: mastery = ≥80% accuracy on topic in last M attempts (specify threshold per curriculum)
- Features: recency-weighted accuracy, streaks, attempts, recent accuracy, time-on-task
- Data: e.g., 50k records, 2k students, 100 topics (add your dataset numbers)
- Models: Logistic Regression (baseline), Random Forest (production candidate)
- Evaluation: holdout by time; report AUC, F1, precision@K; address class imbalance with sampling/weights

Slide 6 — Demo / Results
- Example flow: sample logs -> model -> top 3 weak topics with probabilities -> linked resources
- Example metrics on test set: AUC: 0.78, F1: 0.62 (replace with your numbers and CI)
- Notes: show confusion matrix, precision@top3, and per-topic performance.

Slide 7 — Impact & SDG alignment
- Helps teachers triage interventions, personalizes learning, contributes to improved learning outcomes (SDG4).
- Measurable outcomes: reduced remediation time, increased topic mastery % after targeted practice.

Slide 8 — Roadmap & Ask
- Pilot: anonymized classroom logs, 3-month pilot, measure pre/post mastery and teacher adoption.
- Integrations: LMS/curriculum links, teacher UI.
- Ask: pilot partners, labeled datasets, mentorship for deployment, compute access.
- Risks & mitigations: privacy (anonymize, consent), bias (per-topic performance monitoring), label noise (human review).