```markdown
# Learning-Gap Detector — Article Submission (Week 2 — SDG 4: Quality Education)

Title: Identifying Learning Gaps with Lightweight Supervised Models to Support Quality Education

Problem statement
-----------------
Quality education (SDG 4) requires that learners master core topics so that downstream skills and learning can accumulate. In many contexts teachers and learning platforms lack timely, actionable signals about which topics individual students struggle with. This leads to generalized remediation that is less effective and wastes instructional time.

Our solution
------------
We built a compact prototype — the Learning-Gap Detector — that predicts topic-level mastery from simple quiz logs. The model produces per-student, per-topic mastery probabilities and surfaces the weakest topics for targeted remediation. This supports personalized learning, efficient teacher interventions, and helps direct learning resources where they are most needed.

Data & features
---------------
Input logs (one row per attempt) include: student_id, timestamp, question_id, topic, correct (0/1), elapsed_time_seconds. From these logs the pipeline builds snapshots per (student, topic) including:
- attempts: total attempts on that topic
- recency-weighted accuracy (exponential decay)
- recent accuracy (last 30 days)
- streak of consecutive corrects
- average elapsed time per attempt
These features are fast to compute using vectorized pandas operations and are meaningful for mastery prediction.

Model & training
----------------
We provide two lightweight models:
- Logistic Regression baseline (with StandardScaler),
- Random Forest classifier (200 trees, depth-limited).
Evaluation metrics include AUC, accuracy, precision, recall, and F1. The scripts print results and save the chosen model to models/.

Demo & UI
---------
A Streamlit app lets users upload logs (or use synthetic demo data), pick a student, and view top-k weak topics together with suggested resource links. The demo produces screenshots for submission and helps teachers explore predictions.

Results (example)
-----------------
On the synthetic dataset the Random Forest typically outperforms logistic regression (higher AUC). Exact numbers depend on synthetic parameters or the real dataset used; the training script prints metrics after training.

Ethical considerations
----------------------
- Privacy: Do not store student identifiers in plain text when sharing. Obtain consent and minimize personally identifiable data.
- Fairness: Validate across cohorts (gender, socioeconomic status, language). Models trained on biased logs may amplify inequities.
- Usage: Predictions should support teacher decisions, not replace them. Present probabilities with uncertainty and ask for teacher validation before high-stakes actions.

Limitations & future work
------------------------
- Synthetic data is a placeholder: real pilot data is necessary for meaningful evaluation.
- Sequence models (RNN/Transformer) may capture more temporal nuance but require more data.
- Curriculum metadata (topic prerequisites, difficulty) would improve personalized recommendations.
- Integrate A/B testing in classrooms to measure learning gains from targeted remediation.

How to reproduce
----------------
1. pip install -r requirements.txt
2. python src/train.py --use_synthetic
3. streamlit run app/streamlit_app.py
4. Replace synthetic logs with real logs (anonymized) and re-train.

Conclusion
----------
The Learning-Gap Detector demonstrates that a small, interpretable supervised pipeline can deliver actionable mastery probabilities to support SDG 4. By surfacing weak topics per student, teachers and platforms can provide targeted instruction and improve learning outcomes more efficiently.
```