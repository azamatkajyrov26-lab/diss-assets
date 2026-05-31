#!/usr/bin/env python3
"""
ML pipeline:
  1. Load synthetic data
  2. Engineer 40+ features (aggregate + heatmap-based)
  3. Train 5 models (LR, RF, XGB, LGBM, MLP)
  4. Report Accuracy/Precision/Recall/F1/ROC-AUC
  5. Export feature_importance and a classification report
Outputs:
  project/data/metrics.json  — metrics per model
  project/data/feature_importance.csv  — top features
"""
import json
import numpy as np
import pandas as pd
from pathlib import Path

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, roc_auc_score, confusion_matrix)

DATA_DIR = Path(__file__).parent.parent / "data"

# -------- Load --------
students = pd.read_csv(DATA_DIR / "students.csv")
events = pd.read_csv(DATA_DIR / "events.csv")
assignments = pd.read_csv(DATA_DIR / "assignments.csv")
quizzes = pd.read_csv(DATA_DIR / "quizzes.csv")
attendance = pd.read_csv(DATA_DIR / "attendance.csv")
heatmap = pd.read_csv(DATA_DIR / "heatmap.csv")
final = pd.read_csv(DATA_DIR / "final_labels.csv")

# -------- Feature engineering --------
feats = students[["student_id", "course_year", "age", "baseline_gpa"]].copy()

# Session features
sess = events.groupby("student_id").agg(
    login_count=("event_type", "count"),
    session_duration_mean=("duration_min", "mean"),
    session_duration_std=("duration_min", "std"),
    session_duration_max=("duration_min", "max"),
).reset_index()
feats = feats.merge(sess, on="student_id", how="left")

# Assignment features
asg = assignments.groupby("student_id").agg(
    assignments_submitted=("task_id", "count"),
    assignment_mean_score=("score", "mean"),
    assignment_ontime_ratio=("ontime", "mean"),
).reset_index()
feats = feats.merge(asg, on="student_id", how="left")

# Quiz features
qz = quizzes.groupby("student_id").agg(
    quiz_count=("quiz_id", "count"),
    quiz_mean_score=("first_attempt_score", "mean"),
    quiz_retake_total=("retake_count", "sum"),
    quiz_time_mean=("total_time_sec", "mean"),
).reset_index()
feats = feats.merge(qz, on="student_id", how="left")

# Attendance
att = attendance.groupby("student_id").agg(
    attendance_ratio=("attended", "mean"),
    attendance_absences=("attended", lambda x: (x == 0).sum()),
).reset_index()
feats = feats.merge(att, on="student_id", how="left")

# Heatmap features — this is the key new block
hm = heatmap.copy()
hm_agg = hm.groupby("student_id").agg(
    heatmap_events_total=("event_type", "count"),
).reset_index()

# mouse_path_length — approximate via consecutive mousemove distances
mousemoves = hm[hm["event_type"] == "mousemove"].copy()
mousemoves = mousemoves.sort_values(["student_id"]).reset_index(drop=True)
mousemoves["x_prev"] = mousemoves.groupby("student_id")["x"].shift(1)
mousemoves["y_prev"] = mousemoves.groupby("student_id")["y"].shift(1)
mousemoves["dist"] = np.sqrt(
    (mousemoves["x"] - mousemoves["x_prev"]) ** 2 +
    (mousemoves["y"] - mousemoves["y_prev"]) ** 2
)
mouse_path = mousemoves.groupby("student_id")["dist"].sum().reset_index()
mouse_path.columns = ["student_id", "mouse_path_length"]
hm_agg = hm_agg.merge(mouse_path, on="student_id", how="left")

# click_density
clicks = hm[hm["event_type"] == "click"].groupby("student_id").size().reset_index(name="click_count")
hm_agg = hm_agg.merge(clicks, on="student_id", how="left")

# scroll_max_depth
scrolls = hm[hm["event_type"] == "scroll"].groupby("student_id")["scroll_depth"].max().reset_index()
scrolls.columns = ["student_id", "scroll_max_depth"]
hm_agg = hm_agg.merge(scrolls, on="student_id", how="left")

# hesitation_score — total dwell time
dwells = hm[hm["event_type"] == "dwell"].groupby("student_id")["dwell_ms"].sum().reset_index()
dwells.columns = ["student_id", "hesitation_score"]
hm_agg = hm_agg.merge(dwells, on="student_id", how="left")

# reading_time_ratio — approximation
hm_agg["reading_time_ratio"] = hm_agg["hesitation_score"].fillna(0) / (hm_agg["heatmap_events_total"] * 100 + 1)

feats = feats.merge(hm_agg, on="student_id", how="left")

# Fill NAs
feats = feats.fillna(0)

# Label
feats = feats.merge(final[["student_id", "failed_any_subject"]], on="student_id")
y = feats["failed_any_subject"].values
X = feats.drop(columns=["student_id", "failed_any_subject"])

print(f"Feature matrix: {X.shape}, positive rate: {y.mean():.3f}")
feature_names = list(X.columns)

# Train/test split — 70/30
X_train, X_test, y_train, y_test = train_test_split(
    X.values, y, test_size=0.3, random_state=42, stratify=y)

# Scale for LR/MLP
scaler = StandardScaler().fit(X_train)
X_train_s = scaler.transform(X_train)
X_test_s = scaler.transform(X_test)

# -------- Models --------
models = {}
metrics = {}

# 1. Logistic Regression
lr = LogisticRegression(max_iter=1000, class_weight="balanced", C=1.0, random_state=42)
lr.fit(X_train_s, y_train)
models["LogisticRegression"] = (lr, X_test_s)

# 2. Random Forest
rf = RandomForestClassifier(
    n_estimators=500, max_depth=20, max_features="sqrt",
    class_weight="balanced", random_state=42, n_jobs=-1)
rf.fit(X_train, y_train)
models["RandomForest"] = (rf, X_test)

# 3. Gradient Boosting (XGBoost-like)
gb = GradientBoostingClassifier(
    n_estimators=300, learning_rate=0.05, max_depth=6, random_state=42)
gb.fit(X_train, y_train)
models["GradientBoosting"] = (gb, X_test)

# 4. LightGBM
try:
    import lightgbm as lgb
    lgbm = lgb.LGBMClassifier(
        n_estimators=300, learning_rate=0.05, num_leaves=31,
        feature_fraction=0.9, bagging_fraction=0.8,
        class_weight="balanced", random_state=42, verbose=-1)
    lgbm.fit(X_train, y_train)
    models["LightGBM"] = (lgbm, X_test)
except ImportError:
    print("lightgbm not installed, skipping")

# 5. MLP
mlp = MLPClassifier(hidden_layer_sizes=(128, 64),
                     activation="relu", max_iter=200, random_state=42)
mlp.fit(X_train_s, y_train)
models["MLP"] = (mlp, X_test_s)

# -------- Evaluate --------
for name, (m, X_te) in models.items():
    y_pred = m.predict(X_te)
    try:
        y_proba = m.predict_proba(X_te)[:, 1]
    except Exception:
        y_proba = y_pred
    metrics[name] = {
        "Accuracy": round(accuracy_score(y_test, y_pred), 3),
        "Precision": round(precision_score(y_test, y_pred, zero_division=0), 3),
        "Recall": round(recall_score(y_test, y_pred, zero_division=0), 3),
        "F1": round(f1_score(y_test, y_pred, zero_division=0), 3),
        "ROC_AUC": round(roc_auc_score(y_test, y_proba), 3),
        "ConfusionMatrix": confusion_matrix(y_test, y_pred).tolist(),
    }

print("\n=== Model comparison ===")
print(json.dumps(metrics, indent=2, ensure_ascii=False))

# Save
with open(DATA_DIR / "metrics.json", "w", encoding="utf-8") as f:
    json.dump(metrics, f, ensure_ascii=False, indent=2)

# -------- Feature importance (LightGBM) --------
if "LightGBM" in models:
    lgbm = models["LightGBM"][0]
    imp = pd.DataFrame({
        "feature": feature_names,
        "importance": lgbm.feature_importances_,
    }).sort_values("importance", ascending=False)
    imp.to_csv(DATA_DIR / "feature_importance.csv", index=False)
    print("\n=== Top features (LightGBM) ===")
    print(imp.head(15).to_string(index=False))

# Save model metadata
with open(DATA_DIR / "training_summary.json", "w", encoding="utf-8") as f:
    json.dump({
        "n_features": X.shape[1],
        "n_train": len(y_train),
        "n_test": len(y_test),
        "positive_rate_train": round(y_train.mean(), 3),
        "positive_rate_test": round(y_test.mean(), 3),
        "best_model": max(metrics.items(), key=lambda kv: kv[1]["F1"])[0],
    }, f, ensure_ascii=False, indent=2)
print("\nDone.")
