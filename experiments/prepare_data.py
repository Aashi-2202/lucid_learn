"""
experiments/prepare_data.py
Step 4: Get a real dataset with synthetic timestamps, ready to feed
into the TemporalMetricFrame replay driver (Step 6).
"""
import pandas as pd
from sklearn.datasets import fetch_openml
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LogisticRegression

# --- 1. Fetch Adult Income dataset -----------------------------------------
data = fetch_openml("adult", version=2, as_frame=True)
df = data.frame.copy()

# target column is called "class" in this dataset: '>50K' / '<=50K'
df = df.rename(columns={"class": "y_true"})
df["y_true"] = (df["y_true"].astype(str).str.strip() == ">50K").astype(int)

print("Loaded shape:", df.shape)
print("Columns:", df.columns.tolist())
print(df["y_true"].value_counts())

# --- 2. Assign synthetic, sorted timestamps ---------------------------------
# One record per synthetic hour, starting Jan 1 2026 — gives you a clean
# time-ordered stream to replay through ingest().
#
# NOTE: this is a synthetic replay ordering only. Adult has no real temporal
# field — "timestamp" here means "pretend this record arrived at this point
# in the stream," not "this person entered the dataset at this time." Keep
# that distinction explicit in the report.
start = pd.Timestamp("2026-01-01")
df["timestamp"] = [start + pd.Timedelta(hours=i) for i in range(len(df))]

# We'll use this synthetic timestamp order as the chronological order for
# train/stream splitting below (issue 4) — sort now so "earlier" and "later"
# are unambiguous.
df = df.sort_values("timestamp").reset_index(drop=True)

# --- 3. Explicit feature/sensitive-feature decision (issue 5) ---------------
# DECISION: "sex" is EXCLUDED from the model features and used only as the
# sensitive attribute for Fairlearn. This is the more common fairness-audit
# setup (predict without the protected attribute, then audit by it) and
# keeps this experiment's setup unambiguous. If you later want the
# "sex included in the model" variant, just remove "sex" from this
# exclusion list — but do it deliberately, not by accident.
SENSITIVE_COL = "sex"
non_feature_cols = ("y_true", "timestamp", SENSITIVE_COL)
feature_cols = [c for c in df.columns if c not in non_feature_cols]

categorical_cols = df[feature_cols].select_dtypes(include=["category", "object"]).columns.tolist()
numeric_cols = [c for c in feature_cols if c not in categorical_cols]

print("\nCategorical features:", categorical_cols)
print("Numeric features:", numeric_cols)

# --- 4. Explicit missing-value handling + proper categorical encoding ------
# (issues 1 and 2)
#
# Adult's missing categorical entries typically show up as "?" — normalize
# those to real NaN first so the imputer actually catches them.
df[categorical_cols] = df[categorical_cols].replace("?", pd.NA)

categorical_pipeline = Pipeline(steps=[
    ("impute", SimpleImputer(strategy="constant", fill_value="Missing")),
    ("onehot", OneHotEncoder(handle_unknown="ignore")),
])

numeric_pipeline = Pipeline(steps=[
    ("impute", SimpleImputer(strategy="median")),
])

preprocessor = ColumnTransformer(transformers=[
    ("cat", categorical_pipeline, categorical_cols),
    ("num", numeric_pipeline, numeric_cols),
])

# --- 5. Chronological train / stream split (issue 4) ------------------------
# The dataset is sorted by synthetic timestamp above. Train on the earlier
# slice, predict on the later slice — mirrors "model learns from past data,
# then scores future observations" instead of in-sample prediction (issue 3).
TRAIN_FRACTION = 0.5
split_idx = int(len(df) * TRAIN_FRACTION)

train_df = df.iloc[:split_idx]
stream_df = df.iloc[split_idx:].copy()

X_train = preprocessor.fit_transform(train_df[feature_cols])
X_stream = preprocessor.transform(stream_df[feature_cols])

clf = LogisticRegression(max_iter=200)
clf.fit(X_train, train_df["y_true"])

# y_pred only exists for the "stream" portion — that's the realistic
# out-of-sample prediction this experiment should be scoring.
stream_df["y_pred"] = clf.predict(X_stream)

print(f"\nTrained on {len(train_df)} earlier records, "
      f"predicting on {len(stream_df)} later (stream) records.")

# --- 6. Inject a known drift point (ground truth for Objective 6 later) -----
# Kept as a clearly separate, explicit manipulation step — applied AFTER
# clean baseline y_pred is generated, not mixed into preprocessing (issue 7).
DRIFT_FRACTION_INTO_STREAM = 0.5
drift_point_in_stream = int(len(stream_df) * DRIFT_FRACTION_INTO_STREAM)
drift_global_index = stream_df.index[drift_point_in_stream]

sex_mask = stream_df["sex"] == "Male"
post_drift_mask = stream_df.index >= drift_global_index
drift_idx = stream_df.index[post_drift_mask & sex_mask]

stream_df.loc[drift_idx, "y_pred"] = 1  # inflate positive-prediction rate for this group post-drift

print(f"\nDrift injected at stream row {drift_point_in_stream} "
      f"(timestamp {stream_df.loc[drift_global_index, 'timestamp']}) for sex == 'Male'")

# --- 7. Save for the replay step --------------------------------------------
out_cols = ["timestamp", "y_true", "y_pred", "sex", "age"]
stream_df[out_cols].to_csv("data/adult_stream.csv", index=False)

print("\nSaved data/adult_stream.csv:", stream_df[out_cols].shape)
print(stream_df[out_cols].head())

# Keep drift ground truth visible for later benchmarking (Step 6/Objective 6)
with open("data/drift_ground_truth.txt", "w") as f:
    f.write(f"drift_row_index={drift_global_index}\n")
    f.write(f"drift_timestamp={stream_df.loc[drift_global_index, 'timestamp']}\n")
    f.write("drift_group=sex:Male\n")