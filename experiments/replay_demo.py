"""
experiments/replay_demo.py

Step 6: Replay the prepared, timestamp-ordered dataset through
TemporalMetricFrame — this is the "Incoming record -> Windowing strategy"
loop from the flowchart, running end to end on real data.
"""

import pandas as pd
from fairlearn.metrics import selection_rate
from sklearn.metrics import accuracy_score

from lucid.temporal_metric_frame import TemporalMetricFrame, SlidingWindow, Record

# --- 1. Load the prepared stream --------------------------------------------
df = pd.read_csv("data/adult_stream.csv", parse_dates=["timestamp"])
df = df.sort_values("timestamp").reset_index(drop=True)
print("Loaded stream:", df.shape)

# --- 2. Set up the temporal frame -------------------------------------------
tmf = TemporalMetricFrame(
    metrics={"selection_rate": selection_rate, "accuracy": accuracy_score},
    strategy=SlidingWindow(size=200, step=50),
    n_bootstrap=100,
    random_state=0,
)

# --- 3. Replay the stream, one record at a time -----------------------------
for _, row in df.iterrows():
    tmf.ingest(Record(
        y_true=row["y_true"],
        y_pred=row["y_pred"],
        sensitive_features={"sex": row["sex"]},
        timestamp=row["timestamp"],
    ))

print(f"\nWindows produced: {len(tmf._store)}")

# --- 4. Save results for Step 7 (validation) --------------------------------
history = tmf.history()
history.to_csv("data/history_run1.csv", index=False)
print("\nSaved data/history_run1.csv:", history.shape)
print(history.head(10))