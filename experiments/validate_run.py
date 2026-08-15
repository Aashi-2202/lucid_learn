"""
experiments/validate_run.py

Step 7: Validate the replay output against the flowchart's expectations —
does the injected drift actually show up in the scored windows?
"""

import pandas as pd
import matplotlib.pyplot as plt

history = pd.read_csv("data/history_run1.csv", parse_dates=["window_start", "window_end"])

# --- Sanity check 1: window count in the right ballpark ---------------------
n_windows = history["window_start"].nunique()
print(f"Distinct windows: {n_windows}")

# --- Sanity check 2: values aren't degenerate (all identical / all NaN) -----
sr = history[history["metric"] == "selection_rate"]
print("\nselection_rate summary by group:")
print(sr.groupby("group")["value"].describe())

# --- Sanity check 3: read the drift ground truth ----------------------------
with open("data/drift_ground_truth.txt") as f:
    ground_truth = dict(line.strip().split("=") for line in f if line.strip())
drift_ts = pd.Timestamp(ground_truth["drift_timestamp"])
print(f"\nGround-truth drift injected at: {drift_ts} for {ground_truth['drift_group']}")

# --- Plot: selection_rate by group over time, with drift point marked ------
fig, ax = plt.subplots(figsize=(10, 5))
for group, g in sr.groupby("group"):
    ax.plot(g["window_start"], g["value"], label=f"sex={group}", marker="o", markersize=3)
    ax.fill_between(g["window_start"], g["ci_low"], g["ci_high"], alpha=0.15)

ax.axvline(drift_ts, color="red", linestyle="--", label="injected drift point")
ax.set_xlabel("window start")
ax.set_ylabel("selection rate")
ax.set_title("Selection rate by group over time (Objective 1 validation)")
ax.legend()
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig("data/validation_plot.png", dpi=150)
print("\nSaved data/validation_plot.png")
plt.show() 