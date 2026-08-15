"""
temporal_metric_frame.py

Objective 1: Temporal data-handling and windowing module that extends
Fairlearn's MetricFrame to operate over incoming, time-ordered data.

Design intent:
- Records come in one at a time (or in batches) via `ingest()`.
- A WindowingStrategy decides how records are grouped into evaluation windows.
- For each *closed* window, Fairlearn's MetricFrame is computed once, and the
  result (per-group metric values, bootstrap CI, sample size, window bounds)
  is appended to a time-indexed store.
- AdaptiveWindow is deliberately left as a stub with a naive trigger for now —
  it's meant to be handed a real change signal from the Objective 2 drift
  detector later, via `set_change_trigger()`. Don't over-build it yet.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Callable, Iterable, List, Optional, Sequence, Dict, Any
import numpy as np
import pandas as pd

from fairlearn.metrics import MetricFrame


# ---------------------------------------------------------------------------
# 1. Record model
# ---------------------------------------------------------------------------

@dataclass
class Record:
    y_true: Any
    y_pred: Any
    sensitive_features: Dict[str, Any]   # e.g. {"gender": "F", "age_group": "18-25"}
    timestamp: datetime


# ---------------------------------------------------------------------------
# 2. Windowing strategies
# ---------------------------------------------------------------------------

class WindowingStrategy:
    """Base interface. A strategy owns a buffer of pending records and decides
    when a window is 'closed' and ready to be scored."""

    def add(self, record: Record) -> None:
        raise NotImplementedError

    def pop_ready_windows(self) -> List[List[Record]]:
        """Return any windows that are now closed and ready for MetricFrame.
        Called after every `add()`."""
        raise NotImplementedError


class SlidingWindow(WindowingStrategy):
    """Fixed-size window over the last N records, stepping by `step`.
    step == size  -> non-overlapping (tumbling) windows.
    step  < size  -> overlapping windows.
    """

    def __init__(self, size: int, step: Optional[int] = None):
        self.size = size
        self.step = step or size
        self._buffer: List[Record] = []
        self._since_last_window = 0

    def add(self, record: Record) -> None:
        self._buffer.append(record)
        self._since_last_window += 1
        # keep only what we might still need for the next window
        if len(self._buffer) > self.size:
            self._buffer = self._buffer[-self.size:]

    def pop_ready_windows(self) -> List[List[Record]]:
        windows = []
        if len(self._buffer) >= self.size and self._since_last_window >= self.step:
            windows.append(list(self._buffer[-self.size:]))
            self._since_last_window = 0
        return windows


class FixedTimeWindow(WindowingStrategy):
    """Buckets records into wall-clock intervals, e.g. daily/weekly.
    A window closes once a record arrives with a timestamp past the bucket end.
    """

    def __init__(self, bucket: timedelta):
        self.bucket = bucket
        self._current_start: Optional[datetime] = None
        self._current: List[Record] = []

    def add(self, record: Record) -> None:
        if self._current_start is None:
            self._current_start = self._floor(record.timestamp)
        self._pending = record  # held until pop_ready_windows evaluates it

    def _floor(self, ts: datetime) -> datetime:
        # floor timestamp to the start of its bucket (epoch-aligned)
        epoch = datetime(1970, 1, 1)
        n_buckets = (ts - epoch) // self.bucket
        return epoch + n_buckets * self.bucket

    def pop_ready_windows(self) -> List[List[Record]]:
        record = getattr(self, "_pending", None)
        if record is None:
            return []
        self._pending = None

        bucket_start = self._floor(record.timestamp)
        windows = []

        if self._current_start is None:
            self._current_start = bucket_start

        if bucket_start > self._current_start:
            # the record belongs to a new bucket -> close the old one
            if self._current:
                windows.append(list(self._current))
            self._current = [record]
            self._current_start = bucket_start
        else:
            self._current.append(record)

        return windows


class AdaptiveWindow(WindowingStrategy):
    """Grows while data looks stable; shrinks/resets on a detected change.

    STUB for Objective 1: uses a naive std-dev-jump heuristic as a placeholder
    trigger so the module is testable end-to-end. Replace `self._is_change()`
    with a call into the Objective 2 drift detector once it exists — that's
    the intended extension point (see `set_change_trigger`).
    """

    def __init__(self, min_size: int = 20, max_size: int = 500):
        self.min_size = min_size
        self.max_size = max_size
        self._buffer: List[Record] = []
        self._change_trigger: Optional[Callable[[List[Record], Record], bool]] = None

    def set_change_trigger(self, fn: Callable[[List[Record], Record], bool]) -> None:
        """Plug in a real drift signal later, e.g. from the Page-Hinkley/ADWIN
        layer: fn(current_buffer, incoming_record) -> bool (True = change)."""
        self._change_trigger = fn

    def _naive_is_change(self, incoming: Record) -> bool:
        # Placeholder only: flags a change if the incoming prediction differs
        # from the running mean by > 2 std devs, once we have enough history.
        if len(self._buffer) < self.min_size:
            return False
        preds = np.array([float(r.y_pred) for r in self._buffer])
        mu, sigma = preds.mean(), preds.std() or 1e-9
        return abs(float(incoming.y_pred) - mu) > 2 * sigma

    def add(self, record: Record) -> None:
        trigger = self._change_trigger or self._naive_is_change
        self._pending_change = trigger(self._buffer, record) if self._buffer else False
        self._buffer.append(record)
        if len(self._buffer) > self.max_size:
            self._buffer = self._buffer[-self.max_size:]

    def pop_ready_windows(self) -> List[List[Record]]:
        windows = []
        ready = getattr(self, "_pending_change", False) and len(self._buffer) >= self.min_size
        if ready:
            windows.append(list(self._buffer))
            self._buffer = self._buffer[-self.min_size:]  # keep a small tail, don't fully reset
        return windows


# ---------------------------------------------------------------------------
# 3. TemporalMetricFrame — the Fairlearn-facing wrapper
# ---------------------------------------------------------------------------

@dataclass
class WindowResult:
    window_start: datetime
    window_end: datetime
    n_samples: int
    metric_frame: MetricFrame
    by_group: pd.DataFrame          # metric.by_group, with bootstrap CI columns attached
    overall: pd.Series


class TemporalMetricFrame:
    """Fairlearn-style API: same `metrics` / `sensitive_features` calling
    convention as MetricFrame, plus a windowing strategy and a time-indexed
    result store.

    Example
    -------
    tmf = TemporalMetricFrame(
        metrics={"selection_rate": selection_rate, "accuracy": accuracy_score},
        strategy=SlidingWindow(size=200, step=50),
        n_bootstrap=200,
    )
    for record in stream:
        tmf.ingest(record)
    history = tmf.history()   # tidy DataFrame, one row per (window, group, metric)
    """

    def __init__(
        self,
        metrics: Dict[str, Callable],
        strategy: WindowingStrategy,
        n_bootstrap: int = 200,
        ci: float = 0.95,
        random_state: Optional[int] = None,
    ):
        self.metrics = metrics
        self.strategy = strategy
        self.n_bootstrap = n_bootstrap
        self.ci = ci
        self._rng = np.random.default_rng(random_state)
        self._store: List[WindowResult] = []

    def ingest(self, record: Record) -> None:
        self.strategy.add(record)
        for window_records in self.strategy.pop_ready_windows():
            self._score_window(window_records)

    def ingest_batch(self, records: Iterable[Record]) -> None:
        for r in records:
            self.ingest(r)

    def _score_window(self, records: Sequence[Record]) -> None:
        y_true = [r.y_true for r in records]
        y_pred = [r.y_pred for r in records]
        # supports multiple sensitive features; MetricFrame wants a DataFrame
        sens_df = pd.DataFrame([r.sensitive_features for r in records])

        mf = MetricFrame(
            metrics=self.metrics,
            y_true=y_true,
            y_pred=y_pred,
            sensitive_features=sens_df,
        )

        ci_df = self._bootstrap_ci(y_true, y_pred, sens_df)
        by_group = mf.by_group.join(ci_df) if ci_df is not None else mf.by_group

        self._store.append(WindowResult(
            window_start=records[0].timestamp,
            window_end=records[-1].timestamp,
            n_samples=len(records),
            metric_frame=mf,
            by_group=by_group,
            overall=mf.overall,
        ))

    def _bootstrap_ci(self, y_true, y_pred, sens_df) -> Optional[pd.DataFrame]:
        """Bootstrap CI per (group, metric) so small/sparse windows can be
        flagged as low-confidence downstream (feeds Objective 2's gating)."""
        n = len(y_true)
        if n < 10:  # not enough to bootstrap meaningfully
            return None

        boot_results = []
        idx = np.arange(n)
        for _ in range(self.n_bootstrap):
            sample = self._rng.choice(idx, size=n, replace=True)
            mf = MetricFrame(
                metrics=self.metrics,
                y_true=np.asarray(y_true)[sample],
                y_pred=np.asarray(y_pred)[sample],
                sensitive_features=sens_df.iloc[sample].reset_index(drop=True),
            )
            boot_results.append(mf.by_group)

        stacked = pd.concat(boot_results, keys=range(len(boot_results)))
        alpha = (1 - self.ci) / 2
        lower = stacked.groupby(level=1).quantile(alpha)
        upper = stacked.groupby(level=1).quantile(1 - alpha)
        lower.columns = [f"{c}_ci_low" for c in lower.columns]
        upper.columns = [f"{c}_ci_high" for c in upper.columns]
        return lower.join(upper)

    def history(self) -> pd.DataFrame:
        """Tidy, time-indexed view: one row per (window, group, metric).
        This is the table the Objective 2 drift detector will consume."""
        rows = []
        for w in self._store:
            for group, row in w.by_group.iterrows():
                for metric_name in self.metrics:
                    rows.append({
                        "window_start": w.window_start,
                        "window_end": w.window_end,
                        "n_samples": w.n_samples,
                        "group": group,
                        "metric": metric_name,
                        "value": row[metric_name],
                        "ci_low": row.get(f"{metric_name}_ci_low"),
                        "ci_high": row.get(f"{metric_name}_ci_high"),
                    })
        return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# 4. Smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    from fairlearn.metrics import selection_rate
    from sklearn.metrics import accuracy_score

    rng = np.random.default_rng(0)
    start = datetime(2026, 1, 1)
    records = []
    for i in range(500):
        group = rng.choice(["A", "B"])
        # inject a base-rate shift halfway through, for later drift testing
        bias = 0.1 if i < 250 else (0.4 if group == "A" else 0.1)
        y_true = int(rng.random() < 0.5)
        y_pred = int(rng.random() < (0.5 + bias if group == "A" else 0.5))
        records.append(Record(
            y_true=y_true,
            y_pred=y_pred,
            sensitive_features={"group": group},
            timestamp=start + timedelta(hours=i),
        ))

    print("--- SlidingWindow ---")
    tmf = TemporalMetricFrame(
        metrics={"selection_rate": selection_rate, "accuracy": accuracy_score},
        strategy=SlidingWindow(size=100, step=50),
        n_bootstrap=50,
        random_state=0,
    )
    tmf.ingest_batch(records)
    print(tmf.history().head(10))
    print(f"windows produced: {len(tmf._store)}")

    print("\n--- FixedTimeWindow (daily) ---")
    tmf2 = TemporalMetricFrame(
        metrics={"selection_rate": selection_rate},
        strategy=FixedTimeWindow(bucket=timedelta(days=1)),
        n_bootstrap=30,
        random_state=0,
    )
    tmf2.ingest_batch(records)
    print(tmf2.history().head(10))
    print(f"windows produced: {len(tmf2._store)}")