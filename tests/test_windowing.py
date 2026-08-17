"""
tests/test_windowing.py

Step 8: Minimal unit tests for Objective 1 — confirming each windowing
strategy closes windows at the right cadence, and that history() keeps
a stable schema (Objective 2/3 will build directly on top of this table).
"""

from datetime import datetime, timedelta

import pytest
from sklearn.metrics import accuracy_score

from lucid.temporal_metric_frame import (
    Record,
    SlidingWindow,
    FixedTimeWindow,
    AdaptiveWindow,
    TemporalMetricFrame,
)


def make_records(n, start=None, hours_apart=1, y_true=None, y_pred=None, group="A"):
    """Helper: build n dummy records, evenly spaced in time."""
    start = start or datetime(2026, 1, 1)
    records = []
    for i in range(n):
        records.append(Record(
            y_true=1 if y_true is None else y_true(i),
            y_pred=1 if y_pred is None else y_pred(i),
            sensitive_features={"group": group},
            timestamp=start + timedelta(hours=i * hours_apart),
        ))
    return records


# ---------------------------------------------------------------------------
# SlidingWindow
# ---------------------------------------------------------------------------

def test_sliding_window_produces_expected_count():
    """20 records, size=10, step=5 -> windows close at record 10, 15, 20 => 3 windows."""
    strategy = SlidingWindow(size=10, step=5)
    tmf = TemporalMetricFrame(
        metrics={"accuracy": accuracy_score},
        strategy=strategy,
        n_bootstrap=0,  # skip bootstrap for speed; not what this test checks
    )
    tmf.ingest_batch(make_records(20))
    assert len(tmf._store) == 3


def test_sliding_window_respects_size():
    """Every produced window should contain exactly `size` records."""
    strategy = SlidingWindow(size=10, step=10)
    tmf = TemporalMetricFrame(
        metrics={"accuracy": accuracy_score},
        strategy=strategy,
        n_bootstrap=0,
    )
    tmf.ingest_batch(make_records(30))
    assert all(w.n_samples == 10 for w in tmf._store)


def test_sliding_window_no_window_before_size_reached():
    """Fewer records than `size` -> no window should close yet."""
    strategy = SlidingWindow(size=50, step=10)
    tmf = TemporalMetricFrame(
        metrics={"accuracy": accuracy_score},
        strategy=strategy,
        n_bootstrap=0,
    )
    tmf.ingest_batch(make_records(30))
    assert len(tmf._store) == 0


# ---------------------------------------------------------------------------
# FixedTimeWindow
# ---------------------------------------------------------------------------

def test_fixed_time_window_closes_on_bucket_change():
    """48 hourly records with a 1-day bucket -> 2 buckets total, but the
    LAST bucket only closes when a record from the next bucket arrives.
    With exactly 48 hourly records (day 0 + day 1), only day 0 closes."""
    strategy = FixedTimeWindow(bucket=timedelta(days=1))
    tmf = TemporalMetricFrame(
        metrics={"accuracy": accuracy_score},
        strategy=strategy,
        n_bootstrap=0,
    )
    tmf.ingest_batch(make_records(48, hours_apart=1))
    # day 0 (hours 0-23) closes when the first day-1 record arrives;
    # day 1 never closes because no day-2 record ever arrives (known
    # limitation flagged earlier — no flush() yet).
    assert len(tmf._store) == 1
    assert tmf._store[0].n_samples == 24


def test_fixed_time_window_single_bucket_stays_open():
    """All records within one bucket -> zero windows close (nothing to
    compare against yet)."""
    strategy = FixedTimeWindow(bucket=timedelta(days=7))
    tmf = TemporalMetricFrame(
        metrics={"accuracy": accuracy_score},
        strategy=strategy,
        n_bootstrap=0,
    )
    tmf.ingest_batch(make_records(24, hours_apart=1))  # 1 day, well within a 7-day bucket
    assert len(tmf._store) == 0


# ---------------------------------------------------------------------------
# AdaptiveWindow
# ---------------------------------------------------------------------------

def test_adaptive_window_respects_min_size():
    """No window should close before min_size records have arrived,
    regardless of the change trigger."""
    strategy = AdaptiveWindow(min_size=20, max_size=100)
    strategy.set_change_trigger(lambda buffer, incoming: True)  # always "changed"
    tmf = TemporalMetricFrame(
        metrics={"accuracy": accuracy_score},
        strategy=strategy,
        n_bootstrap=0,
    )
    tmf.ingest_batch(make_records(15))
    assert len(tmf._store) == 0


def test_adaptive_window_closes_on_injected_trigger():
    """With an always-true change trigger and enough records, a window
    should close on essentially every subsequent record."""
    strategy = AdaptiveWindow(min_size=10, max_size=100)
    strategy.set_change_trigger(lambda buffer, incoming: True)
    tmf = TemporalMetricFrame(
        metrics={"accuracy": accuracy_score},
        strategy=strategy,
        n_bootstrap=0,
    )
    tmf.ingest_batch(make_records(30))
    assert len(tmf._store) > 0


# ---------------------------------------------------------------------------
# history() schema stability
# ---------------------------------------------------------------------------

def test_history_schema_is_stable():
    """Objective 2/3 will read this table directly -- lock its columns down
    so a future refactor can't silently break them."""
    strategy = SlidingWindow(size=10, step=10)
    tmf = TemporalMetricFrame(
        metrics={"accuracy": accuracy_score},
        strategy=strategy,
        n_bootstrap=0,
    )
    tmf.ingest_batch(make_records(20))
    history = tmf.history()

    expected_columns = {
        "window_start", "window_end", "n_samples",
        "group", "metric", "value", "ci_low", "ci_high",
    }
    assert set(history.columns) == expected_columns
    assert len(history) == 2  # 2 windows x 1 group x 1 metric


def test_history_empty_when_no_windows_closed():
    """No closed windows yet -> history() should return an empty (but
    correctly shaped) DataFrame, not raise."""
    strategy = SlidingWindow(size=100, step=10)
    tmf = TemporalMetricFrame(
        metrics={"accuracy": accuracy_score},
        strategy=strategy,
        n_bootstrap=0,
    )
    tmf.ingest_batch(make_records(5))
    history = tmf.history()
    assert len(history) == 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])