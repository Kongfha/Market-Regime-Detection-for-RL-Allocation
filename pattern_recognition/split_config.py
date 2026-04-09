"""Shared split configuration for the pattern-recognition pipeline."""

from __future__ import annotations

import pandas as pd


SPLIT_BOUNDS = {
    "warmup": (pd.Timestamp("2014-01-02"), pd.Timestamp("2014-06-30")),
    "train": (pd.Timestamp("2014-07-01"), pd.Timestamp("2020-12-31")),
    "validation": (pd.Timestamp("2021-01-01"), pd.Timestamp("2022-12-30")),
    "locked_test": (pd.Timestamp("2023-01-03"), pd.Timestamp("2026-03-20")),
}

SPLIT_ORDER = ["warmup", "train", "validation", "locked_test"]


def assign_split_stage(dates) -> pd.Series:
    """Assign each date to the configured experimental split.

    Dates outside the configured ranges are returned as <NA> so callers can
    decide whether to drop or flag them.
    """
    date_series = pd.to_datetime(pd.Series(dates))
    split_stage = pd.Series(pd.NA, index=date_series.index, dtype="object")

    for stage, (start, end) in SPLIT_BOUNDS.items():
        mask = (date_series >= start) & (date_series <= end)
        split_stage.loc[mask] = stage

    if hasattr(dates, "index"):
        split_stage.index = dates.index

    return split_stage


def get_split_mask(df: pd.DataFrame, stage: str) -> pd.Series:
    """Return a boolean mask for one split stage."""
    if "week_end" not in df.columns:
        raise KeyError("week_end column is required to assign splits")

    return assign_split_stage(df["week_end"]).eq(stage)


def get_train_mask(df: pd.DataFrame) -> pd.Series:
    """Convenience helper for the training window."""
    return get_split_mask(df, "train")

