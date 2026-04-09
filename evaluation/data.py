from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import pandas as pd

from .config import DEFAULT_DATA_PATH, SplitBoundaries


TARGET_COLUMNS = {
    "spy_weekly_close",
    "tlt_weekly_close",
    "gld_weekly_close",
    "next_return_spy",
    "next_return_tlt",
    "next_return_gld",
    "cash_return",
}
METADATA_COLUMNS = {"week_end", "week_last_trade_date", "source", "eval_split"}
PRICE_PREFIXES = ("spy_", "tlt_", "gld_", "qqq_", "vix_", "tnx_")
TEXT_TOKENS = ("headline", "sentiment", "impact", "relevance", "topic_", "news_")
REGIME_TOKENS = ("regime", "posterior", "state_prob", "hmm")
ENDOGENOUS_COLUMNS = (
    "prev_weight_spy",
    "prev_weight_tlt",
    "prev_weight_gld",
    "prev_weight_cash",
    "portfolio_drawdown",
    "portfolio_volatility",
)


@dataclass(frozen=True)
class FeatureGroups:
    price: tuple[str, ...]
    macro: tuple[str, ...]
    regime: tuple[str, ...]
    text: tuple[str, ...]
    metadata: tuple[str, ...]
    targets: tuple[str, ...]


@dataclass
class EvaluationDataset:
    frame: pd.DataFrame
    feature_groups: FeatureGroups
    return_columns: dict[str, str]
    split_column: str = "eval_split"

    def subset(self, split: str | Sequence[str]) -> "EvaluationDataset":
        if isinstance(split, str):
            split = [split]
        subset_frame = self.frame[self.frame[self.split_column].isin(split)].reset_index(drop=True)
        return EvaluationDataset(
            frame=subset_frame,
            feature_groups=self.feature_groups,
            return_columns=self.return_columns,
            split_column=self.split_column,
        )

    def continuous_columns(self, include_blocks: Sequence[str] = ("price", "macro", "regime", "text")) -> list[str]:
        ordered = []
        for block in include_blocks:
            ordered.extend(getattr(self.feature_groups, block))
        return ordered

    def observation_columns(
        self,
        include_blocks: Sequence[str] = ("price", "macro", "regime", "text"),
        include_endogenous: bool = True,
    ) -> list[str]:
        columns = self.continuous_columns(include_blocks=include_blocks)
        if include_endogenous:
            columns.extend(ENDOGENOUS_COLUMNS)
        return columns

    def state_matrix(self, include_blocks: Sequence[str] = ("price", "macro", "regime", "text")) -> np.ndarray:
        columns = self.continuous_columns(include_blocks=include_blocks)
        if not columns:
            return np.empty((len(self.frame), 0))
        return self.frame.loc[:, columns].to_numpy(dtype=float)

    def rl_input_frame(
        self,
        include_blocks: Sequence[str] = ("price", "macro", "regime", "text"),
        include_targets: bool = True,
    ) -> pd.DataFrame:
        columns = ["week_end", self.split_column]
        columns.extend(self.continuous_columns(include_blocks=include_blocks))
        if include_targets:
            columns.extend(self.return_columns.values())
        return self.frame.loc[:, columns].copy()

    def describe_feature_blocks(self) -> pd.DataFrame:
        rows = []
        for block_name in ("price", "macro", "regime", "text"):
            columns = list(getattr(self.feature_groups, block_name))
            rows.append(
                {
                    "block": block_name,
                    "n_columns": len(columns),
                    "example_columns": ", ".join(columns[:5]) if columns else "(none)",
                }
            )
        return pd.DataFrame(rows)

    def describe_splits(self) -> pd.DataFrame:
        rows = []
        for split_name, group in self.frame.groupby(self.split_column, sort=False):
            rows.append(
                {
                    "split": split_name,
                    "rows": len(group),
                    "start": group["week_end"].min(),
                    "end": group["week_end"].max(),
                }
            )
        return pd.DataFrame(rows)


def load_default_dataset(
    path: str | Path = DEFAULT_DATA_PATH,
    split_boundaries: SplitBoundaries = SplitBoundaries(),
) -> EvaluationDataset:
    frame = pd.read_csv(path, parse_dates=["week_end"])
    frame = frame.sort_values("week_end").reset_index(drop=True)
    frame["cash_return"] = frame["dff_level"].fillna(0.0) / 100.0 / 52.0
    frame["eval_split"] = frame["week_end"].apply(lambda value: _label_split(value, split_boundaries))

    feature_groups = infer_feature_groups(frame.columns)
    return_columns = {
        "SPY": "next_return_spy",
        "TLT": "next_return_tlt",
        "GLD": "next_return_gld",
        "CASH": "cash_return",
    }

    return EvaluationDataset(frame=frame, feature_groups=feature_groups, return_columns=return_columns)


def infer_feature_groups(columns: Iterable[str]) -> FeatureGroups:
    price: list[str] = []
    macro: list[str] = []
    regime: list[str] = []
    text: list[str] = []
    metadata: list[str] = []
    targets: list[str] = []

    for column in columns:
        if column in TARGET_COLUMNS:
            targets.append(column)
        elif column in METADATA_COLUMNS:
            metadata.append(column)
        elif _is_text_feature(column):
            text.append(column)
        elif _is_regime_feature(column):
            regime.append(column)
        elif _is_price_feature(column):
            price.append(column)
        else:
            macro.append(column)

    return FeatureGroups(
        price=tuple(price),
        macro=tuple(macro),
        regime=tuple(regime),
        text=tuple(text),
        metadata=tuple(metadata),
        targets=tuple(targets),
    )


def _label_split(timestamp: pd.Timestamp, boundaries: SplitBoundaries) -> str:
    if timestamp <= pd.Timestamp(boundaries.warmup_end):
        return "warmup"
    if timestamp <= pd.Timestamp(boundaries.train_end):
        return "train"
    if timestamp <= pd.Timestamp(boundaries.validation_end):
        return "validation"
    return "locked_test"


def _is_price_feature(column: str) -> bool:
    return column.startswith(PRICE_PREFIXES)


def _is_text_feature(column: str) -> bool:
    lower = column.lower()
    return any(token in lower for token in TEXT_TOKENS)


def _is_regime_feature(column: str) -> bool:
    lower = column.lower()
    return any(token in lower for token in REGIME_TOKENS)
