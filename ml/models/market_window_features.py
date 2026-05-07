"""Feature, target, and rolling window builders for market-state encoders."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Sequence

import numpy as np
import pandas as pd


TRACKED_ASSETS = ("SPY", "TLT", "GLD")
CONTEXT_TICKERS = ("QQQ", "^VIX", "^TNX")
REQUIRED_TICKERS = TRACKED_ASSETS + CONTEXT_TICKERS

TARGET_COLUMNS = ("next_return_spy", "next_return_tlt", "next_return_gld")
DIRECTION_TARGET_COLUMNS = ("next_direction_spy", "next_direction_tlt", "next_direction_gld")
VOLATILITY_TARGET_COLUMNS = ("next_volatility_spy", "next_volatility_tlt", "next_volatility_gld")
DRAWDOWN_TARGET_COLUMNS = ("next_drawdown_spy", "next_drawdown_tlt", "next_drawdown_gld")
BEST_ASSET_COLUMN = "best_asset_next_week"


@dataclass(frozen=True)
class MarketWindowSamples:
    daily_windows: np.ndarray
    weekly_windows: np.ndarray
    return_targets: np.ndarray
    direction_targets: np.ndarray
    volatility_targets: np.ndarray
    drawdown_targets: np.ndarray
    best_asset_targets: np.ndarray
    metadata: pd.DataFrame
    future_returns: pd.DataFrame
    daily_features: tuple[str, ...]
    weekly_features: tuple[str, ...]


def load_prices(path: str) -> pd.DataFrame:
    prices = pd.read_csv(path, parse_dates=["date"]).sort_values(["date", "symbol"])
    missing = sorted(set(REQUIRED_TICKERS) - set(prices["symbol"].unique()))
    if missing:
        raise ValueError(f"Missing required symbols in price data: {missing}")
    return prices


def make_market_feature_frames(prices: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build daily and weekly feature frames indexed by date/week_end."""

    daily_ohlcv = _pivot_ohlcv(prices)
    daily_features = _build_feature_frame(daily_ohlcv, slope_lag=5)
    daily_features = _add_daily_context_features(daily_features, daily_ohlcv)
    daily_features.index.name = "date"

    weekly_ohlcv = _resample_weekly_ohlcv(prices)
    weekly_features = _build_feature_frame(weekly_ohlcv, slope_lag=4)
    weekly_features = _add_weekly_context_features(weekly_features, weekly_ohlcv)
    weekly_features.index.name = "week_end"
    return daily_features, weekly_features


def build_daily_rolling_pretrain_samples(
    prices: pd.DataFrame,
    daily_features: pd.DataFrame,
    weekly_features: pd.DataFrame,
    daily_lookback: int = 120,
    weekly_lookback: int = 52,
    horizon_days: int = 5,
    split_fn: Callable[[pd.Timestamp], str] | None = None,
) -> MarketWindowSamples:
    """Create daily rolling samples for supervised encoder pretraining.

    Each sample ends on one trading day and predicts the next ``horizon_days``
    trading-day outcomes. Weekly features are cut at the latest available
    weekly bar whose timestamp is not after the decision day.
    """

    if horizon_days <= 0:
        raise ValueError("horizon_days must be positive.")

    daily = daily_features.sort_index()
    weekly = weekly_features.sort_index()
    daily_feature_names = tuple(str(column) for column in daily.columns)
    weekly_feature_names = tuple(str(column) for column in weekly.columns)

    ohlcv = _pivot_ohlcv(prices)
    adj_close = ohlcv["adj_close"].loc[:, TRACKED_ASSETS].sort_index()
    log_returns = np.log(adj_close / adj_close.shift(1))

    daily_values = daily.to_numpy(dtype=np.float32)
    weekly_values = weekly.to_numpy(dtype=np.float32)
    daily_index = daily.index.to_numpy()
    weekly_index = weekly.index.to_numpy()

    daily_windows: list[np.ndarray] = []
    weekly_windows: list[np.ndarray] = []
    return_targets: list[np.ndarray] = []
    direction_targets: list[np.ndarray] = []
    volatility_targets: list[np.ndarray] = []
    drawdown_targets: list[np.ndarray] = []
    best_asset_targets: list[int] = []
    metadata_rows: list[dict[str, object]] = []
    future_rows: list[dict[str, object]] = []

    for price_pos, decision_date in enumerate(adj_close.index):
        future_pos = price_pos + horizon_days
        if future_pos >= len(adj_close):
            break

        date64 = np.datetime64(decision_date)
        daily_end = np.searchsorted(daily_index, date64, side="right")
        weekly_end = np.searchsorted(weekly_index, date64, side="right")
        if daily_end < daily_lookback or weekly_end < weekly_lookback:
            continue

        daily_window = daily_values[daily_end - daily_lookback:daily_end]
        weekly_window = weekly_values[weekly_end - weekly_lookback:weekly_end]
        if not np.isfinite(daily_window).all() or not np.isfinite(weekly_window).all():
            continue

        base_price = adj_close.iloc[price_pos]
        future_price = adj_close.iloc[future_pos]
        future_path = adj_close.iloc[price_pos + 1:future_pos + 1]
        future_log_returns = log_returns.iloc[price_pos + 1:future_pos + 1]

        returns = (future_price / base_price - 1.0).to_numpy(dtype=np.float32)
        directions = (returns > 0.0).astype(np.float32)
        volatility = future_log_returns.std(axis=0).to_numpy(dtype=np.float32)
        drawdown = (future_path.min(axis=0) / base_price - 1.0).to_numpy(dtype=np.float32)
        if (
            not np.isfinite(returns).all()
            or not np.isfinite(directions).all()
            or not np.isfinite(volatility).all()
            or not np.isfinite(drawdown).all()
        ):
            continue

        sample_id = len(daily_windows)
        best_asset = int(np.argmax(returns))
        split = split_fn(pd.Timestamp(decision_date)) if split_fn is not None else "unknown"

        daily_windows.append(daily_window)
        weekly_windows.append(weekly_window)
        return_targets.append(returns)
        direction_targets.append(directions)
        volatility_targets.append(volatility)
        drawdown_targets.append(drawdown)
        best_asset_targets.append(best_asset)
        metadata_rows.append(
            {
                "sample_id": sample_id,
                "decision_date": pd.Timestamp(decision_date),
                "week_end": pd.Timestamp(decision_date).to_period("W-FRI").end_time.normalize(),
                "daily_window_end": pd.Timestamp(daily.index[daily_end - 1]),
                "weekly_window_end": pd.Timestamp(weekly.index[weekly_end - 1]),
                "ticker": "MARKET",
                "split": split,
                "sample_type": "daily_pretrain",
            }
        )
        future_rows.append(
            _future_target_row(
                sample_id=sample_id,
                returns=returns,
                directions=directions,
                volatility=volatility,
                drawdown=drawdown,
                best_asset=best_asset,
            )
        )

    return _make_samples(
        daily_windows=daily_windows,
        weekly_windows=weekly_windows,
        return_targets=return_targets,
        direction_targets=direction_targets,
        volatility_targets=volatility_targets,
        drawdown_targets=drawdown_targets,
        best_asset_targets=best_asset_targets,
        metadata_rows=metadata_rows,
        future_rows=future_rows,
        daily_feature_names=daily_feature_names,
        weekly_feature_names=weekly_feature_names,
        empty_message="No valid daily rolling pretrain samples were created.",
    )


def build_weekly_window_export_samples(
    state_frame: pd.DataFrame,
    daily_features: pd.DataFrame,
    weekly_features: pd.DataFrame,
    daily_lookback: int = 120,
    weekly_lookback: int = 52,
    split_column: str = "eval_split",
) -> MarketWindowSamples:
    """Create weekly windows aligned to the RL state rows for embedding export."""

    required_state_cols = {"week_end", "week_last_trade_date", *TARGET_COLUMNS}
    missing = required_state_cols - set(state_frame.columns)
    if missing:
        raise ValueError(f"Missing required state columns: {sorted(missing)}")

    frame = state_frame.copy()
    frame["week_end"] = pd.to_datetime(frame["week_end"])
    frame["week_last_trade_date"] = pd.to_datetime(frame["week_last_trade_date"])
    frame = frame.sort_values("week_end").reset_index(drop=True)

    daily = daily_features.sort_index()
    weekly = weekly_features.sort_index()
    daily_feature_names = tuple(str(column) for column in daily.columns)
    weekly_feature_names = tuple(str(column) for column in weekly.columns)

    daily_values = daily.to_numpy(dtype=np.float32)
    weekly_values = weekly.to_numpy(dtype=np.float32)
    daily_index = daily.index.to_numpy()
    weekly_index = weekly.index.to_numpy()

    daily_windows: list[np.ndarray] = []
    weekly_windows: list[np.ndarray] = []
    return_targets: list[np.ndarray] = []
    direction_targets: list[np.ndarray] = []
    volatility_targets: list[np.ndarray] = []
    drawdown_targets: list[np.ndarray] = []
    best_asset_targets: list[int] = []
    metadata_rows: list[dict[str, object]] = []
    future_rows: list[dict[str, object]] = []

    for _, row in frame.iterrows():
        target = row.loc[list(TARGET_COLUMNS)].to_numpy(dtype=np.float32)
        if not np.isfinite(target).all():
            continue

        daily_end = np.searchsorted(daily_index, np.datetime64(row["week_last_trade_date"]), side="right")
        weekly_end = np.searchsorted(weekly_index, np.datetime64(row["week_end"]), side="right")
        if daily_end < daily_lookback or weekly_end < weekly_lookback:
            continue

        daily_window = daily_values[daily_end - daily_lookback:daily_end]
        weekly_window = weekly_values[weekly_end - weekly_lookback:weekly_end]
        if not np.isfinite(daily_window).all() or not np.isfinite(weekly_window).all():
            continue

        sample_id = len(daily_windows)
        directions = (target > 0.0).astype(np.float32)
        volatility = np.zeros_like(target, dtype=np.float32)
        drawdown = np.zeros_like(target, dtype=np.float32)
        best_asset = int(np.argmax(target))
        split = str(row[split_column]) if split_column in frame.columns else "unknown"

        daily_windows.append(daily_window)
        weekly_windows.append(weekly_window)
        return_targets.append(target)
        direction_targets.append(directions)
        volatility_targets.append(volatility)
        drawdown_targets.append(drawdown)
        best_asset_targets.append(best_asset)
        metadata_rows.append(
            {
                "sample_id": sample_id,
                "week_end": row["week_end"],
                "week_last_trade_date": row["week_last_trade_date"],
                "daily_window_end": pd.Timestamp(daily.index[daily_end - 1]),
                "weekly_window_end": pd.Timestamp(weekly.index[weekly_end - 1]),
                "ticker": "MARKET",
                "split": split,
                "sample_type": "weekly_export",
            }
        )
        future_row: dict[str, object] = {"sample_id": sample_id}
        for column, value in zip(TARGET_COLUMNS, target):
            future_row[column] = float(value)
        future_rows.append(future_row)

    return _make_samples(
        daily_windows=daily_windows,
        weekly_windows=weekly_windows,
        return_targets=return_targets,
        direction_targets=direction_targets,
        volatility_targets=volatility_targets,
        drawdown_targets=drawdown_targets,
        best_asset_targets=best_asset_targets,
        metadata_rows=metadata_rows,
        future_rows=future_rows,
        daily_feature_names=daily_feature_names,
        weekly_feature_names=weekly_feature_names,
        empty_message="No valid weekly export samples were created.",
    )


def build_window_samples(*args, **kwargs) -> MarketWindowSamples:
    """Backward-compatible alias for weekly export sample construction."""

    return build_weekly_window_export_samples(*args, **kwargs)


def _make_samples(
    daily_windows: list[np.ndarray],
    weekly_windows: list[np.ndarray],
    return_targets: list[np.ndarray],
    direction_targets: list[np.ndarray],
    volatility_targets: list[np.ndarray],
    drawdown_targets: list[np.ndarray],
    best_asset_targets: list[int],
    metadata_rows: list[dict[str, object]],
    future_rows: list[dict[str, object]],
    daily_feature_names: tuple[str, ...],
    weekly_feature_names: tuple[str, ...],
    empty_message: str,
) -> MarketWindowSamples:
    if not daily_windows:
        raise ValueError(empty_message)

    return MarketWindowSamples(
        daily_windows=np.stack(daily_windows).astype(np.float32),
        weekly_windows=np.stack(weekly_windows).astype(np.float32),
        return_targets=np.stack(return_targets).astype(np.float32),
        direction_targets=np.stack(direction_targets).astype(np.float32),
        volatility_targets=np.stack(volatility_targets).astype(np.float32),
        drawdown_targets=np.stack(drawdown_targets).astype(np.float32),
        best_asset_targets=np.asarray(best_asset_targets, dtype=np.int64),
        metadata=pd.DataFrame(metadata_rows),
        future_returns=pd.DataFrame(future_rows),
        daily_features=daily_feature_names,
        weekly_features=weekly_feature_names,
    )


def _future_target_row(
    sample_id: int,
    returns: np.ndarray,
    directions: np.ndarray,
    volatility: np.ndarray,
    drawdown: np.ndarray,
    best_asset: int,
) -> dict[str, object]:
    row: dict[str, object] = {"sample_id": sample_id}
    for column, value in zip(TARGET_COLUMNS, returns):
        row[column] = float(value)
    for column, value in zip(DIRECTION_TARGET_COLUMNS, directions):
        row[column] = float(value)
    for column, value in zip(VOLATILITY_TARGET_COLUMNS, volatility):
        row[column] = float(value)
    for column, value in zip(DRAWDOWN_TARGET_COLUMNS, drawdown):
        row[column] = float(value)
    row[BEST_ASSET_COLUMN] = TRACKED_ASSETS[best_asset]
    return row


def _pivot_ohlcv(prices: pd.DataFrame) -> dict[str, pd.DataFrame]:
    out: dict[str, pd.DataFrame] = {}
    for column in ("open", "high", "low", "close", "adj_close", "volume"):
        out[column] = (
            prices.pivot(index="date", columns="symbol", values=column)
            .sort_index()
            .reindex(columns=REQUIRED_TICKERS)
            .astype(float)
        )
    return out


def _resample_weekly_ohlcv(prices: pd.DataFrame) -> dict[str, pd.DataFrame]:
    frames: dict[str, list[pd.Series]] = {
        "open": [],
        "high": [],
        "low": [],
        "close": [],
        "adj_close": [],
        "volume": [],
    }
    for symbol, group in prices.sort_values("date").groupby("symbol"):
        indexed = group.set_index("date")
        frames["open"].append(indexed["open"].resample("W-FRI").first().rename(symbol))
        frames["high"].append(indexed["high"].resample("W-FRI").max().rename(symbol))
        frames["low"].append(indexed["low"].resample("W-FRI").min().rename(symbol))
        frames["close"].append(indexed["close"].resample("W-FRI").last().rename(symbol))
        frames["adj_close"].append(indexed["adj_close"].resample("W-FRI").last().rename(symbol))
        frames["volume"].append(indexed["volume"].resample("W-FRI").sum().rename(symbol))

    return {
        name: pd.concat(series_list, axis=1).sort_index().reindex(columns=REQUIRED_TICKERS).astype(float)
        for name, series_list in frames.items()
    }


def _build_feature_frame(ohlcv: dict[str, pd.DataFrame], slope_lag: int) -> pd.DataFrame:
    close = ohlcv["close"]
    adj_close = ohlcv["adj_close"]
    open_ = ohlcv["open"]
    high = ohlcv["high"]
    low = ohlcv["low"]
    volume = ohlcv["volume"]

    log_returns = np.log(adj_close / adj_close.shift(1))
    features = pd.DataFrame(index=adj_close.index)

    for symbol in TRACKED_ASSETS:
        prefix = symbol.lower()
        symbol_close = close[symbol]
        symbol_adj = adj_close[symbol]
        symbol_open = open_[symbol]
        symbol_high = high[symbol]
        symbol_low = low[symbol]
        symbol_volume = volume[symbol]
        symbol_log_return = log_returns[symbol]

        features[f"{prefix}_log_return_1d"] = symbol_log_return
        for window in (5, 10, 20, 60, 120):
            features[f"{prefix}_return_{window}d"] = symbol_adj.pct_change(window, fill_method=None)

        candle_range = (symbol_high - symbol_low).replace(0, np.nan)
        features[f"{prefix}_body"] = (symbol_close - symbol_open) / symbol_close
        features[f"{prefix}_range"] = (symbol_high - symbol_low) / symbol_close
        features[f"{prefix}_upper_wick"] = (symbol_high - np.maximum(symbol_open, symbol_close)) / symbol_close
        features[f"{prefix}_lower_wick"] = (np.minimum(symbol_open, symbol_close) - symbol_low) / symbol_close
        features[f"{prefix}_close_pos"] = (symbol_close - symbol_low) / candle_range
        features[f"{prefix}_gap"] = symbol_open / symbol_close.shift(1) - 1.0

        for window in (5, 20, 60):
            features[f"{prefix}_volatility_{window}d"] = symbol_log_return.rolling(window).std()
        true_range = pd.concat(
            [
                symbol_high - symbol_low,
                (symbol_high - symbol_close.shift(1)).abs(),
                (symbol_low - symbol_close.shift(1)).abs(),
            ],
            axis=1,
        ).max(axis=1)
        features[f"{prefix}_ATR_14d"] = true_range.rolling(14).mean() / symbol_close
        features[f"{prefix}_max_drawdown_20d"] = symbol_close / symbol_close.rolling(20).max() - 1.0
        downside = symbol_log_return.where(symbol_log_return < 0.0, 0.0)
        features[f"{prefix}_downside_volatility_20d"] = downside.rolling(20).std()

        volume_log = np.log1p(symbol_volume)
        dollar_volume_log = np.log1p(symbol_close * symbol_volume)
        features[f"{prefix}_volume_ratio_20d"] = symbol_volume / symbol_volume.rolling(20).mean() - 1.0
        features[f"{prefix}_volume_zscore_20d"] = _rolling_zscore(volume_log, 20)
        features[f"{prefix}_dollar_volume"] = dollar_volume_log
        features[f"{prefix}_dollar_volume_zscore"] = _rolling_zscore(dollar_volume_log, 20)

        ma20 = symbol_close.rolling(20).mean()
        ma60 = symbol_close.rolling(60).mean()
        ma120 = symbol_close.rolling(120).mean()
        features[f"{prefix}_close_to_MA20"] = symbol_close / ma20 - 1.0
        features[f"{prefix}_close_to_MA60"] = symbol_close / ma60 - 1.0
        features[f"{prefix}_close_to_MA120"] = symbol_close / ma120 - 1.0
        features[f"{prefix}_MA20_slope"] = ma20 / ma20.shift(slope_lag) - 1.0
        features[f"{prefix}_MA60_slope"] = ma60 / ma60.shift(slope_lag) - 1.0
        features[f"{prefix}_RSI"] = _rsi(symbol_close, 14)
        features[f"{prefix}_MACD_histogram"] = _macd_histogram(symbol_close) / symbol_close

    return features


def _add_daily_context_features(
    features: pd.DataFrame,
    ohlcv: dict[str, pd.DataFrame],
) -> pd.DataFrame:
    adj_close = ohlcv["adj_close"]
    log_returns = np.log(adj_close / adj_close.shift(1))
    out = features.copy()
    qqq_spy_ratio = np.log(adj_close["QQQ"] / adj_close["SPY"])
    out["context_qqq_spy_log_ratio"] = qqq_spy_ratio
    out["context_qqq_spy_ratio_chg_5d"] = qqq_spy_ratio.diff(5)
    out["context_spy_tlt_corr_20d"] = log_returns["SPY"].rolling(20).corr(log_returns["TLT"])
    out["context_spy_gld_corr_20d"] = log_returns["SPY"].rolling(20).corr(log_returns["GLD"])
    out["context_vix_level"] = adj_close["^VIX"]
    out["context_vix_change_5d"] = adj_close["^VIX"].diff(5)
    out["context_tnx_level"] = adj_close["^TNX"]
    out["context_tnx_change_5d"] = adj_close["^TNX"].diff(5)
    return out


def _add_weekly_context_features(
    features: pd.DataFrame,
    ohlcv: dict[str, pd.DataFrame],
) -> pd.DataFrame:
    adj_close = ohlcv["adj_close"]
    log_returns = np.log(adj_close / adj_close.shift(1))
    out = features.copy()
    qqq_spy_ratio = np.log(adj_close["QQQ"] / adj_close["SPY"])
    out["context_qqq_spy_log_ratio"] = qqq_spy_ratio
    out["context_qqq_spy_ratio_chg_5d"] = qqq_spy_ratio.diff(5)
    out["context_spy_tlt_corr_20d"] = log_returns["SPY"].rolling(20).corr(log_returns["TLT"])
    out["context_spy_gld_corr_20d"] = log_returns["SPY"].rolling(20).corr(log_returns["GLD"])
    out["context_vix_level"] = adj_close["^VIX"]
    out["context_vix_change_5d"] = adj_close["^VIX"].diff(5)
    out["context_tnx_level"] = adj_close["^TNX"]
    out["context_tnx_change_5d"] = adj_close["^TNX"].diff(5)
    return out


def _rolling_zscore(series: pd.Series, window: int) -> pd.Series:
    mean = series.rolling(window).mean()
    std = series.rolling(window).std()
    return (series - mean) / std.replace(0, np.nan)


def _rsi(close: pd.Series, window: int = 14) -> pd.Series:
    delta = close.diff()
    gains = delta.clip(lower=0).rolling(window).mean()
    losses = (-delta.clip(upper=0)).rolling(window).mean()
    rs = gains / losses.replace(0, np.nan)
    return 100.0 - (100.0 / (1.0 + rs))


def _macd_histogram(close: pd.Series) -> pd.Series:
    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    macd = ema12 - ema26
    signal = macd.ewm(span=9, adjust=False).mean()
    return macd - signal


def assert_no_target_columns(columns: Sequence[str]) -> None:
    leaked = [
        str(column)
        for column in columns
        if str(column).startswith(("next_return_", "next_direction_", "next_volatility_", "next_drawdown_"))
        or str(column) == BEST_ASSET_COLUMN
    ]
    if leaked:
        raise ValueError(f"Target columns leaked into features: {sorted(leaked)}")

