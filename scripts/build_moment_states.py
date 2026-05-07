#!/usr/bin/env python3
"""Export RL-ready state embeddings from a pretrained MOMENT model."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import torch
from sklearn.decomposition import PCA
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from evaluation.config import SplitBoundaries
from ml.models.market_window_features import (
    assert_no_target_columns,
    build_weekly_window_export_samples,
    load_prices,
    make_market_feature_frames,
)


DEFAULT_PRICE_PATH = ROOT / "data" / "raw" / "yahoo_prices_daily.csv"
DEFAULT_PRICE_MACRO_STATE = ROOT / "data" / "processed" / "model_state_weekly_price_macro.csv"
DEFAULT_HMM_NEWS_STATE = ROOT / "output" / "full_pipeline" / "model_state_weekly_hmm_news.csv"
DEFAULT_OUTPUT_DIR = ROOT / "data" / "processed" / "moment_states"
DEFAULT_MERGED_OUTPUT = ROOT / "output" / "full_pipeline" / "model_state_weekly_hmm_news_moment.csv"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Encode weekly RL state rows with a pretrained MOMENT time-series model."
    )
    parser.add_argument("--price-path", type=Path, default=DEFAULT_PRICE_PATH)
    parser.add_argument("--state-path", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--merged-output", type=Path, default=DEFAULT_MERGED_OUTPUT)
    parser.add_argument("--model-id", type=str, default="AutonLab/MOMENT-1-small")
    parser.add_argument("--daily-lookback", type=int, default=120)
    parser.add_argument("--weekly-lookback", type=int, default=52)
    parser.add_argument("--moment-context-length", type=int, default=512)
    parser.add_argument("--embedding-dim", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument(
        "--precision",
        choices=("auto", "fp32", "fp16"),
        default="auto",
        help="Use fp16 autocast on CUDA by default to reduce GPU memory.",
    )
    parser.add_argument("--device", type=str, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))

    state_path = args.state_path or default_state_path()
    state = pd.read_csv(state_path, parse_dates=["week_end", "week_last_trade_date"], low_memory=False)
    state = ensure_eval_split(state)

    prices = load_prices(str(args.price_path))
    daily_features, weekly_features = make_market_feature_frames(prices)
    assert_no_target_columns(daily_features.columns)
    assert_no_target_columns(weekly_features.columns)

    samples = build_weekly_window_export_samples(
        state_frame=state,
        daily_features=daily_features,
        weekly_features=weekly_features,
        daily_lookback=args.daily_lookback,
        weekly_lookback=args.weekly_lookback,
    )
    daily_scaled, weekly_scaled, scaler_payload = scale_windows(samples)

    precision = resolve_precision(args.precision, device)
    moment = load_moment_model(args.model_id, n_channels=daily_scaled.shape[-1], device=device)
    daily_embeddings = encode_windows(
        moment,
        daily_scaled,
        context_length=args.moment_context_length,
        batch_size=args.batch_size,
        device=device,
        precision=precision,
    )
    weekly_embeddings = encode_windows(
        moment,
        weekly_scaled,
        context_length=args.moment_context_length,
        batch_size=args.batch_size,
        device=device,
        precision=precision,
    )
    raw_embeddings = np.concatenate([daily_embeddings, weekly_embeddings], axis=1).astype(np.float32)
    z_states, reducer_payload = reduce_embeddings(
        raw_embeddings,
        samples.metadata["split"].eq("train").to_numpy(),
        output_dim=args.embedding_dim,
    )

    paths = write_artifacts(
        args=args,
        state_path=state_path,
        samples=samples,
        z_states=z_states,
        raw_dim=raw_embeddings.shape[1],
        scaler_payload=scaler_payload,
        reducer_payload=reducer_payload,
    )
    write_merged_state(state, samples.metadata, z_states, args.merged_output)

    for name, path in paths.items():
        print(f"Saved {name} -> {path}")
    print(f"Saved merged_state -> {args.merged_output}")
    print(f"MOMENT states: rows={len(samples.metadata)} embedding_dim={z_states.shape[1]}")


def default_state_path() -> Path:
    if DEFAULT_HMM_NEWS_STATE.exists():
        return DEFAULT_HMM_NEWS_STATE
    return DEFAULT_PRICE_MACRO_STATE


def ensure_eval_split(frame: pd.DataFrame, boundaries: SplitBoundaries = SplitBoundaries()) -> pd.DataFrame:
    out = frame.sort_values("week_end").reset_index(drop=True).copy()
    if "eval_split" not in out.columns:
        out["eval_split"] = out["week_end"].apply(lambda value: label_split(value, boundaries))
    return out


def label_split(timestamp: pd.Timestamp, boundaries: SplitBoundaries) -> str:
    if timestamp <= pd.Timestamp(boundaries.warmup_end):
        return "warmup"
    if timestamp <= pd.Timestamp(boundaries.train_end):
        return "train"
    if timestamp <= pd.Timestamp(boundaries.validation_end):
        return "validation"
    return "locked_test"


def scale_windows(samples: Any) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    train_mask = samples.metadata["split"].eq("train").to_numpy()
    if not train_mask.any():
        train_mask = np.ones(len(samples.metadata), dtype=bool)

    daily_scaler = StandardScaler()
    weekly_scaler = StandardScaler()
    daily_shape = samples.daily_windows.shape
    weekly_shape = samples.weekly_windows.shape

    daily_scaler.fit(samples.daily_windows[train_mask].reshape(-1, daily_shape[-1]))
    weekly_scaler.fit(samples.weekly_windows[train_mask].reshape(-1, weekly_shape[-1]))

    daily_scaled = daily_scaler.transform(samples.daily_windows.reshape(-1, daily_shape[-1]))
    weekly_scaled = weekly_scaler.transform(samples.weekly_windows.reshape(-1, weekly_shape[-1]))
    daily_scaled = daily_scaled.reshape(daily_shape).astype(np.float32)
    weekly_scaled = weekly_scaled.reshape(weekly_shape).astype(np.float32)

    payload = {
        "daily_mean": daily_scaler.mean_.tolist(),
        "daily_scale": daily_scaler.scale_.tolist(),
        "weekly_mean": weekly_scaler.mean_.tolist(),
        "weekly_scale": weekly_scaler.scale_.tolist(),
        "fit_on": "weekly_rl_train_windows_only",
    }
    return daily_scaled, weekly_scaled, payload


def load_moment_model(model_id: str, n_channels: int, device: torch.device) -> Any:
    try:
        from momentfm import MOMENTPipeline
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "momentfm is required for MOMENT embeddings. Install it with `pip install momentfm`."
        ) from exc

    try:
        model = MOMENTPipeline.from_pretrained(
            model_id,
            model_kwargs={"task_name": "embedding", "n_channels": n_channels},
        )
    except TypeError:
        model = MOMENTPipeline.from_pretrained(model_id, model_kwargs={"task_name": "embedding"})
    model.init()
    model.to(device)
    model.eval()
    return model


def resolve_precision(requested: str, device: torch.device) -> str:
    if requested == "auto":
        return "fp16" if device.type == "cuda" else "fp32"
    if requested == "fp16" and device.type != "cuda":
        raise ValueError("--precision fp16 is only supported on CUDA.")
    return requested


def encode_windows(
    model: Any,
    windows: np.ndarray,
    context_length: int,
    batch_size: int,
    device: torch.device,
    precision: str,
) -> np.ndarray:
    outputs = []
    with torch.no_grad():
        for start in range(0, len(windows), batch_size):
            batch = windows[start:start + batch_size]
            x = prepare_moment_input(batch, context_length=context_length).to(device)
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=precision == "fp16"):
                encoded = model(x_enc=x)
            outputs.append(extract_embedding_tensor(encoded).detach().cpu().numpy())
    return np.concatenate(outputs, axis=0).astype(np.float32)


def prepare_moment_input(windows: np.ndarray, context_length: int) -> torch.Tensor:
    x = np.transpose(windows, (0, 2, 1)).astype(np.float32)
    seq_len = x.shape[-1]
    if seq_len > context_length:
        x = x[:, :, -context_length:]
    elif seq_len < context_length:
        pad_width = context_length - seq_len
        x = np.pad(x, ((0, 0), (0, 0), (pad_width, 0)), mode="constant")
    return torch.from_numpy(x)


def extract_embedding_tensor(output: Any) -> torch.Tensor:
    for name in ("embeddings", "embedding", "last_hidden_state"):
        value = getattr(output, name, None)
        if isinstance(value, torch.Tensor):
            return reduce_embedding_shape(value)
    if isinstance(output, torch.Tensor):
        return reduce_embedding_shape(output)
    if isinstance(output, dict):
        for name in ("embeddings", "embedding", "last_hidden_state"):
            value = output.get(name)
            if isinstance(value, torch.Tensor):
                return reduce_embedding_shape(value)
    raise TypeError(f"Could not find an embedding tensor in MOMENT output type {type(output)!r}.")


def reduce_embedding_shape(embedding: torch.Tensor) -> torch.Tensor:
    if embedding.ndim == 2:
        return embedding
    if embedding.ndim < 2:
        raise ValueError(f"Unexpected embedding shape: {tuple(embedding.shape)}")
    reduce_dims = tuple(range(1, embedding.ndim - 1))
    return embedding.mean(dim=reduce_dims)


def reduce_embeddings(
    raw_embeddings: np.ndarray,
    train_mask: np.ndarray,
    output_dim: int,
) -> tuple[np.ndarray, dict[str, Any]]:
    if not train_mask.any():
        train_mask = np.ones(len(raw_embeddings), dtype=bool)

    output_dim = min(output_dim, raw_embeddings.shape[1], int(train_mask.sum()))
    reducer = make_pipeline(StandardScaler(), PCA(n_components=output_dim, random_state=0))
    reducer.fit(raw_embeddings[train_mask])
    z_states = reducer.transform(raw_embeddings).astype(np.float32)

    pca = reducer.named_steps["pca"]
    payload = {
        "method": "standard_scaler_plus_pca",
        "fit_on": "weekly_rl_train_embeddings_only",
        "raw_embedding_dim": int(raw_embeddings.shape[1]),
        "output_dim": int(output_dim),
        "explained_variance_ratio_sum": float(np.sum(pca.explained_variance_ratio_)),
    }
    return z_states, payload


def write_artifacts(
    args: argparse.Namespace,
    state_path: Path,
    samples: Any,
    z_states: np.ndarray,
    raw_dim: int,
    scaler_payload: dict[str, Any],
    reducer_payload: dict[str, Any],
) -> dict[str, Path]:
    args.output_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "z_states": args.output_dir / "z_states.npy",
        "metadata": args.output_dir / "metadata.parquet",
        "future_returns": args.output_dir / "future_returns.parquet",
        "feature_config": args.output_dir / "feature_config.json",
        "encoding_metrics": args.output_dir / "encoding_metrics.json",
    }

    np.save(paths["z_states"], z_states)
    samples.metadata.to_parquet(paths["metadata"], index=False)
    samples.future_returns.to_parquet(paths["future_returns"], index=False)

    feature_config = {
        "encoder": "moment",
        "model_id": args.model_id,
        "state_path": str(state_path),
        "daily_lookback": args.daily_lookback,
        "weekly_lookback": args.weekly_lookback,
        "moment_context_length": args.moment_context_length,
        "precision": args.precision,
        "daily_features": list(samples.daily_features),
        "weekly_features": list(samples.weekly_features),
        "raw_embedding_dim": int(raw_dim),
        "embedding_dim": int(z_states.shape[1]),
        "scalers": scaler_payload,
        "reducer": reducer_payload,
        "num_export_samples": int(len(samples.metadata)),
    }
    paths["feature_config"].write_text(json.dumps(feature_config, indent=2), encoding="utf-8")

    metrics = {
        "encoder": "moment",
        "model_id": args.model_id,
        "num_export_samples": int(len(samples.metadata)),
        "z_shape": list(z_states.shape),
        "split_counts": samples.metadata["split"].value_counts().to_dict(),
        "reducer": reducer_payload,
    }
    paths["encoding_metrics"].write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    return paths


def write_merged_state(
    state: pd.DataFrame,
    metadata: pd.DataFrame,
    z_states: np.ndarray,
    output_path: Path,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    z_columns = [f"moment_{index:03d}" for index in range(z_states.shape[1])]
    embedding_frame = pd.concat(
        [
            metadata[["week_end"]].reset_index(drop=True),
            pd.DataFrame(z_states, columns=z_columns),
        ],
        axis=1,
    )

    merged = state.drop(columns=[column for column in state.columns if column.startswith("moment_")])
    merged = merged.merge(embedding_frame, on="week_end", how="inner")
    merged.to_csv(output_path, index=False)


if __name__ == "__main__":
    main()
