#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import re
from pathlib import Path

import pandas as pd
import torch

os.environ.setdefault("USE_TF", "0")
from transformers import AutoModelForSequenceClassification, AutoTokenizer, pipeline

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = ROOT / "data" / "raw" / "news" / "all_assets_news_weekly.csv"
DEFAULT_OUTPUT = ROOT / "data" / "raw" / "news_sentiment" / "all_assets_news_weekly_finbert.csv"
MODEL_NAME = "ProsusAI/finbert"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Score weekly news sentiment with FinBERT.")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--batch-size", type=int, default=16)
    return parser.parse_args()


def clean_text(value: object) -> str:
    return re.sub(r"\s+", " ", str(value or "")).strip()


def normalize_text(value: object) -> str:
    text = clean_text(value).lower()
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def strip_publisher_suffix(text: str, publisher: str) -> str:
    text = clean_text(text)
    publisher = clean_text(publisher)
    if not text or not publisher:
        return text

    patterns = [
        rf"\s*(?:-|:|\u2013|\u2014)\s*{re.escape(publisher)}$",
        rf"\s+{re.escape(publisher)}$",
    ]
    for pattern in patterns:
        text = re.sub(pattern, "", text, flags=re.IGNORECASE).strip()
    return text


def choose_finbert_text(title: str, description: str, summary: str, publisher: str) -> str:
    title = strip_publisher_suffix(title, publisher)
    description = strip_publisher_suffix(description, publisher)
    summary = strip_publisher_suffix(summary, publisher)

    title_norm = normalize_text(title)
    description_norm = normalize_text(description)
    summary_norm = normalize_text(summary)

    if summary and summary_norm != title_norm:
        return summary
    if description and description_norm != title_norm:
        return description
    return title or description or summary


def prepare_news(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for column in ["title", "description", "summary", "publisher"]:
        if column not in df.columns:
            df[column] = ""
        df[column] = df[column].fillna("").astype(str)

    df["finbert_text"] = [
        choose_finbert_text(row.title, row.description, row.summary, row.publisher)
        for row in df.itertuples(index=False)
    ]
    return df


def run_finbert(texts: list[str], batch_size: int) -> pd.DataFrame:
    output = pd.DataFrame(
        {
            "finbert_label": [None] * len(texts),
            "finbert_score": [None] * len(texts),
            "finbert_positive": [0.0] * len(texts),
            "finbert_neutral": [0.0] * len(texts),
            "finbert_negative": [0.0] * len(texts),
            "finbert_compound": [0.0] * len(texts),
        }
    )

    indexed_texts = [(idx, text) for idx, text in enumerate(texts) if clean_text(text)]
    if not indexed_texts:
        return output

    device = 0 if torch.cuda.is_available() else -1
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
    classifier = pipeline(
        "text-classification",
        model=model,
        tokenizer=tokenizer,
        device=device,
        framework="pt",
    )

    results = classifier(
        [text for _, text in indexed_texts],
        batch_size=batch_size,
        truncation=True,
        max_length=256,
        top_k=None,
    )

    for (row_idx, _), row_scores in zip(indexed_texts, results):
        score_map = {item["label"].lower(): float(item["score"]) for item in row_scores}
        positive = score_map.get("positive", 0.0)
        neutral = score_map.get("neutral", 0.0)
        negative = score_map.get("negative", 0.0)
        label = max(score_map, key=score_map.get) if score_map else None

        output.at[row_idx, "finbert_label"] = label
        output.at[row_idx, "finbert_score"] = score_map.get(label) if label else None
        output.at[row_idx, "finbert_positive"] = positive
        output.at[row_idx, "finbert_neutral"] = neutral
        output.at[row_idx, "finbert_negative"] = negative
        output.at[row_idx, "finbert_compound"] = positive - negative

    return output


def main() -> None:
    args = parse_args()
    df = pd.read_csv(args.input)
    df = prepare_news(df)
    sentiment_df = run_finbert(df["finbert_text"].tolist(), args.batch_size)

    for column in sentiment_df.columns:
        df[column] = sentiment_df[column]

    args.output.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.output, index=False)

    print("Done.")
    print("Saved to:", args.output)
    print(df[["asset", "title", "finbert_text", "finbert_label", "finbert_score"]].head(10))


if __name__ == "__main__":
    main()
