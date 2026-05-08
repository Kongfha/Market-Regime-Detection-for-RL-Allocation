#!/usr/bin/env python3
from __future__ import annotations

import argparse
import io
import zipfile
from pathlib import Path

import pandas as pd
import requests

ROOT = Path(__file__).resolve().parents[1]
ZIP_URL = (
    "https://huggingface.co/datasets/takala/financial_phrasebank/"
    "resolve/main/data/FinancialPhraseBank-v1.0.zip"
)

AGREEMENT_FILE_MAP = {
    "Sentences_AllAgree.txt": "all_agree",
    "Sentences_75Agree.txt": "agree_75",
    "Sentences_66Agree.txt": "agree_66",
    "Sentences_50Agree.txt": "agree_50",
}


def parse_label_file(file_name: str, payload: bytes) -> pd.DataFrame:
    rows: list[dict[str, str]] = []
    agreement_level = AGREEMENT_FILE_MAP[file_name]
    for raw_line in payload.decode("latin1").splitlines():
        line = raw_line.strip()
        if not line:
            continue
        sentence, label = line.rsplit("@", 1)
        rows.append(
            {
                "agreement_level": agreement_level,
                "sentence": sentence.strip(),
                "label": label.strip(),
                "source": "huggingface_takala_financial_phrasebank",
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Fetch Financial PhraseBank from Hugging Face and convert it to CSV files."
    )
    args = parser.parse_args()

    raw_dir = ROOT / "data" / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)

    response = requests.get(ZIP_URL, timeout=60)
    response.raise_for_status()

    zip_path = raw_dir / "FinancialPhraseBank-v1.0.zip"
    zip_path.write_bytes(response.content)

    archive = zipfile.ZipFile(io.BytesIO(response.content))
    frames: list[pd.DataFrame] = []
    for member in archive.namelist():
        file_name = Path(member).name
        if file_name not in AGREEMENT_FILE_MAP:
            continue

        frame = parse_label_file(file_name, archive.read(member))
        frames.append(frame)

        split_path = raw_dir / f"financial_phrasebank_{AGREEMENT_FILE_MAP[file_name]}.csv"
        frame.to_csv(split_path, index=False)
        print(f"Saved {len(frame)} rows to {split_path}")

    combined = pd.concat(frames, ignore_index=True).sort_values(
        ["agreement_level", "label", "sentence"]
    )
    combined_path = raw_dir / "financial_phrasebank_combined.csv"
    combined.to_csv(combined_path, index=False)

    print(f"Saved archive to {zip_path}")
    print(f"Saved {len(combined)} combined rows to {combined_path}")


if __name__ == "__main__":
    main()
