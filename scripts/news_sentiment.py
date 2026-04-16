import pandas as pd
import torch
from transformers import pipeline

input_csv = "../data/raw/news/all_assets_news_weekly.csv"
output_csv = "../data/raw/news_sentiment/all_assets_news_weekly_finbert.csv"

df = pd.read_csv(input_csv)

# Fill missing values
for col in ["title", "description", "summary"]:
    df[col] = df.get(col, "").fillna("").astype(str)

df["summary"] = df["summary"].where(df["summary"].str.strip() != "", df["description"])
df["summary"] = df["summary"].where(df["summary"].str.strip() != "", df["title"])

df["summary"] = df["summary"].str.replace(r"\s+", " ", regex=True).str.strip()

# MODEL
device = 0 if torch.cuda.is_available() else -1

classifier = pipeline(
    "text-classification",
    model="ProsusAI/finbert",
    tokenizer="ProsusAI/finbert",
    device=device
)

texts = df["summary"].tolist()

# Main prediction
results = classifier(
    texts,
    batch_size=16,
    truncation=True,
    max_length=512
)

df["finbert_label"] = [r["label"].lower() for r in results]
df["finbert_score"] = [r["score"] for r in results]

# Get all class scores
all_scores = classifier(
    texts,
    batch_size=16,
    truncation=True,
    max_length=512,
    top_k=None
)

score_map_list = []
for row in all_scores:
    row_map = {item["label"].lower(): item["score"] for item in row}
    score_map_list.append(row_map)

score_df = pd.DataFrame(score_map_list)

for col in ["positive", "neutral", "negative"]:
    if col not in score_df.columns:
        score_df[col] = 0.0

df["finbert_positive"] = score_df["positive"]
df["finbert_neutral"] = score_df["neutral"]
df["finbert_negative"] = score_df["negative"]

df.to_csv(output_csv, index=False)

print("Done.")
print("Saved to:", output_csv)
print(df[["title", "summary", "finbert_label", "finbert_score"]].head(10))