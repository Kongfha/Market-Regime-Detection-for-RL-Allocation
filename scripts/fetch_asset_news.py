#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import html
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date, timedelta
from difflib import SequenceMatcher
from email.utils import parsedate_to_datetime
from pathlib import Path
from threading import Lock
from urllib.parse import quote, urljoin, urlparse

import requests
from bs4 import BeautifulSoup
from gnews import GNews
from newspaper import Article, Config

ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = ROOT / "data" / "raw" / "news"

DEFAULT_SLEEP_SECONDS = 1.0
DEFAULT_MAX_WORKERS = 2
MAX_RESULTS = 10
MAX_SUMMARY_TRIES = 2
MAX_NEWS_RETRIES = 2
PROGRESS_INTERVAL_SECONDS = 10.0
REQUEST_TIMEOUT = 8
REQUEST_HEADERS = {"User-Agent": "Mozilla/5.0"}

PRINT_LOCK = Lock()
NLP_READY = False

COMMON_BLOCKED = [
    "wolf of wall street",
    "movie",
    "film",
    "box office",
    "obituary",
    "restaurant",
    "wedding",
    "music video",
]

ASSETS = {
    "SPY": {
        "queries": ["S&P 500", "stock market", "Federal Reserve"],
        "terms": ["s p 500", "stock market", "stocks", "equities", "wall street", "federal reserve", "fed"],
        "blocked": [],
    },
    "TLT": {
        "queries": ["Treasury yields", "bond yields", "bond market", "interest rates", "Federal Reserve"],
        "terms": ["treasury", "treasuries", "yield", "yields", "bond", "bond market", "fixed income", "interest rates"],
        "blocked": ["james bond"],
    },
    "GLD": {
        "queries": ["gold prices", "spot gold", "bullion"],
        "terms": ["gold", "gold price", "gold prices", "spot gold", "bullion", "precious metals"],
        "blocked": ["gold coast", "gold medal", "golden globe", "dirty love"],
    },
    "Cash": {
        "queries": ["Federal Reserve", "interest rates", "monetary policy"],
        "terms": ["federal reserve", "fed", "interest rates", "monetary policy", "fed funds", "money market"],
        "blocked": [],
    },
    "QQQ": {
        "queries": [
            "tech stocks",
            "technology stocks",
            "Apple Google Microsoft Facebook",
            "semiconductor stocks",
            "internet companies",
            "software stocks",
            "Nasdaq 100",
        ],
        "terms": [
            "technology",
            "tech",
            "internet",
            "software",
            "semiconductor",
            "chip",
            "apple",
            "google",
            "microsoft",
            "facebook",
            "nvidia",
            "intel",
            "amazon",
            "nasdaq",
        ],
        "blocked": ["breast", "bra", "starbucks cards"],
    },
    "VIX": {
        "queries": ["VIX index", "volatility index", "market volatility", "stock market volatility", "Federal Reserve"],
        "terms": ["vix", "volatility", "fear gauge", "fear", "volatility index", "market volatility"],
        "blocked": [],
    },
    "TNX": {
        "queries": ["10-year Treasury yield", "Treasury yields", "bond yields", "yield curve", "Treasury market", "interest rates"],
        "terms": [
            "10 year treasury yield",
            "treasury yield",
            "10 year note",
            "benchmark yield",
            "treasury yields",
            "bond yields",
            "treasury market",
            "bond market",
            "yield curve",
            "fixed income",
            "interest rates",
            "mortgage rates",
        ],
        "blocked": [],
    },
}

OUTPUT_COLUMNS = [
    "week_end",
    "article_date",
    "asset",
    "title",
    "description",
    "summary",
    "publisher",
    "url",
    "query",
    "query_rank",
    "gnews_rank",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fetch 1 weekly GNews article per asset with summary.")
    parser.add_argument("--market-file", type=Path, default=ROOT / "data" / "processed" / "market_features_weekly.csv")
    parser.add_argument("--start", type=date.fromisoformat)
    parser.add_argument("--end", type=date.fromisoformat)
    parser.add_argument("--asset", action="append", choices=sorted(ASSETS))
    parser.add_argument("--sleep-seconds", type=float, default=DEFAULT_SLEEP_SECONDS)
    parser.add_argument("--max-workers", type=int, default=DEFAULT_MAX_WORKERS)
    return parser.parse_args()


def safe_print(message: str, *, stream: object = sys.stdout) -> None:
    with PRINT_LOCK:
        print(message, file=stream, flush=True)


def clean_text(value: object) -> str:
    text = html.unescape(str(value or ""))
    text = re.sub(r"<[^>]+>", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def normalize_text(*parts: object) -> str:
    text = " ".join(clean_text(part).lower() for part in parts if part)
    text = text.replace("&", " and ")
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def contains_term(text: str, term: str) -> bool:
    normalized = normalize_text(term)
    return f" {normalized} " in f" {text} "


def count_term_hits(text: str, terms: list[str]) -> int:
    return sum(1 for term in terms if contains_term(text, term))


def title_tokens(text: str) -> list[str]:
    return [token for token in normalize_text(text).split() if len(token) >= 4][:6]


def summary_is_boilerplate(text: str) -> bool:
    normalized = normalize_text(text)
    return (
        ("privacy" in normalized and "policy" in normalized)
        or ("terms" in normalized and "conditions" in normalized)
        or ("cookie" in normalized and "policy" in normalized)
    )


def split_sentences(text: str) -> list[str]:
    sentences: list[str] = []
    for sentence in re.split(r"(?<=[.!?])\s+", clean_text(text)):
        sentence = sentence.strip()
        if len(sentence) < 40 or summary_is_boilerplate(sentence):
            continue
        sentences.append(sentence)
    return sentences


def first_sentences(text: str, *, max_sentences: int = 3, max_chars: int = 900) -> str:
    return clean_text(" ".join(split_sentences(text)[:max_sentences]))[:max_chars]


def snippet_summary(description: str) -> str:
    description = clean_text(description)
    return description[:900]


def parse_published_date(value: object) -> date | None:
    raw = clean_text(value)
    if not raw:
        return None
    try:
        return parsedate_to_datetime(raw).date()
    except Exception:
        return None


def format_duration(seconds: float) -> str:
    total_seconds = max(0, int(seconds))
    hours, remainder = divmod(total_seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    if hours:
        return f"{hours:02d}:{minutes:02d}:{seconds:02d}"
    return f"{minutes:02d}:{seconds:02d}"


def ensure_nlp_resources() -> None:
    global NLP_READY
    if NLP_READY:
        return

    try:
        import nltk

        for resource, path in [
            ("punkt", "tokenizers/punkt"),
            ("punkt_tab", "tokenizers/punkt_tab/english"),
            ("stopwords", "corpora/stopwords"),
        ]:
            try:
                nltk.data.find(path)
            except LookupError:
                nltk.download(resource, quiet=True)
    except Exception:
        pass

    NLP_READY = True


def make_client(week_start: date, week_end: date) -> GNews:
    next_day = week_end + timedelta(days=1)
    return GNews(
        language="en",
        country="US",
        max_results=MAX_RESULTS,
        start_date=(week_start.year, week_start.month, week_start.day),
        end_date=(next_day.year, next_day.month, next_day.day),
    )


def fetch_query_items(query: str, week_start: date, week_end: date) -> list[dict]:
    for attempt in range(MAX_NEWS_RETRIES):
        try:
            items = make_client(week_start, week_end).get_news(query)
            if items:
                return items
        except Exception:
            pass
        time.sleep(attempt + 1)
    return []


def resolve_article_url(url: str) -> str:
    parsed = urlparse(url)
    if parsed.hostname != "news.google.com":
        return url

    try:
        response = requests.get(url, headers=REQUEST_HEADERS, timeout=REQUEST_TIMEOUT, allow_redirects=True)
        resolved_url = str(response.url or "").strip()
        if resolved_url and "news.google.com" not in resolved_url and "google.com/sorry" not in resolved_url:
            return resolved_url
    except Exception:
        pass
    return ""


def looks_like_article_url(url: str, hostname: str) -> bool:
    parsed = urlparse(url)
    path = (parsed.path or "/").lower()
    query = (parsed.query or "").lower()

    if not parsed.scheme or parsed.netloc != hostname or path in {"", "/"}:
        return False
    if any(part in path for part in ["/tag/", "/category/", "/author/", "/topic/", "/topics/", "/about", "/contact", "/privacy", "/terms", "/search", "/page/", "/feed"]):
        return False
    if any(part in query for part in ["privacy", "terms", "cookies"]):
        return False
    if any(part in query for part in ["s=", "search=", "query="]) and not ("post_type=news" in query and "p=" in query):
        return False
    return True


def search_article_url(title: str, publisher_url: str) -> str:
    parsed = urlparse(publisher_url)
    tokens = title_tokens(title)
    if not parsed.scheme or not parsed.netloc or not tokens:
        return ""

    base = f"{parsed.scheme}://{parsed.netloc}"
    query = quote(" ".join(tokens))
    search_urls = [f"{base}/?s={query}", f"{base}/search/?query={query}"]

    best_url = ""
    best_score = 0.0
    for search_url in search_urls:
        try:
            response = requests.get(search_url, headers=REQUEST_HEADERS, timeout=REQUEST_TIMEOUT)
        except Exception:
            continue

        if "<" not in response.text:
            continue
        soup = BeautifulSoup(response.text, "html.parser")
        for anchor in soup.select("a[href]"):
            text = clean_text(anchor.get_text(" ", strip=True))
            href = urljoin(base, str(anchor.get("href") or "").strip())
            if not text or not looks_like_article_url(href, parsed.netloc):
                continue

            hits = count_term_hits(normalize_text(text, href), tokens)
            similarity = SequenceMatcher(None, normalize_text(title), normalize_text(text)).ratio()
            score = similarity * 10 + hits
            if score > best_score:
                best_score = score
                best_url = href

        if best_score >= 8:
            return best_url

    return ""


def get_article_summary(article_url: str) -> str:
    if not article_url:
        return ""

    google_news = GNews(language="en", country="US")
    try:
        article = google_news.get_full_article(article_url)
        text = clean_text(article.text)
        if text:
            try:
                ensure_nlp_resources()
                article.nlp()
                summary = clean_text(article.summary)
                if summary and not summary_is_boilerplate(summary):
                    return summary[:900]
            except Exception:
                pass
            summary = first_sentences(text)
            if summary:
                return summary
    except Exception:
        pass

    try:
        config = Config()
        config.browser_user_agent = REQUEST_HEADERS["User-Agent"]
        config.request_timeout = REQUEST_TIMEOUT
        article = Article(article_url, config=config, language="en")
        article.download()
        article.parse()
        text = clean_text(article.text)
        if text:
            summary = first_sentences(text)
            if summary:
                return summary
    except Exception:
        pass

    return ""


def make_row(
    *,
    asset: str,
    week_end: date,
    article_date: date,
    title: str,
    description: str,
    publisher: str,
    url: str,
    query: str,
    query_rank: int,
    gnews_rank: int,
) -> dict:
    return {
        "week_end": week_end.isoformat(),
        "article_date": article_date.isoformat(),
        "asset": asset,
        "title": title,
        "description": description,
        "summary": "",
        "publisher": publisher,
        "url": url,
        "query": query,
        "query_rank": query_rank,
        "gnews_rank": gnews_rank,
    }


def blank_row(asset: str, week_end: date) -> dict:
    return {
        "week_end": week_end.isoformat(),
        "article_date": "",
        "asset": asset,
        "title": "",
        "description": "",
        "summary": "",
        "publisher": "",
        "url": "",
        "query": "",
        "query_rank": "",
        "gnews_rank": "",
    }


def find_best_weekly_article(asset: str, week_end: date) -> dict:
    week_start = week_end - timedelta(days=6)
    terms = ASSETS[asset]["terms"]
    blocked_terms = COMMON_BLOCKED + ASSETS[asset]["blocked"]

    matched: list[tuple[int, int, int, str, dict]] = []
    fallback: list[tuple[int, int, str, dict]] = []

    for query_rank, query in enumerate(ASSETS[asset]["queries"], start=1):
        items = fetch_query_items(query, week_start, week_end)
        if not items:
            continue

        for gnews_rank, item in enumerate(items, start=1):
            article_date = parse_published_date(item.get("published date"))
            if article_date is None or article_date < week_start or article_date > week_end:
                continue

            publisher_raw = item.get("publisher") or {}
            publisher = clean_text(publisher_raw.get("title") if isinstance(publisher_raw, dict) else publisher_raw)
            publisher_url = clean_text(publisher_raw.get("href") if isinstance(publisher_raw, dict) else "")
            title = clean_text(item.get("title"))
            description = clean_text(item.get("description"))
            url = clean_text(item.get("url"))

            match_text = normalize_text(title, description, publisher)
            if any(contains_term(match_text, term) for term in blocked_terms):
                continue

            row = make_row(
                asset=asset,
                week_end=week_end,
                article_date=article_date,
                title=title,
                description=description,
                publisher=publisher,
                url=url,
                query=query,
                query_rank=query_rank,
                gnews_rank=gnews_rank,
            )

            hits = count_term_hits(match_text, terms)
            if hits > 0:
                matched.append((query_rank, -hits, gnews_rank, publisher_url, row))
            else:
                fallback.append((query_rank, gnews_rank, publisher_url, row))

    candidates = [(publisher_url, row) for _, _, _, publisher_url, row in sorted(matched)]
    if not candidates:
        candidates = [(publisher_url, row) for _, _, publisher_url, row in sorted(fallback)]
    if not candidates:
        return blank_row(asset, week_end)

    first_row: dict | None = None
    for publisher_url, row in candidates[:MAX_SUMMARY_TRIES]:
        article_url = resolve_article_url(row["url"]) or search_article_url(row["title"], publisher_url) or row["url"]
        row["url"] = article_url
        row["summary"] = get_article_summary(article_url)
        if row["summary"]:
            return row
        if first_row is None:
            row["summary"] = snippet_summary(row["description"])
            first_row = row

    return first_row or blank_row(asset, week_end)


class ProgressTracker:
    def __init__(self, assets: list[str], week_ends: list[date]) -> None:
        self.total = len(assets) * len(week_ends)
        self.asset_total = len(week_ends)
        self.done = 0
        self.asset_done = {asset: 0 for asset in assets}
        self.started_at = time.time()
        self.last_log_at = 0.0
        self.lock = Lock()

    def tick(self, asset: str, week_end: date) -> None:
        now = time.time()
        with self.lock:
            self.done += 1
            self.asset_done[asset] += 1

            should_log = (
                self.done == 1
                or self.done == self.total
                or self.asset_done[asset] == self.asset_total
                or now - self.last_log_at >= PROGRESS_INTERVAL_SECONDS
            )
            if not should_log:
                return

            elapsed = now - self.started_at
            rate = self.done / elapsed if elapsed > 0 else 0.0
            eta = (self.total - self.done) / rate if rate > 0 else 0.0
            progress = self.done / self.total * 100 if self.total else 100.0

            safe_print(
                f"[progress] {self.done}/{self.total} ({progress:.1f}%)"
                f" | current={asset} {self.asset_done[asset]}/{self.asset_total}"
                f" | week_end={week_end.isoformat()}"
                f" | elapsed={format_duration(elapsed)}"
                f" | eta={format_duration(eta)}"
            )
            self.last_log_at = now


def load_week_ends(path: Path, start: date | None, end: date | None) -> list[date]:
    if not path.exists():
        raise SystemExit(f"Missing market file: {path}")

    week_ends: list[date] = []
    seen: set[date] = set()
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if "week_end" not in (reader.fieldnames or []):
            raise SystemExit(f"`week_end` column not found in {path}")

        for row in reader:
            week_end = date.fromisoformat(row["week_end"])
            if start and week_end < start:
                continue
            if end and week_end > end:
                continue
            if week_end not in seen:
                seen.add(week_end)
                week_ends.append(week_end)

    return sorted(week_ends)


def read_existing_rows(path: Path) -> dict[str, dict]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return {row["week_end"]: row for row in csv.DictReader(handle) if row.get("week_end")}


def merge_existing_rows(rows: list[dict], existing_rows: dict[str, dict]) -> list[dict]:
    merged: list[dict] = []
    for row in rows:
        existing = existing_rows.get(row["week_end"])
        if existing and existing.get("summary") and (not row.get("title") or not row["summary"]):
            merged.append(existing)
        else:
            merged.append(row)
    return merged


def fetch_asset_rows(asset: str, week_ends: list[date], sleep_seconds: float, progress: ProgressTracker) -> tuple[str, list[dict]]:
    rows: list[dict] = []
    for week_end in week_ends:
        rows.append(find_best_weekly_article(asset, week_end))
        progress.tick(asset, week_end)
        if sleep_seconds > 0:
            time.sleep(sleep_seconds)
    return asset, rows


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=OUTPUT_COLUMNS, quoting=csv.QUOTE_ALL)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    if args.start and args.end and args.start > args.end:
        raise SystemExit("--start must be on or before --end")
    if args.max_workers <= 0:
        raise SystemExit("--max-workers must be >= 1")

    week_ends = load_week_ends(args.market_file, args.start, args.end)
    if not week_ends:
        raise SystemExit("No week_end rows found in the selected range.")

    assets = args.asset or list(ASSETS)
    workers = min(args.max_workers, len(assets))
    progress = ProgressTracker(assets, week_ends)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    safe_print(f"Assets: {', '.join(assets)}")
    safe_print(f"Weeks: {week_ends[0]} -> {week_ends[-1]} ({len(week_ends)} week(s))")
    safe_print(f"Source calendar: {args.market_file}")
    safe_print(f"Max workers: {workers}")
    safe_print(f"Total asset-weeks: {len(assets) * len(week_ends)}")

    results: dict[str, list[dict]] = {}
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {
            executor.submit(fetch_asset_rows, asset, week_ends, args.sleep_seconds, progress): asset for asset in assets
        }
        for future in as_completed(futures):
            asset, rows = future.result()
            results[asset] = rows
            safe_print(f"[asset done] {asset}: {len(rows)} week(s)")

    all_rows: list[dict] = []
    for asset in assets:
        asset_path = OUTPUT_DIR / f"{asset.lower()}_news_weekly.csv"
        rows = merge_existing_rows(results[asset], read_existing_rows(asset_path))
        write_csv(asset_path, rows)
        all_rows.extend(rows)

        exact_count = sum(1 for row in rows if row.get("title"))
        blank_count = len(rows) - exact_count
        safe_print(f"\n=== {asset} ===")
        safe_print(f"  exact rows: {exact_count}")
        safe_print(f"  blank rows: {blank_count}")
        safe_print(f"  saved: {asset_path}")

    if len(assets) > 1:
        write_csv(OUTPUT_DIR / "all_assets_news_weekly.csv", all_rows)

    safe_print(f"\nDone. Output directory: {OUTPUT_DIR}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        safe_print("\nInterrupted.", stream=sys.stderr)
        raise SystemExit(130)
