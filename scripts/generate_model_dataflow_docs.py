#!/usr/bin/env python3
from __future__ import annotations

import shutil
import subprocess
from pathlib import Path
from textwrap import dedent

from build_project_datasets import CONTEXT_TICKERS, TRACKED_ASSETS
from fetch_fred_macro_panel import PRESETS

DEFAULT_PRICE_TICKERS = ["SPY", "TLT", "GLD", "QQQ", "^VIX", "^TNX"]
DEFAULT_NEWS_TICKERS = ["SPY", "QQQ", "TLT", "GLD"]

ROOT = Path(__file__).resolve().parents[1]
MARKDOWN_PATH = ROOT / "MODEL_DATAFLOW.md"
TEX_PATH = ROOT / "output" / "pdf" / "model_dataflow_architecture.tex"
PDF_PATH = ROOT / "output" / "pdf" / "model_dataflow_architecture.pdf"

PRICE_FIELDS = [
    "date",
    "symbol",
    "open",
    "high",
    "low",
    "close",
    "adj_close",
    "volume",
]
FRED_FIELDS = [
    "date",
    "series_id",
    "value",
    "frequency",
    "description",
]
NEWS_FIELDS = [
    "title",
    "summary",
    "published_at",
    "publisher",
    "canonical_url",
    "related_tickers",
]
PHRASEBANK_FIELDS = [
    "sentence",
    "label",
]
FRED_THEME_ROWS = [
    (
        "Policy and curve",
        ["DFF", "DGS3MO", "DGS10", "T10Y3M"],
        "Policy stance, front-end rates, long-end rates, inversion risk",
    ),
    (
        "Inflation and risk pricing",
        ["T10YIE", "BAMLH0A0HYM2", "VIXCLS", "DTWEXBGS"],
        "Inflation expectations, credit stress, volatility, dollar tightness",
    ),
    (
        "Liquidity and labor",
        ["WRESBAL", "ICSA", "UNRATE"],
        "System liquidity, weekly labor stress, slower labor confirmation",
    ),
    (
        "Activity and demand",
        ["CPIAUCSL", "CFNAI", "UMCSENT", "PERMIT"],
        "Realized inflation, broad activity, consumer tone, housing lead signal",
    ),
]


def code(value: str) -> str:
    return f"`{value}`"


def latex_escape(text: str) -> str:
    replacements = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
    }
    return "".join(replacements.get(char, char) for char in text)


def latex_code(text: str) -> str:
    return r"\texttt{" + latex_escape(text) + "}"


def markdown_source_sections() -> str:
    price_tickers = ", ".join(code(ticker) for ticker in DEFAULT_PRICE_TICKERS)
    news_tickers = ", ".join(code(ticker) for ticker in DEFAULT_NEWS_TICKERS)
    tradable_assets = ", ".join(code(ticker) for ticker in TRACKED_ASSETS)
    context_tickers = ", ".join(code(ticker) for ticker in CONTEXT_TICKERS)
    price_fields = ", ".join(code(field) for field in PRICE_FIELDS)
    fred_fields = ", ".join(code(field) for field in FRED_FIELDS)
    news_fields = ", ".join(code(field) for field in NEWS_FIELDS)
    phrasebank_fields = ", ".join(code(field) for field in PHRASEBANK_FIELDS)
    core_series = ", ".join(code(series) for series in PRESETS["core"])

    return dedent(
        f"""
        ## Source Specs

        ### 1. Yahoo Finance Price Data
        - extraction_api: {code("yfinance.Ticker.history()")}
        - tickers:
          - tradable_assets: {tradable_assets}
          - context_tickers: {context_tickers}
          - full_download_set: {price_tickers}
        - raw_file: {code("data/raw/yahoo_prices_daily.csv")}
        - raw_fields: {price_fields}
        - feature_outputs:
          - {code("data/processed/market_features_daily.csv")}
          - {code("data/processed/market_features_weekly.csv")}
          - {code("data/processed/weekly_asset_targets.csv")}
        - feature_blocks:
          - asset returns: 1d / 5d / 20d
          - realized volatility: 5d / 20d
          - drawdown / moving-average gap / intraday range / volume z-score
          - cross-asset context: QQQ/SPY ratio, SPY-TLT correlation, SPY-GLD correlation
          - market stress context: VIX level/change, TNX level/change
        - downstream_modules:
          - pattern recognition core input
          - RL state input
          - RL reward target via next-period asset returns

        ### 2. FRED Macro Data
        - extraction_api: {code("https://fred.stlouisfed.org/graph/fredgraph.csv?id=SERIES_ID")}
        - default_repo_preset: {code("core")}
        - optional_duration_preset: {code("core_plus_duration")}
        - recommended_core_series: {core_series}
        - optional_duration_extension: {code("DGS30")}
        - raw_files:
          - {code("data/raw/fred_macro_panel.csv")}
          - {code("data/raw/fred_macro_series_meta.csv")}
        - raw_fields: {fred_fields}
        - feature_output: {code("data/processed/macro_features_weekly.csv")}
        - feature_logic:
          - daily series: available on the same weekly decision row
          - weekly series: lagged by 1 week
          - monthly series: lagged by 1 month
          - transform spec comes from FRED series metadata and is applied causally
        - downstream_modules:
          - economic regime context for the regime detector
          - macro state block for RL
          - ablation axis: price-only vs price+macro

        ### 3. Yahoo Finance News
        - extraction_api: {code("yfinance.Ticker.get_news()")}
        - news_query_tickers: {news_tickers}
        - raw_files:
          - {code("data/raw/yahoo_news_latest.csv")}
          - {code("data/raw/yahoo_news_latest.json")}
        - raw_fields: {news_fields}
        - feature_outputs:
          - {code("data/processed/news_events_enriched.csv")}
          - {code("data/processed/news_features_weekly.csv")}
        - feature_logic:
          - event-level enrichment: sentiment_label, sentiment_score, topic_label, impact_score, relevance_score
          - weekly aggregation: headline_count, sentiment ratios, impact statistics, topic ratios
        - downstream_modules:
          - recent-window multimodal regime experiments
          - qualitative explanation / event overlay
          - optional RL state extension when news coverage exists

        ### 4. Financial PhraseBank
        - extraction_api: {code("Hugging Face dataset download")}
        - raw_files:
          - {code("data/raw/financial_phrasebank_all_agree.csv")}
          - {code("data/raw/financial_phrasebank_agree_75.csv")}
          - {code("data/raw/financial_phrasebank_agree_66.csv")}
          - {code("data/raw/financial_phrasebank_agree_50.csv")}
          - {code("data/raw/financial_phrasebank_combined.csv")}
        - raw_fields: {phrasebank_fields}
        - role_in_project:
          - benchmark / sanity check for the sentiment pipeline
          - not a training source for the trading model
          - useful for validating prompt templates or classifier thresholds
        """
    ).strip()


def build_markdown() -> str:
    fred_theme_lines = []
    for theme, series_list, role in FRED_THEME_ROWS:
        series_text = ", ".join(code(series) for series in series_list)
        fred_theme_lines.append(f"- {theme}: {series_text} -> {role}")

    return (
        dedent(
            """
            # Model Dataflow Architecture

            This file is the LLM-friendly source-of-truth summary for how the project extracts data, engineers features, and maps each data source into pattern-recognition and RL modules.

            ## Objective

            Build a weekly decision dataset that fuses price, macro, and news signals for:
            - market regime detection
            - RL ETF allocation
            - sentiment pipeline benchmarking

            ## Canonical Flow

            1. Yahoo Finance prices -> `data/raw/yahoo_prices_daily.csv` -> `market_features_daily.csv` -> `market_features_weekly.csv`
            2. FRED macro series -> `data/raw/fred_macro_panel.csv` + `fred_macro_series_meta.csv` -> `macro_features_weekly.csv`
            3. Yahoo Finance news -> `data/raw/yahoo_news_latest.csv/json` -> `news_events_enriched.csv` -> `news_features_weekly.csv`
            4. Financial PhraseBank -> raw benchmark CSV files -> sentiment pipeline validation only
            5. Weekly state assembler joins price + macro + targets (+ optional news) into:
               - `data/processed/model_state_weekly_price_macro.csv`
               - `data/processed/model_state_weekly_full.csv`
               - `data/processed/model_state_weekly_recent_full.csv`
            6. Regime detector consumes the weekly state
            7. RL allocator consumes the weekly state, regime output, and next-period returns

            ## Source-to-Module Tree

            ```text
            Yahoo Finance Price Data
              -> data/raw/yahoo_prices_daily.csv
              -> data/processed/market_features_daily.csv
              -> data/processed/market_features_weekly.csv
              -> data/processed/weekly_asset_targets.csv
              -> modules:
                 - regime detector core state
                 - RL state features
                 - RL reward/transition target

            FRED Macro Data
              -> data/raw/fred_macro_panel.csv
              -> data/raw/fred_macro_series_meta.csv
              -> data/processed/macro_features_weekly.csv
              -> modules:
                 - economic regime context
                 - price-only vs price+macro ablation
                 - macro block inside RL state

            Yahoo Finance News
              -> data/raw/yahoo_news_latest.csv
              -> data/raw/yahoo_news_latest.json
              -> data/processed/news_events_enriched.csv
              -> data/processed/news_features_weekly.csv
              -> modules:
                 - recent-window multimodal state
                 - event overlay for interpretation
                 - optional qualitative context for RL

            Financial PhraseBank
              -> data/raw/financial_phrasebank_*.csv
              -> modules:
                 - sentiment benchmark only
                 - not used to train the trading policy
            ```
            """
        ).strip()
        + "\n\n"
        + markdown_source_sections()
        + "\n\n"
        + dedent(
            """
            ## Weekly State Outputs

            ### `data/processed/model_state_weekly_price_macro.csv`
            - main historical training table
            - best first input for HMM / clustering / regime baselines
            - cleanest table for RL experiments with long history

            ### `data/processed/model_state_weekly_full.csv`
            - full joined table with nullable news columns
            - best integration table when code should accept both historical and recent rows

            ### `data/processed/model_state_weekly_recent_full.csv`
            - rows where price + macro + news all exist
            - best table for multimodal demos, recent case studies, and explanation outputs

            ## Model Module Mapping

            - Pattern recognition module:
              - primary inputs: `market_features_weekly.csv` + `macro_features_weekly.csv`
              - optional recent extension: `news_features_weekly.csv`
              - expected outputs: regime label, posterior probabilities, transition behavior
            - RL allocation module:
              - primary inputs: weekly state features + regime outputs + previous allocation
              - reward target: `next_return_spy`, `next_return_tlt`, `next_return_gld`
              - action space idea: interpretable allocation templates over `SPY`, `TLT`, `GLD`, and cash
            - Sentiment benchmark module:
              - input: Financial PhraseBank
              - purpose: verify whether the sentiment pipeline is directionally reasonable before using Yahoo headline aggregates
            - Explanation / analysis module:
              - input: recent weekly multimodal state + enriched headline events
              - purpose: explain why a week looks risk-on, risk-off, panic, or recovery

            ## Causal Alignment Rules

            - decision calendar: weekly (`W-FRI`)
            - daily price and daily FRED series: use the last available value in the decision week
            - weekly FRED series: lag by 1 week before joining
            - monthly FRED series: lag by 1 month before joining
            - news events: convert timestamps to `week_end` and aggregate inside each week
            - targets: next-week returns are shifted forward and must not leak into the current feature row

            ## Recommended FRED Core
            """
        ).strip()
        + "\n"
        + "\n".join(fred_theme_lines)
        + "\n\n"
        + dedent(
            """
            ## DGS30 Recommendation

            - Keep the repo default at the 16-series `core` panel.
            - Use `DGS30` only when the team specifically wants extra duration / term-premium information for `TLT`.
            - The repo now supports `python3 scripts/fetch_fred_macro_panel.py --preset core_plus_duration` when you want that extension.
            - In first-pass modeling, adding `DGS30` is useful only if it improves interpretation or out-of-sample stability beyond what `DGS10` and `T10Y3M` already provide.

            ## Canonical Commands

            ```bash
            PYTHONPATH=./_vendor python3 scripts/fetch_yahoo_seed_data.py --news-count 20
            python3 scripts/fetch_fred_macro_panel.py --preset core
            python3 scripts/fetch_fred_macro_panel.py --preset core_plus_duration
            python3 scripts/fetch_financial_phrasebank.py
            PYTHONPATH=./_vendor python3 scripts/build_project_datasets.py
            python3 scripts/generate_model_dataflow_docs.py
            ```

            ## External References

            - Yahoo Finance via `yfinance`: <https://ranaroussi.github.io/yfinance/index.html>
            - `Ticker.history()`: <https://ranaroussi.github.io/yfinance/reference/api/yfinance.Ticker.history.html>
            - `Ticker.get_news()`: <https://ranaroussi.github.io/yfinance/reference/api/yfinance.Ticker.get_news.html>
            - FRED API overview: <https://fred.stlouisfed.org/docs/api/fred/overview.html>
            - Financial PhraseBank: <https://huggingface.co/datasets/takala/financial_phrasebank>
            """
        ).strip()
        + "\n"
    )


def build_fred_table_tex() -> str:
    rows = []
    for theme, series_list, role in FRED_THEME_ROWS:
        series_text = ", ".join(series_list)
        rows.append(
            " & ".join(
                [
                    latex_escape(theme),
                    latex_escape(series_text),
                    latex_escape(role),
                ]
            )
            + r" \\"
        )
    return "\n".join(rows)


def build_tex() -> str:
    price_tickers = ", ".join(DEFAULT_PRICE_TICKERS)
    news_tickers = ", ".join(DEFAULT_NEWS_TICKERS)
    core_series = ", ".join(PRESETS["core"])
    fred_table_rows = build_fred_table_tex()

    return (
        dedent(
            r"""
            \documentclass[11pt,a4paper]{article}

            \usepackage[utf8]{inputenc}
            \usepackage[T1]{fontenc}
            \usepackage{lmodern}
            \usepackage[a4paper,margin=1.6cm]{geometry}
            \usepackage{graphicx}
            \usepackage{tikz}
            \usetikzlibrary{arrows.meta,positioning,fit,calc}
            \usepackage{tabularx}
            \usepackage{booktabs}
            \usepackage{array}
            \usepackage{enumitem}
            \usepackage{hyperref}
            \usepackage{xcolor}
            \usepackage{titlesec}
            \usepackage{ragged2e}
            \usepackage{setspace}
            \usepackage{float}

            \definecolor{navy}{HTML}{0B3954}
            \definecolor{teal}{HTML}{087E8B}
            \definecolor{gold}{HTML}{F4B942}
            \definecolor{coral}{HTML}{FF6B35}
            \definecolor{slate}{HTML}{34495E}
            \definecolor{softblue}{HTML}{EEF6FA}
            \definecolor{softteal}{HTML}{EAF7F8}
            \definecolor{softgold}{HTML}{FFF7E3}
            \definecolor{softgray}{HTML}{F7FAFC}

            \hypersetup{
              colorlinks=true,
              urlcolor=teal,
              linkcolor=navy,
              pdftitle={Model Dataflow Architecture},
              pdfauthor={Codex},
              pdfsubject={Data extraction and module mapping for market regime detection and RL allocation}
            }

            \setlength{\parindent}{0pt}
            \setlength{\parskip}{0.45em}
            \renewcommand{\arraystretch}{1.18}
            \onehalfspacing

            \titleformat{\section}{\Large\bfseries\color{navy}}{\thesection}{0.6em}{}
            \titleformat{\subsection}{\large\bfseries\color{teal}}{\thesubsection}{0.6em}{}

            \newcolumntype{Y}{>{\RaggedRight\arraybackslash}X}

            \begin{document}

            {\LARGE\bfseries\color{navy}Model Dataflow Architecture\par}
            {\large Repo-aligned extraction, feature engineering, and module mapping for market regime detection plus RL allocation\par}

            \section{End-to-End Diagram}

            \begin{figure}[H]
            \centering
            \scalebox{0.82}{
            \begin{tikzpicture}[
              font=\footnotesize,
              x=1cm,
              y=1cm,
              >={Latex[length=2.5mm]},
              source/.style={draw=navy, fill=softblue, rounded corners, text width=3.05cm, minimum height=2.15cm, align=left, anchor=north west},
              raw/.style={draw=teal!80!black, fill=softteal, rounded corners, text width=3.05cm, minimum height=2.15cm, align=left, anchor=north west},
              feat/.style={draw=gold!70!black, fill=softgold, rounded corners, text width=3.25cm, minimum height=2.15cm, align=left, anchor=north west},
              module/.style={draw=coral!85!black, fill=white, rounded corners, text width=3.35cm, minimum height=2.15cm, align=left, anchor=north west},
              joinbox/.style={draw=slate, fill=softgray, rounded corners, text width=7.55cm, minimum height=2.0cm, align=left, anchor=north west},
              arrow/.style={->, thick, color=slate}
            ]
            \node[source] (price) at (0,0) {
              \textbf{Yahoo Price Data}\\
              \texttt{Ticker.history()}\\
              tradable: SPY, TLT, GLD\\
              context: QQQ, \textasciicircum{}VIX, \textasciicircum{}TNX
            };
            \node[source] (macro) at (0,-2.8) {
              \textbf{FRED Macro Data}\\
              \texttt{fredgraph.csv}\\
              core panel + metadata\\
              optional: DGS30
            };
            \node[source] (news) at (0,-5.6) {
              \textbf{Yahoo News}\\
              \texttt{Ticker.get\_news()}\\
              latest headlines\\
              title, summary, URL
            };
            \node[source] (phrasebank) at (0,-8.4) {
              \textbf{Financial PhraseBank}\\
              benchmark labels only\\
              sentence + sentiment\\
              no trading-model training
            };

            \node[raw] (rawprice) at (3.8,0) {
              \textbf{Raw storage}\\
              folder: \texttt{data/raw}\\
              \texttt{yahoo\_prices\_daily}
            };
            \node[raw] (rawmacro) at (3.8,-2.8) {
              \textbf{Raw storage}\\
              \texttt{fred\_macro\_panel}\\
              \texttt{fred\_macro\_series\_meta}
            };
            \node[raw] (rawnews) at (3.8,-5.6) {
              \textbf{Raw storage}\\
              \texttt{yahoo\_news\_latest}\\
              csv + json snapshot
            };
            \node[raw] (rawphrase) at (3.8,-8.4) {
              \textbf{Raw storage}\\
              \texttt{financial\_phrasebank\_*}\\
              split + combined views
            };

            \node[feat] (featprice) at (7.8,0) {
              \textbf{Feature engineering}\\
              daily price factors\\
              weekly state factors\\
              next-week asset targets
            };
            \node[feat] (featmacro) at (7.8,-2.8) {
              \textbf{Feature engineering}\\
              causal macro transforms\\
              frequency-aware lagging\\
              weekly macro state
            };
            \node[feat] (featnews) at (7.8,-5.6) {
              \textbf{Feature engineering}\\
              headline enrichment\\
              sentiment / topic / impact\\
              weekly news aggregates
            };
            \node[module] (modphrase) at (7.8,-8.4) {
              \textbf{Benchmark module}\\
              validate sentiment logic\\
              calibrate heuristics or prompts\\
              kept outside trading pipeline
            };

            \node[joinbox] (join) at (3.8,-11.25) {
              \textbf{Weekly state assembler}\\
              joins weekly price factors + macro state + next-week targets, then optionally attaches weekly news aggregates\\
              outputs: price\_macro state, full joined state, recent multimodal state
            };
            \node[module] (regime) at (11.9,-10.4) {
              \textbf{Pattern recognition module}\\
              HMM / clustering / regime baseline\\
              output: regime label + posterior\\
              main input: weekly fused state
            };
            \node[module] (rl) at (11.9,-13.0) {
              \textbf{RL allocation module}\\
              input: weekly state + regime output\\
              target: next-week asset returns\\
              action templates over SPY/TLT/GLD/cash
            };
            \node[module] (explain) at (11.9,-15.6) {
              \textbf{Explanation layer}\\
              uses recent full rows + enriched news\\
              qualitative regime narrative\\
              not the primary decision maker
            };

            \coordinate (joininA) at ($(join.north west)+(1.25,0)$);
            \coordinate (joininB) at ($(join.north west)+(3.85,0)$);
            \coordinate (joininC) at ($(join.north west)+(6.4,0)$);

            \draw[arrow] (price.east) -- (rawprice.west);
            \draw[arrow] (macro.east) -- (rawmacro.west);
            \draw[arrow] (news.east) -- (rawnews.west);
            \draw[arrow] (phrasebank.east) -- (rawphrase.west);

            \draw[arrow] (rawprice.east) -- (featprice.west);
            \draw[arrow] (rawmacro.east) -- (featmacro.west);
            \draw[arrow] (rawnews.east) -- (featnews.west);
            \draw[arrow] (rawphrase.east) -- (modphrase.west);

            \draw[arrow] (featprice.south) -- ++(0,-0.35) -| (joininA);
            \draw[arrow] (featmacro.south) -- ++(0,-0.35) -| (joininB);
            \draw[arrow] (featnews.south) -- ++(0,-0.35) -| (joininC);

            \draw[arrow] (join.east) -- (regime.west);
            \draw[arrow] (regime.south) -- (rl.north);
            \draw[arrow] (rl.south) -- (explain.north);
            \draw[arrow] (join.east) |- (explain.west);
            \end{tikzpicture}
            }
            \caption{Data extraction and feature mapping into the regime detector, RL allocator, and explanation modules.}
            \end{figure}

            \section{Source-to-Module Mapping}

            \subsection{Yahoo Finance Price Data}
            \begin{itemize}[leftmargin=1.4em]
              \item Extraction: \texttt{yfinance.Ticker.history()} on """ + latex_escape(price_tickers) + r"""
              \item Raw schema: """ + latex_escape(", ".join(PRICE_FIELDS)) + r"""
              \item Processed outputs: \texttt{market\_features\_daily.csv}, \texttt{market\_features\_weekly.csv}, \texttt{weekly\_asset\_targets.csv}
              \item Model use: regime core state, RL state variables, and next-period return targets.
            \end{itemize}

            \subsection{FRED Macro Data}
            \begin{itemize}[leftmargin=1.4em]
              \item Extraction: official \texttt{fredgraph.csv} endpoint with repo default preset \texttt{core}.
              \item Core series: """ + latex_escape(core_series) + r"""
              \item Optional duration extension: \texttt{core\_plus\_duration} adds \texttt{DGS30}.
              \item Processed output: \texttt{macro\_features\_weekly.csv} with transforms driven by \texttt{frequency} and \texttt{suggested\_transform}.
              \item Model use: economic regime block for pattern recognition and macro context for RL.
            \end{itemize}

            \subsection{Yahoo Finance News}
            \begin{itemize}[leftmargin=1.4em]
              \item Extraction: \texttt{yfinance.Ticker.get\_news()} on """ + latex_escape(news_tickers) + r"""
              \item Raw schema: """ + latex_escape(", ".join(NEWS_FIELDS)) + r"""
              \item Processed outputs: \texttt{news\_events\_enriched.csv} and \texttt{news\_features\_weekly.csv}
              \item Model use: recent-window multimodal regime experiments, event overlays, and explanation support.
            \end{itemize}

            \subsection{Financial PhraseBank}
            \begin{itemize}[leftmargin=1.4em]
              \item Extraction: Hugging Face archive download.
              \item Raw schema: """ + latex_escape(", ".join(PHRASEBANK_FIELDS)) + r"""
              \item Model use: benchmark only for sanity-checking the sentiment pipeline.
            \end{itemize}

            \section{Weekly State Outputs}

            \begin{tabularx}{\textwidth}{@{}p{0.33\textwidth}YY@{}}
            \toprule
            \textbf{Output file} & \textbf{Contents} & \textbf{Main downstream use} \\
            \midrule
            \texttt{model\_state\_weekly\_price\_macro.csv} & Price features + macro features + weekly asset targets & Main historical regime training table and first RL table \\
            \texttt{model\_state\_weekly\_full.csv} & Price + macro + targets + nullable news features & Unified schema for integration code \\
            \texttt{model\_state\_weekly\_recent\_full.csv} & Recent rows where price, macro, and news all exist & Multimodal demos, case studies, and explanation outputs \\
            \bottomrule
            \end{tabularx}

            \section{Recommended FRED Core}

            \begin{tabularx}{\textwidth}{@{}p{0.2\textwidth}p{0.32\textwidth}Y@{}}
            \toprule
            \textbf{Theme} & \textbf{Series} & \textbf{Why it belongs in the state} \\
            \midrule
            """ + fred_table_rows + r"""
            \bottomrule
            \end{tabularx}

            \section{DGS30 Recommendation}

            Keep the default historical panel at the 16-series \texttt{core} preset.
            Add \texttt{DGS30} only when the team wants extra long-duration information for \texttt{TLT} sensitivity or term-premium interpretation.
            The repo now supports:

            \begin{itemize}[leftmargin=1.4em]
              \item \texttt{python3 scripts/fetch\_fred\_macro\_panel.py --preset core}
              \item \texttt{python3 scripts/fetch\_fred\_macro\_panel.py --preset core\_plus\_duration}
            \end{itemize}

            \section{Causal Alignment Rules}

            \begin{itemize}[leftmargin=1.4em]
              \item Decision calendar: weekly (\texttt{W-FRI}).
              \item Daily price and daily FRED series use the last available value in the decision week.
              \item Weekly FRED series are lagged by one week before joining.
              \item Monthly FRED series are lagged by one month before joining.
              \item News events are timestamped, converted to \texttt{week\_end}, and aggregated within each week.
              \item Next-week asset returns stay shifted forward and are not part of the current feature row.
            \end{itemize}

            \section{External References}

            \begin{itemize}[leftmargin=1.4em]
              \item Yahoo Finance via \texttt{yfinance}: \url{https://ranaroussi.github.io/yfinance/index.html}
              \item \texttt{Ticker.history()}: \url{https://ranaroussi.github.io/yfinance/reference/api/yfinance.Ticker.history.html}
              \item \texttt{Ticker.get\_news()}: \url{https://ranaroussi.github.io/yfinance/reference/api/yfinance.Ticker.get_news.html}
              \item FRED API overview: \url{https://fred.stlouisfed.org/docs/api/fred/overview.html}
              \item Financial PhraseBank: \url{https://huggingface.co/datasets/takala/financial_phrasebank}
            \end{itemize}

            \end{document}
            """
        ).strip()
        + "\n"
    )


def compile_pdf(tex_path: Path) -> None:
    tex_path.parent.mkdir(parents=True, exist_ok=True)
    latexmk = shutil.which("latexmk")
    pdflatex = shutil.which("pdflatex")
    tex_name = tex_path.name

    if latexmk:
        subprocess.run(
            [
                latexmk,
                "-pdf",
                "-interaction=nonstopmode",
                "-halt-on-error",
                tex_name,
            ],
            cwd=tex_path.parent,
            check=True,
        )
        return

    if not pdflatex:
        raise RuntimeError("Neither latexmk nor pdflatex is available in PATH.")

    for _ in range(2):
        subprocess.run(
            [
                pdflatex,
                "-interaction=nonstopmode",
                "-halt-on-error",
                tex_name,
            ],
            cwd=tex_path.parent,
            check=True,
        )


def main() -> None:
    MARKDOWN_PATH.write_text(build_markdown(), encoding="utf-8")
    TEX_PATH.parent.mkdir(parents=True, exist_ok=True)
    TEX_PATH.write_text(build_tex(), encoding="utf-8")
    compile_pdf(TEX_PATH)

    print(f"Saved markdown -> {MARKDOWN_PATH}")
    print(f"Saved LaTeX source -> {TEX_PATH}")
    print(f"Saved PDF -> {PDF_PATH}")


if __name__ == "__main__":
    main()
