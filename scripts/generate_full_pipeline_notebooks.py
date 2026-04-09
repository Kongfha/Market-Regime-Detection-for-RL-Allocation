from __future__ import annotations

import json
from pathlib import Path
from textwrap import dedent
from uuid import uuid4


REPO_ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK_DIR = REPO_ROOT / "full_pipeline"


def markdown_cell(source: str) -> dict:
    cleaned = dedent(source).strip("\n") + "\n"
    return {
        "cell_type": "markdown",
        "id": uuid4().hex[:8],
        "metadata": {},
        "source": cleaned.splitlines(keepends=True),
    }


def code_cell(source: str) -> dict:
    cleaned = dedent(source).strip("\n") + "\n"
    return {
        "cell_type": "code",
        "execution_count": None,
        "id": uuid4().hex[:8],
        "metadata": {},
        "outputs": [],
        "source": cleaned.splitlines(keepends=True),
    }


def notebook(cells: list[dict]) -> dict:
    return {
        "cells": cells,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {
                "codemirror_mode": {"name": "ipython", "version": 3},
                "file_extension": ".py",
                "mimetype": "text/x-python",
                "name": "python",
                "nbconvert_exporter": "python",
                "pygments_lexer": "ipython3",
                "version": "3.10",
            },
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


def build_hmm_notebook() -> dict:
    cells = [
        markdown_cell(
            """
            # Full Pipeline 01: HMM Regime Build + Full-Sample Inference

            This notebook is the HMM entrypoint for the merged pipeline.

            - Source of truth for regime selection: `scripts/train_hmm_regimes.py`
            - HMM inputs: `data/processed/model_state_weekly_price_macro.csv`
            - HMM feature preset: `regime_core`
            - HMM selection mode: `pipeline`
            - News inputs: `data/raw/news_sentiment/all_assets_news_weekly_finbert.csv`
            - Outputs written to: `output/full_pipeline/`
            """
        ),
        code_cell(
            """
            from pathlib import Path
            import sys

            import matplotlib.pyplot as plt
            import numpy as np
            import pandas as pd
            import seaborn as sns
            from IPython.display import display

            REPO_ROOT = None
            for candidate in [Path.cwd(), *Path.cwd().parents]:
                if (candidate / "full_pipeline").exists() and (candidate / "scripts").exists():
                    REPO_ROOT = candidate
                    break
            if REPO_ROOT is None:
                raise RuntimeError("Could not locate the repo root.")

            PIPELINE_ROOT = REPO_ROOT / "full_pipeline"
            if str(REPO_ROOT) not in sys.path:
                sys.path.insert(0, str(REPO_ROOT))
            if str(PIPELINE_ROOT) not in sys.path:
                sys.path.insert(0, str(PIPELINE_ROOT))

            from _pipeline_utils import build_full_pipeline_artifacts, OUTPUT_DIR

            plt.style.use("seaborn-v0_8-whitegrid")
            sns.set_palette("deep")
            pd.set_option("display.max_columns", 160)
            pd.set_option("display.width", 180)
            """
        ),
        markdown_cell(
            """
            ## Run the HMM + News Merge Pipeline

            This executes three steps:

            1. Aggregate weekly FinBERT sentiment for SPY, TLT, GLD, VIX, and TNX.
            2. Run the objective-aware HMM selection workflow and refit the chosen model on the development window.
            3. Infer full-sample causal regime probabilities and assemble the merged state table for RL/evaluation.
            """
        ),
        code_cell(
            """
            artifacts = build_full_pipeline_artifacts()

            base_state = artifacts["base_state"]
            news_features = artifacts["news_features"]
            hmm_bundle = artifacts["hmm_bundle"]
            hmm_full = artifacts["hmm_full"]
            merged_state = artifacts["merged_state"]
            artifact_paths = artifacts["artifact_paths"]

            print("Artifacts written to:")
            for name, path in artifact_paths.items():
                print(f"  {name}: {path.relative_to(REPO_ROOT)}")
            """
        ),
        code_cell(
            """
            coverage_df = pd.DataFrame([artifacts["news_coverage"]])
            hmm_validation_df = pd.DataFrame([artifacts["hmm_validation"]])

            print("Weekly FinBERT coverage")
            display(coverage_df)

            print("HMM posterior validation")
            display(hmm_validation_df)
            """
        ),
        code_cell(
            """
            results_df = pd.DataFrame(
                [{k: v for k, v in result.items() if not k.startswith("_")} for result in hmm_bundle.results]
            ).sort_values(
                ["passes_hard_filters", "interpretability_score", "val_ll_per_step"],
                ascending=[False, False, False],
            ).reset_index(drop=True)

            selected_rows = []
            for label, result in [
                ("best_statistical", hmm_bundle.best_statistical),
                ("best_interpretable", hmm_bundle.best_interpretable),
                ("best_k3", hmm_bundle.best_k3),
                ("chosen_for_pipeline", hmm_bundle.chosen),
            ]:
                if result is None:
                    continue
                selected_rows.append(
                    {
                        "role": label,
                        "K": result["K"],
                        "n_pca": result["n_pca"],
                        "cov_type": result["cov_type"],
                        "selected_seed": result["selected_seed"],
                        "val_ll_per_step": result["val_ll_per_step"],
                        "interpretability_score": result["interpretability_score"],
                        "interpretability_tier": result["interpretability_tier"],
                    }
                )

            print("Selected candidates")
            display(pd.DataFrame(selected_rows))

            print("Top grid-search candidates")
            display(
                results_df.loc[
                    :,
                    [
                        "K",
                        "n_pca",
                        "cov_type",
                        "selected_seed",
                        "val_ll_per_step",
                        "passes_hard_filters",
                        "interpretability_score",
                        "interpretability_tier",
                    ],
                ].head(12)
            )

            print("Selection reason:")
            print(hmm_bundle.selection_reason)
            print(f"Feature preset: {hmm_bundle.feature_preset}")
            print(f"Selection mode: {hmm_bundle.selection_mode}")
            print(f"Selection role: {hmm_bundle.selection_role}")
            """
        ),
        code_cell(
            """
            filtered_cols = sorted(column for column in hmm_full.columns if column.startswith("filtered_prob_regime_"))
            regime_counts = hmm_full["regime_filtered"].value_counts().sort_index()

            fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True)
            axes[0].plot(hmm_full["week_end"], hmm_full["regime_filtered"], drawstyle="steps-post", linewidth=1.5)
            axes[0].set_title("Filtered Regime Labels Across the Full Sample")
            axes[0].set_ylabel("Regime")

            for column in filtered_cols:
                axes[1].plot(hmm_full["week_end"], hmm_full[column], label=column)
            axes[1].set_title("Filtered Regime Posterior Probabilities")
            axes[1].set_ylabel("Probability")
            axes[1].set_xlabel("Week End")
            axes[1].legend(loc="upper right", ncol=2, fontsize=9)

            plt.tight_layout()
            plt.show()

            regime_counts.to_frame("n_weeks")
            """
        ),
        code_cell(
            """
            print("Merged state shape:", merged_state.shape)
            print("Merged state columns:", len(merged_state.columns))
            display(merged_state.head(3))

            missing_summary = pd.Series(
                {
                    "missing_regime_rows": int(merged_state["regime_filtered"].isna().sum()),
                    "missing_news_rows": int(
                        merged_state.filter(like="news_finbert_compound_").isna().any(axis=1).sum()
                    ),
                }
            )
            display(missing_summary.to_frame("value"))
            """
        ),
    ]
    return notebook(cells)


def build_rl_notebook() -> dict:
    cells = [
        markdown_cell(
            """
            # Full Pipeline 02: DQN Training with HMM + News Features

            This notebook is the canonical RL training path for the merged pipeline.

            - HMM regime source: `output/full_pipeline/hmm_regimes_full_sample.csv`
            - RL state: price + macro + filtered HMM posteriors + 5 weekly FinBERT features
            - RL algorithm: DQN only
            - Exported outputs: `rl_validation_actions.csv` and `rl_locked_test_actions.csv`
            """
        ),
        code_cell(
            """
            from pathlib import Path
            import random
            import sys

            import matplotlib.pyplot as plt
            import numpy as np
            import pandas as pd
            import torch
            from IPython.display import display

            REPO_ROOT = None
            for candidate in [Path.cwd(), *Path.cwd().parents]:
                if (candidate / "full_pipeline").exists() and (candidate / "scripts").exists():
                    REPO_ROOT = candidate
                    break
            if REPO_ROOT is None:
                raise RuntimeError("Could not locate the repo root.")

            PIPELINE_ROOT = REPO_ROOT / "full_pipeline"
            if str(REPO_ROOT) not in sys.path:
                sys.path.insert(0, str(REPO_ROOT))
            if str(PIPELINE_ROOT) not in sys.path:
                sys.path.insert(0, str(PIPELINE_ROOT))

            from evaluation import EvaluationConfig
            from ml.training_utils import evaluate_episode, train_dqn_finrl
            from _pipeline_utils import (
                OUTPUT_DIR,
                make_rl_env,
                prepare_rl_inputs,
                rollout_agent_on_split,
                save_action_frame,
            )

            SEED = 42
            np.random.seed(SEED)
            random.seed(SEED)
            torch.manual_seed(SEED)
            if torch.cuda.is_available():
                torch.cuda.manual_seed(SEED)

            plt.style.use("seaborn-v0_8-whitegrid")
            pd.set_option("display.max_columns", 160)
            pd.set_option("display.width", 180)
            """
        ),
        code_cell(
            """
            FAST_MODE = True
            TOTAL_TIMESTEPS = 4_000 if FAST_MODE else 30_000
            EVAL_FREQ = 500 if FAST_MODE else 2_000
            EARLY_STOPPING_PATIENCE = 6 if FAST_MODE else 10
            BUFFER_SIZE = 5_000 if FAST_MODE else 10_000
            SEQ_LEN = 4

            print(
                f"FAST_MODE={FAST_MODE} | timesteps={TOTAL_TIMESTEPS} | "
                f"eval_freq={EVAL_FREQ} | buffer={BUFFER_SIZE}"
            )
            """
        ),
        markdown_cell(
            """
            ## Load the Merged RL State

            The state comes from notebook 01 and preserves the evaluation split boundaries:

            - `train` through 2020-12-31
            - `validation` through 2022-12-30
            - `locked_test` after 2022-12-30
            """
        ),
        code_cell(
            """
            prepared = prepare_rl_inputs()
            dataset = prepared["dataset"]
            frame = prepared["frame"]

            display(dataset.describe_splits())
            display(dataset.describe_feature_blocks())

            print("Continuous RL feature columns:", len(prepared["feature_cols"]))
            print("Posterior columns:", prepared["posterior_cols"])
            display(frame.loc[:, ["week_end", "eval_split", *prepared["posterior_cols"][:2]]].head(3))
            """
        ),
        code_cell(
            """
            train_mask = frame["eval_split"] == "train"
            scaled_train = prepared["scaled_features"].loc[train_mask, prepared["feature_cols"]]

            scaling_checks = pd.DataFrame(
                {
                    "train_mean_abs_max": [float(np.abs(scaled_train.mean()).max())],
                    "train_std_min": [float(scaled_train.std(ddof=0).min())],
                    "train_std_max": [float(scaled_train.std(ddof=0).max())],
                }
            )
            display(scaling_checks)
            """
        ),
        code_cell(
            """
            eval_config = EvaluationConfig(
                transaction_cost=0.001,
                risk_penalty=0.05,
                risk_window=12,
            )

            train_env = make_rl_env(prepared, split="train", seq_len=SEQ_LEN, config=eval_config)
            val_env = make_rl_env(prepared, split="validation", seq_len=SEQ_LEN, config=eval_config)
            test_env = make_rl_env(prepared, split="locked_test", seq_len=SEQ_LEN, config=eval_config)

            print("Train action names:", train_env.ACTION_NAMES)
            print("Train observation space:", train_env.observation_space)
            print("Train action space:", train_env.action_space)
            """
        ),
        markdown_cell(
            """
            ## Train the Canonical DQN Agent

            The mainline merged pipeline uses DQN only. PPO/A2C/native-attention remain in the legacy experimentation notebook.
            """
        ),
        code_cell(
            """
            training = train_dqn_finrl(
                train_env=train_env,
                val_env=val_env,
                total_timesteps=TOTAL_TIMESTEPS,
                eval_freq=EVAL_FREQ,
                early_stopping_patience=EARLY_STOPPING_PATIENCE,
                learning_rate=1e-4,
                exploration_fraction=0.15,
                exploration_final_eps=0.05,
                target_update_interval=1_000,
                buffer_size=BUFFER_SIZE,
                batch_size=32,
                device="auto",
            )

            agent = training["agent"]
            print("Best validation reward:", training["best_val_reward"])
            pd.DataFrame(training["val_history"]).head()
            """
        ),
        code_cell(
            """
            validation_eval = evaluate_episode(agent, val_env, deterministic=True)
            locked_test_eval = evaluate_episode(agent, test_env, deterministic=True)

            eval_table = pd.DataFrame(
                [
                    {"split": "validation", **validation_eval},
                    {"split": "locked_test", **locked_test_eval},
                ]
            )
            eval_table.loc[:, ["split", "reward", "length", "cumulative_return", "sharpe_ratio", "max_drawdown"]]
            """
        ),
        code_cell(
            """
            validation_export_env = make_rl_env(prepared, split="validation", seq_len=SEQ_LEN, config=eval_config)
            locked_test_export_env = make_rl_env(prepared, split="locked_test", seq_len=SEQ_LEN, config=eval_config)

            validation_actions = rollout_agent_on_split(agent, validation_export_env, frame, "validation")
            locked_test_actions = rollout_agent_on_split(agent, locked_test_export_env, frame, "locked_test")

            validation_path = save_action_frame(validation_actions, OUTPUT_DIR / "rl_validation_actions.csv")
            locked_test_path = save_action_frame(locked_test_actions, OUTPUT_DIR / "rl_locked_test_actions.csv")

            print("Saved:")
            print(" ", validation_path.relative_to(REPO_ROOT))
            print(" ", locked_test_path.relative_to(REPO_ROOT))

            display(validation_actions.head(3))
            display(locked_test_actions.head(3))
            """
        ),
        code_cell(
            """
            fig, axes = plt.subplots(1, 2, figsize=(14, 4), sharey=True)
            validation_actions["action_name"].value_counts().sort_index().plot(kind="bar", ax=axes[0], title="Validation Actions")
            locked_test_actions["action_name"].value_counts().sort_index().plot(kind="bar", ax=axes[1], title="Locked-Test Actions")
            axes[0].set_ylabel("Count")
            plt.tight_layout()
            plt.show()
            """
        ),
    ]
    return notebook(cells)


def build_evaluation_notebook() -> dict:
    cells = [
        markdown_cell(
            """
            # Full Pipeline 03: Evaluation and Backtest

            This notebook evaluates the canonical DQN policy against the reusable `evaluation/` framework.

            Inputs:

            - `output/full_pipeline/model_state_weekly_hmm_news.csv`
            - `output/full_pipeline/rl_validation_actions.csv`
            - `output/full_pipeline/rl_locked_test_actions.csv`
            """
        ),
        code_cell(
            """
            from pathlib import Path
            import sys

            import matplotlib.pyplot as plt
            import pandas as pd
            from IPython.display import display

            REPO_ROOT = None
            for candidate in [Path.cwd(), *Path.cwd().parents]:
                if (candidate / "evaluation").exists() and (candidate / "full_pipeline").exists():
                    REPO_ROOT = candidate
                    break
            if REPO_ROOT is None:
                raise RuntimeError("Could not locate the repo root.")

            PIPELINE_ROOT = REPO_ROOT / "full_pipeline"
            if str(REPO_ROOT) not in sys.path:
                sys.path.insert(0, str(REPO_ROOT))
            if str(PIPELINE_ROOT) not in sys.path:
                sys.path.insert(0, str(PIPELINE_ROOT))

            from evaluation import (
                BacktestEngine,
                EvaluationConfig,
                all_baseline_policies,
                bootstrap_metric_table,
                default_action_space,
                load_default_dataset,
                plot_equity_curves,
                summary_table,
            )
            from _pipeline_utils import OUTPUT_DIR

            plt.style.use("seaborn-v0_8-whitegrid")
            pd.set_option("display.max_columns", 160)
            pd.set_option("display.width", 180)
            """
        ),
        code_cell(
            """
            state_path = OUTPUT_DIR / "model_state_weekly_hmm_news.csv"
            validation_actions_path = OUTPUT_DIR / "rl_validation_actions.csv"
            locked_test_actions_path = OUTPUT_DIR / "rl_locked_test_actions.csv"

            dataset = load_default_dataset(state_path)
            action_space = default_action_space()
            engine = BacktestEngine(
                dataset=dataset,
                action_space=action_space,
                config=EvaluationConfig(transaction_cost=0.001, risk_penalty=0.05, risk_window=12),
            )

            display(dataset.describe_splits())
            display(dataset.describe_feature_blocks())
            """
        ),
        markdown_cell(
            """
            ## Baseline Benchmarks

            The baselines run on the same merged state table and consume the full block set:

            - price
            - macro
            - regime
            - text
            """
        ),
        code_cell(
            """
            include_blocks = ("price", "macro", "regime", "text")
            baselines = all_baseline_policies(action_space)

            validation_results = engine.run_many(baselines, split="validation", include_blocks=include_blocks)
            locked_test_results = engine.run_many(baselines, split="locked_test", include_blocks=include_blocks)

            print("Validation summary")
            display(summary_table(validation_results))
            print("Locked-test summary")
            display(summary_table(locked_test_results))
            """
        ),
        code_cell(
            """
            validation_actions = pd.read_csv(validation_actions_path, parse_dates=["week_end"])
            locked_test_actions = pd.read_csv(locked_test_actions_path, parse_dates=["week_end"])

            validation_target = dataset.subset("validation").frame
            locked_test_target = dataset.subset("locked_test").frame

            assert len(validation_actions) == len(validation_target), "Validation action count does not match split length."
            assert len(locked_test_actions) == len(locked_test_target), "Locked-test action count does not match split length."

            rl_validation = engine.evaluate_precomputed_actions(
                action_ids=validation_actions["action_id"],
                split="validation",
                include_blocks=include_blocks,
                name="dqn_hmm_news",
            )
            rl_locked_test = engine.evaluate_precomputed_actions(
                action_ids=locked_test_actions["action_id"],
                split="locked_test",
                include_blocks=include_blocks,
                name="dqn_hmm_news",
            )

            combined_validation = summary_table(validation_results + [rl_validation])
            combined_locked_test = summary_table(locked_test_results + [rl_locked_test])

            print("Validation summary with RL")
            display(combined_validation)
            print("Locked-test summary with RL")
            display(combined_locked_test)
            """
        ),
        code_cell(
            """
            bootstrap_metric_table(locked_test_results + [rl_locked_test], metric="sharpe_ratio", n_boot=300, seed=7)
            """
        ),
        code_cell(
            """
            plot_equity_curves(locked_test_results + [rl_locked_test], title="Locked-Test Equity Curves: Baselines vs DQN")
            plt.show()
            """
        ),
        code_cell(
            """
            action_mix = pd.DataFrame(
                {
                    "validation": validation_actions["action_name"].value_counts(),
                    "locked_test": locked_test_actions["action_name"].value_counts(),
                }
            ).fillna(0).astype(int)
            action_mix
            """
        ),
    ]
    return notebook(cells)


def write_notebook(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def main() -> None:
    NOTEBOOK_DIR.mkdir(parents=True, exist_ok=True)
    write_notebook(NOTEBOOK_DIR / "01_hmm_regime_pipeline.ipynb", build_hmm_notebook())
    write_notebook(NOTEBOOK_DIR / "02_rl_dqn_with_hmm_news.ipynb", build_rl_notebook())
    write_notebook(NOTEBOOK_DIR / "03_evaluation_backtest.ipynb", build_evaluation_notebook())
    print(f"Wrote notebooks to {NOTEBOOK_DIR}")


if __name__ == "__main__":
    main()
