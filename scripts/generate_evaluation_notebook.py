from __future__ import annotations

import json
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_PATH = REPO_ROOT / "output" / "jupyter-notebook" / "evaluation_framework.ipynb"


def markdown_cell(source: str) -> dict:
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": source.splitlines(keepends=True),
    }


def code_cell(source: str) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": source.splitlines(keepends=True),
    }


def build_notebook() -> dict:
    cells = [
        markdown_cell(
            "# Evaluation Framework for Market-Regime RL Allocation\n"
            "\n"
            "This notebook exposes the evaluation contract before the RL module exists.\n"
            "\n"
            "- Continuous RL input: price features, macro features, optional regime/text blocks, previous weights, drawdown, volatility.\n"
            "- Discrete RL output: one action id mapped to a portfolio template.\n"
            "- Baselines included now: every fixed action template plus equal weight, momentum rotation, and a rule-based regime proxy.\n"
        ),
        code_cell(
            "from pathlib import Path\n"
            "import sys\n"
            "\n"
            "import matplotlib.pyplot as plt\n"
            "import numpy as np\n"
            "import pandas as pd\n"
            "from IPython.display import display\n"
            "\n"
            "REPO_ROOT = None\n"
            "for candidate in [Path.cwd(), *Path.cwd().parents]:\n"
            "    if (candidate / 'evaluation').exists():\n"
            "        REPO_ROOT = candidate\n"
            "        break\n"
            "if REPO_ROOT is None:\n"
            "    raise RuntimeError('Could not locate repo root containing the evaluation package.')\n"
            "\n"
            "if str(REPO_ROOT) not in sys.path:\n"
            "    sys.path.insert(0, str(REPO_ROOT))\n"
            "\n"
            "from evaluation import (\n"
            "    BacktestEngine,\n"
            "    EvaluationConfig,\n"
            "    all_baseline_policies,\n"
            "    bootstrap_metric_table,\n"
            "    default_action_space,\n"
            "    load_default_dataset,\n"
            "    plot_equity_curves,\n"
            "    summary_table,\n"
            ")\n"
            "\n"
            "plt.style.use('seaborn-v0_8-whitegrid')\n"
            "pd.set_option('display.max_columns', 120)\n"
            "pd.set_option('display.width', 160)\n"
        ),
        markdown_cell(
            "## 1. Load the Weekly Evaluation Dataset\n"
            "\n"
            "The default loader uses `data/processed/model_state_weekly_price_macro.csv` and adds a weekly cash proxy from `dff_level / 100 / 52`.\n"
        ),
        code_cell(
            "dataset = load_default_dataset(REPO_ROOT / 'data/processed/model_state_weekly_price_macro.csv')\n"
            "dataset.describe_splits()\n"
        ),
        code_cell(
            "dataset.describe_feature_blocks()\n"
        ),
        markdown_cell(
            "## 2. Continuous RL Input Contract\n"
            "\n"
            "Static blocks come from the dataset. The environment appends the dynamic fields `prev_weight_*`, `portfolio_drawdown`, and `portfolio_volatility`.\n"
        ),
        code_cell(
            "price_macro_columns = dataset.continuous_columns(include_blocks=('price', 'macro'))\n"
            "price_macro_table = dataset.rl_input_frame(include_blocks=('price', 'macro'))\n"
            "print('Price + macro state shape:', price_macro_table.shape)\n"
            "print('Number of continuous input columns:', len(price_macro_columns))\n"
            "price_macro_table.head(3)\n"
        ),
        code_cell(
            "engine = BacktestEngine(dataset=dataset, action_space=default_action_space(), config=EvaluationConfig())\n"
            "engine.preview_observation(split='locked_test', include_blocks=('price', 'macro')).head(20)\n"
        ),
        markdown_cell(
            "## 3. Discrete Action Space\n"
            "\n"
            "This matches the 7-action proposal and keeps the RL output interpretable.\n"
        ),
        code_cell(
            "action_space = default_action_space()\n"
            "action_space.to_frame()\n"
        ),
        markdown_cell(
            "## 4. Baselines Available Before RL Training\n"
            "\n"
            "These are the benchmark strategies the RL agent should beat on validation and locked test. "
            "The notebook now evaluates every fixed action template in the action space as well as the dynamic heuristic baselines.\n"
        ),
        code_cell(
            "baselines = all_baseline_policies(action_space)\n"
            "baseline_catalog = pd.DataFrame({'baseline': [policy.name for policy in baselines]})\n"
            "print(f'Evaluating {len(baselines)} baselines')\n"
            "display(baseline_catalog)\n"
            "\n"
            "validation_results = engine.run_many(baselines, split='validation', include_blocks=('price', 'macro'))\n"
            "locked_test_results = engine.run_many(baselines, split='locked_test', include_blocks=('price', 'macro'))\n"
            "\n"
            "validation_summary = summary_table(validation_results)\n"
            "locked_test_summary = summary_table(locked_test_results)\n"
            "\n"
            "print('Validation summary')\n"
            "display(validation_summary)\n"
            "print('Locked test summary')\n"
            "display(locked_test_summary)\n"
        ),
        code_cell(
            "bootstrap_metric_table(locked_test_results, metric='sharpe_ratio', n_boot=300, seed=7)\n"
        ),
        code_cell(
            "plot_equity_curves(locked_test_results, title='Locked-Test Baseline Equity Curves')\n"
            "plt.show()\n"
        ),
        markdown_cell(
            "## 5. How to Plug In RL Output Later\n"
            "\n"
            "When your RL module is ready, pass one discrete action id per week. The evaluation layer converts those ids into weights, applies turnover costs, and computes the portfolio metrics.\n"
        ),
        code_cell(
            "locked_test_dataset = dataset.subset('locked_test')\n"
            "stub_action_ids = np.full(len(locked_test_dataset.frame), action_space.name_to_id['balanced_60_30_10'], dtype=int)\n"
            "stub_result = engine.evaluate_precomputed_actions(\n"
            "    action_ids=stub_action_ids,\n"
            "    split='locked_test',\n"
            "    include_blocks=('price', 'macro'),\n"
            "    name='stub_rl_balanced',\n"
            ")\n"
            "\n"
            "summary_table(locked_test_results + [stub_result])\n"
        ),
        markdown_cell(
            "## 6. Expected RL Interface\n"
            "\n"
            "Use the framework with the following contract:\n"
            "\n"
            "- Input to the RL model: `X_t` from `dataset.continuous_columns(...)` plus previous weights and risk context.\n"
            "- Output from the RL model: integer action id in `[0, 6]`.\n"
            "- Evaluation target: next-week returns from `next_return_spy`, `next_return_tlt`, `next_return_gld`, and derived `cash_return`.\n"
            "- Primary score: Sharpe ratio on validation and locked test, with drawdown and turnover as diagnostics.\n"
        ),
    ]

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


def main() -> None:
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    notebook = build_notebook()
    OUTPUT_PATH.write_text(json.dumps(notebook, indent=2), encoding="utf-8")
    print(f"Wrote {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
