# Advanced Market Regime Detection: A Gaussian HMM & XAI Framework

> Status: historical / experimental pipeline.
>
> This folder is **not** the current source of truth for regime detection.
> The active baseline is `scripts/train_hmm_regimes.py`, and the canonical
> merged notebook workflow lives in `full_pipeline/`.

This repository implements a high-dimension, automated pipeline for detecting latent market regimes using **Gaussian Hidden Markov Models (HMM)**. Optimized via **Bayesian Information Criterion (BIC)** and interpreted through **Explainable AI (XAI)** heatmaps, this framework provides a robust state-space for Reinforcement Learning (RL) agents in multi-asset allocation.

---

## 🔬 Mathematical Foundations

### 1. Hidden Markov Model (HMM) Specification
The market is modeled as a discrete-time, first-order Markov process where the underlying state $S_t \in \{1, \dots, K\}$ is unobserved. The system is defined by the triplet $\lambda = (A, B, \pi)$:

- **Initial State Distribution ($\pi$):** $\pi_i = P(S_1 = i)$, the probability of starting in regime $i$.
- **Transition Probability Matrix ($A$):** $a_{ij} = P(S_t = j | S_{t-1} = i)$, representing the probability of transitioning from regime $i$ to $j$.
- **Emission Distribution ($B$):** Since observations $O_t$ are continuous, we use a Gaussian emission model:
  $$P(O_t | S_t = i) = \mathcal{N}(O_t | \mu_i, \Sigma_i)$$
  where $\mu_i$ is the mean vector and $\Sigma_i$ is the covariance matrix for regime $i$.

### 2. Feature Selection: Greedy BIC Forward Selection
To avoid the *Curse of Dimensionality*, we select the optimal feature subset $\mathcal{F}^* \subset \mathcal{F}_{all}$ by minimizing the **Bayesian Information Criterion (BIC)**. 

$$BIC = -2 \ln(\hat{L}) + k \ln(n)$$
- $\hat{L}$: Maximum likelihood estimated via the **Baum-Welch Algorithm** (Expectation-Maximization).
- $k$: Degrees of freedom, calculated as:
  $$k = (N_{states}^2 - 1) + (N_{states} \times N_{features}) + (N_{states} \times N_{features})$$
  *(Assuming diagonal covariance for selection stability)*.
- **Algorithm:** The pipeline performs a **Sequential Forward Selection (SFS)**. It starts with an empty set and iteratively adds the feature that yields the maximum decrease in BIC, stopping when $\Delta BIC > 0$.

### 3. Dimensionality Reduction & Decorrelation (PCA)
To handle multicollinearity $(\rho > 0.8)$ among macroeconomic indicators, we apply **Principal Component Analysis (PCA)**.
- **De-correlation:** PCA transforms the selected features into orthogonal components ($PC_n$).
- **Numerical Stability:** HMMs with diagonal covariance matrices assume feature independence. PCA-transformed inputs satisfy this assumption, preventing singular covariance matrices during the M-step of the EM algorithm.

### 4. Fixed-Split Protocol
The benchmark is evaluated with a strict chronological split:
- train for feature selection, scaler/PCA fitting, and HMM fitting
- validation for tuning reward weights, actions, and hyperparameters
- locked test for final out-of-sample reporting only

---

## 🛤 Automated Pipeline Architecture

The workflow is orchestrated via `run_pattern_recognition.py` across 7 specialized modules:

### Phase 1: Feature Engineering & Selection
- **`01_prepare_data.py`:** Harmonizes FRED (Macro), Yahoo (Market), and ETF returns. Implements **Forward-Fill (FF)** to ensure strict causality.
- **`01b_feature_filtering.py`:** Executes the **Greedy BIC Search** on the training window only. It evaluates 50+ candidates (e.g., Yield Spreads, Jobless Claims, Sentiment) to find the statistically optimal 8-10 drivers.
- **`02_scaling_and_pca.py`:** Fits the **StandardScaler** and **PCA** on the training window only, then transforms the full sample with the frozen preprocessing objects.

### Phase 2: Latent State Induction
- **`03_train_hmm.py`:** Trained using the **Baum-Welch** algorithm on the training window only. The model converges when the log-likelihood gain falls below a predefined epsilon.
- **`04_analyze_regimes.py`:** Decodes the hidden sequence using the **Viterbi Algorithm** for analysis and visualization. Regime naming is calibrated on the training window, and the separate price-trend label is derived from trailing momentum:
  $$S^* = \arg \max_{S} P(S | O, \lambda)$$

### Phase 3: Interpretability & XAI
- **`06_diagnostics_and_interpretability.py`:** Implements **Feature Saliency Analysis** via Z-Scores:
  $$Z_{i,j} = \frac{\mu_{i,j} - \bar{\mu}_j}{\sigma_j}$$
  - $\mu_{i,j}$: Mean of feature $j$ in regime $i$.
  - $\bar{\mu}_j, \sigma_j$: Global mean and standard deviation of feature $j$.
  - **Outcome:** A heatmap visualizing the "Economic Signature" of each regime (e.g., Bear markets characterized by HIGH VIX and LOW Consumer Sentiment).

---

## 📂 Output Directory Schema

- `/data/`: Cleaned time-series and the final `model_state_with_regimes.csv`.
- `/models/`: Serialized `StandardScaler`, `PCA`, and `GaussianHMM` objects.
- `/plots/`:
    - `main_regimes/`: SPY Price overlay with separate trend and regime bands.
    - `stats/`: Cross-sectional return/volatility distributions.
    - `model_diagnostics/`: BIC optimization curves and Transition Matrix heatmaps.

---

## 🧩 Reinforcement Learning (RL) Integration

The output features provide a high-fidelity **State Space ($S$)** for an RL Agent:
1. **Discrete State:** The `regime_id` acts as a cluster-based state for Q-Learning or PPO policies.
2. **Confidence Scores:** The smoothed posterior probabilities $P(S_t | O_{1:T})$ (derived via the Forward-Backward algorithm) provide the agent with a measure of **Regime Uncertainty**.
3. **Continuous State:** The PCA components serve as filtered, de-noised environmental observations for deep-RL architectures.
4. **Trend Label:** The separate `trend_name` column is for interpretation and visualization only. It is not the same thing as the HMM regime.

---

## 🚀 Execution & Requirements

```bash
# Execute the historical alternate pipeline
python run_pattern_recognition.py
```

**Required Libraries:** `hmmlearn`, `scikit-learn`, `statsmodels`, `pandas`, `seaborn`.
