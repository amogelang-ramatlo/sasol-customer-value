# Zindi Challenge: Customer Value & Opportunity Prediction

---

## 1. Evaluation & Scoring

This challenge utilizes a **multi-metric evaluation** combining regression and classification objectives.

### Targets
1. **`CLV_fuel`**: Log-transformed, normalized customer lifetime value (fuel volume in litres over a 3-month horizon). Evaluated with standard **RMSE**.
2. **`CLV_nonfuel`**: Log-transformed, normalized customer lifetime value (non-fuel spend in rands over a 3-month horizon). Evaluated with standard **RMSE**.
3. **`Opportunity`**: Combined opportunity type-and-category labels (including standalone `Inactivity` and `Stable`). Evaluated with **Weighted F1 Score**.

### Target Transformation Formula
For both regression targets, raw 3-month customer totals $y$ are transformed into submission targets $z$:

$$z = \frac{\ln(1 + y)}{s}$$

- $y$: Raw 3-month total (fuel in litres, non-fuel in rands).
- $s$: Fixed population standard deviation of $\ln(1 + y)$ calculated strictly on a published historical calibration set prior to the test cutoff.
- Submissions require transformed values $z$. **Do not apply the transformation twice.**
- Recovering raw predictions:
  $$y = \exp(s \cdot z) - 1$$
- Note: A raw value of $0$ transforms to $0$.

### Leaderboard Metric Formula

$$\text{Final Score} = 0.3 \times \text{RMSE}_{\text{CLV\_fuel}} + 0.3 \times \text{RMSE}_{\text{CLV\_nonfuel}} + 0.4 \times \text{Weighted F1}_{\text{Opportunity}}$$

| Component | Target Type | Evaluation Metric | Weight |
|---|---|---|---|
| `CLV_fuel` | Continuous ($z_{\text{fuel}}$) | RMSE | 0.30 |
| `CLV_nonfuel` | Continuous ($z_{\text{nonfuel}}$) | RMSE | 0.40 / 0.30 (0.30) |
| `Opportunity` | Multi-class Label | Weighted F1 | 0.40 |

---

## 2. Key Competition Rules

- **Eligibility:** South African residents only (prizes require SA residency verification).
- **Team Size:** Solo only (Max team size = 1).
- **Submissions:**
  - Maximum **10 submissions per day**.
  - Maximum **200 submissions total**.
  - Select **2 final submissions** for private leaderboard scoring (default is top 2 public).
- **Split:**
  - **Public Leaderboard:** ~30% of the test set.
  - **Private Leaderboard:** ~70% of the test set.
- **Tooling Constraints:**
  - Open-source packages and languages only.
  - Publicly accessible pre-trained models allowed.
  - **AutoML tools are strictly prohibited.**
  - No paid APIs, closed services, or platforms requiring a credit card.
- **Reproducibility & Verification:**
  - Top 10 finalists must submit code within 48 hours of request.
  - Deterministic code is required: **Always fix random seeds.**
  - Code must run end-to-end without custom proprietary libraries.

---

## 3. Recommended Roadmap: What to Start With First

Here is a structured sequence to get off the ground quickly and avoid critical pipeline mistakes.

### Step 1: Data Audit & Baseline Sanity Check (Day 1)
- **Confirm the scaling constant $s$:** Identify whether $s_{\text{fuel}}$ and $s_{\text{nonfuel}}$ are already provided in the starter pack, or if you need to extract them from the specified historical calibration subset.
- **Inspect Target Distributions:**
  - Plot distributions of raw $y$ vs. transformed $z$.
  - Check zero-inflation rate (many customers will have 0 spend/litres).
  - Inspect class balance for `Opportunity` (class frequencies for `Inactivity`, `Stable`, and combined categories).
- **Create an End-to-End Dummy Submission:**
  - Submit median/mean values for regression and the majority class for classification to ensure the submission format passes Zindi’s validator without errors.

### Step 2: Build a Local Validation Strategy (Day 1–2)
Because this is a multi-target problem with potential temporal dynamics:
- **Validation Split:** If customer records are time-based, use an **out-of-time (temporal) split** that mirrors the train-to-test cutoff. If strictly cross-sectional per customer, use a **Stratified K-Fold** stratified on the `Opportunity` label.
- **Custom Metric Evaluator:** Write an evaluation function that mirrors the competition metric locally:
  $$\text{Score}_{\text{local}} = 0.3 \times \text{RMSE}(z_1, \hat{z}_1) + 0.3 \times \text{RMSE}(z_2, \hat{z}_2) + 0.4 \times \text{F1}_{\text{weighted}}(c, \hat{c})$$
  *Never tune solely on public leaderboard feedback.*

### Step 3: Minimal Viable Baseline Model (Day 2–3)
- Use standard gradient boosting (LightGBM or CatBoost).
- Train **three separate models**:
  1. `LGBMRegressor` for `CLV_fuel` (optimizing RMSE directly on $z$).
  2. `LGBMRegressor` for `CLV_nonfuel` (optimizing RMSE directly on $z$).
  3. `LGBMClassifier` for `Opportunity` (optimizing multi-class log loss / weighted F1).
- Submit this baseline to establish your benchmark anchor on both local CV and the public leaderboard.

### Step 4: Iterative Feature Engineering (Days 4+)
Focus where the business value lies:
- **RFM Metrics:** Recency, Frequency, Monetary values split across fuel vs. non-fuel transactions (e.g., 30-day, 60-day, 90-day aggregations).
- **Ratios & Proportions:** Non-fuel spend to fuel spend ratio, fuel volume per visit, transaction frequency trends (acceleration/deceleration).
- **Activity flags:** Days since last purchase, churn risk signals (directly relevant to `Inactivity` and `Stable` labels).
- **Multi-task correlation:** Note that high non-fuel spend often correlates with fuel volume; predictions from one model can sometimes serve as an out-of-fold feature for another.