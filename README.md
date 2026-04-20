# AutoResearch_Prediction_Optimization
An autoresearch project to optimize prompts used to do LLM-based predictions.

## Overview
This project evaluates how well a large language model (LLM) can generate probabilistic forecasts for real-world events compared to community forecasts from Metaculus.

The task is:

> Given a question and a forecast date, predict the probability (0–100%) that the event resolves YES.

Performance is evaluated using the **Brier score**.

---

## Data

* Source: Hand-picked binary questions from **Metaculus**
* Domain: Primarily political and economic forecasting
* Format: CSV file (`RawData.csv`)

Each row contains:

* `Question`
* `Forecast_Date`
* `ForecastDate_Probability` (community forecast)
* `Resolution` (Yes/No outcome)

---

## Project Structure

```
.
├── RawData.csv                # original dataset
├── train.csv                 # cleaned 60% dataset (generated)
├── holdout_40_unused.csv     # unused 40% holdout set
├── clean.py                  # data cleaning + splitting
├── baseline.py               # LLM baseline evaluation
├── train_predictions.csv     # full predictions output
├── train_valid_predictions.csv  # parsed + valid predictions
└── README.md
```

---

## Setup

### 1. Install Codex CLI

Using Homebrew:

```bash
brew install codex
```

Then authenticate:

```bash
codex
```

Follow the login instructions in your terminal.

---

## Usage

### Step 1: Prepare Data

Generate the cleaned training dataset:

```bash
python clean.py
```

This will:

* Split data into **60% train / 40% holdout**
* Save:

  * `train.csv` (used for baseline)
  * `holdout_40_unused.csv` (not used yet)

---

### Step 2: Run Baseline

```bash
python baseline.py
```

This will:

* Prompt Codex for each question
* Extract:

  * Probability (0–100)
  * One paragraph reasoning
* Save predictions to:

  * `train_predictions.csv`
* Compute:

  * LLM Brier score
  * Community Brier score

---

## Model Prompt

The model is given the following fixed instruction:

```
Do not use any information after [Forecast_Date].

Generate:
- A probability from 0–100 that the question resolves YES
- One paragraph of reasoning
```

Output format is strictly enforced:

```
Probability: <number>
Reasoning: <one paragraph>
```

---

## Evaluation

We use the **Brier score**:

[
(p - y)^2
]

Where:

* ( p ) = predicted probability
* ( y ) = actual outcome (0 or 1)

We compute:

* **LLM Brier Score**
* **Community Brier Score** (from `ForecastDate_Probability`)

---

## Notes

* The 40% holdout set is **not used** in this stage (reserved for future evaluation)
* All logic (prompt, parsing, evaluation) is fixed

---

## Limitations

* Uses Codex CLI via subprocess (slow, not ideal for scaling)
* Parsing relies on strict output formatting
* Small dataset (~100 questions)
* Not completely reproducible as it prompts codex which is random

---

## Future Work

* Add more data to train/test (from Kalshi or Metaculus)
* Add batching for faster inference
* Introduce prompt optimization / autoresearch loop
* Evaluate on held-out test set
* Replace Codex CLI with direct API calls (potentially)
