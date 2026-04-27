# autoresearch (prompt optimization)

This is an experiment to have the LLM improve its own forecasting ability by optimizing its prompting strategy.

---

## Setup

To set up a new experiment, work with the user to:

- Agree on a run tag: propose a tag based on today's date (e.g. `apr27`). The branch `autoresearch/<tag>` must not already exist — this is a fresh run.

- Create the branch:

git checkout -b autoresearch/<tag>

- Read the in-scope files:
  - model.py — runs evaluation using the current prompt  
  - prompt.txt — the prompt to optimize (**this is the ONLY file you modify**)  
  - data/train.csv — forecasting dataset  
  - outputs/train_valid_predictions.csv — baseline + community predictions  

- Verify data exists:
  - Ensure data/ contains the required CSV files  
  - If not, tell the human to generate them  

- Confirm and go:
  - Once setup is verified, begin experimentation.

---

## Experimentation

Each optimization iteration first reads the latest prediction CSV, asks Codex CLI to rewrite `prompt.txt`, then runs a full evaluation of the dataset using the revised prompt.

Start the optimization loop with:

python optimize_prompt.py

The loop starts from the existing baseline:

- `outputs/train_valid_predictions.csv`

Each optimization run produces:
- Brier score (printed to stdout)  
- Saved outputs in outputs/run_XXX/  

---

## What you CAN do

- Modify prompt.txt
- Add, remove, or rewrite instructions
- Change reasoning style
- Add calibration guidance
- Encourage or discourage extreme probabilities

---

## What you CANNOT do

- Modify model.py
- Modify dataset files in data/
- Change evaluation logic (Brier score is fixed)
- Add external tools, APIs, or retrieval

---

## Goal

Minimize:

Brier Score = mean((predicted_prob - outcome)^2)

Lower is better.

---

## Output format

Each run is saved to:

outputs/run_XXX/

Containing:

- predictions.csv — model outputs  
- prompt.txt — prompt used  
- metrics.json — Brier score and metadata  

---

## Logging results

Each run automatically writes:

- `outputs/run_XXX/predictions.csv`
- `outputs/run_XXX/prompt.txt`
- `outputs/run_XXX/metrics.json`

Use these rules when deciding whether `prompt.txt` should keep the modified prompt:

- Use keep if Brier improves  
- Use discard if worse or equal  
- Use crash if parsing or execution fails  

---

## Run budget

Run exactly 3 optimization iterations after the existing baseline:

- iteration 1: optimize from `outputs/train_valid_predictions.csv`, then save `outputs/run_XXX/`
- iteration 2: optimize from the latest `outputs/run_XXX/predictions.csv`, then save another run
- iteration 3: optimize from the latest `outputs/run_XXX/predictions.csv`, then save another run

After the third optimization iteration, stop and report the best score and which prompt was kept.

---

## The experiment loop

Repeat until 3 optimization iterations have completed:

1. Read the current best prompt (`prompt.txt`)
2. Read the latest prediction CSV:
   - First iteration: `outputs/train_valid_predictions.csv`
   - Later iterations: latest `outputs/run_XXX/predictions.csv`
3. Ask Codex CLI to propose a prompt modification from the current prompt and latest prediction errors
4. Save modified prompt  

Run:

python model.py

5. Save predictions and metrics under `outputs/run_XXX/`
6. Extract Brier score from output  
7. Compare with best score:
   - If improved → keep new prompt
   - Else → revert to previous prompt

8. Use the newest `outputs/run_XXX/predictions.csv` as evidence for the next iteration
9. Stop after the third optimization iteration  

---

## Prompt optimization strategies

Try:

- Adding calibration instructions (“avoid overconfidence”)
- Encouraging base rate reasoning
- Asking for uncertainty awareness
- Structuring reasoning (step-by-step)
- Penalizing extreme probabilities
- Encouraging comparison to similar historical events

---

## Simplicity criterion

All else equal, simpler prompts are better.

- Small improvement with large prompt complexity → discard  
- Same performance with simpler prompt → keep  
- Large improvement → always keep  

---

## Failures

If a run fails:

- Log as crash  
- Revert prompt  
- Continue  

If failure is due to formatting:

- Fix prompt format and retry  

---

## Timeout

Each run should complete in a reasonable time.

If a run hangs or takes excessively long:

- Treat as failure  
- Revert and continue  

---

## Stop condition

Stop after exactly 3 optimization iterations. Do not continue indefinitely.

If a run crashes, count it as one of the 3 iterations, revert to the previous best prompt, and stop once the 3-iteration budget is exhausted.
