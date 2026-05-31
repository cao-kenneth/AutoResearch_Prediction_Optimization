# Scaffold Optimizer Report

Generated: 2026-05-31T13:54:26.378860
Mode: OpenAI API
Generated candidate configs: 556
Acceptance threshold: 0.005
Cost penalty per extra call: 0.002

Best cost-adjusted validation config: `p1_sk_pm0_agg_mean_ref_none` (Brier 0.131979, calls 1, score 0.131979).

## Decisions

- Loaded 670 prior optimizer prediction rows for 10 already-evaluated configs.
- Candidate search space: 556 valid generated configs under the five-call budget.
- Iteration 11: selected p1_sk_pm0_agg_mean_ref_rare. Rationale: The strongest observed validation performer is the skeptic, no-premortem, mean, no-refinement scaffold, but the error report shows severe underprediction on rare-but-real YES outcomes. Among the available candidates, adding the rare_yes_audit is the most targeted way to address that failure mode while keeping cost low (2 calls, same as other cheap options). It is also the closest low-cost variant to the best-performing scaffold, so it offers the best expected Brier improvement under the cost-adjusted budget.
- Iteration 11: evaluated 67 prediction rows in 41.0s.
- Rejected `p1_sk_pm0_agg_mean_ref_rare`: cost-adjusted validation score improved by only -0.008965; threshold is 0.005000.
- Iteration 12: selected p1_sk_pm0_agg_mean_ref_res. Rationale: The skeptic, no-premortem, mean-aggregation family is the strongest observed performer on both dev and validation, with p1_sk_pm0_agg_mean_ref_none already best on validation and p1_sk_pm0_agg_mean_ref_rare best on dev. Given the stated failure modes, resolution_audit is the most plausible refinement to improve deadline/resolution handling while keeping cost at only 2 calls. It is a low-risk upgrade over the current best config and cheaper than judge-based or multi-perspective options.
- Iteration 12: evaluated 67 prediction rows in 38.2s.
- Rejected `p1_sk_pm0_agg_mean_ref_res`: cost-adjusted validation score improved by only -0.015659; threshold is 0.005000.
- Iteration 13: selected p1_sk_pm0_agg_mean_ref_self. Rationale: The skeptic, no-premortem, mean-aggregation family is clearly strongest on both dev and validation. The best observed config is p1_sk_pm0_agg_mean_ref_none, but the error report shows severe underprediction on rare-but-real YES outcomes, which rare_yes_audit helped on dev while keeping the same perspective/aggregation. Among the available options, p1_sk_pm0_agg_mean_ref_self is the cheapest 2-call variant in the same strong family and is a reasonable hedge against overconfidence without paying the larger 3+ call penalty. I would avoid judge-based or multi-perspective configs because their validation results are weaker and the extra-call Brier penalty makes them less attractive under budget.
- Iteration 13: evaluated 67 prediction rows in 43.1s.
- Rejected `p1_sk_pm0_agg_mean_ref_self`: cost-adjusted validation score improved by only -0.025715; threshold is 0.005000.
- Iteration 14: selected p1_sk_pm0_agg_judge_ref_none. Rationale: The skeptic, no-premortem, judge aggregation variant is the closest higher-capacity extension of the current best config while staying relatively cheap at 2 calls. Validation shows the skeptic family is strongest overall, and judge aggregation may help with the observed short-window overconfidence and rare-event misses without paying the larger 3+ call penalty. I’m avoiding extra-call-heavy configs because the cost-adjusted penalty makes them harder to justify absent stronger validation evidence.
- Iteration 14: evaluated 67 prediction rows in 34.3s.
- Rejected `p1_sk_pm0_agg_judge_ref_none`: cost-adjusted validation score improved by only -0.002506; threshold is 0.005000.
- Iteration 15: selected p1_sk_pm0_agg_judge_ref_self. Rationale: The strongest observed dev/validation performer among the listed candidates is the skeptic, no-premortem family, and the judge aggregation appears to address the current model’s main failure mode: rare-but-real YES underprediction and some short-window overconfidence. Compared with the current best p1_sk_pm0_agg_mean_ref_none, judge-based skeptic configs slightly improve validation Brier despite extra cost, and the cost-adjusted penalty is modest relative to the likely calibration gain. I chose the self-critique refinement over none/rare/res because it is the most plausible low-risk way to improve resolution/deadline handling without jumping to even more expensive multi-perspective scaffolds.
- Iteration 15: evaluated 67 prediction rows in 60.4s.
- Rejected `p1_sk_pm0_agg_judge_ref_self`: cost-adjusted validation score improved by only -0.023041; threshold is 0.005000.
- Iteration 16: selected p1_sk_pm0_agg_judge_ref_rare. Rationale: The strongest validated pattern is skeptic + premortem off + judge/rare_yes_audit: it is among the best dev performers in the provided results, and it directly targets the observed failure mode of underpredicting rare-but-real YES outcomes. Although it costs 3 calls, the expected calibration gain appears worth the modest 0.004 Brier penalty versus a 1-call config. I avoided higher-call variants because the incremental evidence for extra refinement is weak and the budget favors cheaper options.
- Iteration 16: evaluated 67 prediction rows in 61.4s.
- Rejected `p1_sk_pm0_agg_judge_ref_rare`: cost-adjusted validation score improved by only -0.020336; threshold is 0.005000.
- Iteration 17: selected p1_sk_pm0_agg_judge_ref_res. Rationale: The skeptic single-perspective configs are clearly strongest on both dev and validation, and the best validation result among the listed candidates is p1_sk_pm0_agg_mean_ref_none. However, the error report shows the current best underpredicts rare-but-real YES outcomes and makes some resolution mistakes; adding a resolution audit is the most targeted fix. Among the available options, p1_sk_pm0_agg_judge_ref_res is the closest higher-cost variant that directly addresses resolution/deadline errors while staying within the same skeptical framing that already performs best. I’m choosing this over cheaper alternatives because the known failure modes suggest the extra audit call is justified despite the cost penalty.
- Iteration 17: evaluated 67 prediction rows in 54.2s.
- Rejected `p1_sk_pm0_agg_judge_ref_res`: cost-adjusted validation score improved by only -0.009156; threshold is 0.005000.
- Iteration 18: selected p1_sk_pm1_agg_mean_ref_none. Rationale: The strongest observed pattern is that the skeptic perspective with mean aggregation and no refinement is already best on validation, and it is the cheapest validated-style option at 1 call. It directly addresses the main failure mode of underpredicting rare-but-real YES outcomes better than the current best by likely being less overconfidently low than the current skeptic setup, while avoiding extra-call penalties. More expensive refinements and judge-based variants did not show enough consistent validation gain to justify their cost under the 0.002 Brier-per-extra-call penalty.
- Iteration 18: evaluated 67 prediction rows in 19.0s.
- Rejected `p1_sk_pm1_agg_mean_ref_none`: cost-adjusted validation score improved by only -0.011738; threshold is 0.005000.
- Iteration 19: selected p1_sk_pm1_agg_mean_ref_rare. Rationale: The skeptic + premortem + mean + rare_yes_audit config is the best low-cost hedge against the observed failure mode of underpredicting rare-but-real YES outcomes. It is only 2 calls/question, so the cost penalty is modest, and on validation it is competitive with the best skeptic variants while adding an explicit rare-YES audit that should help on the hardest false-negative cases without the extra cost of judge-based 3+ call configs. I would not pay for higher-call judge variants given the budget penalty and the weak validation evidence that they beat the cheaper skeptic mean family.
- Iteration 19: evaluated 67 prediction rows in 41.8s.
- Rejected `p1_sk_pm1_agg_mean_ref_rare`: cost-adjusted validation score improved by only -0.028045; threshold is 0.005000.
- Iteration 20: selected p1_sk_pm1_agg_mean_ref_res. Rationale: The skeptic single-perspective mean config is already the strongest low-cost family on validation, and adding premortem plus resolution audit is a plausible way to address the observed failure modes around deadline/resolution mistakes without paying for a full judge stack. It stays at 2 calls, so the cost penalty is modest, and it is more defensible than higher-call judge variants given the budget. I prefer it over the current best because the current config shows severe underprediction on rare YES outcomes and some resolution errors; resolution audit should help specifically with those cases while keeping cost controlled.
- Iteration 20: evaluated 67 prediction rows in 38.2s.
- Rejected `p1_sk_pm1_agg_mean_ref_res`: cost-adjusted validation score improved by only -0.038997; threshold is 0.005000.

## Validation Summary

| Config | Brier | Calls/question | Cost-adjusted score |
| --- | ---: | ---: | ---: |
| `p1_sk_pm0_agg_mean_ref_none` | 0.131979 | 1 | 0.131979 |
| `p1_sk_pm0_agg_judge_ref_none` | 0.132485 | 2 | 0.134485 |
| `p1_sk_pm0_agg_mean_ref_rare` | 0.138944 | 2 | 0.140944 |
| `p1_sk_pm0_agg_judge_ref_res` | 0.137135 | 3 | 0.141135 |
| `p1_sk_pm1_agg_mean_ref_none` | 0.143718 | 1 | 0.143718 |
| `p1_sk_pm0_agg_mean_ref_res` | 0.145638 | 2 | 0.147638 |
| `baseline_one_shot` | 0.148303 | 1 | 0.148303 |
| `self_critique` | 0.147497 | 2 | 0.149497 |
| `premortem_one_shot` | 0.150247 | 1 | 0.150247 |
| `p1_sk_pm0_agg_judge_ref_rare` | 0.148315 | 3 | 0.152315 |
| `p1_br_pm1_agg_mean_ref_res` | 0.152041 | 2 | 0.154041 |
| `p1_sk_pm0_agg_judge_ref_self` | 0.151021 | 3 | 0.155021 |
| `p1_br_pm0_agg_mean_ref_res` | 0.153021 | 2 | 0.155021 |
| `p1_sk_pm0_agg_mean_ref_self` | 0.155694 | 2 | 0.157694 |
| `three_perspectives_judge` | 0.152147 | 4 | 0.158147 |
| `p1_br_pm0_agg_judge_ref_rare` | 0.155598 | 3 | 0.159598 |
| `p1_sk_pm1_agg_mean_ref_rare` | 0.158024 | 2 | 0.160024 |
| `p1_br_pm0_agg_judge_ref_none` | 0.160112 | 2 | 0.162112 |
| `p1_br_pm0_agg_judge_ref_self` | 0.161774 | 3 | 0.165774 |
| `p1_br_pm1_agg_mean_ref_rare` | 0.164498 | 2 | 0.166498 |
| `p1_br_pm0_agg_judge_ref_res` | 0.163468 | 3 | 0.167468 |
| `p1_br_pm1_agg_mean_ref_self` | 0.166710 | 2 | 0.168710 |
| `p1_sk_pm1_agg_mean_ref_res` | 0.168976 | 2 | 0.170976 |
| `p1_br_pm0_agg_mean_ref_rare` | 0.176344 | 2 | 0.178344 |
