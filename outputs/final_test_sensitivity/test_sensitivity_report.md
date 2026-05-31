# Final Test Sensitivity Report

Generated: 2026-05-31T14:15:54.885124
Mode: OpenAI API
Temperatures tested for LLM configs: 0.0, 0.7

This is a post-hoc sensitivity analysis and was not used to select the final scaffold.

Best test result in this analysis: `community` at temperature `fixed` with Brier 0.144584.

| Temperature | Config | Valid rows | Failures | Brier | Calls/question |
| --- | --- | ---: | ---: | ---: | ---: |
| fixed | `community` | 45 | 0 | 0.144584 | 0 |
| fixed | `guess_50_50` | 45 | 0 | 0.250000 | 0 |
| 0.0 | `baseline_one_shot` | 45 | 0 | 0.196264 | 1 |
| 0.0 | `premortem_one_shot` | 45 | 0 | 0.214019 | 1 |
| 0.0 | `self_critique` | 45 | 0 | 0.195632 | 2 |
| 0.0 | `three_perspectives_judge` | 45 | 0 | 0.202684 | 4 |
| 0.0 | `p1_sk_pm0_agg_mean_ref_none` | 44 | 1 | 0.209552 | 1 |
| 0.7 | `baseline_one_shot` | 45 | 0 | 0.215162 | 1 |
| 0.7 | `premortem_one_shot` | 45 | 0 | 0.223961 | 1 |
| 0.7 | `self_critique` | 45 | 0 | 0.188136 | 2 |
| 0.7 | `three_perspectives_judge` | 45 | 0 | 0.202614 | 4 |
| 0.7 | `p1_sk_pm0_agg_mean_ref_none` | 45 | 0 | 0.246891 | 1 |
