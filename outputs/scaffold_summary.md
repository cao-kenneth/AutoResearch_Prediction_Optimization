# Scaffold Comparison Summary

Generated: 2026-05-31T09:44:23.011776
Mode: OpenAI API

Best validation scaffold: `self_critique` with Brier 0.147497 across 34 valid rows.

| Split | Config | Valid rows | Failures | Brier | Calls/question |
| --- | --- | ---: | ---: | ---: | ---: |
| dev | `baseline_one_shot` | 33 | 0 | 0.276224 | 1 |
| dev | `premortem_one_shot` | 33 | 0 | 0.247773 | 1 |
| dev | `self_critique` | 33 | 0 | 0.255970 | 2 |
| dev | `three_perspectives_judge` | 33 | 0 | 0.259070 | 4 |
| validation | `baseline_one_shot` | 34 | 0 | 0.148303 | 1 |
| validation | `premortem_one_shot` | 34 | 0 | 0.150247 | 1 |
| validation | `self_critique` | 34 | 0 | 0.147497 | 2 |
| validation | `three_perspectives_judge` | 34 | 0 | 0.152147 | 4 |
