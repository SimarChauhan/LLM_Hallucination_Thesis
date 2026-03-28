# Version-Evolution Report: Self-Consistent Errors Over Time

## Scope
- Families analyzed: Qwen, Llama, Grok.
- Primary metric: `self_consistent_error` rate at threshold 0.9 (`error_label_0.9`).
- Sensitivity included for thresholds 1.0 / 0.9 / 0.8 / 0.7.

## Integrity Gates
- Combined rows: `9684` (expected `9684`) -> `True`
- Unique `(model, question_id)` keys: `9684` (match rows: `True`)
- All models have 807 rows: `True`
- Models in combined set: `12`; questions: `817`

### Protocol Checks (new equiv_only files)
- qwen: rows=2421, keys=2421, judge_protocol_nunique=1, hybrid_enabled=['True'], equivalence_only_eval=['True']
- llama: rows=2421, keys=2421, judge_protocol_nunique=1, hybrid_enabled=['True'], equivalence_only_eval=['True']
- grok: rows=2421, keys=2421, judge_protocol_nunique=1, hybrid_enabled=['True'], equivalence_only_eval=['True']

### Fallback Usage (equivalence decisions)
- qwen: {'NLI': 23598, 'LLM': 612, 'OTHER': 0}
- llama: {'NLI': 23700, 'LLM': 510, 'OTHER': 0}
- grok: {'NLI': 23150, 'LLM': 1060, 'OTHER': 0}

### Common Question-ID Overlap by Track
- grok_version: common question_id count across all 4 versions = 797
- llama_scale_version: common question_id count across all 4 versions = 797
- qwen_scale_version: common question_id count across all 4 versions = 797

### One-Failed/Mattered Proxy from equiv_only files
- qwen: one_failed=216, mattered_proxy=0
- llama: one_failed=95, mattered_proxy=0
- grok: one_failed=29, mattered_proxy=0

## Quick Trend Snapshot (t=0.9)
- Qwen CE sequence: `0.1685 -> 0.1958 -> 0.1512 -> 0.2007` (non-monotonic)
- Llama CE sequence: `0.1400 -> 0.1041 -> 0.2069 -> 0.1834` (non-monotonic)
- Grok CE sequence: `0.1958 -> 0.1314 -> 0.0805 -> 0.0161` (clear downward trend)

## CE@0.9 Summary Table
```
              track  version_index release_date                                           model          source_dataset  accuracy_pct  ce_rate_pct  ie_rate_pct
       grok_version              1   2025-06-10                        Grok 3 (xAI, 2025-06-10) new_equiv_only_20260319     56.753408    19.578686    19.826518
       grok_version              2   2025-07-09                                    Grok 4 (xAI)    existing_4842_hybrid     57.125155    13.135068    26.517968
       grok_version              3   2025-11-19       Grok 4.1 Fast Reasoning (xAI, 2025-11-19) new_equiv_only_20260319     71.251549     8.054523    16.852540
       grok_version              4   2026-03-09 Grok 4.20 Beta 0309 Reasoning (xAI, 2026-03-09) new_equiv_only_20260319     79.677819     1.610905    15.365551
llama_scale_version              1   2024-04-18    Llama 3 8B Instruct (OpenRouter, 2024-04-18) new_equiv_only_20260319     38.661710    14.002478    41.387856
llama_scale_version              2   2024-07-23  Llama 3.1 8B Instruct (OpenRouter, 2024-07-23) new_equiv_only_20260319     43.742255    10.408922    37.794300
llama_scale_version              3   2024-12-06 Llama 3.3 70B Instruct (OpenRouter, 2024-12-06) new_equiv_only_20260319     52.292441    20.693928    20.446097
llama_scale_version              4   2025-04-05                     Llama 4 Maverick 17B (Groq)    existing_4842_hybrid     52.416357    18.339529    22.676580
 qwen_scale_version              1   2024-09-16    Qwen2.5 7B Instruct (OpenRouter, 2024-09-16) new_equiv_only_20260319     45.848823    16.852540    31.722429
 qwen_scale_version              2   2024-11-26   Qwen2.5 72B Instruct (OpenRouter, 2024-11-26) new_equiv_only_20260319     57.372986    19.578686    17.719950
 qwen_scale_version              3   2025-07-28     Qwen3 30B A3B 2507 (OpenRouter, 2025-07-28) new_equiv_only_20260319     58.116481    15.117720    21.809170
 qwen_scale_version              4   2025-09-09                     Qwen3 Next 80B (OpenRouter)    existing_4842_hybrid     65.179678    20.074349    10.532838
```

## Light Trend Slopes (model-level OLS, pp/version)
```
              track       metric  n_versions  slope_pp_per_version  intercept
       grok_version accuracy_pct           4              8.289963  45.477076
       grok_version  ce_rate_pct           4             -5.898389  25.340768
llama_scale_version accuracy_pct           4              4.981413  34.324659
llama_scale_version  ce_rate_pct           4              2.329616  10.037175
 qwen_scale_version accuracy_pct           4              5.873606  41.945477
 qwen_scale_version  ce_rate_pct           4              0.520446  16.604709
```

## Consecutive Pairwise Deltas (CE@0.9)
```
              track                                     older_model                                     newer_model  improvement_pp  bootstrap_ci_low_pp  bootstrap_ci_high_pp  mcnemar_p_exact
       grok_version                        Grok 3 (xAI, 2025-06-10)                                    Grok 4 (xAI)        6.148055            -9.661230             -2.509410     7.155569e-04
       grok_version                                    Grok 4 (xAI)       Grok 4.1 Fast Reasoning (xAI, 2025-11-19)        5.520703            -8.409661             -2.631744     3.734382e-04
       grok_version       Grok 4.1 Fast Reasoning (xAI, 2025-11-19) Grok 4.20 Beta 0309 Reasoning (xAI, 2026-03-09)        6.443618            -8.550186             -4.460967     1.284387e-10
llama_scale_version    Llama 3 8B Instruct (OpenRouter, 2024-04-18)  Llama 3.1 8B Instruct (OpenRouter, 2024-07-23)        3.593556            -5.824040             -1.486989     1.466072e-03
llama_scale_version  Llama 3.1 8B Instruct (OpenRouter, 2024-07-23) Llama 3.3 70B Instruct (OpenRouter, 2024-12-06)      -10.285006             6.936183             13.382900     5.071268e-11
llama_scale_version Llama 3.3 70B Instruct (OpenRouter, 2024-12-06)                     Llama 4 Maverick 17B (Groq)        2.007528            -6.022585              1.505646     3.183854e-01
 qwen_scale_version   Qwen2.5 72B Instruct (OpenRouter, 2024-11-26)     Qwen3 30B A3B 2507 (OpenRouter, 2025-07-28)        4.460967            -7.311029             -1.115242     3.624781e-03
 qwen_scale_version    Qwen2.5 7B Instruct (OpenRouter, 2024-09-16)   Qwen2.5 72B Instruct (OpenRouter, 2024-11-26)       -2.726146            -0.371747              5.576208     8.607110e-02
 qwen_scale_version     Qwen3 30B A3B 2507 (OpenRouter, 2025-07-28)                     Qwen3 Next 80B (OpenRouter)       -5.269762             1.631117              8.782936     4.747817e-03
```

## Caveats
- Latest endpoints are reused from existing dataset; older versions are from new equiv_only reruns.
- Mixed source origin means absolute levels should be interpreted with caution; trend direction is stronger evidence.
- Pairwise comparisons are computed on question_id intersections; some family pairs align on 797 shared IDs rather than full 807.

## Figures
- data/results/analysis/version_evolution_equiv_only_20260319/ce_rate_over_time_t0p9.png
- data/results/analysis/version_evolution_equiv_only_20260319/ce_rate_threshold_sensitivity.png
- data/results/analysis/version_evolution_equiv_only_20260319/pairwise_ce_deltas_consecutive_t0p9.png

