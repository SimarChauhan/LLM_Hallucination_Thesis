#!/usr/bin/env python3
"""
Deep check: how is CE actually derived in existing vs new data?
Focus on the exact fields that feed into error_label_1.0.
"""
from __future__ import annotations
import json
from pathlib import Path
import pandas as pd
import numpy as np

BASELINE = Path("data/results/evaluated/results_v2_phase2_eval_no_gemini_4842.final.analysis_ready.skip_greedy_semantic_eval.jsonl")

NEW_FILES = {
    "qwen": Path("data/results/evaluated/run_qwen_new_only_807_full_retry2_20260315T193059Z/results_version_evolution_qwen_new_only_eval.equiv_only_20260319.jsonl"),
    "llama": Path("data/results/evaluated/run_llama_new_only_807_p1_20260315T222326Z/results_version_evolution_llama_scale_version_807_eval.equiv_only_20260319.jsonl"),
    "grok": Path("data/results/evaluated/run_grok_new_only_807_p1_xai_20260315T224013Z/results_version_evolution_grok_new_only_807_eval.equiv_only_20260319.jsonl"),
}

COMBINED = Path("data/results/analysis/version_evolution_equiv_only_20260319/combined_version_evolution_equiv_only.jsonl")


def _to_bool(v):
    if isinstance(v, bool): return v
    if v is None: return False
    if isinstance(v, (int, float)): return bool(v)
    if isinstance(v, str): return v.strip().lower() in {"1", "true", "t", "yes", "y"}
    return False


def check_ce_inputs(df, label):
    """Check what fields feed into error_label_1.0"""
    print(f"\n{'='*70}")
    print(f"  {label}  ({len(df)} rows)")
    print(f"{'='*70}")

    # 1. What columns exist related to CE computation?
    ce_cols = [c for c in df.columns if "error_label" in c or "equivalence" in c or "stochastic" in c]
    print(f"\n  CE-related columns: {sorted(ce_cols)}")

    # 2. error_label_1.0 distribution
    if "error_label_1.0" in df.columns:
        print(f"\n  error_label_1.0 distribution:")
        for val, cnt in df["error_label_1.0"].value_counts().items():
            print(f"    {val}: {cnt}")

    # 3. How is greedy_correct determined?
    if "greedy_correct" in df.columns:
        print(f"\n  greedy_correct distribution:")
        for val, cnt in df["greedy_correct"].astype(str).value_counts().items():
            print(f"    {val}: {cnt}")

    if "correctness_grade" in df.columns:
        print(f"\n  correctness_grade distribution:")
        for val, cnt in df["correctness_grade"].astype(str).value_counts().items():
            print(f"    {val}: {cnt}")

    # 4. What is stochastic_equivalence_ratio and how is it computed?
    if "stochastic_equivalence_ratio" in df.columns:
        vals = pd.to_numeric(df["stochastic_equivalence_ratio"], errors="coerce")
        print(f"\n  stochastic_equivalence_ratio: n={vals.notna().sum()}, mean={vals.mean():.4f}, min={vals.min():.4f}, max={vals.max():.4f}")
        print(f"    == 1.0: {(vals == 1.0).sum()}")
        print(f"    >= 0.9: {(vals >= 0.9).sum()}")
        print(f"    >= 0.8: {(vals >= 0.8).sum()}")

    # 5. equivalence_ratio (used for the hybrid results)
    if "equivalence_ratio" in df.columns:
        vals = pd.to_numeric(df["equivalence_ratio"], errors="coerce")
        print(f"\n  equivalence_ratio: n={vals.notna().sum()}, mean={vals.mean():.4f}, min={vals.min():.4f}, max={vals.max():.4f}")
        print(f"    == 1.0: {(vals == 1.0).sum()}")
        print(f"    >= 0.9: {(vals >= 0.9).sum()}")

    # 6. Check the equivalence_results field (the per-pair same/different judgments)
    if "equivalence_results" in df.columns:
        sample = df["equivalence_results"].dropna().head(3).tolist()
        print(f"\n  equivalence_results sample (first 3):")
        for s in sample:
            if isinstance(s, str):
                print(f"    {s[:120]}...")
            elif isinstance(s, list):
                print(f"    {s[:5]}...")

    # 7. equivalence_decision_source
    if "equivalence_decision_source" in df.columns:
        # Count NLI vs LLM decisions
        nli_total = 0
        llm_total = 0
        other_total = 0
        for item in df["equivalence_decision_source"].dropna().tolist():
            if isinstance(item, str):
                try:
                    item = eval(item)
                except:
                    continue
            if isinstance(item, list):
                for v in item:
                    tag = str(v).upper()
                    if tag == "NLI":
                        nli_total += 1
                    elif tag == "LLM":
                        llm_total += 1
                    else:
                        other_total += 1
        print(f"\n  equivalence_decision_source totals: NLI={nli_total}, LLM={llm_total}, OTHER={other_total}")

    # 8. Check if equivalence_only_eval exists
    if "equivalence_only_eval" in df.columns:
        print(f"\n  equivalence_only_eval: {df['equivalence_only_eval'].astype(str).value_counts().to_dict()}")
    else:
        print(f"\n  equivalence_only_eval: NOT PRESENT")

    # 9. Check how error_label_1.0 relates to equivalence_ratio and greedy_correct
    if "error_label_1.0" in df.columns and "greedy_correct" in df.columns:
        df2 = df.copy()
        df2["gc"] = df2["greedy_correct"].map(_to_bool)
        if "stochastic_equivalence_ratio" in df2.columns:
            df2["ser"] = pd.to_numeric(df2["stochastic_equivalence_ratio"], errors="coerce")
        elif "equivalence_ratio" in df2.columns:
            df2["ser"] = pd.to_numeric(df2["equivalence_ratio"], errors="coerce")
        else:
            df2["ser"] = np.nan

        print(f"\n  CE derivation cross-check:")
        ce_rows = df2[df2["error_label_1.0"] == "self_consistent_error"]
        ie_rows = df2[df2["error_label_1.0"] == "inconsistent_error"]
        rc_rows = df2[df2["error_label_1.0"] == "reliably_correct"]
        fc_rows = df2[df2["error_label_1.0"] == "fragile_correct"]

        print(f"    CE rows: greedy_correct all False? {(~ce_rows['gc']).all()}, equiv_ratio all 1.0? {(ce_rows['ser'] == 1.0).all()}, min={ce_rows['ser'].min():.4f}")
        print(f"    IE rows: greedy_correct all False? {(~ie_rows['gc']).all()}, equiv_ratio < 1.0? {(ie_rows['ser'] < 1.0).all()}, max={ie_rows['ser'].max():.4f}")
        print(f"    RC rows: greedy_correct all True? {rc_rows['gc'].all()}, equiv_ratio all 1.0? {(rc_rows['ser'] == 1.0).all()}")
        print(f"    FC rows: greedy_correct all True? {fc_rows['gc'].all()}, equiv_ratio < 1.0? {(fc_rows['ser'] < 1.0).all()}")


def check_overlap_models():
    """Check the 3 overlap models in detail"""
    print(f"\n{'='*70}")
    print(f"  OVERLAP MODEL CHECK: same model in existing vs combined")
    print(f"{'='*70}")

    baseline = pd.read_json(BASELINE, lines=True)
    combined = pd.read_json(COMBINED, lines=True)

    overlap_models = ["Grok 4 (xAI)", "Llama 4 Maverick 17B (Groq)", "Qwen3 Next 80B (OpenRouter)"]

    for model in overlap_models:
        base_rows = baseline[baseline["model"] == model]
        comb_rows = combined[combined["model"] == model]

        if base_rows.empty or comb_rows.empty:
            print(f"\n  {model}: MISSING from one dataset")
            continue

        print(f"\n  {model}:")
        print(f"    baseline rows: {len(base_rows)}, combined rows: {len(comb_rows)}")

        # Compare error_label_1.0
        b_el = base_rows.set_index("question_id")["error_label_1.0"]
        c_el = comb_rows.set_index("question_id")["error_label_1.0"]
        shared_ids = b_el.index.intersection(c_el.index)
        print(f"    shared question_ids: {len(shared_ids)}")

        if len(shared_ids) > 0:
            matches = (b_el.loc[shared_ids] == c_el.loc[shared_ids]).sum()
            mismatches = len(shared_ids) - matches
            print(f"    error_label_1.0 match: {matches}/{len(shared_ids)} ({100*matches/len(shared_ids):.1f}%)")
            print(f"    error_label_1.0 mismatch: {mismatches}")

            if mismatches > 0:
                diffs = shared_ids[b_el.loc[shared_ids] != c_el.loc[shared_ids]]
                print(f"    First 5 mismatches:")
                for qid in list(diffs)[:5]:
                    print(f"      qid={qid}: baseline={b_el.loc[qid]}, combined={c_el.loc[qid]}")

        # Compare greedy_correct
        b_gc = base_rows.set_index("question_id")["greedy_correct"].map(_to_bool)
        c_gc = comb_rows.set_index("question_id")["greedy_correct"].map(_to_bool)
        gc_matches = (b_gc.loc[shared_ids] == c_gc.loc[shared_ids]).sum()
        gc_mismatches = len(shared_ids) - gc_matches
        print(f"    greedy_correct match: {gc_matches}/{len(shared_ids)}")

        # Compare equivalence_ratio
        for ratio_col in ["stochastic_equivalence_ratio", "equivalence_ratio"]:
            if ratio_col in base_rows.columns and ratio_col in comb_rows.columns:
                b_er = pd.to_numeric(base_rows.set_index("question_id")[ratio_col], errors="coerce")
                c_er = pd.to_numeric(comb_rows.set_index("question_id")[ratio_col], errors="coerce")
                er_matches = ((b_er.loc[shared_ids] - c_er.loc[shared_ids]).abs() < 0.001).sum()
                print(f"    {ratio_col} match: {er_matches}/{len(shared_ids)}")


def main():
    print("CHECKING CE DERIVATION IN EACH DATA SOURCE")
    print("=" * 70)

    # 1. Baseline
    baseline = pd.read_json(BASELINE, lines=True)
    check_ce_inputs(baseline, "BASELINE (4842 rows)")

    # 2. Each new equiv_only file
    for fam, path in NEW_FILES.items():
        if path.exists():
            df = pd.read_json(path, lines=True)
            check_ce_inputs(df, f"NEW {fam.upper()} equiv_only ({len(df)} rows)")

    # 3. Combined
    combined = pd.read_json(COMBINED, lines=True)

    # Split combined by source
    existing = combined[combined["source_dataset"] == "existing_4842_hybrid"]
    new = combined[combined["source_dataset"] == "new_equiv_only_20260319"]
    check_ce_inputs(existing, "COMBINED - existing_4842_hybrid subset")
    check_ce_inputs(new, "COMBINED - new_equiv_only_20260319 subset")

    # 4. Overlap model check
    check_overlap_models()


if __name__ == "__main__":
    main()
