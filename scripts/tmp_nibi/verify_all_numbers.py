#!/usr/bin/env python3
"""
Comprehensive number verification for the professor report.
Checks every claimed number against source data.
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import pandas as pd

BASELINE = Path("data/results/evaluated/results_v2_phase2_eval_no_gemini_4842.final.analysis_ready.skip_greedy_semantic_eval.jsonl")
ANALYSIS_DIR = Path("data/results/analysis/version_evolution_equiv_only_20260319")
COMBINED = ANALYSIS_DIR / "combined_version_evolution_equiv_only.jsonl"

def _to_bool(v):
    if isinstance(v, bool): return v
    if v is None: return False
    if isinstance(v, (int, float)): return bool(v)
    if isinstance(v, str): return v.strip().lower() in {"1", "true", "t", "yes", "y"}
    return False

def verify_baseline():
    print("=" * 80)
    print("BASELINE VERIFICATION (4842 JSONL)")
    print("=" * 80)
    df = pd.read_json(BASELINE, lines=True)
    print(f"\nTotal rows: {len(df)}")

    df["is_correct"] = df["greedy_correct"].map(_to_bool)
    df["grade"] = df["correctness_grade"].astype(str).str.strip().str.upper()

    correct = (df["grade"] == "CORRECT").sum()
    incorrect = (df["grade"] == "INCORRECT").sum()
    not_attempted = (df["grade"] == "NOT_ATTEMPTED").sum()
    print(f"Correct: {correct} ({100*correct/len(df):.1f}%)")
    print(f"Incorrect: {incorrect} ({100*incorrect/len(df):.1f}%)")
    print(f"Not attempted: {not_attempted} ({100*not_attempted/len(df):.1f}%)")
    print(f"Sum check: {correct + incorrect + not_attempted} == {len(df)}: {correct + incorrect + not_attempted == len(df)}")

    print("\n--- CE/IE at each threshold (t=1.0, 0.9, 0.8, 0.7) ---")
    for t in ["1.0", "0.9", "0.8", "0.7"]:
        col = f"error_label_{t}"
        if col not in df.columns:
            print(f"  {col}: MISSING")
            continue
        ce = (df[col].astype(str) == "self_consistent_error").sum()
        ie = (df[col].astype(str) == "inconsistent_error").sum()
        ce_total_pct = 100 * ce / len(df)
        ce_wrong_pct = 100 * ce / incorrect if incorrect > 0 else 0
        print(f"  t={t}: CE={ce}, IE={ie}, CE% total={ce_total_pct:.1f}%, CE share of wrong={ce_wrong_pct:.1f}%")

    print("\n--- Per-model breakdown (t=1.0) ---")
    col10 = "error_label_1.0"
    df["is_ce_10"] = df[col10].astype(str).eq("self_consistent_error")
    df["is_ie_10"] = df[col10].astype(str).eq("inconsistent_error")

    for model, mdf in df.groupby("model"):
        n = len(mdf)
        acc = mdf["is_correct"].mean()
        n_correct = mdf["is_correct"].sum()
        n_wrong = (mdf["grade"] == "INCORRECT").sum()
        n_na = (mdf["grade"] == "NOT_ATTEMPTED").sum()
        ce = mdf["is_ce_10"].sum()
        ie = mdf["is_ie_10"].sum()
        ce_rate = 100 * ce / n
        ce_share_wrong = 100 * ce / n_wrong if n_wrong > 0 else 0
        print(f"  {model}:")
        print(f"    n={n}, correct={n_correct} ({100*acc:.1f}%), wrong={n_wrong}, NA={n_na}")
        print(f"    CE={ce} ({ce_rate:.2f}%), IE={ie}, CE share of wrong={ce_share_wrong:.1f}%")

    print("\n--- AUROC verification ---")
    from sklearn.metrics import roc_auc_score
    subset = df[(df["grade"] == "CORRECT") | (df["grade"] == "INCORRECT")].copy()
    subset["is_ce_binary"] = subset["is_ce_10"].astype(int)
    print(f"  Subset for AUROC: {len(subset)} rows (CORRECT + INCORRECT only)")

    if "disagreement_score" in subset.columns:
        valid_dis = subset.dropna(subset=["disagreement_score"])
        if len(valid_dis) > 0:
            auc_dis = roc_auc_score(valid_dis["is_ce_binary"], valid_dis["disagreement_score"])
            print(f"  Disagreement AUROC: {auc_dis:.3f}")
    if "semantic_entropy" in subset.columns:
        valid_se = subset.dropna(subset=["semantic_entropy"])
        if len(valid_se) > 0:
            auc_se = roc_auc_score(valid_se["is_ce_binary"], valid_se["semantic_entropy"])
            print(f"  Semantic entropy AUROC: {auc_se:.3f}")

    # also compute on full dataset (not just correct+incorrect)
    full_valid_dis = df.dropna(subset=["disagreement_score"]) if "disagreement_score" in df.columns else pd.DataFrame()
    if len(full_valid_dis) > 0:
        auc_dis_full = roc_auc_score(full_valid_dis["is_ce_10"].astype(int), full_valid_dis["disagreement_score"])
        print(f"  Disagreement AUROC (full 4842): {auc_dis_full:.3f}")
    full_valid_se = df.dropna(subset=["semantic_entropy"]) if "semantic_entropy" in df.columns else pd.DataFrame()
    if len(full_valid_se) > 0:
        auc_se_full = roc_auc_score(full_valid_se["is_ce_10"].astype(int), full_valid_se["semantic_entropy"])
        print(f"  Semantic entropy AUROC (full 4842): {auc_se_full:.3f}")


def verify_version_evolution():
    print("\n" + "=" * 80)
    print("VERSION-EVOLUTION VERIFICATION")
    print("=" * 80)

    summary = pd.read_csv(ANALYSIS_DIR / "model_summary_t1p0.csv")
    pairwise = pd.read_csv(ANALYSIS_DIR / "pairwise_deltas_t1p0.csv")
    checks = json.loads((ANALYSIS_DIR / "validation_checks.json").read_text())

    print(f"\nCombined rows: {checks['combined_rows']} (expected 9684)")
    print(f"n_models: {checks['n_models']}, all 807: {checks['all_models_have_807_rows']}")

    print("\n--- Model summary t=1.0 ---")
    for _, r in summary.sort_values(["track", "version_index"]).iterrows():
        print(f"  {r['family']} v{int(r['version_index'])}: {r['model']}")
        print(f"    Acc={100*r['accuracy']:.1f}%, CE={100*r['ce_rate']:.2f}%, IE={100*r['ie_rate']:.2f}%")
        print(f"    source={r['source_dataset']}, protocol={r['protocol_group']}")

    print("\n--- Consecutive pairwise CE deltas (t=1.0) ---")
    ce_consec = pairwise[(pairwise["metric"] == "ce_rate") & (pairwise["consecutive_pair"])].copy()
    ce_consec = ce_consec.sort_values(["track", "older_model"])
    for _, r in ce_consec.iterrows():
        print(f"  {r['track']}: {r['older_model']} -> {r['newer_model']}")
        print(f"    n_paired={int(r['n_paired_questions'])}, improvement_pp={r['improvement_pp']:.2f}, p={r['mcnemar_p_exact']:.2e}")

    print("\n--- Protocol check: are Study 1 and Study 2 comparable? ---")
    for fam, info in checks["new_file_protocol_checks"].items():
        print(f"  {fam} new files: protocol={info['judge_protocol_values']}")
        print(f"    hybrid_enabled={info['hybrid_enabled_values']}")
        print(f"    equiv_only={info['equivalence_only_eval_values']}")

    print("\n--- Source dataset for each model ---")
    for _, r in summary.sort_values(["track", "version_index"]).iterrows():
        print(f"  {r['model']}: source={r['source_dataset']}, protocol_group={r['protocol_group']}")


def check_protocol_comparability():
    print("\n" + "=" * 80)
    print("APPLES-TO-APPLES CHECK: PROTOCOL COMPARISON")
    print("=" * 80)

    df_base = pd.read_json(BASELINE, lines=True)
    print(f"\nBaseline JSONL columns related to protocol:")
    proto_cols = [c for c in df_base.columns if any(k in c.lower() for k in ["protocol", "judge", "hybrid", "equiv", "nli"])]
    for c in proto_cols:
        vals = df_base[c].dropna().astype(str).unique()
        print(f"  {c}: {sorted(vals)[:5]} ({'...' if len(vals) > 5 else ''})")

    if "judge_protocol" in df_base.columns:
        print(f"\n  Baseline judge_protocol distribution:")
        for val, cnt in df_base["judge_protocol"].value_counts().items():
            print(f"    {val}: {cnt}")

    if "hybrid_enabled" in df_base.columns:
        print(f"\n  Baseline hybrid_enabled distribution:")
        for val, cnt in df_base["hybrid_enabled"].value_counts().items():
            print(f"    {val}: {cnt}")

    if "equivalence_only_eval" in df_base.columns:
        print(f"\n  Baseline equivalence_only_eval distribution:")
        for val, cnt in df_base["equivalence_only_eval"].value_counts().items():
            print(f"    {val}: {cnt}")

    # Check the equiv_only files
    checks = json.loads((ANALYSIS_DIR / "validation_checks.json").read_text())
    print(f"\n  New equiv_only files:")
    for fam, info in checks["new_file_protocol_checks"].items():
        print(f"    {fam}: protocol={info['judge_protocol_values']}, hybrid={info['hybrid_enabled_values']}, equiv_only={info['equivalence_only_eval_values']}")

    # Check the existing_4842_hybrid models in combined
    combined = pd.read_json(COMBINED, lines=True)
    existing = combined[combined["source_dataset"] == "existing_4842_hybrid"]
    new = combined[combined["source_dataset"] == "new_equiv_only_20260319"]
    print(f"\n  Combined dataset breakdown:")
    print(f"    existing_4842_hybrid rows: {len(existing)}")
    print(f"    new_equiv_only_20260319 rows: {len(new)}")

    if "judge_protocol" in existing.columns:
        print(f"\n  Existing rows judge_protocol:")
        for val, cnt in existing["judge_protocol"].value_counts().items():
            print(f"    {val}: {cnt}")

    if "judge_protocol" in new.columns:
        print(f"\n  New rows judge_protocol:")
        for val, cnt in new["judge_protocol"].value_counts().items():
            print(f"    {val}: {cnt}")

    if "equivalence_only_eval" in existing.columns:
        print(f"\n  Existing rows equivalence_only_eval:")
        for val, cnt in existing["equivalence_only_eval"].astype(str).value_counts().items():
            print(f"    {val}: {cnt}")

    if "equivalence_only_eval" in new.columns:
        print(f"\n  New rows equivalence_only_eval:")
        for val, cnt in new["equivalence_only_eval"].astype(str).value_counts().items():
            print(f"    {val}: {cnt}")

    # Check how error_label_1.0 was computed for existing vs new
    print(f"\n  error_label_1.0 distribution in existing rows:")
    if "error_label_1.0" in existing.columns:
        for val, cnt in existing["error_label_1.0"].value_counts().items():
            print(f"    {val}: {cnt}")
    print(f"  error_label_1.0 distribution in new rows:")
    if "error_label_1.0" in new.columns:
        for val, cnt in new["error_label_1.0"].value_counts().items():
            print(f"    {val}: {cnt}")

    # check stochastic_equivalence_ratio source
    print(f"\n  Stochastic equivalence ratio stats:")
    for label, sub in [("existing", existing), ("new", new)]:
        if "stochastic_equivalence_ratio" in sub.columns:
            vals = sub["stochastic_equivalence_ratio"].dropna()
            print(f"    {label}: n={len(vals)}, mean={vals.mean():.4f}, min={vals.min():.4f}, max={vals.max():.4f}")


if __name__ == "__main__":
    verify_baseline()
    verify_version_evolution()
    check_protocol_comparability()
