#!/usr/bin/env python3
"""
Comprehensive validation of all numbers in the v2_thesis_hybrid_analysis_report.tex
against the source data file.
"""

import json
import pandas as pd
import numpy as np
from pathlib import Path
from collections import defaultdict
from itertools import combinations

DATA_FILE = Path("data/results/evaluated/results_v2_phase2_eval_no_gemini_4842.final.analysis_ready.skip_greedy_semantic_eval.jsonl")

def load_data():
    """Load the JSONL data into a DataFrame."""
    records = []
    with open(DATA_FILE) as f:
        for line in f:
            records.append(json.loads(line))
    return pd.DataFrame(records)

def check_value(name, computed, reported, tolerance=0.15):
    """Check if computed value matches reported value within tolerance."""
    if isinstance(computed, float) and isinstance(reported, float):
        diff = abs(computed - reported)
        match = diff <= tolerance
        status = "✓" if match else "✗"
        print(f"  {status} {name}: computed={computed:.2f}, reported={reported:.2f}, diff={diff:.2f}")
    else:
        match = computed == reported
        status = "✓" if match else "✗"
        print(f"  {status} {name}: computed={computed}, reported={reported}")
    return match

def main():
    print("="*80)
    print("COMPREHENSIVE REPORT NUMBER VALIDATION")
    print("="*80)
    
    print("\nLoading data...")
    df = load_data()
    
    all_checks = []
    
    # ============================================
    # SECTION 1: Basic counts
    # ============================================
    print("\n" + "="*80)
    print("SECTION 1: BASIC COUNTS")
    print("="*80)
    
    row_count = len(df)
    all_checks.append(check_value("Row count", row_count, 4842))
    
    question_count = df['question_id'].nunique()
    all_checks.append(check_value("Question count", question_count, 807))
    
    model_count = df['model'].nunique()
    all_checks.append(check_value("Model count", model_count, 6))
    
    # Check for duplicates
    duplicates = df.duplicated(subset=['question_id', 'model']).sum()
    all_checks.append(check_value("Duplicate count", duplicates, 0))
    
    # ============================================
    # SECTION 2: Overall rates (Table 6 in report)
    # ============================================
    print("\n" + "="*80)
    print("SECTION 2: OVERALL RATES (Table 6)")
    print("="*80)
    
    # Correctness distribution
    correct_count = (df['correctness_grade'] == 'CORRECT').sum()
    incorrect_count = (df['correctness_grade'] == 'INCORRECT').sum()
    not_attempted_count = (df['correctness_grade'] == 'NOT_ATTEMPTED').sum()
    
    accuracy = 100 * correct_count / row_count
    incorrect_rate = 100 * incorrect_count / row_count
    not_attempted_rate = 100 * not_attempted_count / row_count
    
    all_checks.append(check_value("Accuracy %", accuracy, 63.3))
    all_checks.append(check_value("Incorrect rate %", incorrect_rate, 32.2))
    all_checks.append(check_value("NOT_ATTEMPTED rate %", not_attempted_rate, 4.5))
    
    # CE/IE at threshold 1.0
    ce_count_1_0 = (df['error_label_1.0'] == 'self_consistent_error').sum()
    ie_count_1_0 = (df['error_label_1.0'] == 'inconsistent_error').sum()
    
    ce_rate_1_0 = 100 * ce_count_1_0 / row_count
    ie_rate_1_0 = 100 * ie_count_1_0 / row_count
    
    all_checks.append(check_value("CE rate (t=1.0) %", ce_rate_1_0, 13.5))
    all_checks.append(check_value("IE rate (t=1.0) %", ie_rate_1_0, 18.7))
    
    # CE share among incorrect (Table 7)
    ce_share_incorrect = 100 * ce_count_1_0 / incorrect_count
    ie_share_incorrect = 100 * ie_count_1_0 / incorrect_count
    
    all_checks.append(check_value("CE share among incorrect %", ce_share_incorrect, 42.1))
    all_checks.append(check_value("IE share among incorrect %", ie_share_incorrect, 57.9))
    
    print(f"\n  Raw counts: CORRECT={correct_count}, INCORRECT={incorrect_count}, NOT_ATTEMPTED={not_attempted_count}")
    print(f"  CE count (t=1.0)={ce_count_1_0}, IE count (t=1.0)={ie_count_1_0}")
    
    # ============================================
    # SECTION 3: Per-model accuracy (Table 8)
    # ============================================
    print("\n" + "="*80)
    print("SECTION 3: PER-MODEL ACCURACY (Table 8)")
    print("="*80)
    
    reported_accuracy = {
        'Claude Opus 4.6 (Anthropic)': 74.7,
        'GPT-5.2 (OpenAI)': 69.8,
        'Qwen3 Next 80B (OpenRouter)': 65.2,
        'DeepSeek V3.2 (DeepSeek)': 60.3,
        'Grok 4 (xAI)': 57.1,
        'Llama 4 Maverick 17B (Groq)': 52.4,
    }
    
    for model in df['model'].unique():
        model_df = df[df['model'] == model]
        model_n = len(model_df)
        model_correct = (model_df['correctness_grade'] == 'CORRECT').sum()
        computed_acc = 100 * model_correct / model_n
        reported_acc = reported_accuracy.get(model, 0)
        all_checks.append(check_value(f"{model[:20]} accuracy", computed_acc, reported_acc))
    
    # ============================================
    # SECTION 4: Per-model CE share (Table 9)
    # ============================================
    print("\n" + "="*80)
    print("SECTION 4: PER-MODEL ERROR CHARACTER (Table 9, threshold 1.0)")
    print("="*80)
    
    reported_ce_share = {
        'Qwen3 Next 80B (OpenRouter)': 59.9,
        'Claude Opus 4.6 (Anthropic)': 53.8,
        'DeepSeek V3.2 (DeepSeek)': 51.8,
        'Llama 4 Maverick 17B (Groq)': 37.8,
        'GPT-5.2 (OpenAI)': 28.1,
        'Grok 4 (xAI)': 27.5,
    }
    
    reported_incorrect_pct = {
        'Qwen3 Next 80B (OpenRouter)': 30.6,
        'Claude Opus 4.6 (Anthropic)': 20.9,
        'DeepSeek V3.2 (DeepSeek)': 34.2,
        'Llama 4 Maverick 17B (Groq)': 41.0,
        'GPT-5.2 (OpenAI)': 26.9,
        'Grok 4 (xAI)': 39.7,
    }
    
    for model in df['model'].unique():
        model_df = df[df['model'] == model]
        model_incorrect = (model_df['correctness_grade'] == 'INCORRECT').sum()
        model_ce = (model_df['error_label_1.0'] == 'self_consistent_error').sum()
        
        computed_incorrect_pct = 100 * model_incorrect / len(model_df)
        computed_ce_share = 100 * model_ce / model_incorrect if model_incorrect > 0 else 0
        
        reported_inc = reported_incorrect_pct.get(model, 0)
        reported_ce = reported_ce_share.get(model, 0)
        
        all_checks.append(check_value(f"{model[:20]} incorrect%", computed_incorrect_pct, reported_inc))
        all_checks.append(check_value(f"{model[:20]} CE share", computed_ce_share, reported_ce))
    
    # ============================================
    # SECTION 5: Cross-model overlap (Table in Section 11)
    # ============================================
    print("\n" + "="*80)
    print("SECTION 5: CROSS-MODEL CE OVERLAP (threshold 0.9)")
    print("="*80)
    
    # Build question-level CE lookup at threshold 0.9
    df_ce_09 = df[df['error_label_0.9'] == 'self_consistent_error']
    question_ce_models = defaultdict(set)
    for _, row in df_ce_09.iterrows():
        question_ce_models[row['question_id']].add(row['model'])
    
    # Reported overlap values from the report
    reported_overlaps = {
        ('DeepSeek V3.2 (DeepSeek)', 'Qwen3 Next 80B (OpenRouter)'): 86,
        ('DeepSeek V3.2 (DeepSeek)', 'Llama 4 Maverick 17B (Groq)'): 83,
        ('Llama 4 Maverick 17B (Groq)', 'Qwen3 Next 80B (OpenRouter)'): 79,
        ('Claude Opus 4.6 (Anthropic)', 'DeepSeek V3.2 (DeepSeek)'): 73,
        ('DeepSeek V3.2 (DeepSeek)', 'Grok 4 (xAI)'): 66,
        ('Grok 4 (xAI)', 'Qwen3 Next 80B (OpenRouter)'): 61,
        ('Grok 4 (xAI)', 'Llama 4 Maverick 17B (Groq)'): 58,
        ('Claude Opus 4.6 (Anthropic)', 'Llama 4 Maverick 17B (Groq)'): 51,
        ('DeepSeek V3.2 (DeepSeek)', 'GPT-5.2 (OpenAI)'): 51,
        ('Claude Opus 4.6 (Anthropic)', 'Qwen3 Next 80B (OpenRouter)'): 50,
        ('GPT-5.2 (OpenAI)', 'Qwen3 Next 80B (OpenRouter)'): 48,
        ('GPT-5.2 (OpenAI)', 'Llama 4 Maverick 17B (Groq)'): 46,
        ('Claude Opus 4.6 (Anthropic)', 'Grok 4 (xAI)'): 45,
        ('GPT-5.2 (OpenAI)', 'Grok 4 (xAI)'): 43,
        ('Claude Opus 4.6 (Anthropic)', 'GPT-5.2 (OpenAI)'): 41,
    }
    
    models = df['model'].unique().tolist()
    for (model_a, model_b), reported_overlap in reported_overlaps.items():
        # Count questions where both have CE
        overlap = sum(1 for qid, ce_models in question_ce_models.items() 
                     if model_a in ce_models and model_b in ce_models)
        all_checks.append(check_value(f"Overlap {model_a[:10]}--{model_b[:10]}", overlap, reported_overlap))
    
    # Per-model CE counts at threshold 0.9
    print("\n  Per-model CE counts at threshold 0.9:")
    reported_ce_counts_09 = {
        'Claude Opus 4.6 (Anthropic)': 100,
        'DeepSeek V3.2 (DeepSeek)': 168,
        'GPT-5.2 (OpenAI)': 79,
        'Grok 4 (xAI)': 106,
        'Llama 4 Maverick 17B (Groq)': 148,
        'Qwen3 Next 80B (OpenRouter)': 162,
    }
    for model in models:
        ce_count = (df[(df['model'] == model) & (df['error_label_0.9'] == 'self_consistent_error')]).shape[0]
        reported = reported_ce_counts_09.get(model, 0)
        all_checks.append(check_value(f"  {model[:20]} CE count (0.9)", ce_count, reported))
    
    # ============================================
    # SECTION 6: Threshold sensitivity (Table in report)
    # ============================================
    print("\n" + "="*80)
    print("SECTION 6: THRESHOLD SENSITIVITY")
    print("="*80)
    
    reported_ce_counts = {
        0.7: 919,
        0.8: 841,
        0.9: 763,
        1.0: 656,
    }
    
    reported_ie_counts = {
        0.7: 641,
        0.8: 719,
        0.9: 797,
        1.0: 904,
    }
    
    for threshold in [0.7, 0.8, 0.9, 1.0]:
        col = f'error_label_{threshold}'
        ce_count = (df[col] == 'self_consistent_error').sum()
        ie_count = (df[col] == 'inconsistent_error').sum()
        
        all_checks.append(check_value(f"CE count (t={threshold})", ce_count, reported_ce_counts[threshold]))
        all_checks.append(check_value(f"IE count (t={threshold})", ie_count, reported_ie_counts[threshold]))
    
    # ============================================
    # SECTION 7: Five-category breakdown at 0.9 (Table 5)
    # ============================================
    print("\n" + "="*80)
    print("SECTION 7: FIVE-CATEGORY BREAKDOWN (threshold 0.9)")
    print("="*80)
    
    # Reported values from Table 5
    reported_breakdown_09 = {
        'Claude Opus 4.6 (Anthropic)': {'Corr+Same': 48.8, 'Corr+Diff': 25.9, 'Inc+Same': 12.4, 'Inc+Diff': 8.6, 'N/A': 4.3},
        'DeepSeek V3.2 (DeepSeek)': {'Corr+Same': 42.9, 'Corr+Diff': 17.5, 'Inc+Same': 20.8, 'Inc+Diff': 13.4, 'N/A': 5.5},
        'GPT-5.2 (OpenAI)': {'Corr+Same': 32.7, 'Corr+Diff': 37.1, 'Inc+Same': 9.8, 'Inc+Diff': 17.1, 'N/A': 3.3},
        'Grok 4 (xAI)': {'Corr+Same': 17.6, 'Corr+Diff': 39.5, 'Inc+Same': 13.1, 'Inc+Diff': 26.5, 'N/A': 3.2},
        'Llama 4 Maverick 17B (Groq)': {'Corr+Same': 32.0, 'Corr+Diff': 20.4, 'Inc+Same': 18.3, 'Inc+Diff': 22.7, 'N/A': 6.6},
        'Qwen3 Next 80B (OpenRouter)': {'Corr+Same': 46.3, 'Corr+Diff': 18.8, 'Inc+Same': 20.1, 'Inc+Diff': 10.5, 'N/A': 4.2},
    }
    
    for model in df['model'].unique():
        model_df = df[df['model'] == model]
        n = len(model_df)
        
        rel_corr = (model_df['error_label_0.9'] == 'reliably_correct').sum()
        frag_corr = (model_df['error_label_0.9'] == 'fragile_correct').sum()
        sc_err = (model_df['error_label_0.9'] == 'self_consistent_error').sum()
        inc_err = (model_df['error_label_0.9'] == 'inconsistent_error').sum()
        not_att = (model_df['error_label_0.9'] == 'not_attempted').sum()
        
        computed = {
            'Corr+Same': 100 * rel_corr / n,
            'Corr+Diff': 100 * frag_corr / n,
            'Inc+Same': 100 * sc_err / n,
            'Inc+Diff': 100 * inc_err / n,
            'N/A': 100 * not_att / n,
        }
        
        reported = reported_breakdown_09.get(model, {})
        print(f"\n  {model}:")
        for key in ['Corr+Same', 'Corr+Diff', 'Inc+Same', 'Inc+Diff', 'N/A']:
            all_checks.append(check_value(f"    {key}", computed[key], reported.get(key, 0)))
    
    # ============================================
    # SECTION 8: Entropy quintile analysis
    # ============================================
    print("\n" + "="*80)
    print("SECTION 8: SEMANTIC ENTROPY QUINTILE ANALYSIS (threshold 1.0)")
    print("="*80)
    
    # Filter to answered rows only
    df_answered = df[df['correctness_grade'].isin(['CORRECT', 'INCORRECT'])].copy()
    print(f"  Answered rows: {len(df_answered)}")
    
    # Create quintile bins (handle duplicate edges)
    df_answered['entropy_quintile'] = pd.qcut(df_answered['semantic_entropy'].rank(method='first'), q=5, labels=['Q1', 'Q2', 'Q3', 'Q4', 'Q5'])
    
    reported_quintiles = {
        'Q1': {'total': 925, 'incorrect': 270, 'ce': 265, 'ce_share': 98.1},
        'Q2': {'total': 924, 'incorrect': 302, 'ce': 293, 'ce_share': 97.0},
        'Q3': {'total': 925, 'incorrect': 253, 'ce': 98, 'ce_share': 38.7},
        'Q4': {'total': 924, 'incorrect': 325, 'ce': 0, 'ce_share': 0.0},
        'Q5': {'total': 925, 'incorrect': 410, 'ce': 0, 'ce_share': 0.0},
    }
    
    for q in ['Q1', 'Q2', 'Q3', 'Q4', 'Q5']:
        q_df = df_answered[df_answered['entropy_quintile'] == q]
        total = len(q_df)
        incorrect = (q_df['correctness_grade'] == 'INCORRECT').sum()
        # Entropy stratification in the thesis tables is reported at strict CE threshold 1.0.
        ce = (q_df['error_label_1.0'] == 'self_consistent_error').sum()
        ce_share = 100 * ce / incorrect if incorrect > 0 else 0
        
        reported = reported_quintiles[q]
        print(f"\n  {q}:")
        all_checks.append(check_value(f"    Total rows", total, reported['total'], tolerance=5))
        all_checks.append(check_value(f"    Incorrect", incorrect, reported['incorrect'], tolerance=5))
        all_checks.append(check_value(f"    CE count", ce, reported['ce'], tolerance=5))
        all_checks.append(check_value(f"    CE share %", ce_share, reported['ce_share'], tolerance=2))
    
    # ============================================
    # SECTION 9: Label partition check
    # ============================================
    print("\n" + "="*80)
    print("SECTION 9: LABEL PARTITION CHECK (threshold 0.9)")
    print("="*80)
    
    # Reported in validation checklist
    reported_partition = {
        'reliably_correct': 1778,
        'fragile_correct': 1285,
        'self_consistent_error': 763,
        'inconsistent_error': 797,
        'not_attempted': 219,
    }
    
    for label, reported_count in reported_partition.items():
        computed_count = (df['error_label_0.9'] == label).sum()
        all_checks.append(check_value(f"{label}", computed_count, reported_count))
    
    # ============================================
    # FINAL SUMMARY
    # ============================================
    print("\n" + "="*80)
    print("FINAL SUMMARY")
    print("="*80)
    
    passed = sum(all_checks)
    total = len(all_checks)
    failed = total - passed
    
    print(f"\n  Total checks: {total}")
    print(f"  Passed: {passed}")
    print(f"  Failed: {failed}")
    
    if failed == 0:
        print("\n  ✓✓✓ ALL NUMBERS VERIFIED CORRECTLY ✓✓✓")
    else:
        print(f"\n  ✗✗✗ {failed} DISCREPANCIES FOUND ✗✗✗")

if __name__ == "__main__":
    main()
