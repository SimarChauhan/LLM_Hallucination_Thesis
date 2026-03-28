#!/usr/bin/env python3
"""Generate a thesis-style LaTeX report and figures from analysis outputs."""

from __future__ import annotations

import json
import math
import os
import re
import shutil
import subprocess
from collections import Counter
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns


REPO_ROOT = Path(__file__).resolve().parents[1]
BASE = Path(os.environ.get("PROJECT_ROOT", str(REPO_ROOT)))
ANALYSIS_DIR = BASE / 'data/results/analysis/final_analysis_ready'
FINAL_JSONL_CANDIDATES = [
    BASE / 'data/results/evaluated/results_v2_phase2_eval_no_gemini_4842.final.analysis_ready.jsonl',
    BASE / 'data/results/evaluated/results_v2_phase2_eval_no_gemini_4842.final.analysis_ready.skip_greedy_semantic_eval.jsonl',
]
TRUTHFULQA_CSV = BASE / 'TruthfulQA.csv'

OUT_DIR = ANALYSIS_DIR / 'latex_report'
FIG_DIR = OUT_DIR / 'figures'
REPORT_THRESHOLDS = [1.0, 0.9, 0.8]


def first_existing(paths: List[Path]) -> Path:
    for path in paths:
        if path.exists():
            return path
    raise FileNotFoundError(
        "Could not find an evaluated JSONL. Checked: "
        + ", ".join(str(p) for p in paths)
    )


def tex_escape(value: object) -> str:
    text = str(value)
    repl = {
        '\\': r'\textbackslash{}',
        '&': r'\&',
        '%': r'\%',
        '$': r'\$',
        '#': r'\#',
        '_': r'\_',
        '{': r'\{',
        '}': r'\}',
        '~': r'\textasciitilde{}',
        '^': r'\textasciicircum{}',
    }
    for src, dst in repl.items():
        text = text.replace(src, dst)
    return text


def short_model(name: str) -> str:
    mapping = {
        'Claude Opus 4.6 (Anthropic)': 'Claude Opus 4.6',
        'DeepSeek V3.2 (DeepSeek)': 'DeepSeek V3.2',
        'GPT-5.2 (OpenAI)': 'GPT-5.2',
        'Grok 4 (xAI)': 'Grok 4',
        'Llama 4 Maverick 17B (Groq)': 'Llama 4 Maverick',
        'Qwen3 Next 80B (OpenRouter)': 'Qwen3 Next 80B',
    }
    return mapping.get(name, name)


def pct(x: float) -> str:
    if pd.isna(x):
        return 'N/A'
    return f"{100.0 * float(x):.1f}\\%"


def fmt(x: float, digits: int = 3) -> str:
    if pd.isna(x):
        return 'N/A'
    return f"{float(x):.{digits}f}"


def setup_dirs() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)


def load_data() -> Dict[str, object]:
    final_jsonl = first_existing(FINAL_JSONL_CANDIDATES)
    summary = json.loads((ANALYSIS_DIR / 'thesis_deep_analysis_summary.json').read_text(encoding='utf-8'))
    model_df = pd.read_csv(ANALYSIS_DIR / 'thesis_deep_model_metrics.csv')
    group_df = pd.read_csv(ANALYSIS_DIR / 'thesis_deep_group_metrics.csv')
    cat_by_model_df = pd.read_csv(ANALYSIS_DIR / 'thesis_deep_category_by_model.csv')
    cat_agg_df = pd.read_csv(ANALYSIS_DIR / 'thesis_deep_category_aggregate.csv')
    rank_sim_df = pd.read_csv(ANALYSIS_DIR / 'thesis_deep_category_rank_similarity.csv')
    pair_df = pd.read_csv(ANALYSIS_DIR / 'thesis_deep_pairwise_significance.csv')

    records = [json.loads(line) for line in final_jsonl.read_text(encoding='utf-8').splitlines() if line.strip()]
    final_df = pd.DataFrame(records)

    truthfulqa = pd.read_csv(TRUTHFULQA_CSV).reset_index(drop=False).rename(
        columns={'index': 'q_idx', 'Category': 'category', 'Type': 'question_type'}
    )

    qid_re = re.compile(r'truthfulqa_csv_(\d+)$')

    def qid_to_idx(qid: object) -> float:
        if not isinstance(qid, str):
            return math.nan
        m = qid_re.search(qid)
        return float(m.group(1)) if m else math.nan

    final_df['q_idx'] = final_df['question_id'].map(qid_to_idx)
    final_df = final_df.merge(
        truthfulqa[['q_idx', 'category', 'question_type', 'Question']],
        on='q_idx',
        how='left',
    )

    return {
        'summary': summary,
        'model_df': model_df,
        'group_df': group_df,
        'cat_by_model_df': cat_by_model_df,
        'cat_agg_df': cat_agg_df,
        'rank_sim_df': rank_sim_df,
        'pair_df': pair_df,
        'final_df': final_df,
        'truthfulqa_df': truthfulqa,
    }


def add_short_names(model_df: pd.DataFrame) -> pd.DataFrame:
    out = model_df.copy()
    out['model_short'] = out['model'].map(short_model)
    return out


def build_pairwise_matrix(pair_df: pd.DataFrame, metric: str, models: List[str]) -> pd.DataFrame:
    mat = pd.DataFrame(np.nan, index=models, columns=models)
    sub = pair_df[pair_df['metric'] == metric]
    for _, row in sub.iterrows():
        a = row['model_a']
        b = row['model_b']
        d = float(row['delta_a_minus_b'])
        mat.loc[a, b] = d
        mat.loc[b, a] = -d
    arr = mat.to_numpy(copy=True)
    np.fill_diagonal(arr, 0.0)
    return pd.DataFrame(arr, index=models, columns=models)


def plot_model_rates(model_df: pd.DataFrame) -> None:
    df = model_df.copy()
    df['model_short'] = df['model'].map(short_model)
    df = df.sort_values('accuracy', ascending=False)

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8))
    sns.barplot(data=df, x='model_short', y='accuracy', ax=axes[0], color='#2a9d8f')
    axes[0].set_title('Accuracy by Model')
    axes[0].set_ylabel('Accuracy')
    axes[0].set_xlabel('')
    axes[0].tick_params(axis='x', rotation=35)

    sns.barplot(data=df, x='model_short', y='self_consistent_rate_total', ax=axes[1], color='#e76f51')
    axes[1].set_title('Self-Consistent Error Rate (All Rows)')
    axes[1].set_ylabel('Rate')
    axes[1].set_xlabel('')
    axes[1].tick_params(axis='x', rotation=35)

    sns.barplot(data=df, x='model_short', y='not_attempted_rate', ax=axes[2], color='#457b9d')
    axes[2].set_title('NOT_ATTEMPTED Rate')
    axes[2].set_ylabel('Rate')
    axes[2].set_xlabel('')
    axes[2].tick_params(axis='x', rotation=35)

    for ax in axes:
        ax.set_ylim(0, 1)
        ax.grid(axis='y', alpha=0.3)

    fig.tight_layout()
    fig.savefig(FIG_DIR / 'fig_model_rates.png', dpi=220, bbox_inches='tight')
    plt.close(fig)


def plot_error_composition(model_df: pd.DataFrame) -> None:
    df = model_df.copy()
    df['model_short'] = df['model'].map(short_model)
    df = df.sort_values('model_short').reset_index(drop=True)

    sc = df['self_consistent_0_9'] / df['n']
    inc = df['inconsistent_0_9'] / df['n']
    na = df['not_attempted'] / df['n']

    x = np.arange(len(df))
    width = 0.7

    fig, ax = plt.subplots(figsize=(11, 5.5))
    ax.bar(x, sc, width=width, label='Self-consistent error', color='#e76f51')
    ax.bar(x, inc, width=width, bottom=sc, label='Inconsistent error', color='#f4a261')
    ax.bar(x, na, width=width, bottom=sc + inc, label='NOT_ATTEMPTED', color='#457b9d')

    ax.set_xticks(x)
    ax.set_xticklabels(df['model_short'], rotation=30, ha='right')
    ax.set_ylim(0, 1)
    ax.set_ylabel('Fraction of all rows')
    ax.set_title('Error Composition by Model')
    ax.legend(loc='upper right')
    ax.grid(axis='y', alpha=0.3)

    fig.tight_layout()
    fig.savefig(FIG_DIR / 'fig_error_composition.png', dpi=220, bbox_inches='tight')
    plt.close(fig)


def plot_group_comparison(group_df: pd.DataFrame) -> None:
    melted = group_df.melt(
        id_vars=['group'],
        value_vars=['accuracy', 'self_consistent_rate_total', 'not_attempted_rate'],
        var_name='metric',
        value_name='value',
    )
    metric_map = {
        'accuracy': 'Accuracy',
        'self_consistent_rate_total': 'SC rate (all rows)',
        'not_attempted_rate': 'NOT_ATTEMPTED rate',
    }
    melted['metric'] = melted['metric'].map(metric_map)

    fig, ax = plt.subplots(figsize=(9, 5))
    sns.barplot(data=melted, x='metric', y='value', hue='group', ax=ax, palette='Set2')
    ax.set_ylabel('Rate')
    ax.set_xlabel('')
    ax.set_ylim(0, 1)
    ax.set_title('Closed API vs Open-weight API Group Comparison')
    ax.grid(axis='y', alpha=0.3)
    ax.legend(title='Group')

    fig.tight_layout()
    fig.savefig(FIG_DIR / 'fig_group_comparison.png', dpi=220, bbox_inches='tight')
    plt.close(fig)


def plot_top_categories(cat_agg_df: pd.DataFrame) -> None:
    df = cat_agg_df.copy().sort_values('mean_sc_rate_total', ascending=False).head(12)

    fig, ax = plt.subplots(figsize=(10, 6.5))
    sns.barplot(data=df, y='category', x='mean_sc_rate_total', color='#d62828', ax=ax)

    for idx, row in df.reset_index(drop=True).iterrows():
        ax.text(
            row['mean_sc_rate_total'] + 0.005,
            idx,
            f"n={int(round(row['support_questions_per_model_mean']))}",
            va='center',
            fontsize=9,
        )

    ax.set_xlim(0, min(1.0, float(df['mean_sc_rate_total'].max()) + 0.1))
    ax.set_xlabel('Mean self-consistent error rate (all rows)')
    ax.set_ylabel('')
    ax.set_title('Top Category Vulnerabilities (Mean Across Models)')
    ax.grid(axis='x', alpha=0.3)

    fig.tight_layout()
    fig.savefig(FIG_DIR / 'fig_top_categories.png', dpi=220, bbox_inches='tight')
    plt.close(fig)


def plot_category_heatmap(cat_by_model_df: pd.DataFrame) -> None:
    top_cats = (
        cat_by_model_df.groupby('category')['sc_rate_total']
        .mean()
        .sort_values(ascending=False)
        .head(15)
        .index
    )
    sub = cat_by_model_df[cat_by_model_df['category'].isin(top_cats)].copy()
    sub['model_short'] = sub['model'].map(short_model)
    piv = sub.pivot(index='category', columns='model_short', values='sc_rate_errors')

    fig, ax = plt.subplots(figsize=(10.5, 7.2))
    sns.heatmap(
        piv,
        cmap='YlOrRd',
        linewidths=0.4,
        linecolor='white',
        cbar_kws={'label': 'SC rate (of errors)'},
        ax=ax,
    )
    ax.set_title('Category x Model Heatmap (SC rate among incorrect rows)')
    ax.set_xlabel('')
    ax.set_ylabel('')

    fig.tight_layout()
    fig.savefig(FIG_DIR / 'fig_category_heatmap.png', dpi=220, bbox_inches='tight')
    plt.close(fig)


def plot_pairwise_heatmaps(pair_df: pd.DataFrame, model_df: pd.DataFrame) -> None:
    models = model_df.sort_values('accuracy', ascending=False)['model'].tolist()
    labels = [short_model(m) for m in models]

    acc_mat = build_pairwise_matrix(pair_df, 'acc', models)
    sc_mat = build_pairwise_matrix(pair_df, 'sc', models)

    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.4))

    sns.heatmap(
        acc_mat * 100,
        annot=True,
        fmt='.1f',
        cmap='RdBu_r',
        center=0,
        xticklabels=labels,
        yticklabels=labels,
        cbar_kws={'label': 'Delta (percentage points)'},
        ax=axes[0],
    )
    axes[0].set_title('Pairwise Accuracy Delta (A - B)')

    sns.heatmap(
        sc_mat * 100,
        annot=True,
        fmt='.1f',
        cmap='RdBu_r',
        center=0,
        xticklabels=labels,
        yticklabels=labels,
        cbar_kws={'label': 'Delta (percentage points)'},
        ax=axes[1],
    )
    axes[1].set_title('Pairwise SC-Rate Delta (A - B)')

    for ax in axes:
        ax.tick_params(axis='x', rotation=35)
        ax.tick_params(axis='y', rotation=0)

    fig.tight_layout()
    fig.savefig(FIG_DIR / 'fig_pairwise_heatmaps.png', dpi=220, bbox_inches='tight')
    plt.close(fig)


def plot_judge_diagnostics(final_df: pd.DataFrame) -> Dict[str, object]:
    patterns = Counter()
    slot_parse = Counter()
    slot_api = Counter()
    slot_ok = Counter()

    for statuses in final_df['correctness_judge_statuses'].tolist():
        if not isinstance(statuses, list):
            continue
        patterns[tuple(statuses)] += 1
        for i, s in enumerate(statuses):
            if s == 'OK':
                slot_ok[i] += 1
            elif s == 'PARSE_FAILED':
                slot_parse[i] += 1
            elif s == 'API_FAILED':
                slot_api[i] += 1

    pat_df = pd.DataFrame(
        [{'pattern': ' | '.join(k), 'count': v} for k, v in patterns.items()]
    ).sort_values('count', ascending=False).head(6)

    slot_df = pd.DataFrame(
        {
            'slot': [1, 2, 3],
            'OK': [slot_ok.get(0, 0), slot_ok.get(1, 0), slot_ok.get(2, 0)],
            'PARSE_FAILED': [slot_parse.get(0, 0), slot_parse.get(1, 0), slot_parse.get(2, 0)],
            'API_FAILED': [slot_api.get(0, 0), slot_api.get(1, 0), slot_api.get(2, 0)],
        }
    )

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6))
    sns.barplot(data=pat_df, y='pattern', x='count', ax=axes[0], color='#6d597a')
    axes[0].set_title('Top Judge Status Patterns')
    axes[0].set_xlabel('Rows')
    axes[0].set_ylabel('')

    slot_m = slot_df.melt(id_vars=['slot'], var_name='status', value_name='count')
    sns.barplot(data=slot_m, x='slot', y='count', hue='status', ax=axes[1])
    axes[1].set_title('Judge Status Counts by Slot')
    axes[1].set_xlabel('Judge slot')
    axes[1].set_ylabel('Rows')
    axes[1].legend(title='Status')

    for ax in axes:
        ax.grid(axis='x' if ax is axes[0] else 'y', alpha=0.3)

    fig.tight_layout()
    fig.savefig(FIG_DIR / 'fig_judge_diagnostics.png', dpi=220, bbox_inches='tight')
    plt.close(fig)

    two_ok = 0
    agree = 0
    for _, row in final_df.iterrows():
        grades = row.get('correctness_judge_grades')
        statuses = row.get('correctness_judge_statuses')
        if not isinstance(grades, list) or not isinstance(statuses, list):
            continue
        ok = [g for g, s in zip(grades, statuses) if s == 'OK' and g is not None]
        if len(ok) == 2:
            two_ok += 1
            if ok[0] == ok[1]:
                agree += 1

    agree_rate = (agree / two_ok) if two_ok else math.nan

    return {
        'pattern_df': pat_df,
        'slot_df': slot_df,
        'two_ok_rows': two_ok,
        'two_ok_agree': agree,
        'two_ok_agree_rate': agree_rate,
    }


def plot_question_type_comparison(final_df: pd.DataFrame) -> pd.DataFrame:
    df = final_df[~final_df['question_type'].isna()].copy()
    df['is_correct'] = df['greedy_correct'].map(lambda x: bool(x) if isinstance(x, bool) else str(x).lower() == 'true')
    df['is_sc'] = df['error_label_0.9'].eq('self_consistent_error')
    df['is_na'] = df['correctness_grade'].eq('NOT_ATTEMPTED')

    rows = []
    for qtype, sub in df.groupby('question_type'):
        incorrect = (~sub['is_correct']).sum()
        rows.append(
            {
                'question_type': qtype,
                'rows': len(sub),
                'accuracy': sub['is_correct'].mean(),
                'sc_rate_total': sub['is_sc'].mean(),
                'sc_rate_errors': (sub['is_sc'].sum() / incorrect) if incorrect else math.nan,
                'na_rate': sub['is_na'].mean(),
            }
        )
    out = pd.DataFrame(rows).sort_values('question_type')

    melt = out.melt(
        id_vars=['question_type'],
        value_vars=['accuracy', 'sc_rate_total', 'na_rate'],
        var_name='metric',
        value_name='value',
    )
    metric_map = {
        'accuracy': 'Accuracy',
        'sc_rate_total': 'SC rate (all rows)',
        'na_rate': 'NOT_ATTEMPTED rate',
    }
    melt['metric'] = melt['metric'].map(metric_map)

    fig, ax = plt.subplots(figsize=(8.8, 4.8))
    sns.barplot(data=melt, x='metric', y='value', hue='question_type', ax=ax, palette='Set1')
    ax.set_ylim(0, 1)
    ax.set_xlabel('')
    ax.set_ylabel('Rate')
    ax.set_title('Adversarial vs Non-Adversarial Performance')
    ax.grid(axis='y', alpha=0.3)
    ax.legend(title='Question type')

    fig.tight_layout()
    fig.savefig(FIG_DIR / 'fig_question_type_comparison.png', dpi=220, bbox_inches='tight')
    plt.close(fig)

    return out


def compute_threshold_frames(final_df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    labels = [
        'reliably_correct',
        'fragile_correct',
        'self_consistent_error',
        'inconsistent_error',
        'not_attempted',
    ]
    threshold_rows = []
    threshold_model_rows = []

    for threshold in REPORT_THRESHOLDS:
        col = f"error_label_{threshold:.1f}"
        if col not in final_df.columns:
            continue

        total = len(final_df)
        full_labels = final_df[col].astype(str)
        row: Dict[str, float] = {'threshold': threshold, 'rows': total}
        for label in labels:
            count = int((full_labels == label).sum())
            row[f'{label}_count'] = count
            row[f'{label}_rate'] = (count / total) if total else math.nan
        threshold_rows.append(row)

        for model, sub in final_df.groupby('model'):
            labels_sub = sub[col].astype(str)
            n = len(sub)
            sc_count = int((labels_sub == 'self_consistent_error').sum())
            threshold_model_rows.append(
                {
                    'model': model,
                    'model_short': short_model(model),
                    'threshold': threshold,
                    'rows': n,
                    'self_consistent_count': sc_count,
                    'self_consistent_rate': (sc_count / n) if n else math.nan,
                }
            )

    threshold_df = pd.DataFrame(threshold_rows).sort_values('threshold', ascending=False)
    threshold_model_df = pd.DataFrame(threshold_model_rows)
    return {'threshold_df': threshold_df, 'threshold_model_df': threshold_model_df}


def plot_threshold_sensitivity(threshold_model_df: pd.DataFrame) -> None:
    if threshold_model_df.empty:
        return

    tdf = threshold_model_df.copy()
    tdf['threshold_label'] = tdf['threshold'].map(lambda x: f"{x:.1f}")

    anchor = (
        tdf[tdf['threshold'].eq(0.9)]
        .sort_values('self_consistent_rate')
        ['model_short']
        .tolist()
    )
    if not anchor:
        anchor = sorted(tdf['model_short'].dropna().unique().tolist())

    fig, ax = plt.subplots(figsize=(11.2, 6.2))
    sns.barplot(
        data=tdf,
        x='model_short',
        y='self_consistent_rate',
        hue='threshold_label',
        hue_order=[f"{thr:.1f}" for thr in REPORT_THRESHOLDS],
        order=anchor,
        palette=['#264653', '#2a9d8f', '#e9c46a'],
        ax=ax,
    )
    ax.set_ylim(0, 1)
    ax.set_xlabel('')
    ax.set_ylabel('Self-consistent error rate (all rows)')
    ax.set_title('Threshold Sensitivity by Model (1.0 vs 0.9 vs 0.8)')
    ax.tick_params(axis='x', rotation=30)
    ax.grid(axis='y', alpha=0.3)
    ax.legend(title='Threshold')

    fig.tight_layout()
    fig.savefig(FIG_DIR / 'fig_threshold_sensitivity.png', dpi=220, bbox_inches='tight')
    plt.close(fig)


def generate_figures(data: Dict[str, object]) -> Dict[str, object]:
    sns.set_theme(style='whitegrid')

    model_df = add_short_names(data['model_df'])
    group_df = data['group_df']
    cat_agg_df = data['cat_agg_df']
    cat_by_model_df = data['cat_by_model_df']
    pair_df = data['pair_df']
    final_df = data['final_df']
    threshold_frames = compute_threshold_frames(final_df)

    plot_model_rates(model_df)
    plot_error_composition(model_df)
    plot_group_comparison(group_df)
    plot_top_categories(cat_agg_df)
    plot_category_heatmap(cat_by_model_df)
    plot_pairwise_heatmaps(pair_df, model_df)
    judge_diag = plot_judge_diagnostics(final_df)
    qtype_df = plot_question_type_comparison(final_df)
    plot_threshold_sensitivity(threshold_frames['threshold_model_df'])

    return {
        'judge_diag': judge_diag,
        'question_type_df': qtype_df,
        'threshold_df': threshold_frames['threshold_df'],
        'threshold_model_df': threshold_frames['threshold_model_df'],
    }


def build_missing_question_table(summary: Dict[str, object], truthfulqa_df: pd.DataFrame) -> pd.DataFrame:
    missing = summary['quality']['truthfulqa_missing_indices']
    if not missing:
        return pd.DataFrame(columns=['idx', 'category', 'question'])

    sub = truthfulqa_df[truthfulqa_df['q_idx'].isin(missing)].copy()
    sub = sub[['q_idx', 'category', 'Question']].rename(columns={'q_idx': 'idx', 'Question': 'question'})
    sub = sub.sort_values('idx')
    return sub


def table_model_rows(model_df: pd.DataFrame) -> str:
    lines = []
    for _, r in model_df.sort_values('accuracy', ascending=False).iterrows():
        lines.append(
            f"{tex_escape(short_model(r['model']))} & {int(r['n'])} & "
            f"{100*r['accuracy']:.1f} [{100*r['accuracy_ci_low']:.1f}, {100*r['accuracy_ci_high']:.1f}] & "
            f"{100*r['self_consistent_rate_total']:.1f} & {100*r['self_consistent_rate_of_errors']:.1f} & "
            f"{100*r['not_attempted_rate']:.1f} \\\\")
    return '\n'.join(lines)


def table_category_rows(cat_agg_df: pd.DataFrame) -> str:
    lines = []
    top = cat_agg_df.sort_values('mean_sc_rate_total', ascending=False).head(12)
    for _, r in top.iterrows():
        support = int(round(r['support_questions_per_model_mean']))
        lines.append(
            f"{tex_escape(r['category'])} & {100*r['mean_sc_rate_total']:.1f} & "
            f"{100*r['mean_sc_rate_errors']:.1f} & {support} \\\\")
    return '\n'.join(lines)


def table_significance_rows(pair_df: pd.DataFrame, metric: str, top_n: int = 10) -> str:
    sub = pair_df[(pair_df['metric'] == metric) & (pair_df['p_exact_mcnemar'] < 0.05)].copy()
    sub = sub.sort_values('p_exact_mcnemar').head(top_n)
    if sub.empty:
        return r'\multicolumn{3}{c}{No pairwise differences were significant at $p<0.05$.} \\'

    lines = []
    for _, r in sub.iterrows():
        pair = f"{short_model(r['model_a'])} vs {short_model(r['model_b'])}"
        lines.append(
            f"{tex_escape(pair)} & {100*r['delta_a_minus_b']:+.1f} & {r['p_exact_mcnemar']:.2e} \\\\")
    return '\n'.join(lines)


def table_missing_rows(missing_df: pd.DataFrame) -> str:
    if missing_df.empty:
        return r'\multicolumn{3}{l}{No missing TruthfulQA indices.} \\'

    lines = []
    for _, r in missing_df.iterrows():
        q = str(r['question'])
        if len(q) > 88:
            q = q[:85] + '...'
        lines.append(
            f"{int(r['idx'])} & {tex_escape(r['category'])} & {tex_escape(q)} \\\\")
    return '\n'.join(lines)


def table_threshold_rows(threshold_df: pd.DataFrame) -> str:
    if threshold_df.empty:
        return r'\multicolumn{6}{c}{Threshold columns missing in input data.} \\'

    lines = []
    for _, r in threshold_df.sort_values('threshold', ascending=False).iterrows():
        lines.append(
            f"{r['threshold']:.1f} & "
            f"{100*r['reliably_correct_rate']:.1f} & "
            f"{100*r['fragile_correct_rate']:.1f} & "
            f"{100*r['self_consistent_error_rate']:.1f} & "
            f"{100*r['inconsistent_error_rate']:.1f} & "
            f"{100*r['not_attempted_rate']:.1f} \\\\"
        )
    return '\n'.join(lines)


def write_bib() -> None:
    bib = r"""@article{tan2025tooconsistent,
  title={Too Consistent to Detect: A Study of Self-Consistent Errors in LLMs},
  author={Tan, Hanlin and Sun, Fei and Liu, Shuo and Su, Dan and others},
  journal={arXiv preprint arXiv:2505.17656},
  year={2025}
}

@inproceedings{manakul2023selfcheckgpt,
  title={SelfCheckGPT: Zero-Resource Black-Box Hallucination Detection for Generative Large Language Models},
  author={Manakul, Potsawee and Liusie, Adian and Gales, Mark},
  booktitle={Proceedings of EMNLP 2023},
  pages={9004--9017},
  year={2023}
}

@article{farquhar2024semantic,
  title={Detecting Hallucinations in Large Language Models Using Semantic Entropy},
  author={Farquhar, Sebastian and Kossen, Jens and Kuhn, Lorenz and Gal, Yarin},
  journal={Nature},
  volume={630},
  number={8017},
  pages={625--630},
  year={2024}
}

@inproceedings{wang2023sac3,
  title={SAC3: Reliable Hallucination Detection in Black-Box Language Models via Semantic-Aware Cross-Checking},
  author={Wang, Yile and Yu, Yixuan and Zhang, Yue and others},
  booktitle={Findings of EMNLP 2023},
  year={2023}
}

@inproceedings{chuang2024interrogatellm,
  title={InterrogateLLM: Zero-Resource Hallucination Evaluation for Generative Large Language Models},
  author={Chuang, Yung-Sung and others},
  booktitle={Proceedings of ACL 2024},
  year={2024}
}

@inproceedings{zhou2025agser,
  title={AGSER: Agentic Semantic Similarity Self-Reflection for Hallucination Detection in LLMs},
  author={Zhou, Biao and others},
  booktitle={Proceedings of EMNLP 2025},
  year={2025}
}

@inproceedings{lin2022truthfulqa,
  title={TruthfulQA: Measuring How Models Mimic Human Falsehoods},
  author={Lin, Stephanie and Hilton, Jacob and Evans, Owain},
  booktitle={Proceedings of ACL 2022},
  pages={3214--3252},
  year={2022}
}

@article{ji2023survey,
  title={Survey of Hallucination in Natural Language Generation},
  author={Ji, Ziwei and Lee, Nayeon and Frieske, Rob and others},
  journal={ACM Computing Surveys},
  volume={55},
  number={12},
  pages={1--38},
  year={2023}
}

@article{kuhn2023semanticuncertainty,
  title={Semantic Uncertainty: Linguistic Invariances for Uncertainty Estimation in Natural Language Generation},
  author={Kuhn, Lorenz and Gal, Yarin and Farquhar, Sebastian},
  journal={arXiv preprint arXiv:2302.09664},
  year={2023}
}

@article{he2020deberta,
  title={DeBERTa: Decoding-Enhanced BERT with Disentangled Attention},
  author={He, Pengcheng and Liu, Xiaodong and Gao, Jianfeng and Chen, Weizhu},
  journal={arXiv preprint arXiv:2006.03654},
  year={2020}
}
"""
    (OUT_DIR / 'references.bib').write_text(bib, encoding='utf-8')


def write_tex(data: Dict[str, object], extras: Dict[str, object]) -> None:
    summary = data['summary']
    quality = summary['quality']
    checks = summary['proposal_checks']
    model_df = data['model_df']
    group_df = data['group_df']
    cat_agg_df = data['cat_agg_df']
    rank_sim_df = data['rank_sim_df']
    pair_df = data['pair_df']
    truthfulqa_df = data['truthfulqa_df']
    qtype_df = extras['question_type_df']
    judge_diag = extras['judge_diag']
    threshold_df = extras['threshold_df']
    threshold_model_df = extras['threshold_model_df']

    overall_accuracy = model_df['correct'].sum() / quality['rows_total']
    overall_na = quality['rows_not_attempted'] / quality['rows_total']
    threshold_lookup = {float(r['threshold']): r for _, r in threshold_df.iterrows()} if not threshold_df.empty else {}
    overall_sc_1_0 = float(threshold_lookup[1.0]['self_consistent_error_rate']) if 1.0 in threshold_lookup else math.nan
    overall_sc_0_9 = float(threshold_lookup[0.9]['self_consistent_error_rate']) if 0.9 in threshold_lookup else math.nan
    overall_sc_0_8 = float(threshold_lookup[0.8]['self_consistent_error_rate']) if 0.8 in threshold_lookup else math.nan

    rank_mean = rank_sim_df['spearman_rho_sc_rate_errors'].mean()
    rank_min = rank_sim_df['spearman_rho_sc_rate_errors'].min()
    rank_max = rank_sim_df['spearman_rho_sc_rate_errors'].max()

    closed = group_df[group_df['group'] == 'closed_api'].iloc[0]
    openw = group_df[group_df['group'] == 'open_weight_api'].iloc[0]

    qtype_lines = []
    for _, r in qtype_df.iterrows():
        qtype_lines.append(
            f"\\item \\textbf{{{tex_escape(r['question_type'])}}}: accuracy {100*r['accuracy']:.1f}\\%, "
            f"SC(all rows) {100*r['sc_rate_total']:.1f}\\%, SC(of errors) {100*r['sc_rate_errors']:.1f}\\%, "
            f"NOT\\_ATTEMPTED {100*r['na_rate']:.1f}\\%."
        )

    model_release_bullets = [
        r"\item Claude Opus 4.6: February 5, 2026",
        r"\item GPT-5.2: December 11, 2025",
        r"\item Qwen3 Next 80B: September 9, 2025 (first public checkpoint)",
        r"\item DeepSeek V3.2: December 1, 2025",
        r"\item Grok 4: July 9, 2025",
        r"\item Llama 4 Maverick: April 5, 2025",
    ]

    rank_1_0 = []
    rank_0_9 = []
    rank_0_8 = []
    if not threshold_model_df.empty:
        rank_1_0 = (
            threshold_model_df[threshold_model_df['threshold'].eq(1.0)]
            .sort_values('self_consistent_rate')
            ['model_short']
            .tolist()
        )
        rank_0_9 = (
            threshold_model_df[threshold_model_df['threshold'].eq(0.9)]
            .sort_values('self_consistent_rate')
            ['model_short']
            .tolist()
        )
        rank_0_8 = (
            threshold_model_df[threshold_model_df['threshold'].eq(0.8)]
            .sort_values('self_consistent_rate')
            ['model_short']
            .tolist()
        )

    def rank_line(order: List[str]) -> str:
        return ", ".join(order) if order else "N/A"

    missing_df = build_missing_question_table(summary, truthfulqa_df)

    tex = f"""\\documentclass[11pt]{{article}}
\\usepackage[a4paper,margin=1in]{{geometry}}
\\usepackage{{graphicx}}
\\usepackage{{booktabs}}
\\usepackage{{longtable}}
\\usepackage{{array}}
\\usepackage{{float}}
\\usepackage{{hyperref}}
\\usepackage{{xcolor}}

\\title{{Self-Consistent Error Analysis Across Six LLM APIs\\\\
\\large Proposal-Aligned Results Report}}
\\author{{Simranjeet Singh}}
\\date{{February 18, 2026}}

\\begin{{document}}
\\maketitle

\\begin{{abstract}}
In this work, we evaluate self-consistent factual errors in six large language model APIs using a black-box pipeline with repeated sampling, semantic equivalence checks, and ensemble judging. The final analysis set contains {quality['rows_total']} rows ({quality['unique_questions']} questions across {quality['unique_models']} models) and a complete cross-model comparison matrix. Overall accuracy is {100*overall_accuracy:.2f}\\%. The self-consistent error rate over all rows is {100*overall_sc_1_0:.2f}\\% at threshold 1.0, {100*overall_sc_0_9:.2f}\\% at threshold 0.9, and {100*overall_sc_0_8:.2f}\\% at threshold 0.8. This document reports model-level prevalence, category-level vulnerability, pairwise significance testing, reliability diagnostics, and explicit alignment with the thesis proposal research questions.
\\end{{abstract}}

\\section{{Analysis Scope and Data Gate}}
\\textbf{{Input file.}} \\texttt{{{tex_escape(str(FINAL_JSONL))}}}

\\textbf{{Quality gate summary.}}
\\begin{{itemize}}
\\item Total rows: {quality['rows_total']} = {quality['unique_questions']} questions $\\times$ {quality['unique_models']} models.
\\item Complete matrix for cross-comparison: {str(quality['matrix_complete'])}.
\\item NOT\\_ATTEMPTED rows: {quality['rows_not_attempted']} ({100*overall_na:.2f}\\%).
\\item Parse-failed judge rows: {quality['rows_any_parse_failed']}; API-failed judge rows: {quality['rows_any_api_failed']}.
\\item Adjudicated rows: {quality['rows_adjudicated']}; unresolved/no-judge rows: {quality['rows_unresolved_or_no_judge']}.
\\item TruthfulQA coverage: {quality['truthfulqa_questions_covered']}/{quality['truthfulqa_questions_total_csv']} (10 filtered-out indices; listed in Appendix A).
\\end{{itemize}}

\\paragraph{{Model release dates (timeline context).}}
\\begin{{itemize}}
{chr(10).join(model_release_bullets)}
\\end{{itemize}}
These dates do not imply causality, but they add useful temporal context when interpreting cross-model differences.

\\section{{Research Question Alignment}}
\\begin{{itemize}}
\\item \\textbf{{RQ1 (prevalence across models):}} Supported by this dataset.
\\item \\textbf{{RQ2 (black-box viability):}} Supported by this dataset.
\\item \\textbf{{RQ3 (black-box vs white-box AUROC/cost):}} Not computable from current file because no hidden-state probing outputs are present.
\\item \\textbf{{RQ4 (category vulnerability):}} Supported by this dataset.
\\end{{itemize}}

\\section{{Headline Results}}
\\paragraph{{What "same meaning" and "+" mean.}}
In labels like \\texttt{{Correct+Same}} or \\texttt{{Incorrect+Same}}, the \\textbf{{+}} symbol means \\textbf{{AND}}. \\textbf{{Correct/Incorrect}} refers to the judged grade of the greedy answer. \\textbf{{Same/Different}} refers to whether the model's sampled answers are semantically equivalent to the greedy answer (paraphrases count), based on an NLI equivalence judge. At threshold $t$ (e.g., 0.9), we count a row as \\textbf{{Same}} if $\\#same/(\\#same+\\#different) \\ge t$; comparisons labeled \\texttt{{unclear}} are excluded from the ratio.
\\begin{{itemize}}
\\item Overall accuracy: {100*overall_accuracy:.2f}\\%.
\\item Self-consistent error rate (all rows): {100*overall_sc_1_0:.2f}\\% at threshold 1.0, {100*overall_sc_0_9:.2f}\\% at threshold 0.9, and {100*overall_sc_0_8:.2f}\\% at threshold 0.8.
\\item Greedy-correct rows where all sampled answers were semantically different from greedy: {checks['greedy_correct_but_all_samples_different']} rows.
\\end{{itemize}}

\\begin{{figure}}[H]
  \\centering
  \\includegraphics[width=0.99\\textwidth]{{figures/fig_model_rates.png}}
  \\caption{{Per-model headline rates: accuracy, self-consistent error rate (all rows), and NOT\\_ATTEMPTED rate.}}
\\end{{figure}}

\\begin{{table}}[H]
\\centering
\\caption{{Per-model performance and error profile}}
\\begin{{tabular}}{{lrrrrr}}
\\toprule
Model & N & Accuracy [95\\% CI] & SC(all) & SC(of errors) & NA \\\\
\\midrule
{table_model_rows(model_df)}
\\bottomrule
\\end{{tabular}}
\\end{{table}}

\\section{{Threshold Sensitivity (Including 1.0)}}
I keep 0.9 as my primary operating threshold, but I explicitly report 1.0 and 0.8 so the reader can see exactly how sensitive the conclusions are to this choice.

\\begin{{table}}[H]
\\centering
\\caption{{Overall label distribution across thresholds (percent of all rows)}}
\\begin{{tabular}}{{lrrrrr}}
\\toprule
Threshold & Correct+Same & Correct+Different & Incorrect+Same & Incorrect+Different & NOT\\_ATTEMPTED \\\\
\\midrule
{table_threshold_rows(threshold_df)}
\\bottomrule
\\end{{tabular}}
\\end{{table}}

\\begin{{figure}}[H]
  \\centering
  \\includegraphics[width=0.92\\textwidth]{{figures/fig_threshold_sensitivity.png}}
  \\caption{{Per-model self-consistent error rate at thresholds 1.0, 0.9, and 0.8. Lower is better.}}
\\end{{figure}}

\\paragraph{{How I read this in one pass.}}
As the threshold loosens (1.0 to 0.9 to 0.8), more rows qualify as ``same meaning,'' so self-consistent error rates rise. The key ranking is stable enough for interpretation: the low-SC end remains led by GPT-5.2, and the high-SC end remains DeepSeek/Qwen-heavy. Concretely, the model order by SC rate is:
\\begin{{itemize}}
\\item Threshold 1.0 (low to high): {tex_escape(rank_line(rank_1_0))}
\\item Threshold 0.9 (low to high): {tex_escape(rank_line(rank_0_9))}
\\item Threshold 0.8 (low to high): {tex_escape(rank_line(rank_0_8))}
\\end{{itemize}}
This is why I treat threshold 0.9 as a strong default, while keeping 1.0 and 0.8 as mandatory sensitivity checks.

\\section{{Closed-API vs Open-Weight API Comparison}}
For interpretive comparison, we grouped models as: closed APIs (Claude Opus 4.6, GPT-5.2, Grok 4) and open-weight APIs (DeepSeek V3.2, Llama 4 Maverick, Qwen3 Next 80B).

\\begin{{itemize}}
\\item Closed API group: accuracy {100*closed['accuracy']:.1f}\\%, SC(all rows) {100*closed['self_consistent_rate_total']:.1f}\\%, SC(of errors) {100*closed['self_consistent_rate_of_errors']:.1f}\\%.
\\item Open-weight API group: accuracy {100*openw['accuracy']:.1f}\\%, SC(all rows) {100*openw['self_consistent_rate_total']:.1f}\\%, SC(of errors) {100*openw['self_consistent_rate_of_errors']:.1f}\\%.
\\end{{itemize}}

\\begin{{figure}}[H]
  \\centering
  \\includegraphics[width=0.82\\textwidth]{{figures/fig_group_comparison.png}}
  \\caption{{Group-level comparison between closed APIs and open-weight APIs.}}
\\end{{figure}}

\\section{{Category Vulnerability Analysis (RQ4)}}
Category-level vulnerability is computed from TruthfulQA categories using mean self-consistent error rate across models. The strongest vulnerable categories are shown below.

\\begin{{table}}[H]
\\centering
\\caption{{Top categories by mean self-consistent error rate}}
\\begin{{tabular}}{{lrrr}}
\\toprule
Category & Mean SC(all) & Mean SC(of errors) & Questions \\\\
\\midrule
{table_category_rows(cat_agg_df)}
\\bottomrule
\\end{{tabular}}
\\end{{table}}

\\begin{{figure}}[H]
  \\centering
  \\includegraphics[width=0.49\\textwidth]{{figures/fig_top_categories.png}}\\hfill
  \\includegraphics[width=0.49\\textwidth]{{figures/fig_category_heatmap.png}}
  \\caption{{Category-level self-consistent error structure.}}
\\end{{figure}}

\\textbf{{Cross-model rank stability.}} Spearman correlation of category vulnerability ranks has mean {rank_mean:.3f}, minimum {rank_min:.3f}, and maximum {rank_max:.3f}. This indicates moderate shared structure with meaningful model-specific variation.

\\section{{Pairwise Significance Testing}}
We used exact McNemar tests on paired question-level outcomes.

\\begin{{table}}[H]
\\centering
\\caption{{Most significant pairwise differences for accuracy}}
\\begin{{tabular}}{{p{{8.5cm}}rr}}
\\toprule
Pair & $\\Delta$ accuracy (pp) & $p$ (exact McNemar) \\\\
\\midrule
{table_significance_rows(pair_df, 'acc', top_n=10)}
\\bottomrule
\\end{{tabular}}
\\end{{table}}

\\begin{{table}}[H]
\\centering
\\caption{{Most significant pairwise differences for self-consistent error rate}}
\\begin{{tabular}}{{p{{8.5cm}}rr}}
\\toprule
Pair & $\\Delta$ SC rate (pp) & $p$ (exact McNemar) \\\\
\\midrule
{table_significance_rows(pair_df, 'sc', top_n=10)}
\\bottomrule
\\end{{tabular}}
\\end{{table}}

\\begin{{figure}}[H]
  \\centering
  \\includegraphics[width=0.95\\textwidth]{{figures/fig_pairwise_heatmaps.png}}
  \\caption{{Pairwise deltas for accuracy and self-consistent error rate (A-B, percentage points).}}
\\end{{figure}}

\\section{{Reliability and Failure Diagnostics}}
Judge status diagnostics show that failures are concentrated in one judge slot (slot 2 parse failures), while two-OK disagreements are rare.

\\begin{{itemize}}
\\item Two-OK rows: {judge_diag['two_ok_rows']}
\\item Two-OK agreement count: {judge_diag['two_ok_agree']}
\\item Two-OK agreement rate: {100*judge_diag['two_ok_agree_rate']:.2f}\\%
\\end{{itemize}}

\\begin{{figure}}[H]
  \\centering
  \\includegraphics[width=0.95\\textwidth]{{figures/fig_judge_diagnostics.png}}
  \\caption{{Judge status pattern and per-slot failure distribution.}}
\\end{{figure}}

\\section{{Adversarial vs Non-Adversarial Split}}
To align with TruthfulQA structure, we compared outcomes by question type.

\\begin{{itemize}}
{chr(10).join(qtype_lines)}
\\end{{itemize}}

\\begin{{figure}}[H]
  \\centering
  \\includegraphics[width=0.82\\textwidth]{{figures/fig_question_type_comparison.png}}
  \\caption{{Performance split by TruthfulQA question type.}}
\\end{{figure}}

\\section{{Interpretation in Proposal Tone}}
In this work, the black-box pipeline is practically successful for measuring self-consistent errors across multiple providers. At the same time, the results show that consistency should not be treated as truth. A model can remain stable and still be wrong, and in some rows the greedy answer is correct while sampled outputs are semantically divergent. This matches the core risk pattern discussed in recent literature: consistency alone is not enough for reliable hallucination detection \\cite{{manakul2023selfcheckgpt,farquhar2024semantic,tan2025tooconsistent}}.

The category analysis also supports the proposal claim that vulnerability is not uniform across topics. Some categories repeatedly trigger stable wrong answers across models, while others are relatively safer. This means practical deployment should use category-aware safeguards instead of one global policy.

\\section{{Limitations and Explicit Gap to Proposal}}
\\begin{{itemize}}
\\item Proposal expected Google among commercial providers; this dataset does not include a Google model.
\\item Proposal RQ3 (black-box vs white-box AUROC/cost) cannot be finalized from this file because hidden-state probing outputs are absent.
\\item The evaluated subset includes 807/817 TruthfulQA questions. This is acceptable if documented as filtered subset.
\\item Category results with small support should be reported with support counts to avoid over-interpretation.
\\end{{itemize}}

\\section{{Conclusion}}
The analysis is thesis-safe for RQ1, RQ2, and RQ4: prevalence is quantified, black-box viability is demonstrated, and category vulnerabilities are measured with significance analysis and diagnostics. For RQ3, a white-box probing run is still required before making comparative AUROC claims.

\\appendix
\\section{{Missing TruthfulQA Indices in This Evaluation}}
\\begin{{longtable}}{{rll}}
\\toprule
Index & Category & Question (truncated) \\\\
\\midrule
{table_missing_rows(missing_df)}
\\bottomrule
\\end{{longtable}}

\\section{{Reproducibility Notes}}
This LaTeX report and all figures were generated directly from:
\\begin{{itemize}}
\\item \\texttt{{{tex_escape(str(ANALYSIS_DIR / 'thesis_deep_analysis_summary.json'))}}}
\\item \\texttt{{{tex_escape(str(ANALYSIS_DIR / 'thesis_deep_model_metrics.csv'))}}}
\\item \\texttt{{{tex_escape(str(ANALYSIS_DIR / 'thesis_deep_category_aggregate.csv'))}}}
\\item \\texttt{{{tex_escape(str(ANALYSIS_DIR / 'thesis_deep_pairwise_significance.csv'))}}}
\\item \\texttt{{{tex_escape(str(FINAL_JSONL))}}}
\\end{{itemize}}

\\bibliographystyle{{plain}}
\\bibliography{{references}}

\\end{{document}}
"""

    (OUT_DIR / 'thesis_analysis_report.tex').write_text(tex, encoding='utf-8')


def compile_pdf() -> Dict[str, object]:
    tex_file = OUT_DIR / 'thesis_analysis_report.tex'
    if not tex_file.exists():
        return {'compiled': False, 'reason': 'tex_missing'}

    if shutil.which('pdflatex') is None:
        return {'compiled': False, 'reason': 'pdflatex_not_found'}

    cmds = [
        ['pdflatex', '-interaction=nonstopmode', '-halt-on-error', tex_file.name],
        ['bibtex', 'thesis_analysis_report'],
        ['pdflatex', '-interaction=nonstopmode', '-halt-on-error', tex_file.name],
        ['pdflatex', '-interaction=nonstopmode', '-halt-on-error', tex_file.name],
    ]

    tex_cache = OUT_DIR / '.texcache'
    tex_fonts = OUT_DIR / '.texfonts'
    tex_cache.mkdir(parents=True, exist_ok=True)
    tex_fonts.mkdir(parents=True, exist_ok=True)
    env = dict(**os.environ)
    env['TEXMFVAR'] = str(tex_cache)
    env['VARTEXFONTS'] = str(tex_fonts)

    for cmd in cmds:
        proc = subprocess.run(
            cmd,
            cwd=OUT_DIR,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            env=env,
            check=False,
        )
        if proc.returncode != 0:
            (OUT_DIR / 'compile_error.log').write_text(proc.stdout, encoding='utf-8')
            return {
                'compiled': False,
                'reason': 'command_failed',
                'failed_cmd': ' '.join(cmd),
                'log': str(OUT_DIR / 'compile_error.log'),
            }

    err_log = OUT_DIR / 'compile_error.log'
    if err_log.exists():
        err_log.unlink()

    return {
        'compiled': True,
        'pdf': str(OUT_DIR / 'thesis_analysis_report.pdf'),
    }


def main() -> None:
    setup_dirs()
    data = load_data()
    extras = generate_figures(data)
    write_bib()
    write_tex(data, extras)
    status = compile_pdf()

    result = {
        'tex_file': str(OUT_DIR / 'thesis_analysis_report.tex'),
        'bib_file': str(OUT_DIR / 'references.bib'),
        'figure_dir': str(FIG_DIR),
        'compiled': status,
    }
    (OUT_DIR / 'build_report_status.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
