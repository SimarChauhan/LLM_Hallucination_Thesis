#!/usr/bin/env python3
from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

OUT_DIR = Path('data/results/analysis/version_evolution_equiv_only_20260319')
OUT_DIR.mkdir(parents=True, exist_ok=True)

OLD_PATH = Path('data/results/evaluated/results_v2_phase2_eval_no_gemini_4842.final.analysis_ready.skip_greedy_semantic_eval.jsonl')
NEW_PATHS = [
    Path('data/results/evaluated/run_qwen_new_only_807_full_retry2_20260315T193059Z/results_version_evolution_qwen_new_only_eval.equiv_only_20260319.jsonl'),
    Path('data/results/evaluated/run_llama_new_only_807_p1_20260315T222326Z/results_version_evolution_llama_scale_version_807_eval.equiv_only_20260319.jsonl'),
    Path('data/results/evaluated/run_grok_new_only_807_p1_xai_20260315T224013Z/results_version_evolution_grok_new_only_807_eval.equiv_only_20260319.jsonl'),
]

OLD_KEEP = {
    'Qwen3 Next 80B (OpenRouter)': dict(track='qwen_scale_version', family='qwen', release_date='2025-09-09', version_index=4, provider='openrouter', model_id='qwen/qwen3-next-80b-a3b-instruct'),
    'Llama 4 Maverick 17B (Groq)': dict(track='llama_scale_version', family='llama', release_date='2025-04-05', version_index=4, provider='groq', model_id='meta-llama/llama-4-maverick-17b-128e-instruct'),
    'Grok 4 (xAI)': dict(track='grok_version', family='grok', release_date='2025-07-09', version_index=2, provider='xai', model_id='grok-4-fast-non-reasoning'),
}

NEW_META = {
    'Qwen2.5 7B Instruct (OpenRouter, 2024-09-16)': dict(track='qwen_scale_version', family='qwen', release_date='2024-09-16', version_index=1),
    'Qwen2.5 72B Instruct (OpenRouter, 2024-11-26)': dict(track='qwen_scale_version', family='qwen', release_date='2024-11-26', version_index=2),
    'Qwen3 30B A3B 2507 (OpenRouter, 2025-07-28)': dict(track='qwen_scale_version', family='qwen', release_date='2025-07-28', version_index=3),
    'Llama 3 8B Instruct (OpenRouter, 2024-04-18)': dict(track='llama_scale_version', family='llama', release_date='2024-04-18', version_index=1),
    'Llama 3.1 8B Instruct (OpenRouter, 2024-07-23)': dict(track='llama_scale_version', family='llama', release_date='2024-07-23', version_index=2),
    'Llama 3.3 70B Instruct (OpenRouter, 2024-12-06)': dict(track='llama_scale_version', family='llama', release_date='2024-12-06', version_index=3),
    'Grok 3 (xAI, 2025-06-10)': dict(track='grok_version', family='grok', release_date='2025-06-10', version_index=1),
    'Grok 4.1 Fast Reasoning (xAI, 2025-11-19)': dict(track='grok_version', family='grok', release_date='2025-11-19', version_index=3),
    'Grok 4.20 Beta 0309 Reasoning (xAI, 2026-03-09)': dict(track='grok_version', family='grok', release_date='2026-03-09', version_index=4),
}

TRACK_LABELS = {
    'qwen_scale_version': 'Qwen Scale+Version',
    'llama_scale_version': 'Llama Scale+Version',
    'grok_version': 'Grok Version',
}
COLORS = {
    'qwen_scale_version': '#9c6644',
    'llama_scale_version': '#386641',
    'grok_version': '#1d3557',
}

MAIN_THRESHOLD = '0.9'
SENS_THRESHOLDS = ['1.0', '0.9', '0.8', '0.7']
PAIRWISE_BOOTSTRAP_ITERS = int(os.getenv('PAIRWISE_BOOTSTRAP_ITERS', '600'))
TREND_BOOTSTRAP_ITERS = int(os.getenv('TREND_BOOTSTRAP_ITERS', '120'))


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def build_combined_df() -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for rec in load_jsonl(OLD_PATH):
        meta = OLD_KEEP.get(rec.get('model'))
        if not meta:
            continue
        rec = dict(rec)
        rec.update(meta)
        rec['source_dataset'] = 'existing_4842_hybrid'
        rec['protocol_group'] = 'existing_latest_endpoint'
        rows.append(rec)
    for path in NEW_PATHS:
        for rec in load_jsonl(path):
            meta = NEW_META.get(rec.get('model'))
            if not meta:
                continue
            rec = dict(rec)
            rec.update(meta)
            rec['source_dataset'] = 'new_equiv_only_20260319'
            rec['protocol_group'] = 'new_family_rerun'
            rows.append(rec)
    df = pd.DataFrame(rows)
    df['release_date'] = pd.to_datetime(df['release_date'])
    df['version_index'] = pd.to_numeric(df['version_index'])
    return df


def normal_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def exact_binomial_two_sided(k: int, n: int) -> float:
    if n <= 0:
        return 1.0
    tail = 0.0
    for i in range(0, k + 1):
        tail += math.comb(n, i) * (0.5 ** n)
    return min(1.0, 2.0 * tail)


def mcnemar_exact_p(b: int, c: int) -> float:
    n = b + c
    if n == 0:
        return 1.0
    return exact_binomial_two_sided(min(b, c), n)


def bootstrap_mean_ci(values: np.ndarray, num_bootstrap: int = 2000, seed: int = 42) -> Tuple[float, float]:
    if values.size == 0:
        return (float('nan'), float('nan'))
    if values.size == 1:
        v = float(values[0])
        return v, v
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, values.size, size=(num_bootstrap, values.size))
    samples = values[idx].mean(axis=1)
    return float(np.quantile(samples, 0.025)), float(np.quantile(samples, 0.975))


@dataclass
class TrendFit:
    slope: float
    se: float
    z_value: float
    p_value: float
    converged: bool


def fit_logit_question_fe(y: np.ndarray, version_idx: np.ndarray, question_codes: np.ndarray) -> TrendFit | None:
    if y.size == 0 or np.all(y == y[0]):
        return None
    n_questions = int(question_codes.max()) + 1
    if n_questions <= 1:
        return None
    dummies = np.zeros((y.size, n_questions - 1), dtype=float)
    for q in range(1, n_questions):
        dummies[:, q - 1] = (question_codes == q).astype(float)
    X = np.column_stack([version_idx.astype(float), dummies])
    beta = np.zeros(X.shape[1], dtype=float)
    ridge = 1e-6
    converged = False
    for _ in range(80):
        eta = np.clip(X @ beta, -25.0, 25.0)
        p = 1.0 / (1.0 + np.exp(-eta))
        w = np.clip(p * (1.0 - p), 1e-6, None)
        z = eta + (y - p) / w
        xw = X * w[:, None]
        h = X.T @ xw
        h.flat[:: h.shape[0] + 1] += ridge
        rhs = X.T @ (w * z)
        try:
            beta_new = np.linalg.solve(h, rhs)
        except np.linalg.LinAlgError:
            return None
        if float(np.max(np.abs(beta_new - beta))) < 1e-6:
            beta = beta_new
            converged = True
            break
        beta = beta_new
    eta = np.clip(X @ beta, -25.0, 25.0)
    p = 1.0 / (1.0 + np.exp(-eta))
    w = np.clip(p * (1.0 - p), 1e-6, None)
    xw = X * w[:, None]
    h = X.T @ xw
    h.flat[:: h.shape[0] + 1] += ridge
    try:
        cov = np.linalg.inv(h)
    except np.linalg.LinAlgError:
        return None
    slope = float(beta[0])
    se = float(math.sqrt(max(float(cov[0, 0]), 1e-12)))
    z = slope / se if se > 0 else float('nan')
    p_val = 2.0 * (1.0 - normal_cdf(abs(z))) if np.isfinite(z) else float('nan')
    return TrendFit(slope, se, z, p_val, converged)


def bootstrap_trend_ci(track_df: pd.DataFrame, outcome_col: str, n_boot: int = 500, seed: int = 42) -> Tuple[float, float, int]:
    qids = track_df['question_id'].dropna().unique().tolist()
    if len(qids) < 2:
        return float('nan'), float('nan'), 0
    rng = np.random.default_rng(seed)
    slopes = []
    for _ in range(n_boot):
        sampled = rng.choice(qids, size=len(qids), replace=True)
        parts = [track_df[track_df['question_id'] == q] for q in sampled]
        boot = pd.concat(parts, ignore_index=True)
        codes = pd.Categorical(boot['question_id']).codes
        fit = fit_logit_question_fe(boot[outcome_col].to_numpy(dtype=float), boot['version_index'].to_numpy(dtype=float), codes)
        if fit and np.isfinite(fit.slope):
            slopes.append(float(fit.slope))
    if not slopes:
        return float('nan'), float('nan'), 0
    arr = np.asarray(slopes)
    return float(np.quantile(arr, 0.025)), float(np.quantile(arr, 0.975)), int(arr.size)


def compute_model_summary(df: pd.DataFrame, threshold: str) -> pd.DataFrame:
    out = df.copy()
    out['accuracy'] = out['greedy_correct'].astype(bool)
    out['ce_rate'] = out[f'error_label_{threshold}'].eq('self_consistent_error')
    out['ie_rate'] = out[f'error_label_{threshold}'].eq('inconsistent_error')
    res = out.groupby(['track','family','version_index','release_date','model','source_dataset','protocol_group'], dropna=False).agg(
        n_rows=('question_id','size'),
        n_questions=('question_id','nunique'),
        accuracy=('accuracy','mean'),
        ce_rate=('ce_rate','mean'),
        ie_rate=('ie_rate','mean'),
    ).reset_index().sort_values(['track','version_index'])
    for col in ['accuracy','ce_rate','ie_rate']:
        res[f'{col}_pct'] = 100.0 * res[col]
    return res


def compute_pairwise(df: pd.DataFrame, threshold: str) -> pd.DataFrame:
    out = df.copy()
    out['accuracy_bin'] = out['greedy_correct'].astype(bool).astype(int)
    out['ce_bin'] = out[f'error_label_{threshold}'].eq('self_consistent_error').astype(int)
    rows=[]
    for track, track_df in out.groupby('track'):
        models = track_df[['model','version_index']].drop_duplicates().sort_values(['version_index','model'])['model'].tolist()
        for i, older in enumerate(models):
            for j in range(i+1, len(models)):
                newer = models[j]
                left = track_df[track_df['model']==older][['question_id','accuracy_bin','ce_bin']]
                right = track_df[track_df['model']==newer][['question_id','accuracy_bin','ce_bin']]
                merged = left.merge(right, on='question_id', suffixes=('_old','_new'))
                if merged.empty:
                    continue
                for metric, old_col, new_col, high_good in [('accuracy','accuracy_bin_old','accuracy_bin_new',True),('ce_rate','ce_bin_old','ce_bin_new',False)]:
                    old_vals = merged[old_col].to_numpy(dtype=float)
                    new_vals = merged[new_col].to_numpy(dtype=float)
                    diffs = new_vals - old_vals
                    ci_lo, ci_hi = bootstrap_mean_ci(
                        diffs,
                        num_bootstrap=PAIRWISE_BOOTSTRAP_ITERS,
                        seed=42 + i*101 + j*17 + (0 if metric=='accuracy' else 1),
                    )
                    b = int(((old_vals==1) & (new_vals==0)).sum())
                    c = int(((old_vals==0) & (new_vals==1)).sum())
                    delta = float(diffs.mean())
                    improvement = delta if high_good else -delta
                    rows.append({
                        'track': track,
                        'older_model': older,
                        'newer_model': newer,
                        'consecutive_pair': bool(j==i+1),
                        'n_paired_questions': int(len(merged)),
                        'metric': metric,
                        'older_rate': float(old_vals.mean()),
                        'newer_rate': float(new_vals.mean()),
                        'delta_new_minus_old': delta,
                        'delta_new_minus_old_pp': 100.0*delta,
                        'improvement_pp': 100.0*improvement,
                        'bootstrap_ci_low_pp': 100.0*ci_lo,
                        'bootstrap_ci_high_pp': 100.0*ci_hi,
                        'mcnemar_b_old1_new0': b,
                        'mcnemar_c_old0_new1': c,
                        'mcnemar_p_exact': mcnemar_exact_p(b,c),
                    })
    return pd.DataFrame(rows).sort_values(['track','metric','older_model','newer_model'])


def compute_trends(df: pd.DataFrame, threshold: str) -> pd.DataFrame:
    out = df.copy()
    out['accuracy_bin'] = out['greedy_correct'].astype(bool).astype(int)
    out['ce_bin'] = out[f'error_label_{threshold}'].eq('self_consistent_error').astype(int)
    rows=[]
    for track, track_df in out.groupby('track'):
        if track_df['version_index'].nunique() < 3:
            continue
        for metric, col in [('accuracy','accuracy_bin'), ('ce_rate','ce_bin')]:
            sub = track_df[['question_id','version_index',col]].copy()
            if sub[col].nunique() < 2:
                continue
            codes = pd.Categorical(sub['question_id']).codes
            fit = fit_logit_question_fe(sub[col].to_numpy(dtype=float), sub['version_index'].to_numpy(dtype=float), codes)
            if not fit:
                continue
            ci_lo, ci_hi, boots = bootstrap_trend_ci(
                sub.rename(columns={col:'y'}),
                'y',
                n_boot=TREND_BOOTSTRAP_ITERS,
                seed=42 + (0 if metric=='accuracy' else 1000),
            )
            rows.append({
                'track': track,
                'metric': metric,
                'n_rows': int(len(sub)),
                'n_questions': int(sub['question_id'].nunique()),
                'n_versions': int(sub['version_index'].nunique()),
                'slope_per_version': fit.slope,
                'slope_se': fit.se,
                'z_value': fit.z_value,
                'p_value': fit.p_value,
                'odds_ratio_per_version': float(math.exp(fit.slope)),
                'slope_bootstrap_ci_low': ci_lo,
                'slope_bootstrap_ci_high': ci_hi,
                'bootstrap_successes': boots,
                'converged': fit.converged,
            })
    return pd.DataFrame(rows).sort_values(['track','metric'])


def make_timeline_plots(summary_main: pd.DataFrame, summary_all: pd.DataFrame) -> List[Path]:
    paths=[]
    # CE main threshold panel
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8), sharey=False)
    for ax, track in zip(axes, ['qwen_scale_version','llama_scale_version','grok_version']):
        sub = summary_main[summary_main['track']==track].sort_values('version_index')
        ax.plot(sub['version_index'], sub['ce_rate_pct'], marker='o', color=COLORS[track], linewidth=2.2)
        for _, row in sub.iterrows():
            marker = 'D' if row['source_dataset']=='existing_4842_hybrid' else 'o'
            ax.scatter(row['version_index'], row['ce_rate_pct'], color=COLORS[track], s=70, marker=marker, edgecolor='black', linewidth=0.5, zorder=3)
            ax.text(row['version_index'], row['ce_rate_pct']+0.7, f"{row['ce_rate_pct']:.1f}", ha='center', va='bottom', fontsize=8)
        ax.set_title(TRACK_LABELS[track])
        ax.set_xlabel('Version Index')
        ax.grid(alpha=0.25)
        ax.set_xticks(sub['version_index'].tolist())
    axes[0].set_ylabel('Self-Consistent Error Rate (%) at t=0.9')
    fig.suptitle('Self-Consistent Error Rate Over Time', y=1.03, fontsize=14)
    fig.tight_layout()
    p=OUT_DIR/'ce_rate_over_time_t0p9.png'
    fig.savefig(p, dpi=220, bbox_inches='tight')
    plt.close(fig)
    paths.append(p)

    # Accuracy panel
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8), sharey=False)
    for ax, track in zip(axes, ['qwen_scale_version','llama_scale_version','grok_version']):
        sub = summary_main[summary_main['track']==track].sort_values('version_index')
        ax.plot(sub['version_index'], sub['accuracy_pct'], marker='o', color=COLORS[track], linewidth=2.2)
        for _, row in sub.iterrows():
            marker = 'D' if row['source_dataset']=='existing_4842_hybrid' else 'o'
            ax.scatter(row['version_index'], row['accuracy_pct'], color=COLORS[track], s=70, marker=marker, edgecolor='black', linewidth=0.5, zorder=3)
            ax.text(row['version_index'], row['accuracy_pct']+0.7, f"{row['accuracy_pct']:.1f}", ha='center', va='bottom', fontsize=8)
        ax.set_title(TRACK_LABELS[track])
        ax.set_xlabel('Version Index')
        ax.grid(alpha=0.25)
        ax.set_xticks(sub['version_index'].tolist())
    axes[0].set_ylabel('Accuracy (%)')
    fig.suptitle('Accuracy Over Time', y=1.03, fontsize=14)
    fig.tight_layout()
    p=OUT_DIR/'accuracy_over_time.png'
    fig.savefig(p, dpi=220, bbox_inches='tight')
    plt.close(fig)
    paths.append(p)

    # Threshold sensitivity CE panel
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8), sharey=False)
    thresh_colors = {'1.0':'#6c757d','0.9':'#d62828','0.8':'#f77f00','0.7':'#2a9d8f'}
    for ax, track in zip(axes, ['qwen_scale_version','llama_scale_version','grok_version']):
        sub = summary_all[summary_all['track']==track].sort_values(['threshold','version_index'])
        for thr in SENS_THRESHOLDS:
            ss = sub[sub['threshold']==thr].sort_values('version_index')
            ax.plot(ss['version_index'], ss['ce_rate_pct'], marker='o', linewidth=1.8, label=f't={thr}', color=thresh_colors[thr])
        ax.set_title(TRACK_LABELS[track])
        ax.set_xlabel('Version Index')
        ax.grid(alpha=0.25)
        ax.set_xticks(sorted(sub['version_index'].unique()))
    axes[0].set_ylabel('Self-Consistent Error Rate (%)')
    axes[-1].legend(frameon=False, loc='upper right')
    fig.suptitle('Threshold Sensitivity of Self-Consistent Error Rate', y=1.03, fontsize=14)
    fig.tight_layout()
    p=OUT_DIR/'ce_rate_threshold_sensitivity.png'
    fig.savefig(p, dpi=220, bbox_inches='tight')
    plt.close(fig)
    paths.append(p)
    return paths


def summarize_findings(summary_main: pd.DataFrame, pairwise_main: pd.DataFrame, trend_main: pd.DataFrame) -> List[str]:
    lines=[]
    # Grok
    grok = summary_main[summary_main['track']=='grok_version'].sort_values('version_index')
    if not grok.empty:
        first, last = grok.iloc[0], grok.iloc[-1]
        lines.append(f"Grok shows the clearest downward self-consistent-error trend: CE@0.9 falls from {first['ce_rate_pct']:.1f}% ({first['model']}) to {last['ce_rate_pct']:.1f}% ({last['model']}), while accuracy rises from {first['accuracy_pct']:.1f}% to {last['accuracy_pct']:.1f}%.")
    # Llama
    llama = summary_main[summary_main['track']=='llama_scale_version'].sort_values('version_index')
    if not llama.empty:
        peak = llama.loc[llama['ce_rate_pct'].idxmax()]
        last = llama.iloc[-1]
        lines.append(f"Llama is non-monotonic: CE@0.9 dips early, spikes at {peak['model']} ({peak['ce_rate_pct']:.1f}%), then eases slightly to {last['ce_rate_pct']:.1f}% at {last['model']}; accuracy improves overall but largely plateaus between versions 3 and 4.")
    qwen = summary_main[summary_main['track']=='qwen_scale_version'].sort_values('version_index')
    if not qwen.empty:
        first, last = qwen.iloc[0], qwen.iloc[-1]
        best_ce = qwen.loc[qwen['ce_rate_pct'].idxmin()]
        lines.append(f"Qwen accuracy improves overall from {first['accuracy_pct']:.1f}% to {last['accuracy_pct']:.1f}%, but CE@0.9 is non-monotonic and the lowest CE point is {best_ce['model']} ({best_ce['ce_rate_pct']:.1f}%), not the latest endpoint.")
    sig_pairs = pairwise_main[(pairwise_main['metric']=='ce_rate') & (pairwise_main['mcnemar_p_exact'] < 0.05)]
    if not sig_pairs.empty:
        top = sig_pairs.sort_values('improvement_pp', ascending=False).iloc[0]
        lines.append(f"The strongest statistically supported CE reduction in paired questions is {top['older_model']} -> {top['newer_model']} with {top['improvement_pp']:.1f} percentage-point improvement (McNemar p={top['mcnemar_p_exact']:.4g}).")
    return lines


def write_report(df: pd.DataFrame, summary_main: pd.DataFrame, summary_all: pd.DataFrame, pairwise_main: pd.DataFrame, trend_main: pd.DataFrame, figure_paths: List[Path]) -> Path:
    findings = summarize_findings(summary_main, pairwise_main, trend_main)
    lines=[]
    lines.append('# Version-Evolution Report: Self-Consistent Errors Over Time')
    lines.append('')
    lines.append('## Scope')
    lines.append('- Families analyzed: Qwen, Llama, Grok.')
    lines.append('- Outcome of interest: self-consistent error (CE) rate over time.')
    lines.append(f'- Main threshold: CE defined with `error_label_{MAIN_THRESHOLD}`.')
    lines.append('- Sensitivity thresholds also included: 1.0, 0.8, 0.7.')
    lines.append('')
    lines.append('## Data Sources')
    lines.append('- Existing latest endpoints were reused from `results_v2_phase2_eval_no_gemini_4842.final.analysis_ready.skip_greedy_semantic_eval.jsonl`.')
    lines.append('- Older family versions came from the new `equiv_only` reruns dated 2026-03-19.')
    lines.append(f'- Combined rows: {len(df)} across {df["model"].nunique()} models and {df["question_id"].nunique()} paired questions.')
    lines.append('')
    lines.append('## Protocol Note')
    lines.append('- All combined rows use hybrid semantic equivalence with GPT-5.2 fallback on borderline cases.')
    lines.append('- The reused latest endpoints come from the older full hybrid dataset, while the newly added older versions come from lighter equivalence-only reruns.')
    lines.append('- Because the latest endpoints were not regenerated in the same rerun batch, absolute CE levels should be interpreted with that source-dataset caveat in mind. Trend direction is more defensible than tiny absolute differences.')
    lines.append('')
    lines.append('## Main Findings')
    for item in findings:
        lines.append(f'- {item}')
    lines.append('')
    lines.append('## CE@0.9 Summary')
    lines.append(summary_main[['track','version_index','release_date','model','source_dataset','accuracy_pct','ce_rate_pct','ie_rate_pct']].to_markdown(index=False))
    lines.append('')
    lines.append('## Trend Tests')
    if trend_main.empty:
        lines.append('No trend-test rows were produced.')
    else:
        lines.append(trend_main.to_markdown(index=False))
    lines.append('')
    lines.append('## Consecutive Pairwise Deltas (CE@0.9)')
    consec = pairwise_main[(pairwise_main['metric']=='ce_rate') & (pairwise_main['consecutive_pair'])].copy()
    if consec.empty:
        lines.append('No consecutive pairwise rows were produced.')
    else:
        lines.append(consec[['track','older_model','newer_model','improvement_pp','bootstrap_ci_low_pp','bootstrap_ci_high_pp','mcnemar_p_exact']].to_markdown(index=False))
    lines.append('')
    lines.append('## Figures')
    for p in figure_paths:
        lines.append(f'- {p}')
    lines.append('')
    report_path = OUT_DIR / 'report.md'
    report_path.write_text('\n'.join(lines) + '\n', encoding='utf-8')
    return report_path


def main() -> None:
    df = build_combined_df()
    combined_path = OUT_DIR / 'combined_version_evolution_equiv_only.jsonl'
    export_df = df.copy()
    for col in export_df.columns:
        if pd.api.types.is_datetime64_any_dtype(export_df[col]):
            export_df[col] = export_df[col].dt.strftime('%Y-%m-%d')
    with open(combined_path, 'w', encoding='utf-8') as f:
        for rec in export_df.to_dict(orient='records'):
            f.write(json.dumps(rec, ensure_ascii=False) + '\n')

    summaries=[]
    pairwise_frames=[]
    trend_frames=[]
    for thr in SENS_THRESHOLDS:
        sm = compute_model_summary(df, thr)
        sm['threshold']=thr
        sm.to_csv(OUT_DIR / f'model_summary_t{thr.replace(".","p")}.csv', index=False)
        summaries.append(sm)
        pw = compute_pairwise(df, thr)
        pw['threshold']=thr
        pw.to_csv(OUT_DIR / f'pairwise_deltas_t{thr.replace(".","p")}.csv', index=False)
        pairwise_frames.append(pw)
        tr = compute_trends(df, thr)
        tr['threshold']=thr
        tr.to_csv(OUT_DIR / f'trend_tests_t{thr.replace(".","p")}.csv', index=False)
        trend_frames.append(tr)

    summary_all = pd.concat(summaries, ignore_index=True)
    pairwise_all = pd.concat(pairwise_frames, ignore_index=True)
    trend_all = pd.concat(trend_frames, ignore_index=True)
    summary_main = summary_all[summary_all['threshold']==MAIN_THRESHOLD].copy()
    pairwise_main = pairwise_all[pairwise_all['threshold']==MAIN_THRESHOLD].copy()
    trend_main = trend_all[trend_all['threshold']==MAIN_THRESHOLD].copy()
    summary_all.to_csv(OUT_DIR / 'model_summary_all_thresholds.csv', index=False)
    pairwise_all.to_csv(OUT_DIR / 'pairwise_deltas_all_thresholds.csv', index=False)
    trend_all.to_csv(OUT_DIR / 'trend_tests_all_thresholds.csv', index=False)

    figs = make_timeline_plots(summary_main, summary_all)
    report_path = write_report(df, summary_main, summary_all, pairwise_main, trend_main, figs)

    payload = {
        'combined_rows': int(len(df)),
        'models': int(df['model'].nunique()),
        'questions': int(df['question_id'].nunique()),
        'out_dir': str(OUT_DIR),
        'report': str(report_path),
        'figures': [str(p) for p in figs],
    }
    (OUT_DIR / 'analysis_manifest.json').write_text(json.dumps(payload, indent=2), encoding='utf-8')
    print(json.dumps(payload, indent=2))

if __name__ == '__main__':
    main()
