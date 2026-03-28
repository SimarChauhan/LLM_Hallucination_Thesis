#!/usr/bin/env python3
"""
Version-evolution analysis for CE trend studies.

Outputs:
- model_summary.csv
- pairwise_deltas.csv
- trend_tests.csv
- validation_checks.json
"""

from __future__ import annotations

import argparse
import json
import logging
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import yaml


logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def _to_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "t", "yes", "y"}
    return False


def _normal_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def _exact_binomial_two_sided(k: int, n: int) -> float:
    if n <= 0:
        return 1.0
    tail_prob = 0.0
    for i in range(0, k + 1):
        tail_prob += math.comb(n, i) * (0.5 ** n)
    return min(1.0, 2.0 * tail_prob)


def _mcnemar_exact_p(b: int, c: int) -> float:
    n = b + c
    if n == 0:
        return 1.0
    return _exact_binomial_two_sided(min(b, c), n)


def _bootstrap_mean_ci(values: np.ndarray, num_bootstrap: int, seed: int) -> Tuple[float, float]:
    if values.size == 0:
        return (float("nan"), float("nan"))
    if values.size == 1:
        v = float(values[0])
        return (v, v)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, values.size, size=(num_bootstrap, values.size))
    samples = values[idx].mean(axis=1)
    return float(np.quantile(samples, 0.025)), float(np.quantile(samples, 0.975))


def _prepare_model_manifest(config: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    manifest: Dict[str, Dict[str, Any]] = {}
    for key in ("commercial_models", "opensource_models"):
        for entry in config.get(key, []) or []:
            name = str(entry.get("name") or f"{entry.get('provider')}/{entry.get('model')}")
            manifest[name] = {
                "provider": entry.get("provider"),
                "model_id": entry.get("model"),
                "snapshot_id": entry.get("snapshot_id"),
                "release_date": entry.get("release_date"),
                "track": entry.get("track"),
                "family": entry.get("family"),
                "version_index": entry.get("version_index"),
            }
    return manifest


def _resolve_evaluated_path(config: Dict[str, Any], run_id: Optional[str], explicit_input: Optional[str]) -> Path:
    if explicit_input:
        return Path(explicit_input)
    output = config.get("output", {})
    base_dir = Path(output.get("results_dir", "data/results"))
    eval_dir = Path(output.get("evaluated_dir", "evaluated"))
    evaluated_file = output.get("evaluated_file", "results_eval.jsonl")
    immutable = bool(output.get("immutable_runs", True))
    if immutable:
        if not run_id:
            raise ValueError("--run-id is required when input is omitted and immutable_runs=true.")
        return base_dir / eval_dir / run_id / evaluated_file
    return base_dir / eval_dir / evaluated_file


def _load_df(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Evaluated file not found: {path}")
    df = pd.read_json(path, lines=True)
    if df.empty:
        raise ValueError(f"No rows found in evaluated file: {path}")
    return df


def _apply_manifest_defaults(df: pd.DataFrame, manifest: Dict[str, Dict[str, Any]]) -> pd.DataFrame:
    out = df.copy()
    for idx, row in out.iterrows():
        model_name = str(row.get("model", ""))
        model_meta = manifest.get(model_name, {})
        if pd.isna(row.get("model_provider")) and model_meta.get("provider") is not None:
            out.at[idx, "model_provider"] = model_meta.get("provider")
        if pd.isna(row.get("model_id")) and model_meta.get("model_id") is not None:
            out.at[idx, "model_id"] = model_meta.get("model_id")
        if pd.isna(row.get("model_snapshot_id")):
            value = model_meta.get("snapshot_id") or row.get("model_id")
            if value is not None:
                out.at[idx, "model_snapshot_id"] = value
        if pd.isna(row.get("model_release_date")) and model_meta.get("release_date") is not None:
            out.at[idx, "model_release_date"] = model_meta.get("release_date")
        if pd.isna(row.get("model_track")) and model_meta.get("track") is not None:
            out.at[idx, "model_track"] = model_meta.get("track")
        if pd.isna(row.get("model_family")) and model_meta.get("family") is not None:
            out.at[idx, "model_family"] = model_meta.get("family")
        if pd.isna(row.get("model_version_index")) and model_meta.get("version_index") is not None:
            out.at[idx, "model_version_index"] = model_meta.get("version_index")
    return out


def _validate(df: pd.DataFrame, required_samples: int) -> Dict[str, Any]:
    checks: Dict[str, Any] = {
        "total_rows": int(len(df)),
        "required_samples": required_samples,
    }

    if "stochastic_actual_n" in df.columns:
        actual = pd.to_numeric(df["stochastic_actual_n"], errors="coerce").fillna(0).astype(int)
        checks["rows_below_required_samples"] = int((actual < required_samples).sum())
    else:
        checks["rows_below_required_samples"] = None

    def _unique_count(col: str, track_col: str = "model_track") -> Dict[str, int]:
        if col not in df.columns:
            return {}
        grouped = (
            df[[track_col, col]]
            .dropna()
            .groupby(track_col)[col]
            .nunique()
            .to_dict()
        )
        return {str(k): int(v) for k, v in grouped.items()}

    checks["config_hash_unique_per_track"] = _unique_count("config_hash")
    checks["judge_protocol_unique_per_track"] = _unique_count("judge_protocol")
    checks["prompt_version_unique_per_track"] = _unique_count("prompt_version")

    if "question_id" in df.columns:
        question_counts = (
            df.groupby(["model_track", "model"])["question_id"]
            .nunique()
            .reset_index(name="n_questions")
        )
        checks["question_coverage_by_model"] = question_counts.to_dict(orient="records")
    else:
        checks["question_coverage_by_model"] = []

    checks["passes"] = bool(
        (checks["rows_below_required_samples"] in {None, 0})
        and all(v == 1 for v in checks["config_hash_unique_per_track"].values())
        and all(v == 1 for v in checks["judge_protocol_unique_per_track"].values())
        and all(v == 1 for v in checks["prompt_version_unique_per_track"].values())
    )
    return checks


def _model_summary(df: pd.DataFrame, ce_label: str) -> pd.DataFrame:
    out = df.copy()
    out["is_correct"] = out["greedy_correct"].map(_to_bool)
    out["is_ce"] = (out[f"error_label_{ce_label}"] == "self_consistent_error")
    out["is_ie"] = (out[f"error_label_{ce_label}"] == "inconsistent_error")

    group_cols = ["model_track", "model_family", "model", "model_provider", "model_id", "model_snapshot_id", "model_release_date", "model_version_index"]
    existing = [c for c in group_cols if c in out.columns]
    rows = (
        out.groupby(existing, dropna=False)
        .agg(
            n_questions=("question_id", "nunique"),
            n_rows=("question_id", "size"),
            accuracy=("is_correct", "mean"),
            ce_rate=("is_ce", "mean"),
            ie_rate=("is_ie", "mean"),
        )
        .reset_index()
    )
    rows["accuracy_pct"] = 100.0 * rows["accuracy"]
    rows["ce_rate_pct"] = 100.0 * rows["ce_rate"]
    rows["ie_rate_pct"] = 100.0 * rows["ie_rate"]
    if "model_version_index" in rows.columns:
        rows = rows.sort_values(["model_track", "model_version_index", "model"])
    else:
        rows = rows.sort_values(["model_track", "model"])
    return rows


def _ordered_models_for_track(track_df: pd.DataFrame) -> List[str]:
    cols = ["model", "model_version_index"]
    if "model_release_date" in track_df.columns:
        cols.append("model_release_date")
    else:
        track_df = track_df.copy()
        track_df["model_release_date"] = pd.NaT
        cols.append("model_release_date")

    meta = track_df[cols].drop_duplicates().copy()
    meta["model_version_index"] = pd.to_numeric(meta["model_version_index"], errors="coerce")
    meta["model_release_date"] = pd.to_datetime(meta["model_release_date"], errors="coerce")
    meta = meta.sort_values(["model_version_index", "model_release_date", "model"], na_position="last")
    return meta["model"].tolist()


def _pairwise_rows(
    df: pd.DataFrame,
    ce_label: str,
    bootstrap_iters: int,
    seed: int,
) -> pd.DataFrame:
    out = df.copy()
    out["is_correct"] = out["greedy_correct"].map(_to_bool).astype(int)
    out["is_ce"] = (out[f"error_label_{ce_label}"] == "self_consistent_error").astype(int)

    rows: List[Dict[str, Any]] = []
    for track, track_df in out.groupby("model_track", dropna=False):
        ordered_models = _ordered_models_for_track(track_df)
        for i, older_model in enumerate(ordered_models):
            for j in range(i + 1, len(ordered_models)):
                newer_model = ordered_models[j]
                left = track_df[track_df["model"] == older_model][["question_id", "is_correct", "is_ce"]]
                right = track_df[track_df["model"] == newer_model][["question_id", "is_correct", "is_ce"]]
                merged = left.merge(right, on="question_id", how="inner", suffixes=("_old", "_new"))
                if merged.empty:
                    continue

                for metric_name, old_col, new_col, higher_is_better in [
                    ("accuracy", "is_correct_old", "is_correct_new", True),
                    ("ce_rate", "is_ce_old", "is_ce_new", False),
                ]:
                    old_vals = merged[old_col].to_numpy(dtype=float)
                    new_vals = merged[new_col].to_numpy(dtype=float)
                    diffs = new_vals - old_vals
                    ci_low, ci_high = _bootstrap_mean_ci(
                        diffs,
                        num_bootstrap=bootstrap_iters,
                        seed=seed + i * 101 + j * 17 + (0 if metric_name == "accuracy" else 1),
                    )
                    b_only = int(((old_vals == 1) & (new_vals == 0)).sum())
                    c_only = int(((old_vals == 0) & (new_vals == 1)).sum())
                    delta = float(diffs.mean())
                    improvement = delta if higher_is_better else -delta
                    rows.append(
                        {
                            "track": track,
                            "older_model": older_model,
                            "newer_model": newer_model,
                            "consecutive_pair": bool(j == i + 1),
                            "n_paired_questions": int(len(merged)),
                            "metric": metric_name,
                            "older_rate": float(old_vals.mean()),
                            "newer_rate": float(new_vals.mean()),
                            "delta_new_minus_old": delta,
                            "delta_new_minus_old_pp": 100.0 * delta,
                            "improvement_pp": 100.0 * improvement,
                            "bootstrap_ci_low_pp": 100.0 * ci_low,
                            "bootstrap_ci_high_pp": 100.0 * ci_high,
                            "mcnemar_b_old1_new0": b_only,
                            "mcnemar_c_old0_new1": c_only,
                            "mcnemar_p_exact": _mcnemar_exact_p(b_only, c_only),
                        }
                    )
    pairwise = pd.DataFrame(rows)
    if pairwise.empty:
        return pairwise
    return pairwise.sort_values(["track", "metric", "older_model", "newer_model"]).reset_index(drop=True)


@dataclass
class TrendFit:
    slope: float
    se: float
    z_value: float
    p_value: float
    converged: bool


def _fit_logit_question_fe(y: np.ndarray, version_idx: np.ndarray, question_codes: np.ndarray) -> Optional[TrendFit]:
    if y.size == 0:
        return None
    if np.all(y == y[0]):
        return None

    n_questions = int(question_codes.max()) + 1
    if n_questions <= 1:
        return None

    # Design: [version_index, question_dummies (drop first category)]
    dummies = np.zeros((y.size, n_questions - 1), dtype=float)
    for q in range(1, n_questions):
        dummies[:, q - 1] = (question_codes == q).astype(float)
    X = np.column_stack([version_idx.astype(float), dummies])

    beta = np.zeros(X.shape[1], dtype=float)
    ridge = 1e-6
    converged = False
    for _ in range(80):
        eta = X @ beta
        eta = np.clip(eta, -25.0, 25.0)
        p = 1.0 / (1.0 + np.exp(-eta))
        w = np.clip(p * (1.0 - p), 1e-6, None)
        z = eta + (y - p) / w

        xw = X * w[:, None]
        hessian = X.T @ xw
        hessian.flat[:: hessian.shape[0] + 1] += ridge
        rhs = X.T @ (w * z)
        try:
            beta_new = np.linalg.solve(hessian, rhs)
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
    hessian = X.T @ xw
    hessian.flat[:: hessian.shape[0] + 1] += ridge
    try:
        cov = np.linalg.inv(hessian)
    except np.linalg.LinAlgError:
        return None

    slope = float(beta[0])
    se = float(math.sqrt(max(float(cov[0, 0]), 1e-12)))
    z_val = slope / se if se > 0 else float("nan")
    p_val = 2.0 * (1.0 - _normal_cdf(abs(z_val))) if np.isfinite(z_val) else float("nan")
    return TrendFit(slope=slope, se=se, z_value=z_val, p_value=p_val, converged=converged)


def _bootstrap_trend_ci(
    track_df: pd.DataFrame,
    outcome_col: str,
    version_col: str,
    question_col: str,
    n_boot: int,
    seed: int,
) -> Tuple[float, float, int]:
    question_ids = track_df[question_col].dropna().unique().tolist()
    if len(question_ids) < 2:
        return (float("nan"), float("nan"), 0)

    rng = np.random.default_rng(seed)
    slopes: List[float] = []
    for _ in range(n_boot):
        sampled = rng.choice(question_ids, size=len(question_ids), replace=True)
        parts = []
        for q in sampled:
            parts.append(track_df[track_df[question_col] == q])
        boot_df = pd.concat(parts, ignore_index=True)
        q_codes = pd.Categorical(boot_df[question_col]).codes
        fit = _fit_logit_question_fe(
            y=boot_df[outcome_col].to_numpy(dtype=float),
            version_idx=boot_df[version_col].to_numpy(dtype=float),
            question_codes=q_codes,
        )
        if fit is not None and np.isfinite(fit.slope):
            slopes.append(float(fit.slope))

    if not slopes:
        return (float("nan"), float("nan"), 0)
    arr = np.asarray(slopes, dtype=float)
    return float(np.quantile(arr, 0.025)), float(np.quantile(arr, 0.975)), int(arr.size)


def _trend_rows(
    df: pd.DataFrame,
    ce_label: str,
    bootstrap_iters: int,
    seed: int,
) -> pd.DataFrame:
    out = df.copy()
    out["is_correct"] = out["greedy_correct"].map(_to_bool).astype(int)
    out["is_ce"] = (out[f"error_label_{ce_label}"] == "self_consistent_error").astype(int)
    out["model_version_index"] = pd.to_numeric(out["model_version_index"], errors="coerce")
    out = out.dropna(subset=["model_track", "model_version_index", "question_id"]).copy()

    rows: List[Dict[str, Any]] = []
    for track, track_df in out.groupby("model_track", dropna=False):
        n_versions = int(track_df["model_version_index"].nunique())
        if n_versions < 3:
            continue
        for metric_name, col in [("accuracy", "is_correct"), ("ce_rate", "is_ce")]:
            sub = track_df[["question_id", "model_version_index", col]].dropna().copy()
            if sub.empty or int(sub[col].nunique()) < 2:
                continue
            q_codes = pd.Categorical(sub["question_id"]).codes
            fit = _fit_logit_question_fe(
                y=sub[col].to_numpy(dtype=float),
                version_idx=sub["model_version_index"].to_numpy(dtype=float),
                question_codes=q_codes,
            )
            if fit is None:
                continue
            ci_lo, ci_hi, used_boot = _bootstrap_trend_ci(
                track_df=sub.rename(columns={col: "y"}),
                outcome_col="y",
                version_col="model_version_index",
                question_col="question_id",
                n_boot=bootstrap_iters,
                seed=seed + (0 if metric_name == "accuracy" else 4000),
            )
            rows.append(
                {
                    "track": track,
                    "metric": metric_name,
                    "method": "logit_question_fixed_effects",
                    "n_rows": int(len(sub)),
                    "n_questions": int(sub["question_id"].nunique()),
                    "n_versions": n_versions,
                    "slope_per_version": fit.slope,
                    "slope_se": fit.se,
                    "z_value": fit.z_value,
                    "p_value": fit.p_value,
                    "odds_ratio_per_version": float(math.exp(fit.slope)),
                    "slope_bootstrap_ci_low": ci_lo,
                    "slope_bootstrap_ci_high": ci_hi,
                    "bootstrap_successes": used_boot,
                    "converged": fit.converged,
                }
            )
    trends = pd.DataFrame(rows)
    if trends.empty:
        return trends
    return trends.sort_values(["track", "metric"]).reset_index(drop=True)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Analyze CE evolution across ordered model versions.")
    p.add_argument("--config", type=str, default="config.yaml", help="Path to config YAML.")
    p.add_argument("--run-id", type=str, default=None, help="Run ID (required for immutable output if --input is omitted).")
    p.add_argument("--input", type=str, default=None, help="Explicit evaluated JSONL path.")
    p.add_argument("--output-dir", type=str, default=None, help="Output directory for CSV/JSON reports.")
    p.add_argument("--label-threshold", type=str, default="0.9", choices=["1.0", "0.9", "0.8", "0.7"])
    p.add_argument("--bootstrap-iters", type=int, default=2000, help="Bootstrap iterations for pairwise delta CIs.")
    p.add_argument("--trend-bootstrap-iters", type=int, default=500, help="Bootstrap iterations for trend slope CIs.")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--include-incomplete", action="store_true", help="Include incomplete rows in trend/pairwise analysis.")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    with open(args.config, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f) or {}

    evaluated_path = _resolve_evaluated_path(config, args.run_id, args.input)
    logger.info("Loading evaluated rows from %s", evaluated_path)
    df = _load_df(evaluated_path)

    manifest = _prepare_model_manifest(config)
    df = _apply_manifest_defaults(df, manifest)

    required_samples = int(
        (config.get("collection", {}) or {}).get(
            "required_samples",
            ((config.get("inference", {}) or {}).get("stochastic", {}) or {}).get("num_samples", 10),
        )
    )
    validations = _validate(df, required_samples=required_samples)

    work_df = df.copy()
    if not args.include_incomplete and "is_incomplete" in work_df.columns:
        work_df = work_df[~work_df["is_incomplete"].map(_to_bool)].copy()

    ce_col = f"error_label_{args.label_threshold}"
    if ce_col not in work_df.columns:
        raise ValueError(f"Missing required column: {ce_col}")

    model_summary = _model_summary(work_df, ce_label=args.label_threshold)
    pairwise = _pairwise_rows(
        work_df,
        ce_label=args.label_threshold,
        bootstrap_iters=args.bootstrap_iters,
        seed=args.seed,
    )
    trends = _trend_rows(
        work_df,
        ce_label=args.label_threshold,
        bootstrap_iters=args.trend_bootstrap_iters,
        seed=args.seed,
    )

    if args.output_dir:
        out_dir = Path(args.output_dir)
    else:
        run_stub = args.run_id or evaluated_path.stem.replace(".jsonl", "")
        out_dir = Path("data/results/analysis/version_evolution") / run_stub
    out_dir.mkdir(parents=True, exist_ok=True)

    model_path = out_dir / "model_summary.csv"
    pairwise_path = out_dir / "pairwise_deltas.csv"
    trend_path = out_dir / "trend_tests.csv"
    validation_path = out_dir / "validation_checks.json"

    model_summary.to_csv(model_path, index=False)
    pairwise.to_csv(pairwise_path, index=False)
    trends.to_csv(trend_path, index=False)

    payload = {
        "evaluated_path": str(evaluated_path.resolve()),
        "rows_total": int(len(df)),
        "rows_used": int(len(work_df)),
        "threshold": args.label_threshold,
        "include_incomplete": bool(args.include_incomplete),
        "validations": validations,
    }
    validation_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    logger.info("Wrote model summary: %s", model_path)
    logger.info("Wrote pairwise deltas: %s", pairwise_path)
    logger.info("Wrote trend tests: %s", trend_path)
    logger.info("Wrote validation checks: %s", validation_path)


if __name__ == "__main__":
    main()
