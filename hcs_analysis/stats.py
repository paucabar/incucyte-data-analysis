"""Statistical utilities for profiling experiments."""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats as scipy_stats


def _bh_fdr(p_values: np.ndarray) -> np.ndarray:
    """Benjamini-Hochberg FDR correction."""
    n = len(p_values)
    order = np.argsort(p_values)
    adj = p_values[order] * n / np.arange(1, n + 1)
    adj = np.minimum.accumulate(adj[::-1])[::-1]
    out = np.empty(n)
    out[order] = np.clip(adj, 0, 1)
    return out


def compute_map(
    df: pd.DataFrame,
    feature_cols: list[str],
    condition_col: str = "condition",
) -> tuple[float, float]:
    """Mean Average Precision for condition-level replicate reproducibility.

    For each well, positives are all other wells sharing the same condition.
    Similarity is cosine correlation on mean-centred profiles.

    Parameters
    ----------
    df:
        Per-well DataFrame with a condition column and feature columns.
    feature_cols:
        Numeric feature columns representing the well profile.
    condition_col:
        Column identifying experimental condition (default "condition").

    Returns
    -------
    (map_score, random_baseline)
        map_score: mean average precision across all wells.
        random_baseline: expected MAP for a random classifier, approximately
        mean(n_same_condition / (n_wells - 1)). Interpret map_score relative
        to this value — with few replicates it can be close to 0.5.
    """
    X = df[feature_cols].fillna(0).values.astype(float)
    conditions = df[condition_col].values
    n = len(X)

    centered = X - X.mean(axis=1, keepdims=True)
    norms = np.linalg.norm(centered, axis=1, keepdims=True)
    norms = np.where(norms < 1e-12, 1.0, norms)
    X_n = centered / norms
    corr = X_n @ X_n.T

    aps = []
    for i in range(n):
        sims = np.concatenate([corr[i, :i], corr[i, i + 1:]])
        other = np.concatenate([conditions[:i], conditions[i + 1:]])
        rel = (other == conditions[i]).astype(int)
        n_pos = rel.sum()
        if n_pos == 0:
            continue
        order = np.argsort(-sims)
        rel_s = rel[order]
        cum = np.cumsum(rel_s)
        prec = cum / np.arange(1, n)
        aps.append((prec * rel_s).sum() / n_pos)

    map_score = float(np.mean(aps))
    uniq, cnts = np.unique(conditions, return_counts=True)
    random_bl = float(np.mean((cnts - 1) / (n - 1)))
    return map_score, random_bl


def run_stats(
    df: pd.DataFrame,
    feature_cols: list[str],
    control_label: str,
    condition_col: str = "condition",
    sample_col: str = "sample",
) -> pd.DataFrame:
    """Per-feature paired t-test for each non-control condition vs control.

    Replicates are averaged per sample before testing. BH-FDR is applied
    jointly across all features × all conditions, so results are directly
    comparable across conditions.

    Parameters
    ----------
    df:
        Per-well or per-replicate DataFrame with condition and sample columns.
    feature_cols:
        Numeric feature columns to test.
    control_label:
        Value in condition_col that identifies the control group.
    condition_col:
        Column identifying experimental condition (default "condition").
    sample_col:
        Column identifying biological sample (default "sample").

    Returns
    -------
    DataFrame with columns: feature, condition, mean_diff, p_raw, q_bh,
    significant_q05. One row per (feature, non-control condition) pair,
    sorted by p_raw ascending.
    """
    df_sample = (
        df.groupby([sample_col, condition_col])[feature_cols]
        .mean()
        .reset_index()
    )
    ctrl = df_sample[df_sample[condition_col] == control_label].set_index(sample_col)
    treatments = [c for c in df_sample[condition_col].unique() if c != control_label]

    records = []
    for feat in feature_cols:
        for treat in treatments:
            treat_df = df_sample[df_sample[condition_col] == treat].set_index(sample_col)
            paired = ctrl.index.intersection(treat_df.index)
            a = ctrl.loc[paired, feat].values
            b = treat_df.loc[paired, feat].values
            n = len(a)
            if n < 2:
                records.append({
                    "feature": feat, "condition": treat,
                    "mean_diff": np.nan, "p_raw": np.nan,
                })
                continue
            mean_diff = float(b.mean() - a.mean())
            try:
                _, p = scipy_stats.ttest_rel(a, b)
            except Exception:
                p = np.nan
            records.append({"feature": feat, "condition": treat,
                             "mean_diff": mean_diff, "p_raw": p})

    df_stats = pd.DataFrame(records)
    valid = df_stats["p_raw"].notna()
    df_stats["q_bh"] = np.nan
    if valid.any():
        df_stats.loc[valid, "q_bh"] = _bh_fdr(df_stats.loc[valid, "p_raw"].values)
    df_stats["significant_q05"] = df_stats["q_bh"] < 0.05
    return df_stats.sort_values("p_raw").reset_index(drop=True)
