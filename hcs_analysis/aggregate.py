"""Aggregation from per-object to per-well to per-replicate.

Correct aggregation order — always go object → well → replicate:

    df_well = to_well_means(df, metrics=[...])
    df_rep  = to_replicate_means(df_well)

Wells are technical replicates; rows in df_rep represent biological replicates
(one per animal/sample/condition combination). Never aggregate objects directly
to replicate level.

When elapsed_min is present, both functions preserve it as a grouping key so
the same calls work for time-series and single-timepoint data.
"""
from __future__ import annotations

import pandas as pd

# Columns that identify a well uniquely (plus elapsed_min when present)
_WELL_KEYS = ["PlateID", "Well", "sample", "condition", "replicate"]
# Columns that identify a biological replicate (plus elapsed_min when present)
_REP_KEYS  = ["sample", "condition", "replicate"]


def _time_keys(df: pd.DataFrame, base_keys: list[str]) -> list[str]:
    """Append elapsed_min to groupby keys if the column exists."""
    if "elapsed_min" in df.columns:
        return base_keys + ["elapsed_min"]
    return base_keys


def _well_keys(df: pd.DataFrame) -> list[str]:
    """Return well-level groupby keys present and non-null in df."""
    keys = [k for k in _WELL_KEYS if k in df.columns and df[k].notna().any()]
    return _time_keys(df, keys)


def to_well_means(df: pd.DataFrame, metrics: list[str]) -> pd.DataFrame:
    """Aggregate per-object measurements to per-well means.

    Groups by (PlateID, Well, sample, condition, replicate) plus elapsed_min
    if present, then computes the mean of each requested metric column.

    Parameters
    ----------
    df:
        Per-object DataFrame with plate map columns merged.
    metrics:
        Measurement columns to aggregate (e.g. ["Circularity", "Area px^2"]).

    Returns
    -------
    One row per (well [, elapsed_min]) with mean values and an object_count
    column recording how many objects were averaged.
    """
    keys = _well_keys(df)
    grouped = df.groupby(keys)
    result = grouped[metrics].mean()
    result["object_count"] = grouped.size()
    return result.reset_index()


def to_replicate_means(df: pd.DataFrame, metrics: list[str] | None = None) -> pd.DataFrame:
    """Aggregate per-well means to per-biological-replicate means.

    Groups by (sample, condition, replicate) plus elapsed_min if present.
    Input is typically the output of to_well_means() or class_proportions().

    Parameters
    ----------
    df:
        Per-well DataFrame.
    metrics:
        Columns to aggregate. If None, all numeric columns except object_count
        are used.

    Returns
    -------
    One row per (sample, condition, replicate [, elapsed_min]).
    """
    keys = [k for k in _REP_KEYS if k in df.columns]
    keys = _time_keys(df, keys)
    if metrics is None:
        skip = set(keys) | {"object_count"}
        metrics = [c for c in df.select_dtypes("number").columns if c not in skip]
    return df.groupby(keys)[metrics].mean().reset_index()


def class_proportions(
    df: pd.DataFrame,
    class_col: str,
    include_count: bool = True,
) -> pd.DataFrame:
    """Compute per-well proportion of each class.

    Call this after classify_threshold() to convert per-object class labels
    into per-well proportions before aggregating to replicate level. The
    result can be passed directly to to_replicate_means().

    Groups by (PlateID, Well, sample, condition, replicate [, elapsed_min])
    and computes the fraction of objects in each class. Produces a
    proportion_<label> column for each unique value in class_col.

    Parameters
    ----------
    class_col:
        Column added by classify_threshold() (e.g. "morphology_class").
    include_count:
        If True, also add a total_objects column.

    Returns
    -------
    One row per (well [, elapsed_min]) with proportion_* columns (and
    total_objects if include_count=True).

    Example
    -------
    >>> df = classify_threshold(df, "Circularity", 0.5, "Round", "Shape", "morph")
    >>> df_prop = class_proportions(df, class_col="morph")
    >>> df_rep  = to_replicate_means(df_prop, metrics=["proportion_Round"])
    """
    keys = _well_keys(df)
    all_keys = keys + [class_col]

    counts = df.groupby(all_keys).size().reset_index(name="n")
    counts["proportion"] = counts["n"] / counts.groupby(keys)["n"].transform("sum")

    pivot = counts.pivot_table(
        index=keys, columns=class_col, values="proportion", fill_value=0
    )
    pivot.columns = [f"proportion_{c}" for c in pivot.columns]
    pivot = pivot.reset_index()

    if include_count:
        totals = counts.groupby(keys)["n"].sum().reset_index(name="total_objects")
        pivot = pivot.merge(totals, on=keys)

    return pivot


def normalize(
    df: pd.DataFrame,
    feature_cols: list[str],
    control_label: str,
    condition_col: str = "condition",
) -> pd.DataFrame:
    """Z-score each feature relative to the control distribution.

    For each feature, subtracts the control-well mean and divides by the
    control-well standard deviation. All conditions are expressed as deviations
    from the control. Features where control std ≈ 0 are mean-subtracted only.

    Parameters
    ----------
    df:
        Per-well DataFrame with condition and feature columns.
    feature_cols:
        Columns to normalize.
    control_label:
        Value in condition_col that identifies the control wells.
    condition_col:
        Column identifying experimental condition (default "condition").

    Returns
    -------
    Copy of df with feature columns replaced by normalized values.
    """
    df = df.copy()
    ctrl_mask = df[condition_col] == control_label
    ctrl_mean = df.loc[ctrl_mask, feature_cols].mean()
    ctrl_std  = df.loc[ctrl_mask, feature_cols].std(ddof=1)

    for c in feature_cols:
        mu, sigma = ctrl_mean[c], ctrl_std[c]
        df[c] = (df[c] - mu) / sigma if sigma > 1e-10 else df[c] - mu

    return df
