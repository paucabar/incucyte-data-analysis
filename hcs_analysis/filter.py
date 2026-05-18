"""Row-filtering utilities.

All functions return a copy of the filtered DataFrame. They are designed to
be chained:

    df = filter_by(df, Classification="NSC")
    df = filter_by(df, Well=["C4", "C5", "D4", "D5"])
    df = filter_not_null(df, cols=["Branches"])

Standard pandas indexing also works for one-off conditions:

    df = df[df["Circularity"] > 0.3]
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd


def filter_by(df: pd.DataFrame, **kwargs) -> pd.DataFrame:
    """Filter rows by column equality.

    Keyword arguments map column names to values. A list value uses isin();
    a scalar value uses equality.

    Examples
    --------
    >>> filter_by(df, Classification="NSC")
    >>> filter_by(df, Well=["C4", "C5"], sample="Control")
    """
    mask = pd.Series(True, index=df.index)
    for col, val in kwargs.items():
        if isinstance(val, list):
            mask &= df[col].isin(val)
        else:
            mask &= df[col] == val
    return df[mask].copy()


def filter_not_null(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    """Keep only rows where all specified columns are non-null.

    Typically used to restrict skeleton analyses to objects that have a
    detected skeleton:

        df = filter_not_null(df, cols=["Branches"])
        df = df[df["Branches"] > 0]   # additionally exclude zero-skeleton rows
    """
    mask = pd.Series(True, index=df.index)
    for col in cols:
        mask &= df[col].notna()
    return df[mask].copy()


def filter_timepoints(
    df: pd.DataFrame,
    values: list,
    col: str = "elapsed_min",
) -> pd.DataFrame:
    """Keep only rows matching the specified time values.

    Parameters
    ----------
    values:
        List of time values to keep.
    col:
        Column to filter on. Defaults to elapsed_min; pass "Time index" to
        filter by QuPath frame index instead.
    """
    return df[df[col].isin(values)].copy()


def filter_features(
    df: pd.DataFrame,
    feature_cols: list[str],
    nan_thresh: float = 0.20,
    var_thresh: float = 1e-6,
) -> tuple[list[str], pd.DataFrame]:
    """Filter features by NaN fraction and variance.

    Parameters
    ----------
    df:
        Per-well (or per-object) DataFrame.
    feature_cols:
        Candidate feature columns to evaluate.
    nan_thresh:
        Drop features whose NaN fraction across rows exceeds this value
        (default 0.20).
    var_thresh:
        Drop features whose variance falls below this value (default 1e-6).
        Applied after the NaN check.

    Returns
    -------
    (kept_cols, excluded_df)
        kept_cols: feature names that passed both thresholds, in original order.
        excluded_df: DataFrame with columns feature, nan_frac, variance, status
        ("high NaN fraction" or "low variance") for all excluded features.
    """
    nan_frac  = df[feature_cols].isna().mean()
    variances = df[feature_cols].var()

    records = []
    for col in feature_cols:
        nf  = float(nan_frac[col])
        var = float(variances[col])
        if nf > nan_thresh:
            reason = "high NaN fraction"
        elif var < var_thresh:
            reason = "low variance"
        else:
            reason = "kept"
        records.append({"feature": col, "nan_frac": nf, "variance": var, "status": reason})

    df_report   = pd.DataFrame(records)
    kept_cols   = df_report[df_report["status"] == "kept"]["feature"].tolist()
    excluded_df = df_report[df_report["status"] != "kept"].reset_index(drop=True)
    return kept_cols, excluded_df


# Column name aliases recognised in the exclusions file (case-insensitive)
_EXC_WELL_ALIASES = {"well", "metadata_well"}
_EXC_FOV_ALIASES  = {"fov", "metadata_fov", "field"}


def apply_fov_exclusions(
    df: pd.DataFrame,
    exclusions: pd.DataFrame | str | Path,
    well_col: str = "Metadata_Well",
    fov_col: str = "Metadata_FOV",
) -> pd.DataFrame:
    """Drop per-object rows whose (Well, FOV) pair is flagged for exclusion.

    Use this before any aggregation to remove problematic fields-of-view
    (e.g. air bubbles, fluorescence artefacts, focus failures) identified
    during QC review.

    Parameters
    ----------
    df:
        Per-object CellProfiler DataFrame.
    exclusions:
        DataFrame or path to CSV with at least two columns identifying the
        wells and FOVs to exclude. Column names are matched case-insensitively
        against common variants: ``well`` / ``Well`` / ``Metadata_Well`` and
        ``fov`` / ``FOV`` / ``Metadata_FOV`` / ``field``.
    well_col:
        Column in *df* containing well labels (default ``"Metadata_Well"``).
    fov_col:
        Column in *df* containing FOV indices (default ``"Metadata_FOV"``).

    Returns
    -------
    Copy of df with excluded objects removed.
    """
    if not isinstance(exclusions, pd.DataFrame):
        exc_df = pd.read_csv(exclusions)
    else:
        exc_df = exclusions.copy()

    col_lower = {c.lower(): c for c in exc_df.columns}
    exc_well_orig = next((col_lower[k] for k in _EXC_WELL_ALIASES if k in col_lower), None)
    exc_fov_orig  = next((col_lower[k] for k in _EXC_FOV_ALIASES  if k in col_lower), None)

    if exc_well_orig is None or exc_fov_orig is None:
        raise ValueError(
            "exclusions must have a well column (well/Well/Metadata_Well) "
            "and a FOV column (fov/FOV/Metadata_FOV/field). "
            f"Found columns: {list(exc_df.columns)}"
        )

    excluded = set(zip(exc_df[exc_well_orig], exc_df[exc_fov_orig].astype(int)))
    mask = pd.array(
        [(w, int(f)) in excluded for w, f in zip(df[well_col], df[fov_col])]
    )
    n_removed = int(mask.sum())
    df_out = df[~mask].copy()
    print(
        f"FOV QC: removed {n_removed:,} objects ({len(exc_df)} excluded FOVs)  "
        f"— {len(df_out):,} objects remain"
    )
    return df_out
