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
