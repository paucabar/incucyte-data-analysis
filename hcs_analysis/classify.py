"""Object classification utilities.

Classification adds a new column to the DataFrame; it never removes rows.
The result column can then be used with filter_by() or class_proportions().
"""
from __future__ import annotations

import pandas as pd


def classify_threshold(
    df: pd.DataFrame,
    col: str,
    threshold: float,
    below: str,
    above: str,
    result_col: str,
) -> pd.DataFrame:
    """Add a classification column based on a single numeric threshold.

    Objects with col < threshold are labelled below; col >= threshold → above.

    Parameters
    ----------
    col:
        Measurement column to threshold (e.g. "Circularity").
    threshold:
        Decision boundary.
    below:
        Label for values strictly below threshold (e.g. "Round").
    above:
        Label for values at or above threshold (e.g. "Shape").
    result_col:
        Name for the new classification column (e.g. "morphology_class").

    Returns
    -------
    Copy of df with result_col added.

    Example
    -------
    >>> df = classify_threshold(df, col="Circularity", threshold=0.5,
    ...                         below="Round", above="Shape",
    ...                         result_col="morphology_class")
    """
    out = df.copy()
    out[result_col] = above
    out.loc[out[col] < threshold, result_col] = below
    return out
