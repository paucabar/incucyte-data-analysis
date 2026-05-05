"""Plotting functions for Incucyte analysis.

Most functions accept an optional ax parameter so plots can be embedded in
multi-panel figures. When ax is None a new figure is created. These functions
return the Axes object.

Exception: metric_heatmap creates its own figure and returns an object with a
``.fig`` attribute. It uses PyComplexHeatmap if available, otherwise seaborn.
"""
from __future__ import annotations

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def qc_counts(
    df: pd.DataFrame,
    groupby: str = "Well",
    ax: plt.Axes | None = None,
) -> plt.Axes:
    """Bar plot of object counts per group.

    Use as a QC check at the start of every notebook to spot empty wells,
    seeding failures, or outlier object counts before any analysis.

    Parameters
    ----------
    df:
        Per-object DataFrame (before aggregation). For multi-timepoint data,
        counts are summed across all timepoints.
    groupby:
        Column to group by (default "Well").
    """
    n = df[groupby].nunique()
    if ax is None:
        _, ax = plt.subplots(figsize=(max(6, n * 0.6), 4))
    counts = df.groupby(groupby).size().reset_index(name="count")
    sns.barplot(data=counts, x=groupby, y="count", errorbar=None, ax=ax)
    ax.set_ylabel("object count")
    ax.tick_params(axis="x", rotation=45)
    return ax


def time_series(
    df: pd.DataFrame,
    y: str,
    hue: str = "condition",
    style: str | None = None,
    time_col: str = "elapsed_min",
    ax: plt.Axes | None = None,
) -> plt.Axes:
    """Line plot of a metric over time, one line per condition.

    Passes per-replicate (or per-well) data directly to seaborn; the 95 % CI
    band is computed by seaborn from the spread across replicates at each
    timepoint. For biological-replicate CI, aggregate technical replicates to
    one value per (sample, condition, timepoint) before calling this function.

    Parameters
    ----------
    df:
        Per-replicate or per-well DataFrame with a time column.
    y:
        Metric to plot on the y-axis.
    hue:
        Column used for line colour (default "condition").
    style:
        Column used for line style; None disables (default). Use "sample" to
        draw one line per biological replicate instead of a CI band.
    time_col:
        Column used for the x-axis (default "elapsed_min"). Pre-compute an
        "elapsed_h" column for a hours axis.
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(10, 5))
    sns.lineplot(data=df, x=time_col, y=y, hue=hue, style=style, ax=ax)
    ax.set_xlabel(time_col)
    ax.set_ylabel(y)
    handles, labels = ax.get_legend_handles_labels()
    if handles:
        ax.legend(handles, labels, bbox_to_anchor=(1.01, 1), loc="upper left",
                  borderaxespad=0)
    return ax


def boxplot(
    df: pd.DataFrame,
    y: str,
    x: str = "condition",
    hue: str | None = None,
    ax: plt.Axes | None = None,
) -> plt.Axes:
    """Boxplot with jittered individual points.

    Each point is one biological replicate. For unpaired comparisons or when
    pairing is not meaningful.
    """
    n = df[x].nunique()
    if ax is None:
        _, ax = plt.subplots(figsize=(max(4, n * 1.5), 5))
    sns.boxplot(data=df, x=x, y=y, hue=hue, ax=ax, width=0.5,
                flierprops={"marker": ""})
    sns.stripplot(data=df, x=x, y=y, hue=hue, ax=ax,
                  dodge=hue is not None, color="black", size=4, alpha=0.6,
                  legend=False)
    ax.set_xlabel(x)
    ax.set_ylabel(y)
    return ax


def metric_heatmap(
    df: pd.DataFrame,
    metrics: list[str],
    metric_classes: dict[str, str],
    class_colors: dict[str, str],
    row_col: str = "condition",
    row_order: list[str] | None = None,
    sample_col: str | None = "sample",
    condition_colors: dict[str, str] | None = None,
    short_names: dict[str, str] | None = None,
    col_cluster: bool = False,
    title: str | None = None,
    figsize: tuple[int, int] = (14, 8),
):
    """Z-scored heatmap of multiple metrics.

    Rows are biological replicates sorted by condition; columns are metrics
    sorted and colour-coded by metric class. Values are Z-scored per column
    (mean 0 ± 1 SD across all rows). Rows are not clustered so condition
    grouping is preserved.

    Uses PyComplexHeatmap if available (recommended), otherwise falls back to
    seaborn clustermap. In both cases returns an object with a ``.fig``
    attribute — call ``g.fig.savefig(...)`` to save.

    Note: ax parameter is not supported — the heatmap creates its own figure.

    Parameters
    ----------
    df:
        Per-replicate DataFrame containing row_col, sample_col (optional),
        and all metric columns.
    metrics:
        Metric columns to include (must be present in df).
    metric_classes:
        Mapping from metric name to class label, used to sort columns and
        colour the column annotation bar (seaborn fallback only).
    class_colors:
        Mapping from class label to colour hex string.
    row_col:
        Column used for row grouping and ordering (default "condition").
    row_order:
        Display order of row_col values. Defaults to sorted unique values.
    sample_col:
        Column used as secondary row label (typically "sample" or "Animal").
        Set to None to use row_col alone as the index.
    condition_colors:
        Mapping from condition name to colour hex string, used for the row
        annotation sidebar (PyComplexHeatmap path). Defaults to matplotlib
        tab10 colours if None.
    short_names:
        Optional mapping from full metric name to short display label for
        column headers. Applies to both backends.
    col_cluster:
        Whether to cluster columns by similarity (default False).
    title:
        Figure suptitle. None suppresses it.
    figsize:
        Figure size passed to the backend.
    """
    import types

    order = row_order if row_order is not None else sorted(df[row_col].unique())
    order_map = {c: i for i, c in enumerate(order)}

    df_s = df.copy()
    df_s["_ord"] = df_s[row_col].map(order_map)
    sort_by = ["_ord"] + ([sample_col] if sample_col and sample_col in df_s.columns else [])
    df_s = df_s.sort_values(sort_by).drop(columns="_ord")

    present = [m for m in metrics if m in df_s.columns]
    metric_class_s = pd.Series(metric_classes, name="Class").reindex(present)
    sorted_cols = list(metric_class_s.sort_values().index)

    data = df_s[sorted_cols].copy()
    if sample_col and sample_col in df_s.columns:
        data.index = pd.MultiIndex.from_arrays(
            [df_s[row_col].values, df_s[sample_col].values]
        )
    else:
        data.index = df_s[row_col].values

    df_z = data.apply(lambda x: (x - x.mean()) / x.std(), axis=0)

    # Flatten MultiIndex rows to "condition — sample" strings
    if isinstance(df_z.index, pd.MultiIndex):
        flat_index = [f"{a} — {b}" for a, b in df_z.index]
    else:
        flat_index = [str(v) for v in df_z.index]
    df_z.index = flat_index

    # Short column display names
    display_cols = [short_names.get(c, c) for c in sorted_cols] if short_names else sorted_cols
    df_z.columns = display_cols

    # Condition label per row (first part of "condition — sample")
    row_conditions = [
        idx.split(" — ")[0] if " — " in idx else idx for idx in flat_index
    ]
    condition_series = pd.Series(row_conditions, index=flat_index, name=row_col)

    try:
        from PyComplexHeatmap import ClusterMapPlotter, HeatmapAnnotation, anno_simple

        cond_colors_map = condition_colors or {c: f"C{i}" for i, c in enumerate(order)}
        row_ha = HeatmapAnnotation(
            **{row_col: anno_simple(condition_series, colors=cond_colors_map, add_text=False)},
            axis=0,
        )
        fig = plt.figure(figsize=figsize)
        ClusterMapPlotter(
            data=df_z,
            left_annotation=row_ha,
            row_cluster=False,
            col_cluster=col_cluster,
            cmap="coolwarm",
            show_rownames=True,
            show_colnames=True,
            col_names_side="bottom",
        )
        if title is not None:
            plt.suptitle(title, y=1.02)
        return types.SimpleNamespace(fig=fig)

    except ImportError:
        pass

    # Seaborn fallback — column class colour bar
    col_colors_s = pd.Series(
        [class_colors.get(metric_class_s.get(orig), "#cccccc") for orig in sorted_cols],
        index=display_cols,
    )
    g = sns.clustermap(
        df_z,
        cmap="coolwarm",
        row_cluster=False,
        col_cluster=col_cluster,
        col_colors=col_colors_s,
        linewidths=0.5,
        figsize=figsize,
        cbar_kws={"label": "Z-score"},
    )
    for label, color in class_colors.items():
        g.ax_col_dendrogram.bar(0, 0, color=color, label=label, linewidth=0)
    g.ax_col_dendrogram.legend(
        title="Metric class", loc="center", ncol=len(class_colors)
    )
    if title is not None:
        g.fig.suptitle(title, y=1.02)
    return g


def paired_boxplot(
    df: pd.DataFrame,
    y: str,
    condition_col: str = "condition",
    pair_col: str = "sample",
    order: list[str] | None = None,
    ax: plt.Axes | None = None,
) -> plt.Axes:
    """Boxplot with lines connecting matched replicates across conditions.

    Use when the same biological unit (animal, sample) appears under multiple
    conditions. Call after averaging technical replicates so that df has one
    row per (sample, condition).

    Parameters
    ----------
    pair_col:
        Column whose values identify matched pairs (typically "sample").
    order:
        Display order of conditions on the x-axis. Defaults to alphabetical.
    """
    conditions = order if order is not None else sorted(df[condition_col].unique())
    pos = {c: i for i, c in enumerate(conditions)}
    if ax is None:
        _, ax = plt.subplots(figsize=(max(4, len(conditions) * 1.5), 5))
    sns.boxplot(data=df, x=condition_col, y=y, order=conditions, ax=ax,
                width=0.4, flierprops={"marker": ""})
    for _, grp in df.groupby(pair_col):
        grp = grp.set_index(condition_col)
        xs = [pos[c] for c in conditions if c in grp.index]
        ys = [grp.loc[c, y] for c in conditions if c in grp.index]
        ax.plot(xs, ys, color="gray", alpha=0.5, linewidth=1, zorder=2)
        ax.scatter(xs, ys, color="black", s=25, zorder=3)
    ax.set_xlabel(condition_col)
    ax.set_ylabel(y)
    return ax
