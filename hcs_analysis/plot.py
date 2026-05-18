"""Plotting functions for Incucyte analysis.

Most functions accept an optional ax parameter so plots can be embedded in
multi-panel figures. When ax is None a new figure is created. These functions
return the Axes object.

Exception: metric_heatmap creates its own figure and returns an object with a
``.fig`` attribute. It uses PyComplexHeatmap if available, otherwise seaborn.
"""
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
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


def plot_pca(
    df: pd.DataFrame,
    feature_cols: list[str],
    condition_col: str = "condition",
    sample_col: str = "sample",
    replicate_col: str = "replicate",
    condition_colors: dict[str, str] | None = None,
    conditions_order: list[str] | None = None,
    average_replicates: bool = False,
    title: str | None = None,
    ax: plt.Axes | None = None,
) -> tuple[plt.Figure, plt.Axes]:
    """PCA of well profiles coloured by condition.

    Parameters
    ----------
    df:
        Per-well DataFrame with condition, sample, and (optionally) replicate
        columns, plus feature columns.
    feature_cols:
        Numeric feature columns used for PCA.
    condition_col, sample_col, replicate_col:
        Column names for condition, biological sample, and replicate.
    condition_colors:
        Mapping from condition name to hex colour. Defaults to matplotlib tab10.
    conditions_order:
        Display order for the legend. Defaults to sorted unique conditions.
    average_replicates:
        If True, average technical replicates per (condition, sample) before
        PCA. Default False plots individual wells.
    title:
        Axes title. None uses a default.
    ax:
        Existing Axes to plot into. If None, a new figure is created.

    Returns
    -------
    (fig, ax)
    """
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler

    group_cols = [c for c in [condition_col, sample_col] if c in df.columns]

    if average_replicates:
        df_plot = df.groupby(group_cols)[feature_cols].mean().reset_index()
    else:
        df_plot = df.copy()

    X = df_plot[feature_cols].fillna(0).values
    X_scaled = StandardScaler().fit_transform(X)
    pca = PCA(n_components=min(2, X_scaled.shape[0] - 1))
    coords = pca.fit_transform(X_scaled)

    df_plot = df_plot.copy()
    df_plot["_PC1"] = coords[:, 0]
    df_plot["_PC2"] = coords[:, 1] if coords.shape[1] > 1 else 0.0

    order  = conditions_order or sorted(df_plot[condition_col].unique())
    colors = condition_colors or {}

    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 5))
    else:
        fig = ax.figure

    for cond in order:
        sub   = df_plot[df_plot[condition_col] == cond]
        color = colors.get(cond) if colors else None
        ax.scatter(sub["_PC1"], sub["_PC2"], c=color,
                   s=100 if average_replicates else 60, zorder=4, label=cond)
        for _, row in sub.iterrows():
            if average_replicates or replicate_col not in row.index:
                label = str(row[sample_col]) if sample_col in row.index else ""
            else:
                label = (
                    f"{row[sample_col]}r{int(row[replicate_col])}"
                    if sample_col in row.index else ""
                )
            ax.text(row["_PC1"] + 0.02, row["_PC2"], label, fontsize=6, color="#555555")

    ax.legend(title=condition_col, fontsize=9)
    ev = pca.explained_variance_ratio_
    ax.set_xlabel(f"PC1 ({ev[0]:.1%})")
    ax.set_ylabel(f"PC2 ({ev[1]:.1%})" if len(ev) > 1 else "PC2")
    default_title = (
        f"PCA — {'condition profiles (replicates averaged)' if average_replicates else 'well profiles'}"
        f"  ({len(feature_cols)} features)"
    )
    ax.set_title(title if title is not None else default_title)
    return fig, ax


def plot_replicate_correlation(
    df: pd.DataFrame,
    feature_cols: list[str],
    condition_col: str = "condition",
    sample_col: str = "sample",
    replicate_col: str = "replicate",
    condition_colors: dict[str, str] | None = None,
    conditions_order: list[str] | None = None,
) -> plt.Figure:
    """Scatter of replicate-1 vs replicate-2 profiles, one panel per condition.

    Each point is one feature. Samples within a condition are colour-coded;
    Pearson r is computed per sample and annotated in the legend.

    Parameters
    ----------
    df:
        Per-well DataFrame. Must have rows for replicate values 1 and 2.
    feature_cols:
        Feature columns used for the scatter.
    condition_col, sample_col, replicate_col:
        Column names for condition, biological sample, and replicate.
    condition_colors:
        Mapping from condition to hex colour. Used only for the panel title.
    conditions_order:
        Panel order (left to right). Defaults to sorted unique conditions.

    Returns
    -------
    matplotlib Figure.
    """
    from scipy.stats import pearsonr

    order   = conditions_order or sorted(df[condition_col].unique())
    samples = sorted(df[sample_col].unique())
    palette = dict(zip(samples, sns.color_palette("tab10", n_colors=len(samples))))

    fig, axes = plt.subplots(1, len(order), figsize=(6 * len(order), 5), squeeze=False)
    axes = axes[0]

    for ax, cond in zip(axes, order):
        cond_df = df[df[condition_col] == cond]
        all_r1, all_r2 = [], []

        for sample in samples:
            s_df = cond_df[cond_df[sample_col] == sample]
            rep1 = s_df[s_df[replicate_col] == 1]
            rep2 = s_df[s_df[replicate_col] == 2]
            if rep1.empty or rep2.empty:
                continue
            v1 = rep1[feature_cols].values.flatten()
            v2 = rep2[feature_cols].values.flatten()
            mask = np.isfinite(v1) & np.isfinite(v2)
            v1, v2 = v1[mask], v2[mask]
            if len(v1) < 2:
                continue
            r, _ = pearsonr(v1, v2)
            ax.scatter(v1, v2, alpha=0.15, s=4, color=palette[sample],
                       label=f"{sample} (r={r:.2f})")
            all_r1.extend(v1)
            all_r2.extend(v2)

        if all_r1:
            r_all, _ = pearsonr(all_r1, all_r2)
            vmin = min(min(all_r1), min(all_r2))
            vmax = max(max(all_r1), max(all_r2))
            ax.plot([vmin, vmax], [vmin, vmax], "k--", linewidth=0.8, alpha=0.5)
            ax.set_title(f"{cond} — rep 1 vs rep 2\n(overall r={r_all:.3f})", fontsize=10)
        else:
            ax.set_title(f"{cond} — no data")

        ax.set_xlabel("Replicate 1")
        ax.set_ylabel("Replicate 2")
        ax.legend(fontsize=7, markerscale=3)

    plt.suptitle(
        f"Replicate correlation — {len(feature_cols)} features (each point = one feature)",
        y=1.02,
    )
    plt.tight_layout()
    return fig


def plot_stats_volcano(
    df_stats: pd.DataFrame,
    conditions_order: list[str] | None = None,
    q_thresh: float = 0.05,
    n_label: int = 10,
    condition_col: str = "condition",
    title: str | None = None,
) -> plt.Figure:
    """Volcano plot of per-feature statistics from run_stats().

    x-axis: mean difference (treatment − control).
    y-axis: −log10(q_bh).
    One panel per non-control condition. Significant features (q < q_thresh)
    are coloured by direction: orange (up) or blue (down). The top n_label
    features by significance are labelled with their feature name.

    Parameters
    ----------
    df_stats:
        DataFrame returned by run_stats(), with columns feature, condition,
        mean_diff, q_bh.
    conditions_order:
        Panel order. Defaults to sorted unique values of condition_col.
    q_thresh:
        Significance threshold drawn as a horizontal dashed line (default 0.05).
    n_label:
        Number of top-significant features to label per panel (default 10).
        Set to 0 to disable labels.
    condition_col:
        Column identifying the treatment condition (default "condition").
    title:
        Figure suptitle. None uses a default.

    Returns
    -------
    matplotlib Figure.
    """
    order  = conditions_order or sorted(df_stats[condition_col].unique())
    n_cond = len(order)

    fig, axes = plt.subplots(
        1, n_cond,
        figsize=(max(5, 5 * n_cond), 5),
        squeeze=False,
    )
    axes = axes[0]

    y_thresh = -np.log10(q_thresh)

    _C_UP   = "#E74C3C"   # red — significant, positive diff
    _C_DOWN = "#3498DB"   # blue — significant, negative diff
    _C_NS   = "#BDC3C7"   # grey — not significant

    for ax, cond in zip(axes, order):
        sub = df_stats[df_stats[condition_col] == cond].copy()
        sub = sub.dropna(subset=["mean_diff", "q_bh"])

        y_vals = -np.log10(sub["q_bh"].clip(lower=1e-300))
        sig    = sub["q_bh"] < q_thresh
        colors = np.where(
            ~sig, _C_NS,
            np.where(sub["mean_diff"] > 0, _C_UP, _C_DOWN)
        )

        ax.scatter(sub["mean_diff"], y_vals, c=colors, s=6, alpha=0.6, linewidths=0)
        ax.axhline(y_thresh, color="#555555", linewidth=0.8, linestyle="--", alpha=0.7)
        ax.axvline(0, color="#aaaaaa", linewidth=0.6, linestyle=":")

        if n_label > 0 and sig.any():
            top = sub[sig].nlargest(n_label, "q_bh" if False else "mean_diff")
            top = sub[sig].assign(_y=y_vals[sig]).nlargest(n_label, "_y")
            for _, row in top.iterrows():
                ax.text(
                    row["mean_diff"],
                    -np.log10(max(row["q_bh"], 1e-300)) + 0.05,
                    row["feature"].split(" - ")[0] if " - " in row["feature"] else row["feature"][:30],
                    fontsize=4,
                    ha="center",
                    va="bottom",
                    color="#333333",
                )

        n_sig = int(sig.sum())
        ax.set_title(
            f"{cond}\n{n_sig}/{len(sub)} features at q<{q_thresh}",
            fontsize=9,
        )
        ax.set_xlabel("Mean difference (treatment − control)")
        ax.set_ylabel(f"−log₁₀(q_BH)")

        # Legend patches
        import matplotlib.patches as mpatches
        handles = [
            mpatches.Patch(color=_C_UP,   label=f"Up (q<{q_thresh})"),
            mpatches.Patch(color=_C_DOWN, label=f"Down (q<{q_thresh})"),
            mpatches.Patch(color=_C_NS,   label="n.s."),
        ]
        ax.legend(handles=handles, fontsize=7, loc="upper left")

    default_title = f"Volcano plot — per-feature statistics"
    fig.suptitle(title if title is not None else default_title, fontsize=10, y=1.02)
    plt.tight_layout()
    return fig
