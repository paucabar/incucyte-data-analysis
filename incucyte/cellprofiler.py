"""CellProfiler-specific utilities for Cell Painting and profiling analyses.

CellProfiler uses a fixed column naming convention for measurement exports:
    {Compartment}_{MeasurementType}_{Statistic}[_{Channel}[_{Scale}]]

Examples:
    Cells_Intensity_MeanIntensity_Alexa488
    Nuclei_AreaShape_Area
    Cytoplasm_Texture_Variance_HOECHST_3
    Cells_Granularity_1_Mito
    Nuclei_RadialDistribution_MeanFrac_DAPI_0of4

This module provides:
- Feature parsing: extract feature type, compartment, and channel from
  CellProfiler column names.
- Heatmap annotation: build colour-coded column annotation bars for
  seaborn clustermaps, analogous to incucyte.operetta for Harmony exports.
- Plotting: annotated heatmaps for profiling data from CellProfiler exports.

Unlike Harmony exports, CellProfiler channel names are defined by the user
in the CellProfiler pipeline, so no default channel colour map is provided.
Pass a ``channel_colors`` dict to any function; if omitted, colours are
assigned automatically from a matplotlib palette.

Note on Operetta images processed through CellProfiler: Operetta/PerkinElmer
TIFF files use a different naming convention (e.g. r01c01f01p01-ch1sk1fk1fl1.tiff)
from the standard Incucyte pattern parsed by ``load_cellprofiler``. Until an
Operetta CellProfiler dataset is available, the metadata column handling for
that case is unresolved — this module handles only the measurement column
annotation, which is independent of image source.

Usage::

    import hcs_analysis.cellprofiler as cp

    col_colors = cp.cp_col_colors(feature_cols,
                                  channel_colors={"DAPI": "#9B59B6",
                                                  "Alexa488": "#2ECC71"})
    cp.plot_cp_heatmap(df, feature_cols, "heatmap.pdf",
                       channel_colors={"DAPI": "#9B59B6", "Alexa488": "#2ECC71"},
                       condition_colors={"Control": "#17A589", "BMP4": "#E67E22"})
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# ── Default colour maps ───────────────────────────────────────────────────────

# Measurement type → feature type label
FTYPE_MAP: dict[str, str] = {
    "Intensity":          "Intensity",
    "Texture":            "Texture",
    "Granularity":        "Texture",
    "RadialDistribution": "Texture",
    "AreaShape":          "Morphology",
    "ObjectSkeleton":     "Morphology",
}

FTYPE_COLORS: dict[str, str] = {
    "Intensity":  "#F39C12",
    "Texture":    "#8E44AD",
    "Morphology": "#16A085",
    "Other":      "#95A5A6",
}

COMPARTMENT_COLORS: dict[str, str] = {
    "Nuclei":    "#2C3E50",
    "Cells":     "#27AE60",
    "Cytoplasm": "#E67E22",
    "n/a":       "#BDC3C7",
}

# Measurement types that carry a channel name at parts[3] of the column name
_CHANNELLED_TYPES: frozenset[str] = frozenset(
    {"Intensity", "Texture", "Granularity", "RadialDistribution", "Correlation"}
)


# ── Feature parsing ───────────────────────────────────────────────────────────

def parse_cp_feature(col: str) -> tuple[str, str, str]:
    """Parse a CellProfiler column name into (feature_type, compartment, channel).

    Parameters
    ----------
    col:
        CellProfiler measurement column name, e.g.
        ``"Cells_Intensity_MeanIntensity_Alexa488"`` or
        ``"Nuclei_AreaShape_Area"``.

    Returns
    -------
    (feature_type, compartment, channel)
        feature_type: "Intensity", "Texture", "Morphology", or "Other".
        compartment: first segment of the column name (e.g. "Cells", "Nuclei").
        channel: fourth segment for channelled measurement types, else "n/a".
    """
    parts = col.split("_")
    if len(parts) < 2:
        return "Other", "n/a", "n/a"

    compartment = parts[0]
    meas_type   = parts[1]
    ftype       = FTYPE_MAP.get(meas_type, "Other")
    channel     = parts[3] if (meas_type in _CHANNELLED_TYPES and len(parts) >= 4) else "n/a"

    return ftype, compartment, channel


def cp_feature_labels(feature_cols: list[str]) -> pd.DataFrame:
    """Return a DataFrame of annotation label strings for CellProfiler columns.

    Returns
    -------
    DataFrame with columns "Feature type", "Compartment", "Channel".
    Index = feature names.
    """
    rows = [
        dict(zip(("Feature type", "Compartment", "Channel"), parse_cp_feature(c)))
        for c in feature_cols
    ]
    return pd.DataFrame(rows, index=feature_cols)


def _auto_channel_colors(channels: list[str]) -> dict[str, str]:
    """Assign tab10 colours to a list of channel names, preserving order."""
    unique = list(dict.fromkeys(c for c in channels if c != "n/a"))
    palette = sns.color_palette("tab10", n_colors=max(len(unique), 1))
    colors = {ch: palette[i % len(palette)] for i, ch in enumerate(unique)}
    # Convert RGBA tuples to hex
    colors = {
        ch: "#{:02x}{:02x}{:02x}".format(
            int(r * 255), int(g * 255), int(b * 255)
        )
        for ch, (r, g, b) in colors.items()
    }
    colors["n/a"] = "#BDC3C7"
    return colors


def cp_col_colors(
    feature_cols: list[str],
    ftype_colors: dict[str, str] | None = None,
    compartment_colors: dict[str, str] | None = None,
    channel_colors: dict[str, str] | None = None,
) -> pd.DataFrame:
    """Return a DataFrame of hex colours for use as seaborn clustermap col_colors.

    Each column of the returned DataFrame becomes one annotation strip above
    the heatmap. Index = feature names.

    Parameters
    ----------
    ftype_colors:
        Override for feature type colours. See FTYPE_COLORS for defaults.
    compartment_colors:
        Override for compartment colours. See COMPARTMENT_COLORS for defaults.
    channel_colors:
        Mapping from channel name to hex colour. If None, colours are assigned
        automatically from the matplotlib tab10 palette in order of first
        appearance. Pass explicit colours to ensure consistency across plots.
    """
    ft_colors   = ftype_colors       if ftype_colors       is not None else FTYPE_COLORS
    comp_colors = compartment_colors if compartment_colors is not None else COMPARTMENT_COLORS

    parsed = [parse_cp_feature(c) for c in feature_cols]

    if channel_colors is None:
        channel_colors = _auto_channel_colors([ch for _, _, ch in parsed])

    rows = []
    for ftype, compartment, channel in parsed:
        rows.append({
            "Feature type": ft_colors.get(ftype,       "#BDC3C7"),
            "Compartment":  comp_colors.get(compartment, "#BDC3C7"),
            "Channel":      channel_colors.get(channel, "#BDC3C7"),
        })
    return pd.DataFrame(rows, index=feature_cols)


def cp_annotation_handles(
    feature_cols: list[str],
    ftype_colors: dict[str, str] | None = None,
    compartment_colors: dict[str, str] | None = None,
    channel_colors: dict[str, str] | None = None,
) -> list:
    """Return matplotlib patch handles for all annotation levels.

    Suitable for passing to ``fig.legend(handles=...)``. Channel colours are
    derived from the feature columns so the legend matches the heatmap exactly.

    Parameters
    ----------
    feature_cols:
        Same feature columns passed to cp_col_colors.
    ftype_colors, compartment_colors, channel_colors:
        Override default colour maps (same semantics as cp_col_colors).
    """
    ft_colors   = ftype_colors       if ftype_colors       is not None else FTYPE_COLORS
    comp_colors = compartment_colors if compartment_colors is not None else COMPARTMENT_COLORS

    parsed = [parse_cp_feature(c) for c in feature_cols]
    if channel_colors is None:
        channel_colors = _auto_channel_colors([ch for _, _, ch in parsed])

    handles = []
    for label, color in ft_colors.items():
        handles.append(mpatches.Patch(color=color, label=label, linewidth=0))
    handles.append(mpatches.Patch(color="none", label=""))

    seen_comp: set[str] = set()
    for _, compartment, _ in parsed:
        if compartment not in seen_comp:
            seen_comp.add(compartment)
            color = comp_colors.get(compartment, "#BDC3C7")
            handles.append(mpatches.Patch(color=color, label=compartment, linewidth=0))
    handles.append(mpatches.Patch(color="none", label=""))

    seen_ch: set[str] = set()
    for _, _, channel in parsed:
        if channel not in seen_ch:
            seen_ch.add(channel)
            color = channel_colors.get(channel, "#BDC3C7")
            handles.append(mpatches.Patch(color=color, label=channel, linewidth=0))

    return handles


# ── Internal helpers ──────────────────────────────────────────────────────────

def _zscore_matrix(matrix: np.ndarray) -> np.ndarray:
    col_means = np.nanmean(matrix, axis=0)
    col_stds  = np.nanstd(matrix, axis=0, ddof=1)
    col_stds[col_stds < 1e-10] = 1.0
    return (matrix - col_means) / col_stds


def _add_annotation_legend(fig: plt.Figure, handles: list) -> None:
    fig.legend(
        handles=handles,
        loc="lower center",
        ncol=6,
        fontsize=7,
        bbox_to_anchor=(0.5, -0.06),
        title="Column annotations  [Feature type | Compartment | Channel]",
        title_fontsize=8,
        frameon=True,
    )


# ── Plotting ──────────────────────────────────────────────────────────────────

def plot_cp_heatmap(
    df: pd.DataFrame,
    feature_cols: list[str],
    output_path: Path | str,
    condition_col: str = "condition",
    sample_col: str = "sample",
    replicate_col: str = "replicate",
    condition_colors: dict[str, str] | None = None,
    channel_colors: dict[str, str] | None = None,
    average_replicates: bool = False,
    n_top_features: int = 200,
    ftype_colors: dict[str, str] | None = None,
    compartment_colors: dict[str, str] | None = None,
    title: str | None = None,
) -> None:
    """Z-scored annotated clustermap of CellProfiler well profiles.

    Column annotation strips show Feature type, Compartment, and Channel
    parsed from CellProfiler column names.

    Parameters
    ----------
    df:
        Per-well DataFrame with condition, sample, and (optionally) replicate
        columns, plus feature columns.
    feature_cols:
        Numeric feature columns to include.
    output_path:
        Path to save the PDF.
    condition_col, sample_col, replicate_col:
        Column names for condition, biological sample, and replicate.
    condition_colors:
        Mapping from condition to hex colour for the row annotation sidebar.
    channel_colors:
        Mapping from channel name to hex colour. If None, colours are assigned
        automatically. Pass explicit colours for consistency across plots.
    average_replicates:
        If True, average technical replicates per (condition, sample) before
        plotting. Default False keeps individual wells as rows.
    n_top_features:
        Number of highest-variance features to display (default 200).
    ftype_colors, compartment_colors:
        Override default colour maps.
    title:
        Figure suptitle. None generates a default.
    """
    output_path = Path(output_path)
    N = min(n_top_features, len(feature_cols))
    top_feats = df[feature_cols].var().nlargest(N).index.tolist()

    group_cols = [c for c in [condition_col, sample_col] if c in df.columns]

    if average_replicates:
        df_plot = (
            df.groupby(group_cols)[top_feats]
            .mean()
            .reset_index()
            .sort_values(group_cols)
        )
        row_labels = (
            df_plot[condition_col] + " · " + df_plot[sample_col].astype(str)
        ).values
        cond_vals = df_plot[condition_col].values
    else:
        sort_cols = group_cols + ([replicate_col] if replicate_col in df.columns else [])
        df_plot = df.sort_values(sort_cols)
        rep_part = (
            " rep" + df_plot[replicate_col].astype(str)
            if replicate_col in df_plot.columns
            else pd.Series("", index=df_plot.index)
        )
        row_labels = (
            df_plot[condition_col] + " · " + df_plot[sample_col].astype(str) + rep_part
        ).values
        cond_vals = df_plot[condition_col].values

    z = _zscore_matrix(df_plot[top_feats].values.astype(float))
    data_df = pd.DataFrame(z, index=row_labels, columns=top_feats)

    cond_colors_map = condition_colors or {}
    row_colors = pd.Series(
        [cond_colors_map.get(c, "#aaaaaa") for c in cond_vals],
        index=row_labels, name=condition_col,
    )
    col_colors = cp_col_colors(
        top_feats,
        ftype_colors=ftype_colors,
        compartment_colors=compartment_colors,
        channel_colors=channel_colors,
    )

    g = sns.clustermap(
        data_df,
        row_cluster=False,
        col_cluster=True,
        row_colors=row_colors,
        col_colors=col_colors,
        cmap="coolwarm",
        center=0,
        figsize=(max(14, N * 0.08), max(5, len(data_df) * 0.4)),
        cbar_kws={"label": "Z-score"},
        xticklabels=False,
        yticklabels=True,
    )
    g.ax_heatmap.set_yticklabels(
        g.ax_heatmap.get_yticklabels(), rotation=0, fontsize=8,
    )

    default_title = (
        f"{'Condition profiles (replicates averaged)' if average_replicates else 'Well profiles'}"
        f" — top {N} highest-variance features (Z-scored)\n({len(feature_cols)} features total)"
    )
    g.fig.suptitle(title if title is not None else default_title, fontsize=10, y=1.01)
    _add_annotation_legend(g.fig, cp_annotation_handles(
        top_feats,
        ftype_colors=ftype_colors,
        compartment_colors=compartment_colors,
        channel_colors=channel_colors,
    ))
    g.fig.savefig(output_path, format="pdf", bbox_inches="tight")
    plt.close(g.fig)


def plot_cp_heatmap_grouped(
    df: pd.DataFrame,
    feature_cols: list[str],
    output_path: Path | str,
    condition_col: str = "condition",
    sample_col: str = "sample",
    condition_colors: dict[str, str] | None = None,
    channel_colors: dict[str, str] | None = None,
    n_top_features: int = 200,
    ftype_colors: dict[str, str] | None = None,
    compartment_colors: dict[str, str] | None = None,
    title: str | None = None,
) -> None:
    """Z-scored heatmap with columns grouped by Feature type then Compartment.

    Columns are sorted by Feature type → Compartment → descending variance.
    White vertical lines and group labels separate Feature type blocks. Rows
    are replicate-averaged (condition × sample).

    Parameters
    ----------
    df:
        Per-well DataFrame.
    feature_cols:
        Numeric feature columns to include.
    output_path:
        Path to save the PDF.
    condition_col, sample_col:
        Column names for condition and biological sample.
    condition_colors:
        Mapping from condition to hex colour for the row annotation sidebar.
    channel_colors:
        Mapping from channel name to hex colour. If None, auto-assigned.
    n_top_features:
        Number of highest-variance features to display (default 200).
    ftype_colors, compartment_colors:
        Override default colour maps.
    title:
        Figure suptitle. None generates a default.
    """
    output_path = Path(output_path)
    N = min(n_top_features, len(feature_cols))
    top_feats = df[feature_cols].var().nlargest(N).index.tolist()

    labels    = cp_feature_labels(top_feats)
    variances = df[top_feats].var()
    labels["variance"] = variances

    _FTYPE_ORDER = ["Intensity", "Texture", "Morphology", "Other"]
    seen_comp: dict[str, int] = {}
    for _, row in labels.iterrows():
        c = row["Compartment"]
        if c not in seen_comp:
            seen_comp[c] = len(seen_comp)
    _COMP_ORDER = sorted(seen_comp, key=lambda k: seen_comp[k])

    labels["_ft_rank"] = labels["Feature type"].map(
        {v: i for i, v in enumerate(_FTYPE_ORDER)}
    ).fillna(len(_FTYPE_ORDER))
    labels["_comp_rank"] = labels["Compartment"].map(
        {v: i for i, v in enumerate(_COMP_ORDER)}
    ).fillna(len(_COMP_ORDER))
    sorted_feats = (
        labels.sort_values(["_ft_rank", "_comp_rank", "variance"],
                           ascending=[True, True, False])
        .index.tolist()
    )

    ft_series   = labels.loc[sorted_feats, "Feature type"]
    boundaries  = []
    group_spans = []
    prev_ft, block_start = None, 0
    for i, feat in enumerate(sorted_feats):
        ft = ft_series[feat]
        if ft != prev_ft:
            if prev_ft is not None:
                boundaries.append(i)
                group_spans.append((block_start, i, prev_ft))
            block_start = i
            prev_ft = ft
    if prev_ft is not None:
        group_spans.append((block_start, len(sorted_feats), prev_ft))

    group_cols = [c for c in [condition_col, sample_col] if c in df.columns]
    df_agg = (
        df.groupby(group_cols)[sorted_feats]
        .mean()
        .reset_index()
        .sort_values(group_cols)
    )
    row_labels = (df_agg[condition_col] + " · " + df_agg[sample_col].astype(str)).values

    z = _zscore_matrix(df_agg[sorted_feats].values.astype(float))
    data_df = pd.DataFrame(z, index=row_labels, columns=sorted_feats)

    cond_colors_map = condition_colors or {}
    row_colors = pd.Series(
        [cond_colors_map.get(c, "#aaaaaa") for c in df_agg[condition_col].values],
        index=row_labels, name=condition_col,
    )
    col_colors = cp_col_colors(
        sorted_feats,
        ftype_colors=ftype_colors,
        compartment_colors=compartment_colors,
        channel_colors=channel_colors,
    )

    g = sns.clustermap(
        data_df,
        row_cluster=False,
        col_cluster=False,
        row_colors=row_colors,
        col_colors=col_colors,
        cmap="coolwarm",
        center=0,
        figsize=(max(14, N * 0.08), max(4, len(df_agg) * 0.5)),
        cbar_kws={"label": "Z-score"},
        xticklabels=False,
        yticklabels=True,
    )
    g.ax_heatmap.set_yticklabels(
        g.ax_heatmap.get_yticklabels(), rotation=0, fontsize=9,
    )

    for x in boundaries:
        g.ax_heatmap.axvline(x, color="white", linewidth=1.5, zorder=5)
    for start, end, ft_label in group_spans:
        mid = (start + end) / 2
        g.ax_heatmap.text(
            mid, -0.6, ft_label,
            ha="center", va="bottom", fontsize=7, color="#333333",
            transform=g.ax_heatmap.get_xaxis_transform(),
        )

    default_title = (
        f"Condition profiles — columns grouped by Feature type → Compartment "
        f"(top {N}, Z-scored)\n({len(feature_cols)} features total)"
    )
    g.fig.suptitle(title if title is not None else default_title, fontsize=10, y=1.01)
    _add_annotation_legend(g.fig, cp_annotation_handles(
        sorted_feats,
        ftype_colors=ftype_colors,
        compartment_colors=compartment_colors,
        channel_colors=channel_colors,
    ))
    g.fig.savefig(output_path, format="pdf", bbox_inches="tight")
    plt.close(g.fig)
