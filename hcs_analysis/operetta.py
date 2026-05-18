"""Operetta/Harmony-specific utilities for Cell Painting and profiling analyses.

Harmony exports use a fixed column naming convention:
    {Object} - {Region} Region {Channel} {Measurement} - {Aggregation}
e.g. "Nuclei - Nucleus Region Alexa 488 Intensity - Mean per Well"

This module provides:
- Feature parsing: extract feature type, channel, and sub-cellular region
  from Harmony column names.
- Heatmap annotation: build colour-coded column annotation bars for
  seaborn clustermaps.
- Plotting: annotated heatmaps for profiling data from Harmony exports.

Default colour maps reflect a standard 4-channel Cell Painting protocol
(ER / RNA-Phal-Golgi / Mito / DNA). Pass override dicts to any function
to adapt to a different staining panel.

Usage::

    import hcs_analysis.operetta as harmony

    col_colors = harmony.harmony_col_colors(feature_cols)
    harmony.plot_harmony_heatmap(df, feature_cols, "heatmap.pdf",
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

FTYPE_COLORS: dict[str, str] = {
    "Intensity":  "#F39C12",
    "Texture":    "#8E44AD",
    "Morphology": "#16A085",
    "Other":      "#95A5A6",
}

# Key: substring present in Harmony column name
# Value: (display label, hex colour)
CHANNEL_MAP: dict[str, tuple[str, str]] = {
    "Alexa 488":      ("ER",             "#2ECC71"),
    "Alexa 568":      ("RNA/Phal/Golgi", "#E74C3C"),
    "Alexa 647":      ("Mito",           "#3498DB"),
    "HOECHST 33342":  ("DNA",            "#9B59B6"),
}
CHANNEL_NA_COLOR: str = "#BDC3C7"

REGION_COLORS: dict[str, str] = {
    "Nucleus":   "#2C3E50",
    "Cell":      "#27AE60",
    "Cytoplasm": "#E67E22",
    "Membrane":  "#1ABC9C",
    "Ring":      "#C0392B",
    "n/a":       "#BDC3C7",
}

TEXTURE_KEYWORDS: tuple[str, ...] = (
    "SER", "Symmetry", "Profile", "Radial", "Threshold Compactness"
)
MORPHOLOGY_KEYWORDS: tuple[str, ...] = (
    "Area", "Length", "Width", "Perimeter", "Roundness", "Ratio Width", "Axial"
)


# ── Feature parsing ───────────────────────────────────────────────────────────

def parse_harmony_feature(
    col: str,
    channel_map: dict[str, tuple[str, str]] | None = None,
    texture_keywords: tuple[str, ...] | None = None,
    morphology_keywords: tuple[str, ...] | None = None,
) -> tuple[str, str, str]:
    """Parse a Harmony column name into (feature_type, channel_label, region_label).

    Parameters
    ----------
    col:
        Harmony column name, e.g.
        "Nuclei - Nucleus Region Alexa 488 Intensity - Mean per Well".
    channel_map:
        Override for the default CHANNEL_MAP. Keys are substrings present in
        the column name; values are (display_label, hex_colour) tuples.
    texture_keywords, morphology_keywords:
        Override for the default keyword tuples used to classify feature type.

    Returns
    -------
    (feature_type, channel_label, region_label)
        feature_type: "Intensity", "Texture", "Morphology", or "Other".
        channel_label: display label from channel_map, or "n/a".
        region_label: "Nucleus", "Cytoplasm", "Membrane", "Ring", "Cell", or "n/a".
    """
    ch_map   = channel_map         if channel_map         is not None else CHANNEL_MAP
    tex_kw   = texture_keywords    if texture_keywords    is not None else TEXTURE_KEYWORDS
    morph_kw = morphology_keywords if morphology_keywords is not None else MORPHOLOGY_KEYWORDS

    region = "n/a"
    for rname in ("Nucleus", "Cytoplasm", "Membrane", "Ring", "Cell"):
        if f"{rname} Region" in col:
            region = rname
            break

    channel = "n/a"
    for key, (label, _) in ch_map.items():
        if key in col:
            channel = label
            break

    body = col.split(" - ")[1] if " - " in col else col
    if body.startswith("Intensity "):
        ftype = "Intensity"
    elif any(kw in body for kw in tex_kw):
        ftype = "Texture"
    elif any(kw in body for kw in morph_kw):
        ftype = "Morphology"
    else:
        ftype = "Other"

    return ftype, channel, region


def harmony_feature_labels(
    feature_cols: list[str],
    **parse_kwargs,
) -> pd.DataFrame:
    """Return a DataFrame of annotation label strings for a list of Harmony columns.

    Parameters
    ----------
    feature_cols:
        Harmony measurement column names to annotate.
    **parse_kwargs:
        Forwarded to parse_harmony_feature (channel_map, texture_keywords,
        morphology_keywords).

    Returns
    -------
    DataFrame with columns "Feature type", "Channel", "Region". Index = feature names.
    """
    rows = [
        dict(zip(("Feature type", "Channel", "Region"),
                 parse_harmony_feature(c, **parse_kwargs)))
        for c in feature_cols
    ]
    return pd.DataFrame(rows, index=feature_cols)


def harmony_col_colors(
    feature_cols: list[str],
    ftype_colors: dict[str, str] | None = None,
    channel_map: dict[str, tuple[str, str]] | None = None,
    region_colors: dict[str, str] | None = None,
) -> pd.DataFrame:
    """Return a DataFrame of hex colours for use as seaborn clustermap col_colors.

    Each column of the returned DataFrame becomes one annotation strip above the
    heatmap. Index = feature names.

    Parameters
    ----------
    ftype_colors, channel_map, region_colors:
        Override default colour maps. See module-level constants for defaults.
    """
    ft_colors = ftype_colors  if ftype_colors  is not None else FTYPE_COLORS
    ch_map    = channel_map   if channel_map   is not None else CHANNEL_MAP
    rg_colors = region_colors if region_colors is not None else REGION_COLORS

    rows = []
    for col in feature_cols:
        ftype, _, region = parse_harmony_feature(col, channel_map=ch_map)
        ch_color = next(
            (color for key, (_, color) in ch_map.items() if key in col),
            CHANNEL_NA_COLOR,
        )
        rows.append({
            "Feature type": ft_colors.get(ftype, "#BDC3C7"),
            "Channel":      ch_color,
            "Region":       rg_colors.get(region, "#BDC3C7"),
        })
    return pd.DataFrame(rows, index=feature_cols)


def harmony_annotation_handles(
    ftype_colors: dict[str, str] | None = None,
    channel_map: dict[str, tuple[str, str]] | None = None,
    region_colors: dict[str, str] | None = None,
) -> list:
    """Return matplotlib patch handles for all annotation levels.

    Suitable for passing to ``fig.legend(handles=...)``.
    """
    ft_colors = ftype_colors  if ftype_colors  is not None else FTYPE_COLORS
    ch_map    = channel_map   if channel_map   is not None else CHANNEL_MAP
    rg_colors = region_colors if region_colors is not None else REGION_COLORS

    handles = []
    for label, color in ft_colors.items():
        handles.append(mpatches.Patch(color=color, label=label, linewidth=0))
    handles.append(mpatches.Patch(color="none", label=""))
    for _, (label, color) in ch_map.items():
        handles.append(mpatches.Patch(color=color, label=label, linewidth=0))
    handles.append(mpatches.Patch(color=CHANNEL_NA_COLOR, label="n/a", linewidth=0))
    handles.append(mpatches.Patch(color="none", label=""))
    for label, color in rg_colors.items():
        handles.append(mpatches.Patch(color=color, label=label, linewidth=0))
    return handles


# ── Internal helpers ──────────────────────────────────────────────────────────

def _add_annotation_legend(
    fig: plt.Figure,
    handles: list,
) -> None:
    """Three compact legend boxes (Feature type | Channel | Region) below the figure."""
    # Split handles at empty-label separators inserted by harmony_annotation_handles
    groups: list[list] = [[]]
    for h in handles:
        if h.get_label() == "":
            groups.append([])
        else:
            groups[-1].append(h)
    while len(groups) < 3:
        groups.append([])
    ft_h, ch_h, rg_h = groups[0], groups[1], groups[2]

    kw = dict(
        loc="lower left", fontsize=8, title_fontsize=9, frameon=True,
        borderpad=0.4, labelspacing=0.25, handlelength=0.8, handletextpad=0.4,
        ncol=2,
    )
    leg1 = fig.legend(handles=ft_h, title="Feature type",
                      bbox_to_anchor=(0.22, -0.14), **kw)
    fig.add_artist(leg1)
    leg2 = fig.legend(handles=ch_h, title="Channel",
                      bbox_to_anchor=(0.44, -0.14), **kw)
    fig.add_artist(leg2)
    fig.legend(handles=rg_h, title="Region",
               bbox_to_anchor=(0.66, -0.14), **kw)


def _zscore_matrix(matrix: np.ndarray) -> np.ndarray:
    col_means = np.nanmean(matrix, axis=0)
    col_stds  = np.nanstd(matrix, axis=0, ddof=1)
    col_stds[col_stds < 1e-10] = 1.0
    return (matrix - col_means) / col_stds


# ── Plotting ──────────────────────────────────────────────────────────────────

def plot_harmony_heatmap(
    df: pd.DataFrame,
    feature_cols: list[str],
    output_path: Path | str,
    condition_col: str = "condition",
    sample_col: str = "sample",
    replicate_col: str = "replicate",
    condition_colors: dict[str, str] | None = None,
    average_replicates: bool = False,
    n_top_features: int = 200,
    ftype_colors: dict[str, str] | None = None,
    channel_map: dict[str, tuple[str, str]] | None = None,
    region_colors: dict[str, str] | None = None,
    title: str | None = None,
) -> None:
    """Z-scored annotated clustermap of Harmony well profiles.

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
        Defaults to grey for unrecognised conditions.
    average_replicates:
        If True, average technical replicates per (condition, sample) before
        plotting. Default False keeps individual wells as rows.
    n_top_features:
        Number of highest-variance features to display (default 200).
    ftype_colors, channel_map, region_colors:
        Override default Harmony annotation colour maps.
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
    col_colors = harmony_col_colors(
        top_feats,
        ftype_colors=ftype_colors,
        channel_map=channel_map,
        region_colors=region_colors,
    )

    # Cap width at 16 in; compute colors_ratio so row bar = col bar in absolute thickness
    _fig_w = min(max(12, N * 0.03), 16)
    _fig_h = max(8, len(data_df) * 0.6)
    _bar   = 0.10   # target bar thickness in inches for both row and col strips
    g = sns.clustermap(
        data_df,
        row_cluster=False,
        col_cluster=True,
        row_colors=row_colors,
        col_colors=col_colors,
        cmap="coolwarm",
        center=0,
        figsize=(_fig_w, _fig_h),
        colors_ratio=(_bar / _fig_w, _bar / _fig_h),
        cbar_kws={"label": "Z-score", "shrink": 0.4},
        xticklabels=False,
        yticklabels=True,
    )
    g.ax_heatmap.set_yticklabels(
        g.ax_heatmap.get_yticklabels(), rotation=0, fontsize=8,
    )
    # Compact colorbar: shrink to 40 % of its allocated space, centred vertically
    _p = g.cax.get_position()
    g.cax.set_position([_p.x0, _p.y0 + _p.height * 0.3, _p.width, _p.height * 0.4])

    default_title = (
        f"{'Condition profiles (replicates averaged)' if average_replicates else 'Well profiles'}"
        f" — top {N} highest-variance features (Z-scored)\n({len(feature_cols)} features total)"
    )
    g.fig.suptitle(title if title is not None else default_title, fontsize=10, y=1.01)
    _add_annotation_legend(g.fig, harmony_annotation_handles(
        ftype_colors=ftype_colors, channel_map=channel_map, region_colors=region_colors,
    ))
    g.fig.savefig(output_path, format="pdf", bbox_inches="tight")
    plt.close(g.fig)


def plot_harmony_heatmap_grouped(
    df: pd.DataFrame,
    feature_cols: list[str],
    output_path: Path | str,
    condition_col: str = "condition",
    sample_col: str = "sample",
    condition_colors: dict[str, str] | None = None,
    n_top_features: int = 200,
    ftype_colors: dict[str, str] | None = None,
    channel_map: dict[str, tuple[str, str]] | None = None,
    region_colors: dict[str, str] | None = None,
    title: str | None = None,
) -> None:
    """Z-scored heatmap with columns grouped by Feature type then Channel.

    Columns are sorted by Feature type → Channel → descending variance. White
    vertical lines and group labels separate Feature type blocks. Rows are
    replicate-averaged (condition × sample).

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
    n_top_features:
        Number of highest-variance features to display (default 200).
    ftype_colors, channel_map, region_colors:
        Override default Harmony annotation colour maps.
    title:
        Figure suptitle. None generates a default.
    """
    output_path = Path(output_path)
    N = min(n_top_features, len(feature_cols))
    top_feats = df[feature_cols].var().nlargest(N).index.tolist()

    ch_map = channel_map if channel_map is not None else CHANNEL_MAP

    labels    = harmony_feature_labels(top_feats, channel_map=ch_map)
    variances = df[top_feats].var()
    labels["variance"] = variances

    _FTYPE_ORDER = ["Intensity", "Texture", "Morphology", "Other"]
    # Channel order derived from ch_map insertion order, then "n/a"
    seen: set[str] = set()
    _CHANNEL_ORDER: list[str] = []
    for _, (label, _) in ch_map.items():
        if label not in seen:
            _CHANNEL_ORDER.append(label)
            seen.add(label)
    if "n/a" not in seen:
        _CHANNEL_ORDER.append("n/a")

    labels["_ft_rank"] = labels["Feature type"].map(
        {v: i for i, v in enumerate(_FTYPE_ORDER)}
    ).fillna(len(_FTYPE_ORDER))
    labels["_ch_rank"] = labels["Channel"].map(
        {v: i for i, v in enumerate(_CHANNEL_ORDER)}
    ).fillna(len(_CHANNEL_ORDER))
    sorted_feats = (
        labels.sort_values(["_ft_rank", "_ch_rank", "variance"],
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
    col_colors = harmony_col_colors(
        sorted_feats,
        ftype_colors=ftype_colors,
        channel_map=ch_map,
        region_colors=region_colors,
    )

    _fig_w = min(max(12, N * 0.03), 16)
    _fig_h = max(8, len(df_agg) * 0.6)
    _bar   = 0.10
    g = sns.clustermap(
        data_df,
        row_cluster=False,
        col_cluster=False,
        row_colors=row_colors,
        col_colors=col_colors,
        cmap="coolwarm",
        center=0,
        figsize=(_fig_w, _fig_h),
        colors_ratio=(_bar / _fig_w, _bar / _fig_h),
        cbar_kws={"label": "Z-score", "shrink": 0.4},
        xticklabels=False,
        yticklabels=True,
    )
    g.ax_heatmap.set_yticklabels(
        g.ax_heatmap.get_yticklabels(), rotation=0, fontsize=9,
    )
    _p = g.cax.get_position()
    g.cax.set_position([_p.x0, _p.y0 + _p.height * 0.3, _p.width, _p.height * 0.4])

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
        f"Condition profiles — columns grouped by Feature type → Channel "
        f"(top {N}, Z-scored)\n({len(feature_cols)} features total)"
    )
    g.fig.suptitle(title if title is not None else default_title, fontsize=10, y=1.01)
    _add_annotation_legend(g.fig, harmony_annotation_handles(
        ftype_colors=ftype_colors, channel_map=ch_map, region_colors=region_colors,
    ))
    g.fig.savefig(output_path, format="pdf", bbox_inches="tight")
    plt.close(g.fig)
