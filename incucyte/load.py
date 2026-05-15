"""Functions for loading measurement data and plate maps.

Unified metadata schema produced by all loaders
------------------------------------------------
PlateID   : str   vessel code (e.g. "VID9955"); None for legacy multi-channel files
Channel   : str   channel name (e.g. "Phase", "GFP"); "Phase" hardcoded for legacy single-channel
Well      : str   well label (e.g. "C4")
FOV       : int   field-of-view index
Day       : int   from timestamp DDdHHhMMm
Hour      : int
Minute    : int
elapsed_min : int  Day*1440 + Hour*60 + Minute

All loaders preserve the original measurement columns unchanged.
"""
from __future__ import annotations

import re
import warnings

import pandas as pd

# ---------------------------------------------------------------------------
# Filename regexes — four Incucyte export conventions
# ---------------------------------------------------------------------------

# Standard:        VID9955_Phase_C4_1_00d00h10m.tif
_RE_STANDARD = re.compile(
    r"(?P<PlateID>VID\d+)_(?P<Channel>[A-Za-z]+)_(?P<Well>[A-Z]\d{1,2})"
    r"_(?P<FOV>\d+)_(?P<Day>\d+)d(?P<Hour>\d+)h(?P<Minute>\d+)m"
)
# Legacy single-channel:  VID9955_C4_1_00d00h10m.tif  (no Channel field)
_RE_LEGACY_SINGLE = re.compile(
    r"(?P<PlateID>VID\d+)_(?P<Well>[A-Z]\d{1,2})"
    r"_(?P<FOV>\d+)_(?P<Day>\d+)d(?P<Hour>\d+)h(?P<Minute>\d+)m"
)
# Legacy multi-channel:   Phase_C4_1_00d00h10m.tif  (no PlateID)
_RE_LEGACY_MULTI = re.compile(
    r"(?P<Channel>[A-Za-z]+)_(?P<Well>[A-Z]\d{1,2})"
    r"_(?P<FOV>\d+)_(?P<Day>\d+)d(?P<Hour>\d+)h(?P<Minute>\d+)m"
)
# No-timestamp variants — produced by incucyte-fiji-tools stack exports,
# which strip the timestamp from filenames. Tried only after timestamp
# patterns fail, so they never mis-match a standard export.
_RE_STANDARD_NO_TS = re.compile(
    r"(?P<PlateID>VID\d+)_(?P<Channel>[A-Za-z]+)_(?P<Well>[A-Z]\d{1,2})_(?P<FOV>\d+)"
)
_RE_LEGACY_NO_TS = re.compile(
    r"(?P<PlateID>VID\d+)_(?P<Well>[A-Z]\d{1,2})_(?P<FOV>\d+)"
)


def _parse_filename(filename: str) -> dict:
    """Extract metadata from an Incucyte image filename.

    Tries all naming conventions in order. Returns a dict with keys matching
    the unified schema; missing fields are absent (not None). When a timestamp
    is present, Day/Hour/Minute/elapsed_min are included; for no-timestamp
    patterns they are absent — callers must supply timing externally.
    """
    for pattern, defaults in [
        (_RE_STANDARD,       {}),
        (_RE_LEGACY_SINGLE,  {"Channel": "Phase"}),
        (_RE_LEGACY_MULTI,   {"PlateID": None}),
        (_RE_STANDARD_NO_TS, {}),
        (_RE_LEGACY_NO_TS,   {"Channel": "Phase"}),
    ]:
        m = pattern.search(filename)
        if m:
            meta = m.groupdict()
            meta.update({k: v for k, v in defaults.items() if k not in meta})
            meta["FOV"] = int(meta["FOV"])
            if "Day" in meta:
                meta["Day"]    = int(meta["Day"])
                meta["Hour"]   = int(meta["Hour"])
                meta["Minute"] = int(meta["Minute"])
                meta["elapsed_min"] = meta["Day"] * 1440 + meta["Hour"] * 60 + meta["Minute"]
            return meta
    return {}


def load_qupath(
    path: str,
    interval_min: int | None = None,
    start_min: int = 0,
) -> pd.DataFrame:
    """Load a QuPath measurement TSV.

    Parses the Image column to extract unified metadata (PlateID, Channel,
    Well, FOV, Day, Hour, Minute, elapsed_min). All five Incucyte filename
    conventions are supported.

    For filenames that contain a timestamp (standard Incucyte exports),
    elapsed_min is derived directly from Day/Hour/Minute.

    For filenames without a timestamp (incucyte-fiji-tools stack exports),
    elapsed_min is computed as ``start_min + Timepoint * interval_min`` using
    the QuPath Time index column. Both interval_min and start_min must be
    supplied by the caller — QuPath Time index carries no timing metadata.

    Parameters
    ----------
    path:
        Path to the TSV file exported from QuPath.
    interval_min:
        Minutes between consecutive QuPath time indices. Required when any
        filename lacks an embedded timestamp.
    start_min:
        elapsed_min value for time index 0 (default 0). Use this to exclude
        an initial period — e.g. start_min=1440 to make index 0 correspond
        to 24 h elapsed.

    Returns
    -------
    DataFrame with original columns plus parsed metadata columns appended.
    Rows whose Image filename cannot be parsed are kept but metadata columns
    will be NaN — a warning is issued.
    """
    df = pd.read_csv(path, sep="\t")
    df = df.reset_index(drop=True)

    parsed = [_parse_filename(img) for img in df["Image"]]
    meta = pd.DataFrame(parsed, index=df.index)

    unparseable = sum(1 for p in parsed if not p)
    if unparseable:
        warnings.warn(
            f"{unparseable} Image filenames could not be parsed; "
            "metadata columns will be NaN for those rows"
        )

    # Rows that have no embedded timestamp need Timepoint-based elapsed_min
    if "elapsed_min" in meta.columns:
        no_ts = meta["elapsed_min"].isna()
    else:
        no_ts = pd.Series(True, index=df.index)

    if no_ts.any():
        if interval_min is None:
            raise ValueError(
                "interval_min is required: some Image filenames contain no timestamp. "
                "Pass the minutes between consecutive QuPath time indices."
            )
        if "Timepoint" not in df.columns:
            raise ValueError(
                "Timepoint column not found in TSV but filenames have no timestamps"
            )
        if "elapsed_min" not in meta.columns:
            meta["elapsed_min"] = pd.NA
        meta.loc[no_ts, "elapsed_min"] = (
            start_min + df.loc[no_ts, "Timepoint"] * interval_min
        )

    for col in meta.columns:
        df[col] = meta[col]
    return df


def load_cellprofiler(path: str) -> pd.DataFrame:
    """Load a CellProfiler per-object CSV (e.g. CellposeObjects.csv).

    Renames Metadata_* columns to the unified schema and computes elapsed_min.
    All CellProfiler measurement columns are preserved with their original names.

    Parameters
    ----------
    path:
        Path to the CellProfiler CSV (spreadsheet export format).

    Returns
    -------
    DataFrame with unified metadata columns and all CellProfiler measurement
    columns preserved.
    """
    df = pd.read_csv(path)
    rename = {
        "Metadata_PlateID":  "PlateID",
        "Metadata_Channel":  "Channel",
        "Metadata_Well":     "Well",
        "Metadata_FOV":      "FOV",
        "Metadata_Day":      "Day",
        "Metadata_Hour":     "Hour",
        "Metadata_Minute":   "Minute",
    }
    df = df.rename(columns={k: v for k, v in rename.items() if k in df.columns})
    for col in ("Day", "Hour", "Minute", "FOV"):
        if col in df.columns:
            df[col] = df[col].astype(int)
    if all(c in df.columns for c in ("Day", "Hour", "Minute")):
        df["elapsed_min"] = df["Day"] * 1440 + df["Hour"] * 60 + df["Minute"]
    return df


def load_operetta(path: str) -> pd.DataFrame:
    """Load an Operetta/Harmony measurement TXT export.

    Handles both PlateResults.txt (one row per well, aggregated statistics) and
    ObjectResults.txt (one row per cell/object). Both share the same metadata
    preamble and coordinate scheme.

    Parsing steps:
    - Skips the metadata header by scanning for the ``[Data]`` marker; the
      following line is the tab-separated column header.
    - Converts numeric ``Row`` (1–8 → A–H) and ``Column`` (1–12) to a ``Well``
      string (e.g. Row=1, Column=3 → ``'A3'``).
    - Drops plate-management columns that are never features: ``Timepoint``,
      ``Sequence``, ``Time [s]``, ``Compound``, ``Concentration``,
      ``Cell Type``, ``Cell Count``.
    - Retains ``Nuclei - Number of Objects`` for cell-count QC.

    Parameters
    ----------
    path:
        Path to the Harmony TXT export file.

    Returns
    -------
    DataFrame with a ``Well`` column and all measurement columns preserved.
    """
    with open(path, encoding="utf-8") as fh:
        for i, line in enumerate(fh):
            if line.strip() == "[Data]":
                header_row = i + 1  # 0-indexed line number of the column header
                break
        else:
            raise ValueError(f"No [Data] marker found in {path!r}")

    df = pd.read_csv(path, sep="\t", skiprows=header_row, encoding="utf-8")

    df["Well"] = (
        df["Row"].apply(lambda r: chr(ord("A") + int(r) - 1))
        + df["Column"].astype(int).astype(str)
    )
    df = df.drop(columns=["Row", "Column"])

    _DROP = {"Timepoint", "Sequence", "Time [s]", "Compound", "Concentration",
             "Cell Type", "Cell Count"}
    df = df.drop(columns=[c for c in _DROP if c in df.columns])

    return df


def load_plate_map(path: str, plate_ids: list[str] | str | None = None) -> pd.DataFrame:
    """Load a plate_map.csv generated by the Incucyte portal.

    Parameters
    ----------
    path:
        Path to plate_map.csv. Single-plate format: well, sample, condition,
        replicate. Multi-plate format: plate, well, sample, condition, replicate
        (plate is a 1-based integer index).
    plate_ids:
        Maps integer plate index to vessel code (PlateID). Required when the
        CSV contains a plate column. Pass a list where index 0 = plate 1:
        e.g. ["VID9955", "VID9960"]. A single string is accepted for
        single-plate experiments.

    Returns
    -------
    DataFrame with columns: PlateID, Well, sample, condition, replicate.
    PlateID is None for single-plate maps where plate_ids is not provided.
    """
    df = pd.read_csv(path)
    df = df.rename(columns={"well": "Well"})
    if "plate" in df.columns:
        if plate_ids is None:
            raise ValueError(
                "plate_ids is required for multi-plate plate_map.csv "
                "(file contains a 'plate' column)"
            )
        if isinstance(plate_ids, str):
            plate_ids = [plate_ids]
        id_map = {i + 1: vid for i, vid in enumerate(plate_ids)}
        df["PlateID"] = df["plate"].map(id_map)
        df = df.drop(columns=["plate"])
    else:
        if isinstance(plate_ids, str):
            df["PlateID"] = plate_ids
        elif isinstance(plate_ids, list):
            df["PlateID"] = plate_ids[0]
        else:
            df["PlateID"] = None
    return df[["PlateID", "Well", "sample", "condition", "replicate"]]


def merge_plate_map(df: pd.DataFrame, plate_map: pd.DataFrame) -> pd.DataFrame:
    """Merge measurement data with a plate map.

    Joins on (PlateID, Well) when PlateID is present and non-null in both
    DataFrames, or on (Well,) alone for single-plate data. Rows in df whose
    well is absent from the plate map are dropped with a warning (e.g.
    empty/excluded wells).

    Parameters
    ----------
    df:
        Per-object DataFrame from load_qupath() or load_cellprofiler().
    plate_map:
        DataFrame from load_plate_map().

    Returns
    -------
    df with sample, condition, and replicate columns added.
    """
    has_plate = (
        "PlateID" in df.columns
        and "PlateID" in plate_map.columns
        and df["PlateID"].notna().any()
        and plate_map["PlateID"].notna().any()
    )
    keys = ["PlateID", "Well"] if has_plate else ["Well"]
    map_cols = keys + ["sample", "condition", "replicate"]

    n_before = len(df)
    merged = df.merge(plate_map[map_cols], on=keys, how="inner")
    dropped = n_before - len(merged)
    if dropped:
        warnings.warn(
            f"{dropped} rows dropped: well not found in plate map"
        )
    return merged
