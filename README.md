# incucyte-data-analysis

Python package and Jupyter notebook templates for analysing Incucyte live-cell imaging data exported from QuPath or CellProfiler.

## Environment setup

A conda environment file is provided. Run the following from the repository root, with conda available:

```bash
# 1. Create the environment
conda env create -f environment.yml

# 2. Activate it
conda activate hcs-analysis

# 3. Install the package in editable mode
pip install -e .

# 4. Register the Jupyter kernel
python -m ipykernel install --user --name hcs-analysis --display-name "Python (hcs-analysis)"
```

After step 4 the environment appears as **Python (hcs-analysis)** in Jupyter's kernel selector.

> `PyComplexHeatmap` (used by `metric_heatmap`) is included via pip. If installation fails on your platform, remove it from `environment.yml` — the function will fall back to a seaborn clustermap automatically.

## Package — `incucyte`

### Loaders

| Function | Description |
|----------|-------------|
| `load_qupath(path, interval_min, start_min)` | Load a QuPath measurement TSV. Parses all Incucyte filename conventions (standard VID, legacy single/multi-channel). For files without embedded timestamps, supply `interval_min` (minutes between timepoints) and optionally `start_min`. |
| `load_cellprofiler(path)` | Load a CellProfiler per-object CSV. Renames `Metadata_*` columns to unified schema. |
| `load_plate_map(path, plate_ids)` | Load a `plate_map.csv` (columns: `well, sample, condition, replicate`). For multi-plate maps add a `plate` column and pass `plate_ids` as a list of VID strings. |
| `merge_plate_map(df, plate_map)` | Inner-join measurements with plate map on `Well` (or `PlateID + Well` for multi-plate). Rows whose well is absent from the map are dropped with a warning. |

### Filters

| Function | Description |
|----------|-------------|
| `filter_by(df, **kwargs)` | Keep rows where column equals value (e.g. `Classification="NSC"`). |
| `filter_not_null(df, col)` | Drop rows with NaN in `col`. |
| `filter_timepoints(df, min_elapsed, max_elapsed)` | Restrict to a time window (minutes). |

### Classification

| Function | Description |
|----------|-------------|
| `classify_threshold(df, col, threshold, above, below)` | Assign a string class label based on a numeric threshold. |

### Aggregation

| Function | Description |
|----------|-------------|
| `to_well_means(df, metrics)` | Mean per well × timepoint. Groups by `PlateID, Well, sample, condition, replicate, elapsed_min`. |
| `to_replicate_means(df_well, metrics)` | Mean per biological replicate × timepoint (collapses FOVs). |
| `class_proportions(df, class_col, metrics)` | Fraction of each class per well × timepoint. |

### Plots

| Function | Description |
|----------|-------------|
| `time_series(df, y, time_col, hue, ax)` | Line plot with 95 % bootstrapped CI band across replicates. |
| `boxplot(df, y, x, ax)` | Strip + box plot for endpoint comparisons. |
| `paired_boxplot(df, y, x, pair_col, ax)` | Paired strip + box plot (lines connecting matched replicates). |
| `qc_counts(df, groupby, ax)` | Bar chart of object counts per well or condition. |
| `metric_heatmap(df, metrics, ...)` | Z-score heatmap with condition annotation sidebar. Uses PyComplexHeatmap if available, seaborn clustermap otherwise. |

## Plate map format

Single-plate (`plate_map.csv`):

```
well,sample,condition,replicate
A1,S1,Control,1
B1,S2,Control,1
C1,S3,Treatment,1
```

Multi-plate: add a `plate` column (1-based integer) and pass `plate_ids=["VID9955", "VID9960"]` to `load_plate_map`.

## Notebook templates

Located in `notebooks/templates/`. Copy and edit the *Configuration* cell at the top.

| Template | Use case |
|----------|----------|
| `timeseries_qupath.ipynb` | Single-plate QuPath time-series |
| `timeseries_qupath_multiplate.ipynb` | Multi-plate QuPath time-series (single concatenated TSV) |
| `timeseries_qupath_classified.ipynb` | Time-series with threshold-based morphology classification (e.g. Round vs elongated by circularity) |
| `endpoint_qupath.ipynb` | Single-timepoint endpoint analysis |
| `cellprofiler_morphology.ipynb` | CellProfiler shape + skeleton metrics, Z-score heatmap |
