# Pipeline architecture

This document describes how `run_kodagu_dk_pipeline.py` (4,156 lines, single
file) is put together: its stages, data flow, caching behavior, and — the
question this doc leads with — exactly which data is reused from prior
collection versus freshly fetched from Google Earth Engine on every run.

## TL;DR: cached vs. fresh data

The pipeline uses **both**, for two different classes of data, and the
distinction is load-bearing for the science, not incidental:

| Data | Source | Fetched fresh or reused? |
|---|---|---|
| Annual AlphaEarth embeddings (A00–A63, 64 bands) | Uploaded CSV `input_data/kodagu_alphaearth_2017_2024.csv` | **Never fetched from Earth Engine.** Read from disk, hash-validated, used as-is. |
| Annual hydrological state (rainfall totals, soil moisture, ET, etc.) | Uploaded CSV `input_data/kodagu_hydrological_features_2017_2024.csv` | **Never fetched from Earth Engine.** Read from disk, hash-validated, used as-is. |
| Sample coordinates (600 sites) | Uploaded CSV `input_data/sample_coordinates.csv` | **Never fetched from Earth Engine.** Used verbatim to build EE point geometries — not regenerated. |
| Daily CHIRPS rainfall (2017–2024) | `UCSB-CHG/CHIRPS/DAILY` via Earth Engine | **Fetched from Earth Engine**, then cached to Parquet under `01_raw_data/`. Subsequent runs against the same output directory reuse the Parquet cache unless `--refresh-data` is passed. |
| Daily SMAP soil moisture (surface + root zone) | `NASA/SMAP/SPL4SMGP/008` via Earth Engine | Same as CHIRPS: fetched once, cached to Parquet, reused unless `--refresh-data`. |
| Kodagu district boundary | `FAO/GAUL/2015/level2` via Earth Engine | Fetched every time `--stage extract`/`all` runs (cheap single vector query); saved to `01_raw_data/kodagu_boundary.geojson` for reference/plotting. |

So: **the two large annual tables that already exist (AlphaEarth embeddings
and hydrological features) are never re-pulled from Earth Engine — they are
treated as fixed, authoritative, uploaded ground truth.** Earth Engine is
used only to fill the gap those annual tables cannot cover: **daily-resolution
rainfall and soil moisture**, which are required for event extraction and
antecedent-moisture windows but are not derivable from annual totals. Those
daily pulls are themselves cached to disk after the first fetch and reused on
later runs, so Earth Engine is only hit again if the cache is missing or
`--refresh-data` is explicitly passed.

Key evidence in code:

- `run_kodagu_dk_pipeline.py:788-835` (`extract_daily_rainfall`) — checks
  `combined_path.exists() and not refresh` and returns the cached Parquet
  immediately if so; only calls into `ee.ImageCollection(...)` when the cache
  is absent or `--refresh-data` was passed.
- `run_kodagu_dk_pipeline.py:856-915` (`extract_daily_smap`) — identical
  cache-first pattern for SMAP.
- `run_kodagu_dk_pipeline.py:613-653` (`stage_uploaded_annual_inputs`) and
  `522-612` (`validate_uploaded_annual_inputs`) — load, hash, and validate the
  three CSVs; there is no Earth Engine call anywhere in this path.
  AlphaEarth/hydrology data reach the model exclusively through these
  functions.
- `run_kodagu_dk_pipeline.py:918-987`
  (`reconcile_daily_sources_with_uploaded_annual`) — aggregates the freshly
  fetched daily CHIRPS/SMAP up to annual totals and diffs them against the
  uploaded annual values purely as an audit diagnostic
  (`annual_daily_input_reconciliation.csv`). The comparison never feeds back
  into the model; the uploaded annual values remain authoritative regardless
  of the diagnostic's outcome.
- Docstring at `run_kodagu_dk_pipeline.py:16-19` states the intent explicitly:
  "The uploaded annual tables are authoritative... Daily CHIRPS and SMAP are
  downloaded because the uploaded annual tables cannot support rainfall-event
  extraction or antecedent soil-moisture windows."

Beyond raw data, the pipeline caches its own intermediate/derived products
too, keyed by a content hash (see below), so re-running `--stage all` against
the same output directory does not restart from scratch:

- Trained models per `(variant, seed)` are skipped and reloaded from disk if
  `COMPLETED.json`, `model.keras`, `transition_scores.csv`,
  `run_metrics.json`, and `score_calibration.json` already exist
  (`load_completed_run`, line 2292).
- `--stage <name>` can stop after a stage or resume from the previous stage's
  cached outputs (`load_cached_raw`, line 3858), letting you iterate on later
  stages without re-fetching or re-extracting.

## Why this design

1. **Annual embeddings can't be recomputed identically.** AlphaEarth
   embeddings and the annual hydrological feature table were generated once,
   externally, and are treated as the frozen ground truth this whole study is
   built on. Re-deriving them from Earth Engine on every run would risk
   version drift (different AlphaEarth model version, different EE dataset
   revision) silently changing the study's inputs between runs.
2. **Daily granularity genuinely doesn't exist in the annual files.** You
   cannot recover per-day rainfall events or 7-day antecedent soil-moisture
   windows from a single annual total — that data has to come from a daily
   source, which only Earth Engine (CHIRPS/SMAP) provides here.
3. **Caching avoids repeated Earth Engine cost/quota use and non-determinism.**
   Once fetched, daily CHIRPS/SMAP are pinned to Parquet files inside the
   run's own output directory, so re-running analysis stages (events, DK,
   features, train, evaluate, report) never re-hits Earth Engine unless you
   explicitly ask for a refresh.

## Run identity and output layout

Every run writes to:

```
<project-root>/outputs/<run-family>__<config-hash>/
```

`run-family` is a fixed string (`daily_event_dk_uploaded_annual_audit_v1`, or
a `_quicktest` variant). `config-hash` is a SHA-256-derived hash over the full
`PipelineConfig`, plus the **content hashes of the three uploaded CSVs**
(`config_hash`, line 383; `config_dict`, line 353). This means:

- Any change to the code's declared assumptions, or to any of the three
  uploaded CSVs, produces a *new* output directory — nothing is silently
  overwritten or pooled across incompatible inputs.
- The Parquet caches for daily CHIRPS/SMAP live inside this hashed directory
  (`01_raw_data/`), so a fresh config hash naturally forces a fresh Earth
  Engine fetch, while re-running the identical config reuses everything.

Numbered stage folders (`00_manifest` … `10_reports`) hold each stage's
outputs; `OUTPUT_CROSSWALK.md` documents them file-by-file.

## Pipeline stages (`--stage extract|events|dk|features|train|evaluate|report|all`)

The stages run in a strict, gated order inside `main()` (line 3924), each one
optionally short-circuiting via `--stage`:

```
extract  → events → dk → features → train → evaluate → report
```

1. **extract** (`main()` lines 3976–3996)
   - Validate + stage the 3 uploaded CSVs (`stage_uploaded_annual_inputs`).
   - `init_earth_engine` (line 659): authenticate/initialize EE with the
     given Cloud project.
   - `get_kodagu_boundary_and_uploaded_sites` (line 703): fetch the Kodagu
     boundary from `FAO/GAUL/2015/level2` and build EE point features from
     the uploaded coordinates verbatim (never regenerated).
   - `extract_daily_rainfall` / `extract_daily_smap`: cache-checked fetch of
     daily CHIRPS/SMAP, month-by-month, into `01_raw_data/`.
   - `reconcile_daily_sources_with_uploaded_annual`: audit-only diagnostic
     comparing daily-aggregated-to-annual vs. uploaded annual values.

2. **events** (lines 4002–4014)
   - `rainfall_signature_ids`: hashes each site's full daily rainfall
     trajectory to detect exact raster replicas (spatial pseudoreplication
     guard).
   - `grouped_split_manifest`: builds leakage-safe 80/10/10 train/val/test
     splits grouped by rainfall signature (not by raw sample id), so replica
     sites can't straddle splits.
   - `extract_rainfall_events`: wet/dry-day event segmentation (≥1.0 mm wet
     day, ≥1 dry day separation) producing event total, duration, peak
     intensity per event.

3. **dk** (lines 4016–4022)
   - `apply_dragon_king`: per-metric (total/duration/peak) Generalized Pareto
     tail fit above a training-only 90th-percentile threshold, with
     parametric-bootstrap goodness-of-fit gating. A failed GOF gate zeroes
     that metric's DK evidence — no visually-extreme-but-unfit event is
     treated as a Dragon King.

4. **features** (lines 4024–4036)
   - `add_antecedent_soil_moisture`: 7-day pre-event SMAP average per event
     (≥4 valid days required).
   - `select_annual_event_features`: picks one representative event per
     site-year (highest validated DK evidence, tie-broken by
     total/peak/duration).
   - `build_model_feature_table` + `prepare_model_data`: merges uploaded
     annual AlphaEarth/hydrology with the derived event/antecedent features
     into per-branch tensors, with scaling fit strictly on the training
     split.

5. **train** (lines 4038–4101)
   - Trains 4 model variants (`hydrology_only`, `alphaearth_only`,
     `fused_no_dk`, `fused_dk`) × 30 seeds (42–71) — a DK-guided temporal
     attention autoencoder architecture (`build_model`, line 1898).
   - `load_completed_run` (line 2292) reuses a finished run from disk instead
     of retraining, if all of its completion artifacts are already present.
   - `aggregate_variant_scores` + `save_ensemble_outputs`: builds
     per-variant ensembles, then stable/persistence-confirmed regime-shift
     candidates from the `fused_dk` variant (≥80% seed selection frequency
     required for a "stable" candidate; persistence requires next-year score
     ≥ 0.5).

6. **evaluate** (lines 4105–4130)
   - `build_evaluation_tables`: reconstruction error, synthetic
     injection/detection power, bootstrap CIs, cross-evidence, DK validation,
     grouped Moran's I, ablation, run stability, leakage audit, threshold
     sensitivity — all computed without ever calling the synthetic injection
     labels "ground truth."

7. **report** (lines 4132–4152)
   - `plot_all_outputs`: regenerates the full figure set (with pre-existing,
     backward-compatible filenames — see `OUTPUT_CROSSWALK.md`).
   - `make_interactive_map` (optional, `--no-map` to skip): Folium HTML map.
   - `save_final_reports`: analysis report, run manifest, output inventory
     with checksums.

## Module map (by section header in the source)

| Lines | Section | Responsibility |
|---|---|---|
| 66–478 | Config & setup | `PipelineConfig` (all assumptions as dataclass fields), CLI args, path/hash/logging utilities |
| 479–653 | Uploaded-input validation | Hash/shape/key checks on the 3 CSVs; verifies the hydrology file's duplicated A00–A63 columns match the AlphaEarth file exactly |
| 654–989 | Earth Engine extraction | EE init, boundary + site verification, daily CHIRPS/SMAP fetch with local caching, daily-vs-annual reconciliation |
| 990–1184 | Events, groups, splits | Rainfall-signature hashing, grouped leakage-safe splitting, event segmentation |
| 1185–1508 | Dragon-King | GPD tail fits, bootstrap GOF gate, DK evidence scoring, threshold-stability diagnostics |
| 1509–1656 | Antecedent features | 7-day SMAP antecedent window, annual representative-event selection, model feature table |
| 1657–1789 | Model data prep | Robust per-branch scaling, `ModelData` tensor bundle |
| 1790–2336 | Model architecture & training | Keras model (`build_model`), per-run training loop, calibration, completed-run caching/reload |
| 2337–2476 | Synthetic evaluation | Persistent synthetic hydrology-shift injection and detection-power scoring |
| 2477–2649 | Ensemble | Cross-seed aggregation, stability + persistence-confirmed shift logic |
| 2650–3101 | Evaluation/audit | Duplication audit, cluster bootstrap, Spearman cross-evidence, permutation test, Moran's I, ablation, sensitivity |
| 3102–3573 | Comparison & plotting | Cross-variant comparison tables, all diagnostic/result figures |
| 3574–3781 | Map & reports | Interactive Folium map, final analysis report |
| 3782–3923 | Manifest & self-test | Run manifest, output inventory/checksums, dependency-free `--self-test` |
| 3924–4156 | `main()` | Stage orchestration described above |

## Design principles reflected in the code

- **No silent data substitution.** `resolve_uploaded_input` (line 306)
  refuses to guess between a canonical file and a numbered duplicate; it
  raises if the canonical filename isn't found in any checked location.
- **No silent gap-filling.** `extract_daily_rainfall` raises if the CHIRPS
  table doesn't have exactly the expected row count or contains any NaNs —
  it will not proceed with incomplete daily data.
- **Explicit, auditable assumptions.** Every threshold (wet-day mm, event
  separation days, POT percentile, persistence threshold, etc.) is a named
  `PipelineConfig` field with a comment tying it to an "Assumption A#",
  cross-referenced in `ASSUMPTIONS_AND_EDIT_POINTS.md`.
- **Reproducibility over convenience.** The config-hash + input-hash run
  identity (rather than mutable in-place outputs) means two runs can never
  be silently conflated, at the cost of needing `--refresh-data` explicitly
  when you *do* want new daily EE data.
