# Self-contained fetch / analysis pipeline

This folder is a **fully self-contained copy** of the pipeline, split into
two scripts. Everything the scripts need lives inside this one folder —
`input_data/`, `requirements.txt`, and a local copy of
`run_kodagu_dk_pipeline.py` (kept unmodified; reused as a library, never
rewritten). You can copy this whole folder to another machine and run it
there with no dependency on anything outside it.

1. **`fetch_earth_engine_data.py`** — Earth Engine only. Downloads daily
   CHIRPS rainfall and daily SMAP soil moisture for the uploaded coordinates
   and writes them as CSV files. Never touches the uploaded annual
   AlphaEarth/hydrological CSVs.
2. **`run_analysis_pipeline.py`** — analysis only. Never calls Earth Engine.
   Reads the CSVs written by step 1, plus the uploaded annual CSVs from
   `input_data/`, and runs every remaining stage (rainfall events,
   Dragon-King evidence, antecedent soil moisture, model training,
   evaluation, figures/reports) exactly as implemented in
   `run_kodagu_dk_pipeline.py`.

## Folder contents

```
earth_engine_pipeline/
  fetch_earth_engine_data.py     step 1: Earth Engine -> CSV cache
  run_analysis_pipeline.py       step 2: CSV cache -> full analysis
  run_kodagu_dk_pipeline.py      local copy of the original pipeline (library)
  input_data/                    the 3 canonical uploaded CSVs
  requirements.txt               pinned dependencies
  ARCHITECTURE.md                stage-by-stage design reference
  earth_engine_raw_data/         created on first fetch: cached daily CSVs
  outputs/                       created by analysis runs
```

## The caching rule

- **CSV not present** → `fetch_earth_engine_data.py` downloads it from Earth
  Engine and saves it. Requires `--ee-project-id` (or `EE_PROJECT_ID`).
- **CSV already present** → `fetch_earth_engine_data.py` does nothing (no
  Earth Engine call, no project ID required) unless you pass `--refresh`.
- `run_analysis_pipeline.py` **only ever reads** the CSVs; it raises a clear
  error naming the missing files and telling you to run
  `fetch_earth_engine_data.py` if they are not there yet. Run it as many
  times as you like (any `--stage`, with or without `--quick-test`) — the
  same cached CSVs are reused every time.

## Usage

Run both commands from inside this folder (`cd earth_engine_pipeline`):

```bat
:: one-time (or whenever you want fresh Earth Engine data): fetch and cache
python fetch_earth_engine_data.py --ee-project-id YOUR_PROJECT_ID

:: any number of times after that: analyze the cached data
python run_analysis_pipeline.py
python run_analysis_pipeline.py --stage events
python run_analysis_pipeline.py --quick-test
```

To move this to another machine: copy the entire `earth_engine_pipeline`
folder, `pip install -r requirements.txt`, and run the same two commands —
no other files from the parent project are needed.

Both scripts default `--project-root` to the current working directory, and
share the same default cache location, `<project-root>\earth_engine_raw_data\`,
so you normally don't need to pass `--data-dir` to either one. Analysis
outputs land in `<project-root>\outputs\<run-family>__<config-hash>\`, in the
same stage-folder layout documented in `ARCHITECTURE.md`.

## What's identical to the original, and what's simplified

- Every algorithmic stage (event extraction, GPD/Dragon-King fitting,
  antecedent moisture, model architecture, training, ensembling, evaluation,
  plotting, reporting) is the **same code**, imported from the local
  `run_kodagu_dk_pipeline.py` — nothing was rewritten or reimplemented.
- One deliberate simplification: `run_kodagu_dk_pipeline.py`'s own
  `--stage` resume logic re-reads daily data from that specific run's own
  hashed output folder, so resuming a later stage requires having first run
  `--stage extract` (or `all`) inside that same folder. Here, every
  invocation of `run_analysis_pipeline.py` — regardless of `--stage` — reads
  directly from the shared `earth_engine_raw_data` cache instead, since that
  read is cheap and Earth Engine is never involved either way. This makes
  the CSV cache reusable across every run and every `--stage`, matching the
  "fetch once, analyze many times" behavior this split was built for.
- The copy of `run_kodagu_dk_pipeline.py` in this folder and the original
  one level up are identical at time of copying. If you edit the pipeline
  logic, edit both or keep only one canonical copy — they do not sync
  automatically.
