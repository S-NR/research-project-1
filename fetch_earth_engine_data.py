#!/usr/bin/env python3
"""
Step 1 of 2 — fetch daily CHIRPS rainfall and daily SMAP soil moisture from
Google Earth Engine and store them as CSV files.

This script never touches the uploaded annual AlphaEarth or hydrological
tables (those are already-collected, authoritative data — see
ARCHITECTURE.md in this folder). It only fetches what genuinely cannot come
from those annual tables: DAILY rainfall and soil moisture, needed for
rainfall-event extraction and antecedent-moisture windows.

It reuses the exact Earth Engine extraction functions from
run_kodagu_dk_pipeline.py, which lives in this same folder (kept unmodified)
— this script only adds the CSV-based "fetch once, reuse forever" behavior
around them. This whole folder is self-contained (script, library copy,
input_data, requirements.txt) and can be copied to and run on any machine.

    - If the combined CSV outputs already exist in --data-dir, this script
      does nothing at all (no Earth Engine call, no credentials required).
    - Otherwise it authenticates/initializes Earth Engine and downloads the
      data, then writes it out as CSV.
    - Pass --refresh to force a re-download even if the CSVs exist.

Run this first. Then run run_analysis_pipeline.py, which reads only the
CSVs this script writes and never calls Earth Engine itself.

Usage:
    python fetch_earth_engine_data.py --ee-project-id YOUR_PROJECT_ID
    python fetch_earth_engine_data.py --ee-project-id YOUR_PROJECT_ID --refresh
    python fetch_earth_engine_data.py            # re-run with no args: if the
                                                  # CSVs are already there, this
                                                  # just confirms that and exits.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from types import SimpleNamespace

import pandas as pd

# run_kodagu_dk_pipeline.py lives in this same folder and is reused as a
# library, unmodified. This whole folder is self-contained: it can be
# copied to another machine and run from anywhere without the original
# project directory.
import run_kodagu_dk_pipeline as base

DEFAULT_DATA_DIRNAME = "earth_engine_raw_data"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument(
        "--ee-project-id",
        default=os.environ.get("EE_PROJECT_ID"),
        help="Google Cloud project registered for Earth Engine. Only needed "
        "when the CSVs do not already exist in --data-dir.",
    )
    p.add_argument(
        "--project-root",
        type=Path,
        default=Path.cwd(),
        help="Used to find the default --data-dir and the coordinates CSV. "
        "Defaults to the current working directory.",
    )
    p.add_argument(
        "--data-dir",
        type=Path,
        default=None,
        help=f"Where fetched CSVs are stored/read. Default: "
        f"<project-root>/{DEFAULT_DATA_DIRNAME}",
    )
    p.add_argument(
        "--coordinates-csv",
        type=Path,
        default=None,
        help="Optional explicit path to sample_coordinates.csv. Otherwise "
        "resolved the same way run_kodagu_dk_pipeline.py resolves it.",
    )
    p.add_argument(
        "--refresh",
        action="store_true",
        help="Re-download from Earth Engine even if the CSVs already exist.",
    )
    return p.parse_args()


def resolve_data_dir(args: argparse.Namespace) -> Path:
    return args.data_dir or (args.project_root / DEFAULT_DATA_DIRNAME)


def combined_csv_paths(data_dir: Path, cfg: "base.PipelineConfig") -> tuple[Path, Path]:
    rain = data_dir / f"rainfall_daily_{cfg.start_year}_{cfg.end_year}.csv"
    smap = data_dir / f"smap_daily_{cfg.start_year}_{cfg.end_year}.csv"
    return rain, smap


def load_and_validate_coordinates(
    explicit_path: Path | None, project_root: Path, cfg: "base.PipelineConfig"
) -> pd.DataFrame:
    path = base.resolve_uploaded_input(
        explicit_path, project_root, base.DEFAULT_COORDINATES_FILENAME
    )
    coordinates = pd.read_csv(path)
    required = ["sample_id", "longitude", "latitude"]
    missing = [c for c in required if c not in coordinates.columns]
    if missing:
        raise ValueError(f"Coordinates CSV is missing required columns: {missing}")
    coordinates = coordinates[required].copy()
    coordinates["sample_id"] = pd.to_numeric(
        coordinates["sample_id"], errors="raise"
    ).astype(int)
    for column in ("longitude", "latitude"):
        coordinates[column] = pd.to_numeric(coordinates[column], errors="raise")
    if coordinates.isna().any().any() or coordinates.duplicated("sample_id").any():
        raise ValueError("Coordinates must be complete and unique by sample_id.")
    if len(coordinates) != cfg.n_sites:
        raise ValueError(
            f"Expected {cfg.n_sites} coordinate rows, found {len(coordinates)}."
        )
    return coordinates.sort_values("sample_id").reset_index(drop=True)


def main() -> None:
    args = parse_args()
    cfg = base.PipelineConfig()
    data_dir = resolve_data_dir(args)
    data_dir.mkdir(parents=True, exist_ok=True)
    logger = base.setup_logging(data_dir / "fetch_earth_engine_data.log")

    rain_csv, smap_csv = combined_csv_paths(data_dir, cfg)

    if rain_csv.exists() and smap_csv.exists() and not args.refresh:
        logger.info("Daily CHIRPS/SMAP CSVs already exist in %s:", data_dir)
        logger.info("  %s", rain_csv)
        logger.info("  %s", smap_csv)
        logger.info(
            "Nothing to fetch. Pass --refresh to re-download from Earth Engine."
        )
        return

    if not args.ee_project_id:
        raise ValueError(
            "No cached CSVs were found in "
            f"{data_dir}, so Earth Engine must be queried, which requires a "
            "Google Cloud project ID. Pass one with --ee-project-id, or set "
            "the EE_PROJECT_ID environment variable, e.g.:\n"
            "    python fetch_earth_engine_data.py --ee-project-id YOUR_GOOGLE_CLOUD_PROJECT_ID"
        )
    coordinates = load_and_validate_coordinates(
        args.coordinates_csv, args.project_root, cfg
    )
    ee = base.init_earth_engine(args.ee_project_id, logger)

    # get_kodagu_boundary_and_uploaded_sites / extract_daily_rainfall /
    # extract_daily_smap only ever use the `.raw` attribute of the `paths`
    # object they receive, so a minimal stand-in is enough here — this
    # script does not need the full hashed run-output directory tree that
    # run_kodagu_dk_pipeline.py's own `Paths` builds for a complete run.
    ee_paths = SimpleNamespace(raw=data_dir)

    geom, sites = base.get_kodagu_boundary_and_uploaded_sites(
        ee, coordinates, cfg, ee_paths, logger
    )
    rain = base.extract_daily_rainfall(ee, geom, sites, cfg, ee_paths, args.refresh, logger)
    smap = base.extract_daily_smap(ee, geom, sites, cfg, ee_paths, args.refresh, logger)

    # Store the was-missing interpolation flags as 0/1 ints rather than
    # Python bools: CSV round-trips booleans as the strings "True"/"False",
    # and a later `.astype(bool)` on those strings would treat "False" as
    # truthy. Ints round-trip through CSV/`.astype(bool)` correctly.
    smap_out = smap.copy()
    for band in cfg.smap_bands:
        flag = f"{band}_was_missing"
        if flag in smap_out.columns:
            smap_out[flag] = smap_out[flag].astype(int)

    base.save_csv(rain, rain_csv)
    base.save_csv(smap_out, smap_csv)
    logger.info("Saved combined daily rainfall CSV: %s (%d rows)", rain_csv, len(rain))
    logger.info("Saved combined daily SMAP CSV: %s (%d rows)", smap_csv, len(smap_out))
    logger.info("Earth Engine fetch complete. Run run_analysis_pipeline.py next.")


if __name__ == "__main__":
    main()
