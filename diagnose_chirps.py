#!/usr/bin/env python3
"""
One-off diagnostic: reproduce the first CHIRPS reduceRegions() call that
extract_daily_rainfall() makes, and print the raw feature properties Earth
Engine actually returns, so we can see why 'rain_mm_day' goes missing.

Usage:
    python diagnose_chirps.py --ee-project-id YOUR_PROJECT_ID
"""
from __future__ import annotations

import argparse
from datetime import date

import run_kodagu_dk_pipeline as base


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--ee-project-id", required=True)
    args = p.parse_args()

    cfg = base.PipelineConfig()
    logger = base.setup_logging("diagnose_chirps.log")
    ee = base.init_earth_engine(args.ee_project_id, logger)

    import pandas as pd
    from pathlib import Path
    coords_path = Path(__file__).resolve().parent / "input_data" / base.DEFAULT_COORDINATES_FILENAME
    coordinates = pd.read_csv(coords_path)

    from types import SimpleNamespace
    ee_paths = SimpleNamespace(raw=Path("."))
    geom, sites = base.get_kodagu_boundary_and_uploaded_sites(ee, coordinates, cfg, ee_paths, logger)

    chirps = ee.ImageCollection(cfg.chirps_asset).filterBounds(geom)
    day_start = date(cfg.start_year, 1, 1)
    ic = chirps.filterDate(ee.Date(str(day_start)), ee.Date(str(day_start)).advance(1, "day"))
    print("Image count for", day_start, ":", ic.size().getInfo())

    image = ee.Image(ic.first()).select(["precipitation"]).rename(["rain_mm_day"])
    print("Band names after select/rename:", image.bandNames().getInfo())

    small_sites = ee.FeatureCollection(sites.toList(5))
    reduced = image.reduceRegions(
        collection=small_sites,
        reducer=ee.Reducer.first(),
        scale=cfg.chirps_scale_m,
        tileScale=8,
    )
    info = reduced.getInfo()
    print("Raw getInfo() of first 5 reduced features:")
    for feat in info["features"]:
        print(" props:", feat["properties"])

    # Now reproduce the exact ee.data.computeFeatures(...) call used in
    # ee_fc_to_dataframe, on the SAME small collection, to see what the
    # PANDAS_DATAFRAME conversion actually yields.
    dated = reduced.map(lambda f: f.set("date", image.date().format("YYYY-MM-dd")))
    selected = dated.select(["sample_id", "date", "rain_mm_day"])
    result = ee.data.computeFeatures(
        {"expression": selected, "fileFormat": "PANDAS_DATAFRAME", "pageSize": 5000}
    )
    print("Type of computeFeatures result:", type(result))
    if hasattr(result, "columns"):
        print("Columns:", list(result.columns))
        print(result)
    else:
        print(result)


if __name__ == "__main__":
    main()
