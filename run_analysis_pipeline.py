#!/usr/bin/env python3
"""
Step 2 of 2 — run the full Kodagu DK-guided regime-shift analysis using
already-fetched daily CHIRPS/SMAP CSVs plus the uploaded annual
AlphaEarth/hydrological/coordinates CSVs.

This script NEVER calls Earth Engine. If the daily CSVs written by
fetch_earth_engine_data.py are missing, it stops immediately with
instructions to run that script first. If they are already present, they
are reused as-is — no re-fetching, no matter how many times you run this
script or which --stage you pass.

Every analysis stage (rainfall events, Dragon-King evidence, antecedent
soil moisture, model training, evaluation, figures/reports) is the exact
same implementation as run_kodagu_dk_pipeline.py, which lives in this same
folder and is imported unchanged as a library — this script only
re-orchestrates stages 2-7 of that pipeline around CSV-cached daily inputs
instead of a live Earth Engine fetch. See ARCHITECTURE.md in this folder
for the full stage-by-stage description. This whole folder is
self-contained (scripts, library copy, input_data, requirements.txt) and
can be copied to and run on any machine.

Usage:
    python run_analysis_pipeline.py
    python run_analysis_pipeline.py --stage events
    python run_analysis_pipeline.py --quick-test
    python run_analysis_pipeline.py --validate-inputs
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

# run_kodagu_dk_pipeline.py lives in this same folder and is reused as a
# library, unmodified. This whole folder is self-contained: it can be
# copied to another machine and run from anywhere without the original
# project directory.
import run_kodagu_dk_pipeline as base

# Sibling module, same folder.
from fetch_earth_engine_data import DEFAULT_DATA_DIRNAME, combined_csv_paths


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument(
        "--project-root",
        type=Path,
        default=Path.cwd(),
        help="Defaults to the current working directory.",
    )
    p.add_argument(
        "--data-dir",
        type=Path,
        default=None,
        help="Where fetch_earth_engine_data.py stored its CSVs. Default: "
        f"<project-root>/{DEFAULT_DATA_DIRNAME}",
    )
    p.add_argument("--alphaearth-csv", type=Path, default=None)
    p.add_argument("--hydrology-csv", type=Path, default=None)
    p.add_argument("--coordinates-csv", type=Path, default=None)
    p.add_argument(
        "--stage",
        choices=("all", "events", "dk", "features", "train", "evaluate", "report"),
        default="all",
        help="Run all remaining stages or stop after one. There is no "
        "'extract' stage here — use fetch_earth_engine_data.py for that.",
    )
    p.add_argument(
        "--quick-test",
        action="store_true",
        help="Use one seed and two epochs in a separately hashed TEST run family.",
    )
    p.add_argument(
        "--validate-inputs",
        action="store_true",
        help="Validate the three uploaded annual CSVs and exit.",
    )
    p.add_argument(
        "--self-test",
        action="store_true",
        help="Run the local, Earth-Engine-free logic self-test and exit.",
    )
    p.add_argument(
        "--no-map", action="store_true", help="Skip the optional Folium HTML map."
    )
    return p.parse_args()


def load_daily_csvs(
    data_dir: Path, cfg: "base.PipelineConfig"
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rain_csv, smap_csv = combined_csv_paths(data_dir, cfg)
    missing = [str(p) for p in (rain_csv, smap_csv) if not p.exists()]
    if missing:
        raise FileNotFoundError(
            "Daily CHIRPS/SMAP CSVs are missing:\n"
            + "\n".join(missing)
            + "\n\nRun fetch_earth_engine_data.py first, e.g.:\n"
            "    python fetch_earth_engine_data.py --ee-project-id YOUR_PROJECT_ID"
        )
    rain = pd.read_csv(rain_csv, parse_dates=["date"])
    smap = pd.read_csv(smap_csv, parse_dates=["date"])
    rain["sample_id"] = rain["sample_id"].astype(int)
    smap["sample_id"] = smap["sample_id"].astype(int)
    return rain, smap


def main() -> None:  # noqa: PLR0915 - mirrors run_kodagu_dk_pipeline.main() stages 2-7
    args = parse_args()
    if args.self_test:
        base.self_test()
        return

    cfg = base.PipelineConfig()
    input_paths = base.resolve_uploaded_inputs(args)
    _, _, _, input_validation = base.validate_uploaded_annual_inputs(input_paths, cfg)
    if args.validate_inputs:
        print(
            json.dumps(
                {
                    **input_validation,
                    "inputs": {
                        key: {
                            "path": str(path),
                            "size_bytes": path.stat().st_size,
                            "sha256": base.sha256_file(path),
                        }
                        for key, path in input_paths.items()
                    },
                },
                indent=2,
            )
        )
        return

    data_dir = args.data_dir or (args.project_root / DEFAULT_DATA_DIRNAME)
    rain, smap = load_daily_csvs(data_dir, cfg)

    run_cfg = base.config_dict(cfg, args.quick_test)
    run_cfg["uploaded_inputs"] = {
        key: {
            "canonical_filename": {
                "alphaearth": base.DEFAULT_ALPHAEARTH_FILENAME,
                "hydrology": base.DEFAULT_HYDROLOGY_FILENAME,
                "coordinates": base.DEFAULT_COORDINATES_FILENAME,
            }[key],
            "size_bytes": path.stat().st_size,
            "sha256": base.sha256_file(path),
        }
        for key, path in input_paths.items()
    }
    cfg_hash = base.config_hash(run_cfg)
    paths = base.Paths.create(args.project_root, run_cfg["run_family"], cfg_hash)
    base.save_run_manifest(paths, run_cfg, cfg_hash)
    logger = base.setup_logging(paths.manifest / "pipeline.log")
    logger.info("Run root: %s", paths.run_root)
    logger.info("Configuration hash: %s", cfg_hash)
    logger.info(
        "Daily CHIRPS/SMAP loaded from cached CSVs in %s (no Earth Engine call made)",
        data_dir,
    )

    ae, annual_original, coordinates = base.stage_uploaded_annual_inputs(
        input_paths, cfg, paths, logger
    )
    base.reconcile_daily_sources_with_uploaded_annual(
        rain, smap, annual_original, cfg, paths
    )

    rank = base.stage_rank(args.stage)

    # ---------------- Rainfall events + grouped split ----------------
    signatures = base.rainfall_signature_ids(rain, cfg)
    split_manifest = base.grouped_split_manifest(signatures, coordinates, cfg)
    base.save_csv(split_manifest, paths.manifest / "split_manifest.csv")
    events, membership = base.extract_rainfall_events(rain, signatures, cfg)
    base.save_event_outputs(events, membership, split_manifest, cfg, paths)
    events = events.merge(
        split_manifest[["sample_id", "split"]], on="sample_id", how="left"
    )
    if rank == 2:
        base.inventory_outputs(paths)
        logger.info("Rainfall-event stage completed.")
        return

    # ---------------- Dragon-King evidence ----------------
    events, fits = base.apply_dragon_king(events, split_manifest, cfg, paths, logger)
    if rank == 3:
        base.plot_dragon_king_outputs(events, fits, cfg, paths)
        base.inventory_outputs(paths)
        logger.info("Dragon-King stage completed.")
        return

    # ---------------- Antecedent soil moisture + model table ----------------
    event_sm = base.add_antecedent_soil_moisture(events, smap, cfg, paths)
    annual_events = base.select_annual_event_features(
        event_sm, annual_original, cfg, paths
    )
    model_table = base.build_model_feature_table(
        annual_original, annual_events, split_manifest, cfg, paths
    )
    data = base.prepare_model_data(model_table, cfg, paths)
    if rank == 4:
        base.inventory_outputs(paths)
        logger.info("Feature-construction stage completed.")
        return

    # ---------------- Separate compatible model runs ----------------
    run_seeds = [int(x) for x in run_cfg["run_seeds"]]
    variant_runs: dict[str, list[pd.DataFrame]] = {v: [] for v in cfg.model_variants}
    all_metrics: list[dict[str, Any]] = []
    synthetic_frames: list[pd.DataFrame] = []
    representative_scores: dict[str, np.ndarray] | None = None
    representative_history: pd.DataFrame | None = None

    for variant in cfg.model_variants:
        for seed in run_seeds:
            cached = base.load_completed_run(variant, seed, data, cfg, paths)
            if cached is None:
                result = base.train_one_run(variant, seed, data, cfg, run_cfg, paths, logger)
            else:
                logger.info("Reusing completed %s seed %d", variant, seed)
                result = cached
            long, metrics, bundle, scores, calibration = result
            variant_runs[variant].append(long)
            all_metrics.append(metrics)
            if variant == "fused_dk":
                synthetic_frames.append(
                    base.synthetic_detection_evaluation(
                        bundle, calibration, data, cfg, seed, paths
                    )
                )
                if seed == run_seeds[0]:
                    representative_scores = {k: np.asarray(v) for k, v in scores.items()}
                    representative_history = pd.read_csv(
                        paths.runs / variant / f"seed_{seed:04d}" / "training_history.csv"
                    )
            _, keras, _ = base.import_tensorflow()
            keras.backend.clear_session()

    registry = pd.DataFrame(all_metrics)
    base.save_csv(registry, paths.manifest / "run_registry.csv")
    synthetic = pd.concat(synthetic_frames, ignore_index=True)
    variant_ensembles = {
        variant: base.aggregate_variant_scores(frames, cfg)
        for variant, frames in variant_runs.items()
    }
    main_ensemble = base.add_dk_context(variant_ensembles["fused_dk"], annual_events)
    full = base.save_ensemble_outputs(main_ensemble, coordinates, cfg, paths)
    base.create_comparison_outputs(variant_ensembles, full, cfg, paths)

    representative_dir = paths.runs / "fused_dk" / f"seed_{run_seeds[0]:04d}"
    shutil.copy2(
        representative_dir / "model.keras",
        paths.ensemble / "dk_guided_temporal_attention_autoencoder.keras",
    )
    shutil.copy2(
        representative_dir / "training_history.csv",
        paths.ensemble / "dk_training_history.csv",
    )
    if rank == 5:
        base.inventory_outputs(paths)
        logger.info("Training and ensemble stage completed.")
        return

    assert representative_scores is not None and representative_history is not None

    # ---------------- Evaluation ----------------
    evaluation = base.build_evaluation_tables(
        full,
        variant_ensembles,
        all_metrics,
        variant_runs["fused_dk"],
        synthetic,
        split_manifest,
        data,
        fits,
        cfg,
        paths,
    )
    base.save_compatibility_tables(
        full, annual_original, evaluation, representative_scores, data, cfg, paths
    )
    if rank == 6:
        base.inventory_outputs(paths)
        logger.info("Evaluation stage completed.")
        return

    # ---------------- Figures, map, and reports ----------------
    base.plot_all_outputs(
        full,
        variant_ensembles,
        evaluation,
        synthetic,
        events,
        fits,
        annual_original,
        coordinates,
        data,
        representative_scores,
        representative_history,
        cfg,
        paths,
    )
    if not args.no_map:
        base.make_interactive_map(full, coordinates, paths)
    base.save_final_reports(full, evaluation, cfg, run_cfg, cfg_hash, paths)
    base.inventory_outputs(paths)
    logger.info("Complete pipeline finished: %s", paths.run_root)


if __name__ == "__main__":
    main()
