#!/usr/bin/env python3
"""
Audit-corrected Kodagu hydrological regime-shift pipeline.

Workflow implemented exactly in this order:
    Uploaded annual AlphaEarth/hydrological tables + coordinates
    and daily CHIRPS rainfall
    -> rainfall-event extraction
    -> event total, duration, and peak intensity
    -> event-level Dragon-King tests and DK evidence
    -> antecedent SMAP soil moisture + hydrological encoder
    -> DK-guided temporal attention
    -> annual AlphaEarth embedding branch
    -> regime-shift candidates and persistence-confirmed regime shifts

The uploaded annual tables are authoritative for the annual AlphaEarth and
hydrological state variables. Daily CHIRPS and SMAP are downloaded because the
uploaded annual tables cannot support rainfall-event extraction or antecedent
soil-moisture windows. The script deliberately keeps this daily-event method in
a new hashed run family.
It never reads, overwrites, or pools results from the earlier annual-rainfall,
notebook, pipeline, or ablation runs.

Run in the VS Code terminal on Windows:
    python run_kodagu_dk_pipeline.py --ee-project-id YOUR_PROJECT_ID

Run a local dependency-free logic test first:
    python run_kodagu_dk_pipeline.py --self-test

Scientific status
-----------------
The output is an exploratory remote-sensing/model result, not landslide ground
truth and not causal attribution. "Dragon-King" is accepted as model guidance
only when the fitted tail passes the implemented goodness-of-fit gate. Detected
changes remain candidates until they are stable across seeds and pass the
persistence rule.
"""

from __future__ import annotations

import argparse
import calendar
import hashlib
import json
import logging
import math
import os
import platform
import random
import shutil
import sys
import time
import warnings
from dataclasses import asdict, dataclass, field
from datetime import date, timedelta
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import pandas as pd


PIPELINE_VERSION = "1.1.0"


# =============================================================================
# USER ACTIONS AND EXPLICIT ASSUMPTIONS
# =============================================================================

# USER ACTION 1: Replace this placeholder with the Google Cloud project that is
# registered for Earth Engine, OR pass it at runtime with --ee-project-id.
# Example only: DEFAULT_EE_PROJECT_ID = "my-earth-engine-project"
DEFAULT_EE_PROJECT_ID = "REPLACE_WITH_YOUR_GOOGLE_CLOUD_PROJECT_ID"

# USER ACTION 2: Change this only if your Windows project folder is different.
# The raw string prefix r"..." is important for a Windows path.
DEFAULT_PROJECT_ROOT = Path(r"A:\Research\Hydrology\Correct_work")

# USER ACTION 3 (normally no edit is needed): put these three supplied files in
# DEFAULT_PROJECT_ROOT, pass their paths with the command-line options below,
# or keep the packaged copies in this project's input_data folder. The resolver
# checks those locations in that order and never reconstructs daily events from
# annual rainfall totals.
DEFAULT_ALPHAEARTH_FILENAME = "kodagu_alphaearth_2017_2024.csv"
DEFAULT_HYDROLOGY_FILENAME = "kodagu_hydrological_features_2017_2024.csv"
DEFAULT_COORDINATES_FILENAME = "sample_coordinates.csv"


@dataclass(frozen=True)
class PipelineConfig:
    """All methodological assumptions are explicit and saved with every run."""

    # Saved study definition; kept compatible with the previous Kodagu work.
    study_region: str = "Kodagu District, Karnataka, India"
    start_year: int = 2017
    end_year: int = 2024
    n_sites: int = 600

    # Earth Engine assets used in the saved work.
    gaul_asset: str = "FAO/GAUL/2015/level2"
    chirps_asset: str = "UCSB-CHG/CHIRPS/DAILY"
    smap_asset: str = "NASA/SMAP/SPL4SMGP/008"
    alphaearth_bands: tuple[str, ...] = tuple(f"A{i:02d}" for i in range(64))
    # Native/nominal point-sampling scales. Change only if you intentionally
    # want spatial aggregation rather than values at the saved sample points.
    chirps_scale_m: int = 5566
    smap_scale_m: int = 11000

    # ASSUMPTION A1: a wet day has at least 1.0 mm; one or more intervening dry
    # days separates events; single-day events are retained.
    wet_day_threshold_mm: float = 1.0
    min_interevent_dry_days: int = 1
    min_event_duration_days: int = 1

    # ASSUMPTION A2: antecedent wetness is the mean of the seven complete days
    # immediately BEFORE the event begins, excluding all event days. At least
    # four valid days are required. Both saved SMAP bands are retained.
    antecedent_window_days: int = 7
    antecedent_min_valid_days: int = 4
    smap_max_interpolation_gap_days: int = 2
    smap_bands: tuple[str, ...] = ("sm_surface", "sm_rootzone")

    # ASSUMPTION A3: all three event descriptors are tested separately. The
    # continuous event score is Bonferroni-adjusted across the reliable metric
    # tails. A formal DK candidate also uses Bonferroni correction across the
    # three metrics. No DK result is called supported if its GPD fit fails.
    dk_metrics: tuple[str, ...] = (
        "event_total_mm",
        "duration_days",
        "peak_intensity_mm_day",
    )
    dk_threshold_percentile: float = 90.0
    dk_top_k: int = 3
    dk_monte_carlo: int = 5000
    dk_gof_bootstrap: int = 500
    dk_alpha: float = 0.01
    dk_min_regular_exceedances: int = 20
    dk_gof_alpha: float = 0.05
    dk_stability_percentiles: tuple[int, ...] = tuple(range(80, 96))
    dk_attention_weight: float = 2.0

    # ASSUMPTION A4: an event belongs to the year containing its peak-intensity
    # day. The event with the largest validated DK evidence represents that
    # site-year; ties are resolved by total, peak, duration, and start date.
    event_year_rule: str = "peak_date_year"
    annual_event_selection_rule: str = "max_validated_dk_evidence_then_total_peak_duration"

    # Leakage control. A signature is formed from the complete rounded daily
    # CHIRPS trajectory. All sites with an identical trajectory remain in one
    # split, directly preventing raster-replica train/validation duplication.
    rain_signature_round_decimals: int = 3
    train_fraction: float = 0.80
    validation_fraction: float = 0.10
    test_fraction: float = 0.10
    split_seed: int = 42

    # Saved architecture constants.
    d_model: int = 64
    n_heads: int = 4
    ff_dim: int = 128
    dropout: float = 0.10
    gaussian_noise_std: float = 0.05
    learning_rate: float = 1e-3
    batch_size: int = 32
    max_epochs: int = 250
    early_stopping_patience: int = 25
    lr_patience: int = 10
    min_learning_rate: float = 1e-6

    # ASSUMPTION A5: AlphaEarth and hydrology contribute equally to the latent
    # change component and to reconstruction loss. This explicitly prevents 64
    # embedding bands from numerically overwhelming the hydrological variables.
    alphaearth_branch_weight: float = 0.50
    hydrology_branch_weight: float = 0.50
    latent_score_weight: float = 0.70
    attention_score_weight: float = 0.30

    # Detection is calibrated from TRAINING groups only. Stability is assessed
    # across 30 independently initialized compatible runs.
    adaptive_threshold_percentile: float = 95.0
    run_seeds: tuple[int, ...] = tuple(range(42, 72))
    ensemble_selection_frequency: float = 0.80

    # ASSUMPTION A6: a candidate is a confirmed regime shift only when the next
    # annual state persists with score >= 0.5. The final year is not assessable
    # because no subsequent annual state exists and is never auto-confirmed.
    persistence_threshold: float = 0.50

    # ASSUMPTION A7: realistic synthetic tests use a persistent step from its
    # onset through the remaining sequence, with at least two affected years.
    synthetic_fraction: float = 0.15
    synthetic_magnitudes: tuple[float, ...] = (1.0, 2.0, 3.0, 5.0, 8.0, 12.0)
    synthetic_min_affected_years: int = 2
    synthetic_target_branch: str = "hydrology"

    # Spatial tests operate on independent rainfall-signature groups, not all
    # 600 nominal sites. This prevents duplicate pixels inflating significance.
    moran_k_neighbors: int = 8
    moran_permutations: int = 999
    bootstrap_replicates: int = 2000

    # Separate, explicit run family. The configuration hash is appended later.
    run_family: str = "daily_event_dk_uploaded_annual_audit_v1"
    model_variants: tuple[str, ...] = (
        "hydrology_only",
        "alphaearth_only",
        "fused_no_dk",
        "fused_dk",
    )

    # Original five annual variables are preserved exactly as a distinct state
    # subset. Event descriptors are additional hydrological-encoder inputs.
    original_hydrology_features: tuple[str, ...] = (
        "rain_annual_mm",
        "rain_monsoon_mm",
        "sm_surface",
        "sm_rootzone",
        "et_annual_mm",
    )
    event_hydrology_features: tuple[str, ...] = (
        "event_total_mm",
        "event_duration_days",
        "event_peak_intensity_mm_day",
        "antecedent_sm_surface_7d",
        "antecedent_sm_rootzone_7d",
    )


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument(
        "--ee-project-id",
        default=os.environ.get("EE_PROJECT_ID", DEFAULT_EE_PROJECT_ID),
        help="USER ACTION: Google Cloud project registered for Earth Engine.",
    )
    p.add_argument(
        "--project-root",
        type=Path,
        default=DEFAULT_PROJECT_ROOT,
        help=r"Windows project root; default A:\Research\Hydrology\Correct_work",
    )
    p.add_argument(
        "--alphaearth-csv",
        type=Path,
        default=None,
        help=(
            "Optional explicit path to kodagu_alphaearth_2017_2024.csv. "
            "Otherwise search --project-root and packaged input_data."
        ),
    )
    p.add_argument(
        "--hydrology-csv",
        type=Path,
        default=None,
        help=(
            "Optional explicit path to kodagu_hydrological_features_2017_2024.csv. "
            "Otherwise search --project-root and packaged input_data."
        ),
    )
    p.add_argument(
        "--coordinates-csv",
        type=Path,
        default=None,
        help=(
            "Optional explicit path to sample_coordinates.csv. Otherwise search "
            "--project-root and packaged input_data."
        ),
    )
    p.add_argument(
        "--stage",
        choices=("all", "extract", "events", "dk", "features", "train", "evaluate", "report"),
        default="all",
        help="Run all stages or resume from cached preceding outputs.",
    )
    p.add_argument(
        "--quick-test",
        action="store_true",
        help="Use one seed and two epochs in a separately hashed TEST run family.",
    )
    p.add_argument(
        "--self-test",
        action="store_true",
        help="Run local event/DK/split tests without Earth Engine or TensorFlow.",
    )
    p.add_argument(
        "--validate-inputs",
        action="store_true",
        help="Validate the three uploaded CSV inputs and exit without Earth Engine.",
    )
    p.add_argument(
        "--refresh-data",
        action="store_true",
        help=(
            "Re-download daily CHIRPS/SMAP instead of reusing cache. Uploaded "
            "annual CSVs are still used verbatim."
        ),
    )
    p.add_argument(
        "--no-map", action="store_true", help="Skip the optional Folium HTML map."
    )
    return p.parse_args()


def resolve_uploaded_input(
    explicit_path: Path | None,
    project_root: Path,
    canonical_filename: str,
) -> Path:
    """Resolve one supplied input without silently selecting a numbered duplicate."""
    candidates: list[Path] = []
    if explicit_path is not None:
        candidates.append(explicit_path.expanduser())
    candidates.extend(
        [
            project_root / canonical_filename,
            Path(__file__).resolve().parent / "input_data" / canonical_filename,
            Path(__file__).resolve().parent / canonical_filename,
        ]
    )
    checked: list[str] = []
    for candidate in candidates:
        candidate = candidate.resolve()
        checked.append(str(candidate))
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(
        f"Required input {canonical_filename!r} was not found. Checked:\n"
        + "\n".join(checked)
        + "\nCopy the canonical file into the project root or pass its explicit path."
    )


def resolve_uploaded_inputs(args: argparse.Namespace) -> dict[str, Path]:
    return {
        "alphaearth": resolve_uploaded_input(
            args.alphaearth_csv, args.project_root, DEFAULT_ALPHAEARTH_FILENAME
        ),
        "hydrology": resolve_uploaded_input(
            args.hydrology_csv, args.project_root, DEFAULT_HYDROLOGY_FILENAME
        ),
        "coordinates": resolve_uploaded_input(
            args.coordinates_csv, args.project_root, DEFAULT_COORDINATES_FILENAME
        ),
    }


def canonical_json(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), default=str)


def config_dict(cfg: PipelineConfig, quick_test: bool) -> dict[str, Any]:
    from importlib import metadata

    d = asdict(cfg)
    d["pipeline_version"] = PIPELINE_VERSION
    d["pipeline_source_sha256"] = sha256_file(Path(__file__).resolve())
    packages = (
        "earthengine-api",
        "numpy",
        "pandas",
        "pyarrow",
        "scipy",
        "scikit-learn",
        "tensorflow",
        "matplotlib",
        "folium",
    )
    d["software_versions"] = {}
    for package in packages:
        try:
            d["software_versions"][package] = metadata.version(package)
        except metadata.PackageNotFoundError:
            d["software_versions"][package] = "not-installed"
    if quick_test:
        d["run_family"] = cfg.run_family + "_QUICK_TEST_NOT_FOR_SCIENCE"
        d["run_seeds"] = [cfg.run_seeds[0]]
        d["max_epochs"] = 2
    return d


def config_hash(cfg_dict: dict[str, Any]) -> str:
    return hashlib.sha256(canonical_json(cfg_dict).encode("utf-8")).hexdigest()[:12]


@dataclass
class Paths:
    project_root: Path
    run_root: Path
    manifest: Path
    raw: Path
    events: Path
    dk: Path
    antecedent: Path
    model_input: Path
    runs: Path
    ensemble: Path
    evaluation: Path
    figures: Path
    reports: Path

    @classmethod
    def create(cls, project_root: Path, family: str, cfg_hash: str) -> "Paths":
        run_root = project_root / "outputs" / f"{family}__{cfg_hash}"
        p = cls(
            project_root=project_root,
            run_root=run_root,
            manifest=run_root / "00_manifest",
            raw=run_root / "01_raw_data",
            events=run_root / "02_rainfall_events",
            dk=run_root / "03_dragon_king",
            antecedent=run_root / "04_antecedent_soil_moisture",
            model_input=run_root / "05_model_input",
            runs=run_root / "06_runs",
            ensemble=run_root / "07_ensemble",
            evaluation=run_root / "08_evaluation",
            figures=run_root / "09_figures",
            reports=run_root / "10_reports",
        )
        for folder in asdict(p).values():
            Path(folder).mkdir(parents=True, exist_ok=True)
        return p


def setup_logging(path: Path) -> logging.Logger:
    logger = logging.getLogger("kodagu_pipeline")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    fh = logging.FileHandler(path, encoding="utf-8")
    fh.setFormatter(formatter)
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(formatter)
    logger.addHandler(fh)
    logger.addHandler(sh)
    return logger


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, default=str), encoding="utf-8")


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            h.update(chunk)
    return h.hexdigest()


def save_csv(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def save_parquet(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False, compression="zstd")


def set_global_seed(seed: int) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    os.environ.setdefault("TF_DETERMINISTIC_OPS", "1")
    random.seed(seed)
    np.random.seed(seed)


def require_project_id(project_id: str) -> None:
    if not project_id or project_id == DEFAULT_EE_PROJECT_ID:
        raise ValueError(
            "Earth Engine project ID is missing. Edit DEFAULT_EE_PROJECT_ID near "
            "the top of this script or run: python run_kodagu_dk_pipeline.py "
            "--ee-project-id YOUR_GOOGLE_CLOUD_PROJECT_ID"
        )


# =============================================================================
# UPLOADED ANNUAL INPUTS
# =============================================================================


def _coerce_unique_site_year(
    frame: pd.DataFrame,
    required_columns: Sequence[str],
    cfg: PipelineConfig,
    label: str,
) -> pd.DataFrame:
    missing = [column for column in required_columns if column not in frame.columns]
    if missing:
        raise ValueError(f"{label} is missing required columns: {missing}")
    out = frame.copy()
    out["sample_id"] = pd.to_numeric(out["sample_id"], errors="raise").astype(int)
    out["year"] = pd.to_numeric(out["year"], errors="raise").astype(int)
    for column in required_columns:
        if column not in {"sample_id", "year"}:
            out[column] = pd.to_numeric(out[column], errors="raise")
    if out[list(required_columns)].isna().any().any():
        raise ValueError(f"{label} contains missing required values; no silent fill is allowed.")
    if out.duplicated(["sample_id", "year"]).any():
        count = int(out.duplicated(["sample_id", "year"], keep=False).sum())
        raise ValueError(f"{label} contains {count} rows with duplicate site-year keys.")
    expected_years = set(range(cfg.start_year, cfg.end_year + 1))
    actual_years = set(out["year"].unique().tolist())
    if actual_years != expected_years:
        raise ValueError(
            f"{label} years are {sorted(actual_years)}, expected {sorted(expected_years)}."
        )
    expected_rows = cfg.n_sites * len(expected_years)
    if len(out) != expected_rows or out["sample_id"].nunique() != cfg.n_sites:
        raise ValueError(
            f"{label} has {len(out)} rows and {out['sample_id'].nunique()} sites; "
            f"expected {expected_rows} rows and {cfg.n_sites} sites."
        )
    counts = out.groupby("sample_id")["year"].nunique()
    if not (counts == len(expected_years)).all():
        raise ValueError(f"{label} does not contain every year for every site.")
    return out.sort_values(["sample_id", "year"]).reset_index(drop=True)


def validate_uploaded_annual_inputs(
    input_paths: dict[str, Path],
    cfg: PipelineConfig,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Validate the exact uploaded schemas and return canonical model tables."""
    alpha_raw = pd.read_csv(input_paths["alphaearth"])
    hydro_raw = pd.read_csv(input_paths["hydrology"])
    coordinates = pd.read_csv(input_paths["coordinates"])

    alpha_required = ["sample_id", "year", *cfg.alphaearth_bands]
    hydro_required = ["sample_id", "year", *cfg.original_hydrology_features]
    alpha = _coerce_unique_site_year(alpha_raw, alpha_required, cfg, "AlphaEarth CSV")
    hydro = _coerce_unique_site_year(hydro_raw, hydro_required, cfg, "Hydrology CSV")

    coord_required = ["sample_id", "longitude", "latitude"]
    missing_coord = [c for c in coord_required if c not in coordinates.columns]
    if missing_coord:
        raise ValueError(f"Coordinates CSV is missing required columns: {missing_coord}")
    coordinates = coordinates[coord_required].copy()
    coordinates["sample_id"] = pd.to_numeric(
        coordinates["sample_id"], errors="raise"
    ).astype(int)
    for column in ("longitude", "latitude"):
        coordinates[column] = pd.to_numeric(coordinates[column], errors="raise")
    if coordinates.isna().any().any() or coordinates.duplicated("sample_id").any():
        raise ValueError("Coordinates must be complete and unique by sample_id.")
    coordinates = coordinates.sort_values("sample_id").reset_index(drop=True)

    alpha_ids = set(alpha["sample_id"].unique())
    hydro_ids = set(hydro["sample_id"].unique())
    coordinate_ids = set(coordinates["sample_id"].unique())
    if not (alpha_ids == hydro_ids == coordinate_ids) or len(coordinates) != cfg.n_sites:
        raise ValueError("AlphaEarth, hydrology, and coordinate sample_id sets do not match.")
    if not coordinates["longitude"].between(-180, 180).all() or not coordinates[
        "latitude"
    ].between(-90, 90).all():
        raise ValueError("Coordinate values fall outside valid longitude/latitude ranges.")

    keys = ["sample_id", "year"]
    if not alpha[keys].equals(hydro[keys]):
        raise ValueError("AlphaEarth and hydrology site-year keys do not match exactly.")

    # The supplied hydrology file also contains a redundant copy of A00-A63.
    # It is never used as a second embedding source. If present, it must match
    # the separate AlphaEarth file exactly, preventing accidental run mixing.
    redundant_bands = [b for b in cfg.alphaearth_bands if b in hydro_raw.columns]
    redundant_max_abs_difference: float | None = None
    if redundant_bands:
        hydro_alpha = _coerce_unique_site_year(
            hydro_raw,
            ["sample_id", "year", *redundant_bands],
            cfg,
            "Hydrology CSV embedded AlphaEarth copy",
        )
        redundant_max_abs_difference = float(
            np.max(
                np.abs(
                    alpha[redundant_bands].to_numpy(np.float64)
                    - hydro_alpha[redundant_bands].to_numpy(np.float64)
                )
            )
        )
        if redundant_max_abs_difference > 1e-12:
            raise ValueError(
                "The AlphaEarth columns duplicated inside the hydrology CSV do not "
                "match the separate AlphaEarth CSV. Keep those model runs separate."
            )

    annual = alpha[alpha_required].merge(
        hydro[hydro_required], on=keys, how="inner", validate="one_to_one"
    )
    annual = annual[["sample_id", "year", *cfg.alphaearth_bands, *cfg.original_hydrology_features]]
    report = {
        "validation_status": "passed",
        "n_sites": int(annual["sample_id"].nunique()),
        "n_years": int(annual["year"].nunique()),
        "n_site_years": int(len(annual)),
        "alphaearth_dimensions_used": len(cfg.alphaearth_bands),
        "hydrological_variables_used": list(cfg.original_hydrology_features),
        "redundant_alphaearth_columns_in_hydrology_file": len(redundant_bands),
        "redundant_alphaearth_max_abs_difference": redundant_max_abs_difference,
        "daily_rainfall_present_in_uploaded_tables": False,
        "daily_smap_present_in_uploaded_tables": False,
        "daily_data_action": (
            "Fetch daily CHIRPS and daily SMAP at the supplied coordinates; annual "
            "totals are not disaggregated or treated as daily observations."
        ),
    }
    return alpha[alpha_required], annual, coordinates, report


def stage_uploaded_annual_inputs(
    input_paths: dict[str, Path],
    cfg: PipelineConfig,
    paths: Paths,
    logger: logging.Logger,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    alpha, annual, coordinates, report = validate_uploaded_annual_inputs(
        input_paths, cfg
    )
    for key, canonical_name in (
        ("alphaearth", DEFAULT_ALPHAEARTH_FILENAME),
        ("hydrology", DEFAULT_HYDROLOGY_FILENAME),
        ("coordinates", DEFAULT_COORDINATES_FILENAME),
    ):
        destination = paths.raw / canonical_name
        if input_paths[key].resolve() != destination.resolve():
            shutil.copy2(input_paths[key], destination)
    write_json(
        paths.manifest / "uploaded_input_validation.json",
        {
            **report,
            "inputs": {
                key: {
                    "source_path": str(value),
                    "filename": value.name,
                    "size_bytes": value.stat().st_size,
                    "sha256": sha256_file(value),
                }
                for key, value in input_paths.items()
            },
        },
    )
    logger.info(
        "Validated uploaded annual tables: %d sites x %d years (%d rows)",
        report["n_sites"],
        report["n_years"],
        report["n_site_years"],
    )
    return alpha, annual, coordinates


# =============================================================================
# EARTH ENGINE EXTRACTION
# =============================================================================


def init_earth_engine(project_id: str, logger: logging.Logger):
    require_project_id(project_id)
    try:
        import ee
    except ImportError as exc:
        raise RuntimeError("Install requirements.txt before running extraction.") from exc

    try:
        ee.Initialize(project=project_id)
    except Exception:
        logger.info("Earth Engine authentication is required; opening browser flow.")
        ee.Authenticate(auth_mode="localhost")
        ee.Initialize(project=project_id)
    if ee.Number(1).add(1).getInfo() != 2:
        raise RuntimeError("Earth Engine initialization check failed.")
    logger.info("Earth Engine initialized with project %s", project_id)
    return ee


def ee_fc_to_dataframe(ee, fc, selectors: Sequence[str]) -> pd.DataFrame:
    """Download a FeatureCollection directly as a paginated pandas DataFrame."""
    selected = fc.select(list(selectors))
    result = ee.data.computeFeatures(
        {
            "expression": selected,
            "fileFormat": "PANDAS_DATAFRAME",
            "pageSize": 5000,
        }
    )
    if isinstance(result, pd.DataFrame):
        df = result.reset_index(drop=True)
    elif isinstance(result, dict) and "features" in result:
        df = pd.DataFrame([x.get("properties", {}) for x in result["features"]])
    else:
        df = pd.DataFrame(result)
    for column in ("geo", ".geo", "system:index"):
        if column in df.columns and column not in selectors:
            df = df.drop(columns=column)
    missing = [c for c in selectors if c not in df.columns]
    if missing:
        raise RuntimeError(f"Earth Engine download omitted columns: {missing}")
    return df[list(selectors)]


def get_kodagu_boundary_and_uploaded_sites(
    ee,
    coordinates: pd.DataFrame,
    cfg: PipelineConfig,
    paths: Paths,
    logger,
):
    boundary_file = paths.raw / "kodagu_boundary.geojson"

    l2 = ee.FeatureCollection(cfg.gaul_asset)
    karnataka = (
        l2.filter(ee.Filter.eq("ADM0_NAME", "India"))
        .filter(ee.Filter.eq("ADM1_NAME", "Karnataka"))
    )
    names = [str(x) for x in karnataka.aggregate_array("ADM2_NAME").distinct().getInfo()]
    exact = [x for x in names if x.strip().lower() in {"kodagu", "coorg"}]
    if not exact:
        raise RuntimeError(f"Kodagu/Coorg was not found in GAUL names: {names}")
    kodagu_fc = karnataka.filter(ee.Filter.eq("ADM2_NAME", exact[0]))
    geom = kodagu_fc.geometry()

    # Use the supplied coordinates verbatim. Regenerating Earth Engine random
    # points would risk attaching daily data to the wrong annual embedding rows.
    features = [
        ee.Feature(
            ee.Geometry.Point([float(row.longitude), float(row.latitude)]),
            {"sample_id": int(row.sample_id)},
        )
        for row in coordinates.itertuples(index=False)
    ]
    sites = ee.FeatureCollection(features)
    inside_count = int(sites.filterBounds(geom).size().getInfo())
    if inside_count != cfg.n_sites:
        raise RuntimeError(
            f"Only {inside_count}/{cfg.n_sites} supplied coordinates intersect the "
            "GAUL Kodagu boundary. Verify that the coordinate file belongs to these annual tables."
        )
    boundary_file.write_text(json.dumps(kodagu_fc.getInfo()), encoding="utf-8")
    logger.info("Saved boundary and verified %d uploaded sites", len(coordinates))
    return geom, sites


def monthly_periods(start: date, end_exclusive: date) -> Iterable[tuple[date, date]]:
    current = date(start.year, start.month, 1)
    while current < end_exclusive:
        last_day = calendar.monthrange(current.year, current.month)[1]
        next_month = current + timedelta(days=last_day)
        yield max(current, start), min(next_month, end_exclusive)
        current = next_month


def reduce_collection_by_day(
    ee,
    image_collection,
    sites,
    value_bands: Sequence[str],
    output_names: Sequence[str],
    scale_m: int,
    date_start: date,
    date_end_exclusive: date,
):
    """Reduce one image per day to all sites and return one flat collection."""
    start = ee.Date(str(date_start))
    end = ee.Date(str(date_end_exclusive))
    ic = image_collection.filterDate(start, end)
    count = int(ic.size().getInfo())
    if count == 0:
        raise RuntimeError(f"No images between {date_start} and {date_end_exclusive}")
    images = ic.toList(count)

    def reduce_one(i):
        image = ee.Image(images.get(i)).select(list(value_bands)).rename(list(output_names))
        date_text = image.date().format("YYYY-MM-dd")
        reduced = image.reduceRegions(
            collection=sites,
            reducer=ee.Reducer.first(),
            scale=scale_m,
            tileScale=8,
        )
        # For a single-band image, reduceRegions names the output field after
        # the reducer ("first"), not the band, so it must be renamed
        # explicitly. For a multi-band image it already uses the band names.
        if len(output_names) == 1:
            reduced = reduced.map(
                lambda f: f.set(output_names[0], f.get("first"))
            )
        return reduced.map(lambda f: f.set("date", date_text))

    nested = ee.List.sequence(0, count - 1).map(reduce_one)
    return ee.FeatureCollection(nested).flatten()


def extract_daily_rainfall(
    ee, geom, sites, cfg: PipelineConfig, paths: Paths, refresh: bool, logger
) -> pd.DataFrame:
    combined_path = paths.raw / "rainfall_daily_2017_2024.parquet"
    if combined_path.exists() and not refresh:
        logger.info("Reusing cached daily rainfall: %s", combined_path)
        return pd.read_parquet(combined_path)

    chirps = ee.ImageCollection(cfg.chirps_asset).filterBounds(geom)
    frames: list[pd.DataFrame] = []
    start = date(cfg.start_year, 1, 1)
    end = date(cfg.end_year + 1, 1, 1)
    for month_start, month_end in monthly_periods(start, end):
        logger.info("Downloading CHIRPS daily rainfall %s", month_start.strftime("%Y-%m"))
        fc = reduce_collection_by_day(
            ee,
            chirps,
            sites,
            ["precipitation"],
            ["rain_mm_day"],
            cfg.chirps_scale_m,
            month_start,
            month_end,
        )
        frame = ee_fc_to_dataframe(ee, fc, ["sample_id", "date", "rain_mm_day"])
        frames.append(frame)

    rain = pd.concat(frames, ignore_index=True)
    rain["sample_id"] = pd.to_numeric(rain["sample_id"]).astype(int)
    rain["date"] = pd.to_datetime(rain["date"])
    rain["rain_mm_day"] = pd.to_numeric(rain["rain_mm_day"], errors="coerce")
    rain = rain.sort_values(["sample_id", "date"]).reset_index(drop=True)

    expected_days = len(pd.date_range(start, end - timedelta(days=1), freq="D"))
    expected_rows = cfg.n_sites * expected_days
    if len(rain) != expected_rows or rain["rain_mm_day"].isna().any():
        raise RuntimeError(
            f"CHIRPS table is incomplete: rows={len(rain)} expected={expected_rows}, "
            f"missing rainfall={int(rain['rain_mm_day'].isna().sum())}. No silent fill is allowed."
        )
    if (rain["rain_mm_day"] < 0).any():
        raise RuntimeError("CHIRPS contains negative daily rainfall values.")

    save_parquet(rain, combined_path)
    for year, frame in rain.groupby(rain["date"].dt.year):
        save_csv(frame, paths.raw / f"rainfall_daily_{int(year)}.csv")
    logger.info("Saved %d complete daily rainfall observations", len(rain))
    return rain


def make_smap_daily_collection(ee, smap, start_day: date, end_exclusive: date, bands):
    images = []
    d = start_day
    while d < end_exclusive:
        s = ee.Date(str(d))
        e = s.advance(1, "day")
        image = (
            smap.filterDate(s, e)
            .select(list(bands))
            .mean()
            .rename(list(bands))
            .set("system:time_start", s.millis())
        )
        images.append(image)
        d += timedelta(days=1)
    return ee.ImageCollection.fromImages(images)


def extract_daily_smap(
    ee, geom, sites, cfg: PipelineConfig, paths: Paths, refresh: bool, logger
) -> pd.DataFrame:
    processed_path = paths.raw / "smap_daily_processed.parquet"
    raw_path = paths.raw / "smap_daily_raw.parquet"
    if processed_path.exists() and not refresh:
        logger.info("Reusing cached daily SMAP: %s", processed_path)
        return pd.read_parquet(processed_path)

    # Fetch the antecedent buffer before 2017 so January events are evaluable.
    start = date(cfg.start_year, 1, 1) - timedelta(days=cfg.antecedent_window_days)
    end = date(cfg.end_year + 1, 1, 1)
    smap = ee.ImageCollection(cfg.smap_asset).filterBounds(geom)
    frames: list[pd.DataFrame] = []
    for month_start, month_end in monthly_periods(start, end):
        logger.info("Downloading daily-mean SMAP %s", month_start.strftime("%Y-%m"))
        daily = make_smap_daily_collection(
            ee, smap, month_start, month_end, cfg.smap_bands
        )
        fc = reduce_collection_by_day(
            ee,
            daily,
            sites,
            cfg.smap_bands,
            cfg.smap_bands,
            cfg.smap_scale_m,
            month_start,
            month_end,
        )
        frames.append(
            ee_fc_to_dataframe(ee, fc, ["sample_id", "date", *cfg.smap_bands])
        )

    sm = pd.concat(frames, ignore_index=True)
    sm["sample_id"] = pd.to_numeric(sm["sample_id"]).astype(int)
    sm["date"] = pd.to_datetime(sm["date"])
    for band in cfg.smap_bands:
        sm[band] = pd.to_numeric(sm[band], errors="coerce")
    sm = sm.sort_values(["sample_id", "date"]).reset_index(drop=True)
    save_parquet(sm, raw_path)

    # ASSUMPTION A8: within-site linear interpolation is limited to short,
    # internal gaps. Original missingness flags are retained for auditing and
    # interpolated days do not count toward the antecedent minimum-valid rule.
    for band in cfg.smap_bands:
        sm[f"{band}_was_missing"] = sm[band].isna()
        sm[band] = sm.groupby("sample_id", group_keys=False)[band].transform(
            lambda s: s.interpolate(
                limit=cfg.smap_max_interpolation_gap_days,
                limit_direction="forward",
                limit_area="inside",
            )
        )
    if sm[list(cfg.smap_bands)].isna().any().any():
        raise RuntimeError("SMAP still contains missing values after within-site interpolation.")
    save_parquet(sm, processed_path)
    for year, frame in sm.groupby(sm["date"].dt.year):
        save_csv(frame, paths.raw / f"smap_daily_{int(year)}.csv")
    logger.info("Saved %d daily SMAP site-date observations", len(sm))
    return sm


def reconcile_daily_sources_with_uploaded_annual(
    rain: pd.DataFrame,
    smap: pd.DataFrame,
    annual_uploaded: pd.DataFrame,
    cfg: PipelineConfig,
    paths: Paths,
) -> None:
    """Save diagnostics only; uploaded annual values remain authoritative."""
    r = rain.copy()
    r["year"] = r["date"].dt.year
    annual_rain = (
        r.groupby(["sample_id", "year"], as_index=False)["rain_mm_day"]
        .sum()
        .rename(columns={"rain_mm_day": "rain_annual_mm_from_daily_chirps"})
    )
    monsoon = (
        r[r["date"].dt.month.isin([6, 7, 8, 9])]
        .groupby(["sample_id", "year"], as_index=False)["rain_mm_day"]
        .sum()
        .rename(columns={"rain_mm_day": "rain_monsoon_mm_from_daily_chirps"})
    )
    s = smap[
        smap["date"].dt.year.between(cfg.start_year, cfg.end_year)
    ].copy()
    s["year"] = s["date"].dt.year
    annual_sm = s.groupby(["sample_id", "year"], as_index=False)[
        list(cfg.smap_bands)
    ].mean()
    annual_sm = annual_sm.rename(
        columns={band: f"{band}_from_daily_smap" for band in cfg.smap_bands}
    )
    keys = ["sample_id", "year"]
    audit = annual_uploaded[
        ["sample_id", "year", "rain_annual_mm", "rain_monsoon_mm", *cfg.smap_bands]
    ].merge(annual_rain, on=keys, how="left", validate="one_to_one")
    audit = audit.merge(monsoon, on=keys, how="left", validate="one_to_one")
    audit = audit.merge(annual_sm, on=keys, how="left", validate="one_to_one")
    for variable, daily_variable in (
        ("rain_annual_mm", "rain_annual_mm_from_daily_chirps"),
        ("rain_monsoon_mm", "rain_monsoon_mm_from_daily_chirps"),
        ("sm_surface", "sm_surface_from_daily_smap"),
        ("sm_rootzone", "sm_rootzone_from_daily_smap"),
    ):
        audit[f"{variable}_difference_daily_minus_uploaded"] = (
            audit[daily_variable] - audit[variable]
        )
    save_csv(audit, paths.raw / "annual_daily_input_reconciliation.csv")
    summary_rows = []
    for variable, daily_variable in (
        ("rain_annual_mm", "rain_annual_mm_from_daily_chirps"),
        ("rain_monsoon_mm", "rain_monsoon_mm_from_daily_chirps"),
        ("sm_surface", "sm_surface_from_daily_smap"),
        ("sm_rootzone", "sm_rootzone_from_daily_smap"),
    ):
        difference = audit[daily_variable] - audit[variable]
        summary_rows.append(
            {
                "variable": variable,
                "n_compared": int(difference.notna().sum()),
                "mean_difference_daily_minus_uploaded": float(difference.mean()),
                "mean_absolute_difference": float(difference.abs().mean()),
                "max_absolute_difference": float(difference.abs().max()),
                "pearson_correlation": float(audit[[variable, daily_variable]].corr().iloc[0, 1]),
                "role_in_pipeline": "diagnostic_only_uploaded_annual_is_authoritative",
            }
        )
    save_csv(
        pd.DataFrame(summary_rows),
        paths.raw / "annual_daily_input_reconciliation_summary.csv",
    )


# =============================================================================
# RAINFALL EVENTS, DUPLICATE GROUPS, AND LEAKAGE-FREE SPLITS
# =============================================================================


def rainfall_signature_ids(rain: pd.DataFrame, cfg: PipelineConfig) -> pd.DataFrame:
    """Hash complete CHIRPS histories so identical raster replicas stay together."""
    rows = []
    for sample_id, frame in rain.groupby("sample_id", sort=True):
        values = np.round(
            frame.sort_values("date")["rain_mm_day"].to_numpy(np.float64),
            cfg.rain_signature_round_decimals,
        ).astype("<f4")
        digest = hashlib.sha256(values.tobytes()).hexdigest()[:16]
        rows.append({"sample_id": int(sample_id), "rain_signature_id": digest})
    return pd.DataFrame(rows)


def grouped_split_manifest(
    signatures: pd.DataFrame,
    coordinates: pd.DataFrame,
    cfg: PipelineConfig,
) -> pd.DataFrame:
    from sklearn.model_selection import GroupShuffleSplit

    if not math.isclose(
        cfg.train_fraction + cfg.validation_fraction + cfg.test_fraction,
        1.0,
        rel_tol=0,
        abs_tol=1e-9,
    ):
        raise ValueError("Train, validation, and test fractions must sum to 1.")

    base = signatures.merge(coordinates, on="sample_id", how="left").sort_values(
        "sample_id"
    )
    if base["rain_signature_id"].nunique() < 10:
        raise RuntimeError("Too few independent rainfall signatures for grouped splitting.")

    indices = np.arange(len(base))
    groups = base["rain_signature_id"].to_numpy()
    first = GroupShuffleSplit(
        n_splits=1, test_size=cfg.test_fraction, random_state=cfg.split_seed
    )
    train_val_idx, test_idx = next(first.split(indices, groups=groups))
    val_relative = cfg.validation_fraction / (
        cfg.train_fraction + cfg.validation_fraction
    )
    second = GroupShuffleSplit(
        n_splits=1, test_size=val_relative, random_state=cfg.split_seed + 1
    )
    train_rel, val_rel = next(
        second.split(train_val_idx, groups=groups[train_val_idx])
    )
    train_idx = train_val_idx[train_rel]
    val_idx = train_val_idx[val_rel]

    split = np.full(len(base), "", dtype=object)
    split[train_idx] = "train"
    split[val_idx] = "validation"
    split[test_idx] = "test"
    base["split"] = split

    leakage = base.groupby("rain_signature_id")["split"].nunique().max()
    if leakage != 1:
        raise AssertionError("A rainfall signature leaked across data splits.")
    return base.reset_index(drop=True)


def extract_rainfall_events(
    rain: pd.DataFrame, signatures: pd.DataFrame, cfg: PipelineConfig
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Extract contiguous wet-day events and their three required descriptors."""
    signature_map = signatures.set_index("sample_id")["rain_signature_id"].to_dict()
    event_rows: list[dict[str, Any]] = []
    member_rows: list[dict[str, Any]] = []

    for sample_id, frame in rain.groupby("sample_id", sort=True):
        frame = frame.sort_values("date").reset_index(drop=True)
        wet = frame[frame["rain_mm_day"] >= cfg.wet_day_threshold_mm].copy()
        if wet.empty:
            continue

        current_indices: list[int] = []
        previous_wet_date: pd.Timestamp | None = None
        event_counter = 0

        def finalize(indices: list[int]) -> None:
            nonlocal event_counter
            if not indices:
                return
            w = wet.loc[indices].sort_values("date")
            start_date = pd.Timestamp(w["date"].iloc[0])
            end_date = pd.Timestamp(w["date"].iloc[-1])
            duration = int((end_date - start_date).days + 1)
            if duration < cfg.min_event_duration_days:
                return
            event_counter += 1
            peak_idx = w["rain_mm_day"].idxmax()
            peak_date = pd.Timestamp(w.loc[peak_idx, "date"])
            event_id = f"S{int(sample_id):04d}_E{event_counter:05d}"
            signature = signature_map[int(sample_id)]
            independent_key = (
                f"{signature}_{start_date:%Y%m%d}_{end_date:%Y%m%d}"
            )
            total = float(w["rain_mm_day"].sum())
            peak = float(w["rain_mm_day"].max())
            event_rows.append(
                {
                    "event_id": event_id,
                    "sample_id": int(sample_id),
                    "rain_signature_id": signature,
                    "independent_event_key": independent_key,
                    "event_start_date": start_date,
                    "event_end_date": end_date,
                    "peak_date": peak_date,
                    "event_year": int(peak_date.year),
                    "event_total_mm": total,
                    "duration_days": duration,
                    "n_wet_days": int(len(w)),
                    "peak_intensity_mm_day": peak,
                }
            )
            for _, row in w.iterrows():
                member_rows.append(
                    {
                        "event_id": event_id,
                        "sample_id": int(sample_id),
                        "date": row["date"],
                        "rain_mm_day": float(row["rain_mm_day"]),
                    }
                )

        for idx, row in wet.iterrows():
            d = pd.Timestamp(row["date"])
            if previous_wet_date is None:
                current_indices = [idx]
            else:
                dry_days = int((d - previous_wet_date).days - 1)
                if dry_days >= cfg.min_interevent_dry_days:
                    finalize(current_indices)
                    current_indices = [idx]
                else:
                    current_indices.append(idx)
            previous_wet_date = d
        finalize(current_indices)

    events = pd.DataFrame(event_rows)
    membership = pd.DataFrame(member_rows)
    if events.empty:
        raise RuntimeError("No rainfall events were extracted.")
    events["is_independent_representative"] = ~events.duplicated(
        "independent_event_key", keep="first"
    )
    return events.sort_values(["sample_id", "event_start_date"]).reset_index(
        drop=True
    ), membership.sort_values(["sample_id", "date"]).reset_index(drop=True)


def save_event_outputs(
    events: pd.DataFrame,
    membership: pd.DataFrame,
    split_manifest: pd.DataFrame,
    cfg: PipelineConfig,
    paths: Paths,
) -> None:
    events = events.merge(
        split_manifest[["sample_id", "split"]], on="sample_id", how="left"
    )
    save_csv(events, paths.events / "rainfall_events.csv")
    save_parquet(membership, paths.events / "rainfall_event_membership.parquet")
    summary = (
        events.groupby(["event_year", "split"], as_index=False)
        .agg(
            n_events=("event_id", "count"),
            n_independent_events=("is_independent_representative", "sum"),
            mean_event_total_mm=("event_total_mm", "mean"),
            median_duration_days=("duration_days", "median"),
            maximum_peak_mm_day=("peak_intensity_mm_day", "max"),
        )
        .sort_values(["event_year", "split"])
    )
    save_csv(summary, paths.events / "rainfall_events_by_year.csv")
    write_json(
        paths.events / "rainfall_event_definition.json",
        {
            "wet_day_threshold_mm": cfg.wet_day_threshold_mm,
            "min_interevent_dry_days": cfg.min_interevent_dry_days,
            "min_event_duration_days": cfg.min_event_duration_days,
            "event_year_rule": cfg.event_year_rule,
            "note": "Assumptions are editable in PipelineConfig.",
        },
    )


# =============================================================================
# DRAGON-KING TAIL FITTING AND VALIDATED EVIDENCE
# =============================================================================


@dataclass
class DKMetricFit:
    metric: str
    status: str
    threshold: float = math.nan
    n_input: int = 0
    n_exceedances: int = 0
    n_regular: int = 0
    shape: float = math.nan
    scale: float = math.nan
    bootstrap_ks_statistic: float = math.nan
    bootstrap_ks_p_value: float = math.nan
    model_adequate: bool = False
    candidate_keys: list[str] = field(default_factory=list)
    candidate_values: list[float] = field(default_factory=list)
    candidate_rank_p_values: list[float] = field(default_factory=list)
    candidate_bonferroni_p_values: list[float] = field(default_factory=list)
    candidate_supported: list[bool] = field(default_factory=list)


def _fit_gpd_regular(values: np.ndarray, top_k: int):
    from scipy import stats

    threshold_exceedances = np.sort(values)[::-1]
    regular = threshold_exceedances[top_k:]
    shape, _, scale = stats.genpareto.fit(regular, floc=0)
    return float(shape), float(scale), regular


def parametric_bootstrap_ks(
    regular: np.ndarray, shape: float, scale: float, n_bootstrap: int, seed: int
) -> tuple[float, float]:
    """KS goodness-of-fit with refitting in every bootstrap replicate."""
    from scipy import stats

    observed = float(stats.kstest(regular, "genpareto", args=(shape, 0, scale)).statistic)
    rng = np.random.default_rng(seed)
    simulated_stats = []
    for _ in range(n_bootstrap):
        sim = stats.genpareto.rvs(
            shape, loc=0, scale=scale, size=len(regular), random_state=rng
        )
        try:
            c, _, s = stats.genpareto.fit(sim, floc=0)
            simulated_stats.append(
                float(stats.kstest(sim, "genpareto", args=(c, 0, s)).statistic)
            )
        except Exception:
            continue
    if not simulated_stats:
        return observed, math.nan
    p = (1 + np.sum(np.asarray(simulated_stats) >= observed)) / (
        1 + len(simulated_stats)
    )
    return observed, float(p)


def fit_dk_metric(
    train_representatives: pd.DataFrame,
    metric: str,
    cfg: PipelineConfig,
    seed: int,
) -> DKMetricFit:
    from scipy import stats

    frame = train_representatives[["independent_event_key", metric]].dropna().copy()
    frame = frame.sort_values(metric, ascending=False).reset_index(drop=True)
    values = frame[metric].to_numpy(np.float64)
    if len(values) == 0:
        return DKMetricFit(metric=metric, status="no_training_events")
    threshold = float(np.percentile(values, cfg.dk_threshold_percentile))
    tail = frame[frame[metric] > threshold].copy()
    exceedances = tail[metric].to_numpy(np.float64) - threshold
    k = min(cfg.dk_top_k, max(0, len(exceedances) - cfg.dk_min_regular_exceedances))
    result = DKMetricFit(
        metric=metric,
        status="initialized",
        threshold=threshold,
        n_input=len(values),
        n_exceedances=len(exceedances),
    )
    if k < 1:
        result.status = "insufficient_exceedances"
        return result

    order = np.argsort(exceedances)[::-1]
    sorted_exc = exceedances[order]
    tail_sorted = tail.iloc[order].reset_index(drop=True)
    candidates = sorted_exc[:k]
    regular = sorted_exc[k:]
    result.n_regular = len(regular)
    try:
        shape, _, scale = stats.genpareto.fit(regular, floc=0)
    except Exception as exc:
        result.status = f"fit_failed:{type(exc).__name__}"
        return result
    if not np.isfinite(shape) or not np.isfinite(scale) or scale <= 0:
        result.status = "invalid_fit"
        return result

    ks_d, ks_p = parametric_bootstrap_ks(
        regular, float(shape), float(scale), cfg.dk_gof_bootstrap, seed
    )
    rng = np.random.default_rng(seed + 1000)
    simulations = stats.genpareto.rvs(
        shape,
        loc=0,
        scale=scale,
        size=(cfg.dk_monte_carlo, len(exceedances)),
        random_state=rng,
    )
    simulated_ranked = np.sort(simulations, axis=1)[:, ::-1][:, :k]
    rank_p = [
        float((1 + np.sum(simulated_ranked[:, i] >= candidates[i])) / (cfg.dk_monte_carlo + 1))
        for i in range(k)
    ]
    bonf = [min(1.0, p * len(cfg.dk_metrics)) for p in rank_p]
    adequate = bool(np.isfinite(ks_p) and ks_p >= cfg.dk_gof_alpha)
    supported = [bool(adequate and p < cfg.dk_alpha) for p in bonf]

    result.status = "ok" if adequate else "gpd_gof_failed"
    result.shape = float(shape)
    result.scale = float(scale)
    result.bootstrap_ks_statistic = ks_d
    result.bootstrap_ks_p_value = ks_p
    result.model_adequate = adequate
    result.candidate_keys = tail_sorted["independent_event_key"].iloc[:k].tolist()
    result.candidate_values = (candidates + threshold).astype(float).tolist()
    result.candidate_rank_p_values = rank_p
    result.candidate_bonferroni_p_values = bonf
    result.candidate_supported = supported
    return result


def score_metric_tail(values: pd.Series, fit: DKMetricFit) -> np.ndarray:
    from scipy import stats

    x = pd.to_numeric(values, errors="coerce").to_numpy(np.float64)
    evidence = np.zeros(len(x), dtype=np.float64)
    if not np.isfinite(fit.shape) or not np.isfinite(fit.scale):
        return evidence
    mask = np.isfinite(x) & (x > fit.threshold)
    evidence[mask] = stats.genpareto.cdf(
        x[mask] - fit.threshold, fit.shape, loc=0, scale=fit.scale
    )
    return np.clip(evidence, 0.0, 1.0 - np.finfo(float).eps)


def dk_threshold_stability(
    train_representatives: pd.DataFrame, cfg: PipelineConfig
) -> pd.DataFrame:
    from scipy import stats

    rows = []
    for metric in cfg.dk_metrics:
        x = train_representatives[metric].dropna().to_numpy(np.float64)
        for pct in cfg.dk_stability_percentiles:
            threshold = float(np.percentile(x, pct))
            exc = x[x > threshold] - threshold
            if len(exc) < cfg.dk_min_regular_exceedances:
                rows.append(
                    {
                        "metric": metric,
                        "percentile": pct,
                        "threshold": threshold,
                        "n_exceedances": len(exc),
                        "gpd_shape": math.nan,
                        "gpd_scale": math.nan,
                    }
                )
                continue
            try:
                shape, _, scale = stats.genpareto.fit(exc, floc=0)
            except Exception:
                shape, scale = math.nan, math.nan
            rows.append(
                {
                    "metric": metric,
                    "percentile": pct,
                    "threshold": threshold,
                    "n_exceedances": len(exc),
                    "gpd_shape": shape,
                    "gpd_scale": scale,
                }
            )
    return pd.DataFrame(rows)


def apply_dragon_king(
    events: pd.DataFrame,
    split_manifest: pd.DataFrame,
    cfg: PipelineConfig,
    paths: Paths,
    logger: logging.Logger,
) -> tuple[pd.DataFrame, list[DKMetricFit]]:
    events = events.drop(columns=["split"], errors="ignore").merge(
        split_manifest[["sample_id", "split"]], on="sample_id", how="left"
    )
    train_rep = events[
        (events["split"] == "train") & events["is_independent_representative"]
    ].copy()
    fits = [
        fit_dk_metric(train_rep, metric, cfg, cfg.split_seed + i * 100)
        for i, metric in enumerate(cfg.dk_metrics)
    ]

    raw_tail_p = []
    supported_key_to_metrics: dict[str, list[str]] = {}
    candidate_key_to_metrics: dict[str, list[str]] = {}
    for fit in fits:
        raw_evidence = score_metric_tail(events[fit.metric], fit)
        events[f"{fit.metric}_tail_evidence_raw"] = raw_evidence
        events[f"{fit.metric}_tail_evidence_validated"] = (
            raw_evidence if fit.model_adequate else np.zeros(len(events))
        )
        raw_tail_p.append(1.0 - raw_evidence)
        for key, supported in zip(fit.candidate_keys, fit.candidate_supported):
            candidate_key_to_metrics.setdefault(key, []).append(fit.metric)
            if supported:
                supported_key_to_metrics.setdefault(key, []).append(fit.metric)

    reliable_fits = [f for f in fits if f.model_adequate]
    if reliable_fits:
        reliable_p = np.vstack(
            [
                1.0 - events[f"{f.metric}_tail_evidence_raw"].to_numpy(float)
                for f in reliable_fits
            ]
        )
        adjusted_p = np.minimum(1.0, len(reliable_fits) * reliable_p.min(axis=0))
        events["dk_evidence_score"] = 1.0 - adjusted_p
    else:
        events["dk_evidence_score"] = 0.0
        logger.warning(
            "No event-metric GPD passed the goodness-of-fit gate; DK attention is disabled."
        )

    events["dk_reliable_metric_count"] = len(reliable_fits)
    events["is_dragon_king_candidate"] = events["independent_event_key"].isin(
        candidate_key_to_metrics
    )
    events["is_dragon_king_supported"] = events["independent_event_key"].isin(
        supported_key_to_metrics
    )
    events["dk_candidate_metrics"] = events["independent_event_key"].map(
        lambda k: ";".join(candidate_key_to_metrics.get(k, []))
    )
    events["dk_supported_metrics"] = events["independent_event_key"].map(
        lambda k: ";".join(supported_key_to_metrics.get(k, []))
    )
    events["dk_causal_attribution"] = False
    events["dk_interpretation"] = np.where(
        events["is_dragon_king_supported"],
        "statistically supported tail exception; not causal attribution",
        "continuous tail context only; not causal attribution",
    )
    save_csv(events, paths.dk / "dragon_king_events.csv")

    fit_rows = []
    candidate_rows = []
    for fit in fits:
        row = asdict(fit)
        for key in (
            "candidate_keys",
            "candidate_values",
            "candidate_rank_p_values",
            "candidate_bonferroni_p_values",
            "candidate_supported",
        ):
            row.pop(key)
        fit_rows.append(row)
        for rank, key in enumerate(fit.candidate_keys):
            candidate_rows.append(
                {
                    "metric": fit.metric,
                    "rank": rank + 1,
                    "independent_event_key": key,
                    "candidate_value": fit.candidate_values[rank],
                    "rank_monte_carlo_p": fit.candidate_rank_p_values[rank],
                    "bonferroni_p": fit.candidate_bonferroni_p_values[rank],
                    "gpd_model_adequate": fit.model_adequate,
                    "is_dragon_king_supported": fit.candidate_supported[rank],
                }
            )
    fit_df = pd.DataFrame(fit_rows)
    candidates_df = pd.DataFrame(candidate_rows)
    save_csv(fit_df, paths.dk / "dragon_king_metric_fits.csv")
    save_csv(candidates_df, paths.dk / "dragon_king_candidates.csv")
    save_csv(fit_df, paths.evaluation / "05_dk_tail_validation.csv")

    stability = dk_threshold_stability(train_rep, cfg)
    save_csv(stability, paths.dk / "dk_tail_threshold_stability.csv")
    summary = [
        "DRAGON-KING EVENT-TAIL ANALYSIS",
        "================================",
        "Fit sample: independent rainfall signatures in the training split only.",
        "No result is interpreted as causal attribution.",
        "",
    ]
    for fit in fits:
        summary.extend(
            [
                f"Metric: {fit.metric}",
                f"Status: {fit.status}",
                f"POT threshold: {fit.threshold:.6g}",
                f"Exceedances/regular: {fit.n_exceedances}/{fit.n_regular}",
                f"GPD shape/scale: {fit.shape:.6g}/{fit.scale:.6g}",
                f"Bootstrap KS p-value: {fit.bootstrap_ks_p_value:.6g}",
                f"GPD accepted for attention: {fit.model_adequate}",
                f"Supported DK candidates: {sum(fit.candidate_supported)}",
                "",
            ]
        )
    (paths.dk / "dragon_king_summary.txt").write_text(
        "\n".join(summary), encoding="utf-8"
    )
    return events, fits


# =============================================================================
# ANTECEDENT SOIL MOISTURE AND ANNUAL MODEL TABLE
# =============================================================================


def add_antecedent_soil_moisture(
    events: pd.DataFrame,
    smap: pd.DataFrame,
    cfg: PipelineConfig,
    paths: Paths,
) -> pd.DataFrame:
    s = smap.sort_values(["sample_id", "date"]).copy()
    for band in cfg.smap_bands:
        missing_flag = f"{band}_was_missing"
        observed_col = f"{band}_observed_only"
        if missing_flag in s:
            s[observed_col] = s[band].mask(s[missing_flag].astype(bool))
        else:
            # Cached legacy input may not contain flags. In that case finite
            # values are treated as observed and this fact remains auditable.
            s[observed_col] = s[band]
        s[f"antecedent_{band}_7d"] = s.groupby("sample_id")[observed_col].transform(
            lambda x: x.shift(1).rolling(
                cfg.antecedent_window_days,
                min_periods=cfg.antecedent_min_valid_days,
            ).mean()
        )
        s[f"antecedent_{band}_valid_days"] = s.groupby("sample_id")[observed_col].transform(
            lambda x: x.shift(1).rolling(
                cfg.antecedent_window_days,
                min_periods=1,
            ).count()
        )
    cols = ["sample_id", "date"]
    for band in cfg.smap_bands:
        cols.extend([f"antecedent_{band}_7d", f"antecedent_{band}_valid_days"])
    lookup = s[cols].rename(columns={"date": "event_start_date"})
    out = events.merge(lookup, on=["sample_id", "event_start_date"], how="left")
    save_csv(out, paths.antecedent / "event_antecedent_soil_moisture.csv")
    return out


def select_annual_event_features(
    events: pd.DataFrame, annual_original: pd.DataFrame, cfg: PipelineConfig, paths: Paths
) -> pd.DataFrame:
    ranked = events.sort_values(
        [
            "sample_id",
            "event_year",
            "dk_evidence_score",
            "event_total_mm",
            "peak_intensity_mm_day",
            "duration_days",
            "event_start_date",
        ],
        ascending=[True, True, False, False, False, False, True],
    )
    selected = ranked.drop_duplicates(["sample_id", "event_year"], keep="first").copy()
    selected = selected.rename(
        columns={
            "event_year": "year",
            "duration_days": "event_duration_days",
            "peak_intensity_mm_day": "event_peak_intensity_mm_day",
            "antecedent_sm_surface_7d": "antecedent_sm_surface_7d",
            "antecedent_sm_rootzone_7d": "antecedent_sm_rootzone_7d",
        }
    )
    keep = [
        "sample_id",
        "year",
        "event_id",
        "event_start_date",
        "event_end_date",
        "event_total_mm",
        "event_duration_days",
        "event_peak_intensity_mm_day",
        "antecedent_sm_surface_7d",
        "antecedent_sm_rootzone_7d",
        "dk_evidence_score",
        "dk_reliable_metric_count",
        "is_dragon_king_candidate",
        "is_dragon_king_supported",
    ]
    selected = selected[keep]

    grid = annual_original[["sample_id", "year", "sm_surface", "sm_rootzone"]].copy()
    annual = grid.merge(selected, on=["sample_id", "year"], how="left")
    annual["has_selected_event"] = annual["event_id"].notna()
    for col in ("event_total_mm", "event_duration_days", "event_peak_intensity_mm_day"):
        annual[col] = annual[col].fillna(0.0)
    annual["dk_evidence_score"] = annual["dk_evidence_score"].fillna(0.0)
    annual["dk_reliable_metric_count"] = annual["dk_reliable_metric_count"].fillna(0).astype(int)
    annual["is_dragon_king_candidate"] = annual["is_dragon_king_candidate"].fillna(False)
    annual["is_dragon_king_supported"] = annual["is_dragon_king_supported"].fillna(False)

    # Events without a valid 7-day window use that site's annual SMAP mean. The
    # fallback is explicit and flagged; it is not fitted using validation/test.
    annual["antecedent_sm_fallback_used"] = annual[
        ["antecedent_sm_surface_7d", "antecedent_sm_rootzone_7d"]
    ].isna().any(axis=1)
    annual["antecedent_sm_surface_7d"] = annual[
        "antecedent_sm_surface_7d"
    ].fillna(annual["sm_surface"])
    annual["antecedent_sm_rootzone_7d"] = annual[
        "antecedent_sm_rootzone_7d"
    ].fillna(annual["sm_rootzone"])
    annual = annual.drop(columns=["sm_surface", "sm_rootzone"])
    save_csv(annual, paths.antecedent / "annual_event_features.csv")
    return annual


def build_model_feature_table(
    annual_original: pd.DataFrame,
    annual_events: pd.DataFrame,
    split_manifest: pd.DataFrame,
    cfg: PipelineConfig,
    paths: Paths,
) -> pd.DataFrame:
    event_cols = [
        "sample_id",
        "year",
        "event_id",
        "event_start_date",
        "event_end_date",
        *cfg.event_hydrology_features,
        "dk_evidence_score",
        "dk_reliable_metric_count",
        "has_selected_event",
        "is_dragon_king_candidate",
        "is_dragon_king_supported",
        "antecedent_sm_fallback_used",
    ]
    table = annual_original.merge(
        annual_events[event_cols], on=["sample_id", "year"], how="left"
    )
    table = table.merge(
        split_manifest[["sample_id", "rain_signature_id", "split"]],
        on="sample_id",
        how="left",
    )
    expected = cfg.n_sites * (cfg.end_year - cfg.start_year + 1)
    numeric = [*cfg.alphaearth_bands, *cfg.original_hydrology_features, *cfg.event_hydrology_features, "dk_evidence_score"]
    if len(table) != expected or table[numeric].isna().any().any():
        raise RuntimeError("Final annual model feature table is incomplete.")
    save_csv(table, paths.model_input / "model_feature_table.csv")
    return table.sort_values(["sample_id", "year"]).reset_index(drop=True)


# =============================================================================
# TRAINING-ONLY PREPROCESSING AND MODEL TENSORS
# =============================================================================


@dataclass
class RobustBranchScaler:
    feature_names: list[str]
    median: np.ndarray
    iqr: np.ndarray

    @classmethod
    def fit(cls, x: np.ndarray, feature_names: Sequence[str]) -> "RobustBranchScaler":
        flat = x.reshape(-1, x.shape[-1]).astype(np.float64)
        median = np.nanmedian(flat, axis=0)
        q25 = np.nanpercentile(flat, 25, axis=0)
        q75 = np.nanpercentile(flat, 75, axis=0)
        iqr = q75 - q25
        iqr[~np.isfinite(iqr) | (np.abs(iqr) < 1e-8)] = 1.0
        if not np.isfinite(median).all():
            raise RuntimeError("A training feature has no finite observations.")
        return cls(list(feature_names), median, iqr)

    def transform(self, x: np.ndarray) -> np.ndarray:
        out = x.astype(np.float64).copy()
        for j in range(out.shape[-1]):
            bad = ~np.isfinite(out[..., j])
            out[..., j][bad] = self.median[j]
        out = (out - self.median) / self.iqr
        if not np.isfinite(out).all():
            raise RuntimeError("Non-finite value produced during robust scaling.")
        return out.astype(np.float32)

    def to_dict(self) -> dict[str, Any]:
        return {
            "feature_names": self.feature_names,
            "training_median": self.median.tolist(),
            "training_iqr": self.iqr.tolist(),
            "fit_scope": "training rainfall-signature groups only",
        }


@dataclass
class ModelData:
    sample_ids: np.ndarray
    years: list[int]
    alphaearth_raw: np.ndarray
    hydrology_raw: np.ndarray
    alphaearth: np.ndarray
    hydrology: np.ndarray
    dk_evidence: np.ndarray
    split_labels: np.ndarray
    train_idx: np.ndarray
    val_idx: np.ndarray
    test_idx: np.ndarray
    ae_scaler: RobustBranchScaler
    hydro_scaler: RobustBranchScaler
    table: pd.DataFrame


def prepare_model_data(
    table: pd.DataFrame, cfg: PipelineConfig, paths: Paths
) -> ModelData:
    years = list(range(cfg.start_year, cfg.end_year + 1))
    sample_ids = np.sort(table["sample_id"].unique()).astype(int)
    if len(sample_ids) != cfg.n_sites:
        raise RuntimeError("Model table does not contain the configured number of sites.")
    if not np.array_equal(sample_ids, np.arange(cfg.n_sites)):
        raise RuntimeError("sample_id must be contiguous from 0 to n_sites-1.")

    ordered = table.sort_values(["sample_id", "year"]).reset_index(drop=True)
    n_years = len(years)
    ae_cols = list(cfg.alphaearth_bands)
    hydro_cols = [*cfg.original_hydrology_features, *cfg.event_hydrology_features]
    ae_raw = ordered[ae_cols].to_numpy(float).reshape(cfg.n_sites, n_years, len(ae_cols))
    hydro_raw = ordered[hydro_cols].to_numpy(float).reshape(
        cfg.n_sites, n_years, len(hydro_cols)
    )
    dk = ordered["dk_evidence_score"].to_numpy(float).reshape(cfg.n_sites, n_years)
    site_meta = ordered.drop_duplicates("sample_id").sort_values("sample_id")
    split_labels = site_meta["split"].to_numpy(str)
    train_idx = np.flatnonzero(split_labels == "train")
    val_idx = np.flatnonzero(split_labels == "validation")
    test_idx = np.flatnonzero(split_labels == "test")
    if min(len(train_idx), len(val_idx), len(test_idx)) == 0:
        raise RuntimeError("At least one grouped split is empty.")

    ae_scaler = RobustBranchScaler.fit(ae_raw[train_idx], ae_cols)
    hydro_scaler = RobustBranchScaler.fit(hydro_raw[train_idx], hydro_cols)
    ae_scaled = ae_scaler.transform(ae_raw)
    hydro_scaled = hydro_scaler.transform(hydro_raw)

    np.save(paths.model_input / "X_alphaearth_raw.npy", ae_raw.astype(np.float32))
    np.save(paths.model_input / "X_hydrology_raw.npy", hydro_raw.astype(np.float32))
    np.save(paths.model_input / "X_alphaearth.npy", ae_scaled)
    np.save(paths.model_input / "X_hydrology.npy", hydro_scaled)
    np.save(paths.model_input / "dk_evidence.npy", dk.astype(np.float32))
    np.save(
        paths.model_input / "X_model.npy",
        np.concatenate([ae_scaled, hydro_scaled], axis=-1),
    )
    (paths.model_input / "feature_columns.txt").write_text(
        "\n".join([*ae_cols, *hydro_cols]), encoding="utf-8"
    )
    write_json(paths.model_input / "alphaearth_scaler.json", ae_scaler.to_dict())
    write_json(paths.model_input / "hydrology_scaler.json", hydro_scaler.to_dict())
    write_json(
        paths.model_input / "tensor_shapes.json",
        {
            "alphaearth": list(ae_scaled.shape),
            "hydrology": list(hydro_scaled.shape),
            "dk_evidence": list(dk.shape),
            "years": years,
        },
    )
    return ModelData(
        sample_ids=sample_ids,
        years=years,
        alphaearth_raw=ae_raw,
        hydrology_raw=hydro_raw,
        alphaearth=ae_scaled,
        hydrology=hydro_scaled,
        dk_evidence=dk.astype(np.float32),
        split_labels=split_labels,
        train_idx=train_idx,
        val_idx=val_idx,
        test_idx=test_idx,
        ae_scaler=ae_scaler,
        hydro_scaler=hydro_scaler,
        table=ordered,
    )


# =============================================================================
# DK-GUIDED, BRANCH-BALANCED TEMPORAL ATTENTION AUTOENCODER
# =============================================================================


def import_tensorflow():
    try:
        import tensorflow as tf
        from tensorflow import keras
        from tensorflow.keras import layers
    except ImportError as exc:
        raise RuntimeError("TensorFlow is required for the training stage.") from exc
    return tf, keras, layers


def register_custom_layers():
    tf, keras, layers = import_tensorflow()

    @keras.utils.register_keras_serializable(package="kodagu_dk_audit")
    class LearnedPositionEncoding(layers.Layer):
        def __init__(self, sequence_length: int, d_model: int, **kwargs):
            super().__init__(**kwargs)
            self.sequence_length = int(sequence_length)
            self.d_model = int(d_model)
            self.embedding = layers.Embedding(self.sequence_length, self.d_model)

        def call(self, x):
            positions = tf.range(tf.shape(x)[1])
            return x + self.embedding(positions)[tf.newaxis, :, :]

        def get_config(self):
            config = super().get_config()
            config.update(
                {"sequence_length": self.sequence_length, "d_model": self.d_model}
            )
            return config

    @keras.utils.register_keras_serializable(package="kodagu_dk_audit")
    class DKGuidedAttention(layers.Layer):
        def __init__(
            self,
            n_heads: int,
            d_model: int,
            dropout: float,
            dk_weight: float,
            **kwargs,
        ):
            super().__init__(**kwargs)
            if d_model % n_heads != 0:
                raise ValueError("d_model must be divisible by n_heads.")
            self.n_heads = int(n_heads)
            self.d_model = int(d_model)
            self.head_dim = int(d_model // n_heads)
            self.dropout_rate = float(dropout)
            self.dk_weight = float(dk_weight)
            self.q = layers.Dense(d_model, name="query_projection")
            self.k = layers.Dense(d_model, name="key_projection")
            self.v = layers.Dense(d_model, name="value_projection")
            self.o = layers.Dense(d_model, name="output_projection")
            self.dropout = layers.Dropout(dropout)

        def _heads(self, x):
            batch = tf.shape(x)[0]
            seq = tf.shape(x)[1]
            x = tf.reshape(x, [batch, seq, self.n_heads, self.head_dim])
            return tf.transpose(x, [0, 2, 1, 3])

        def call(self, inputs, training=False):
            x, dk_evidence = inputs
            q = self._heads(self.q(x))
            k = self._heads(self.k(x))
            v = self._heads(self.v(x))
            logits = tf.matmul(q, k, transpose_b=True) / tf.math.sqrt(
                tf.cast(self.head_dim, tf.float32)
            )
            logits = logits + (
                self.dk_weight * dk_evidence[:, tf.newaxis, tf.newaxis, :]
            )
            weights = tf.nn.softmax(logits, axis=-1)
            dropped = self.dropout(weights, training=training)
            out = tf.matmul(dropped, v)
            out = tf.transpose(out, [0, 2, 1, 3])
            out = tf.reshape(out, [tf.shape(out)[0], tf.shape(out)[1], self.d_model])
            return self.o(out), weights

        def get_config(self):
            config = super().get_config()
            config.update(
                {
                    "n_heads": self.n_heads,
                    "d_model": self.d_model,
                    "dropout": self.dropout_rate,
                    "dk_weight": self.dk_weight,
                }
            )
            return config

    return tf, keras, layers, LearnedPositionEncoding, DKGuidedAttention


@dataclass
class ModelBundle:
    model: Any
    analysis_model: Any
    variant: str
    seed: int


def build_model(
    cfg: PipelineConfig,
    sequence_length: int,
    ae_dim: int,
    hydro_dim: int,
    variant: str,
    seed: int,
) -> ModelBundle:
    tf, keras, layers, Position, DKAttention = register_custom_layers()
    set_global_seed(seed)
    tf.keras.utils.set_random_seed(seed)

    ae_input = keras.Input((sequence_length, ae_dim), name="alphaearth_input")
    hydro_input = keras.Input((sequence_length, hydro_dim), name="hydrology_input")
    dk_input = keras.Input((sequence_length,), name="dk_evidence_input")

    ae_noisy = layers.GaussianNoise(cfg.gaussian_noise_std, name="ae_noise")(ae_input)
    hydro_noisy = layers.GaussianNoise(
        cfg.gaussian_noise_std, name="hydrology_noise"
    )(hydro_input)
    ae_latent = layers.Dense(cfg.d_model, name="alphaearth_branch")(ae_noisy)
    hydro_latent = layers.Dense(cfg.d_model, name="hydrological_encoder")(
        hydro_noisy
    )

    if variant == "hydrology_only":
        # A zero-weight connector keeps the AlphaEarth input/layer in the
        # Functional graph for common diagnostics without contributing signal.
        ae_zero = layers.Rescaling(0.0, name="alphaearth_zero_connector")(ae_latent)
        fused = layers.Add(name="hydrology_only_connector")([hydro_latent, ae_zero])
        dk_weight = 0.0
    elif variant == "alphaearth_only":
        hydro_zero = layers.Rescaling(0.0, name="hydrology_zero_connector")(hydro_latent)
        fused = layers.Add(name="alphaearth_only_connector")([ae_latent, hydro_zero])
        dk_weight = 0.0
    elif variant == "fused_no_dk":
        fused = layers.Add(name="equal_branch_fusion")([ae_latent, hydro_latent])
        fused = layers.Rescaling(0.5, name="equal_branch_average")(fused)
        dk_weight = 0.0
    elif variant == "fused_dk":
        fused = layers.Add(name="equal_branch_fusion")([ae_latent, hydro_latent])
        fused = layers.Rescaling(0.5, name="equal_branch_average")(fused)
        dk_weight = cfg.dk_attention_weight
    else:
        raise ValueError(f"Unknown model variant: {variant}")

    fused = Position(sequence_length, cfg.d_model, name="temporal_position")(fused)
    attention_layer = DKAttention(
        cfg.n_heads,
        cfg.d_model,
        cfg.dropout,
        dk_weight,
        name="dk_guided_temporal_attention",
    )
    attention_out, attention_weights = attention_layer([fused, dk_input])
    x = layers.Add(name="attention_residual")([fused, attention_out])
    x = layers.LayerNormalization(epsilon=1e-6, name="attention_norm")(x)
    ff = layers.Dense(cfg.ff_dim, activation=tf.nn.gelu, name="ffn_1")(x)
    ff = layers.Dropout(cfg.dropout, name="ffn_dropout")(ff)
    ff = layers.Dense(cfg.d_model, name="ffn_2")(ff)
    x = layers.Add(name="ffn_residual")([x, ff])
    x = layers.LayerNormalization(epsilon=1e-6, name="ffn_norm")(x)
    latent_state = layers.Dense(
        cfg.d_model, activation="linear", name="latent_state"
    )(x)
    decoder = layers.Dense(cfg.ff_dim, activation=tf.nn.gelu, name="decoder")(
        latent_state
    )
    ae_output = layers.Dense(ae_dim, name="alphaearth_reconstruction")(decoder)
    hydro_output = layers.Dense(hydro_dim, name="hydrology_reconstruction")(decoder)

    inputs = [ae_input, hydro_input, dk_input]
    model = keras.Model(
        inputs,
        [ae_output, hydro_output],
        name=f"Kodagu_{variant}_temporal_autoencoder",
    )
    model.compile(
        optimizer=keras.optimizers.Adam(cfg.learning_rate),
        loss={
            "alphaearth_reconstruction": "mse",
            "hydrology_reconstruction": "mse",
        },
        loss_weights={
            "alphaearth_reconstruction": cfg.alphaearth_branch_weight,
            "hydrology_reconstruction": cfg.hydrology_branch_weight,
        },
    )
    analysis_model = keras.Model(
        inputs,
        [
            ae_output,
            hydro_output,
            ae_latent,
            hydro_latent,
            latent_state,
            attention_weights,
        ],
        name=f"{variant}_analysis_model",
    )
    return ModelBundle(model=model, analysis_model=analysis_model, variant=variant, seed=seed)


def robust_reference(values: np.ndarray) -> tuple[float, float]:
    values = np.asarray(values, dtype=np.float64)
    med = float(np.nanmedian(values))
    iqr = float(np.nanpercentile(values, 75) - np.nanpercentile(values, 25))
    if not np.isfinite(iqr) or abs(iqr) < 1e-8:
        iqr = 1.0
    return med, iqr


def robust_apply(values: np.ndarray, reference: tuple[float, float]) -> np.ndarray:
    return (np.asarray(values, dtype=np.float64) - reference[0]) / reference[1]


@dataclass
class ScoreCalibration:
    ae_reference: tuple[float, float]
    hydro_reference: tuple[float, float]
    attention_reference: tuple[float, float]
    threshold: float

    def to_dict(self):
        return {
            "ae_change_median_iqr": list(self.ae_reference),
            "hydrology_change_median_iqr": list(self.hydro_reference),
            "attention_change_median_iqr": list(self.attention_reference),
            "adaptive_threshold": self.threshold,
            "threshold_scope": "training rainfall-signature groups only",
        }


def score_from_analysis_outputs(
    outputs: Sequence[np.ndarray],
    variant: str,
    train_idx: np.ndarray,
    cfg: PipelineConfig,
    calibration: ScoreCalibration | None = None,
) -> tuple[dict[str, np.ndarray], ScoreCalibration]:
    ae_recon, hydro_recon, ae_latent, hydro_latent, latent, attention = outputs
    ae_change = np.linalg.norm(np.diff(ae_latent, axis=1), axis=2)
    hydro_change = np.linalg.norm(np.diff(hydro_latent, axis=1), axis=2)
    latent_change = np.linalg.norm(np.diff(latent, axis=1), axis=2)
    mean_attention = attention.mean(axis=1)
    attention_change = np.linalg.norm(np.diff(mean_attention, axis=1), axis=2)
    attention_received = attention.mean(axis=(1, 2))

    if calibration is None:
        ae_ref = robust_reference(ae_change[train_idx].ravel())
        hydro_ref = robust_reference(hydro_change[train_idx].ravel())
        attention_ref = robust_reference(attention_change[train_idx].ravel())
    else:
        ae_ref = calibration.ae_reference
        hydro_ref = calibration.hydro_reference
        attention_ref = calibration.attention_reference

    ae_z = robust_apply(ae_change, ae_ref)
    hydro_z = robust_apply(hydro_change, hydro_ref)
    attention_z = robust_apply(attention_change, attention_ref)
    if variant == "alphaearth_only":
        balanced_latent_z = ae_z
    elif variant == "hydrology_only":
        balanced_latent_z = hydro_z
    else:
        balanced_latent_z = (
            cfg.alphaearth_branch_weight * ae_z
            + cfg.hydrology_branch_weight * hydro_z
        )
    adaptive = (
        cfg.latent_score_weight * balanced_latent_z
        + cfg.attention_score_weight * attention_z
    )
    if calibration is None:
        threshold = float(
            np.percentile(
                adaptive[train_idx].ravel(), cfg.adaptive_threshold_percentile
            )
        )
        calibration = ScoreCalibration(ae_ref, hydro_ref, attention_ref, threshold)

    return {
        "alphaearth_reconstruction": ae_recon,
        "hydrology_reconstruction": hydro_recon,
        "alphaearth_latent": ae_latent,
        "hydrology_latent": hydro_latent,
        "latent_state": latent,
        "attention_weights": attention,
        "attention_received": attention_received,
        "alphaearth_change": ae_change,
        "hydrology_change": hydro_change,
        "latent_change": latent_change,
        "attention_change": attention_change,
        "balanced_latent_change_z": balanced_latent_z,
        "adaptive_shift_score": adaptive,
    }, calibration


def persistence_scores(latent: np.ndarray) -> np.ndarray:
    n, t, _ = latent.shape
    out = np.full((n, t - 1), np.nan, dtype=np.float64)
    for transition in range(t - 2):
        jump = np.linalg.norm(latent[:, transition + 1] - latent[:, transition], axis=1)
        next_change = np.linalg.norm(
            latent[:, transition + 2] - latent[:, transition + 1], axis=1
        )
        valid = jump > 1e-8
        out[valid, transition] = 1.0 - next_change[valid] / jump[valid]
    return out


def scores_to_long(
    scores: dict[str, np.ndarray],
    calibration: ScoreCalibration,
    data: ModelData,
    variant: str,
    seed: int,
) -> pd.DataFrame:
    transition_years = data.years[1:]
    persist = persistence_scores(scores["latent_state"])
    rows = []
    for i, sample_id in enumerate(data.sample_ids):
        for j, year in enumerate(transition_years):
            score = float(scores["adaptive_shift_score"][i, j])
            rows.append(
                {
                    "sample_id": int(sample_id),
                    "year": int(year),
                    "split": data.split_labels[i],
                    "variant": variant,
                    "seed": int(seed),
                    "alphaearth_change": float(scores["alphaearth_change"][i, j]),
                    "hydrology_change": float(scores["hydrology_change"][i, j]),
                    "latent_change": float(scores["latent_change"][i, j]),
                    "balanced_latent_change_z": float(
                        scores["balanced_latent_change_z"][i, j]
                    ),
                    "attention_change": float(scores["attention_change"][i, j]),
                    "adaptive_score": score,
                    "threshold": calibration.threshold,
                    "score_minus_threshold": score - calibration.threshold,
                    "is_candidate": bool(score >= calibration.threshold),
                    "persistence_score": float(persist[i, j])
                    if np.isfinite(persist[i, j])
                    else math.nan,
                }
            )
    return pd.DataFrame(rows)


def train_one_run(
    variant: str,
    seed: int,
    data: ModelData,
    cfg: PipelineConfig,
    run_cfg: dict[str, Any],
    paths: Paths,
    logger: logging.Logger,
) -> tuple[pd.DataFrame, dict[str, Any], ModelBundle, dict[str, np.ndarray], ScoreCalibration]:
    tf, keras, _ = import_tensorflow()
    run_dir = paths.runs / variant / f"seed_{seed:04d}"
    run_dir.mkdir(parents=True, exist_ok=True)
    completed = run_dir / "COMPLETED.json"

    bundle = build_model(
        cfg,
        len(data.years),
        data.alphaearth.shape[-1],
        data.hydrology.shape[-1],
        variant,
        seed,
    )
    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=cfg.early_stopping_patience,
            restore_best_weights=True,
            verbose=0,
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=cfg.lr_patience,
            min_lr=cfg.min_learning_rate,
            verbose=0,
        ),
    ]
    logger.info("Training %s seed %d", variant, seed)
    history = bundle.model.fit(
        [
            data.alphaearth[data.train_idx],
            data.hydrology[data.train_idx],
            data.dk_evidence[data.train_idx],
        ],
        [
            data.alphaearth[data.train_idx],
            data.hydrology[data.train_idx],
        ],
        validation_data=(
            [
                data.alphaearth[data.val_idx],
                data.hydrology[data.val_idx],
                data.dk_evidence[data.val_idx],
            ],
            [data.alphaearth[data.val_idx], data.hydrology[data.val_idx]],
        ),
        epochs=int(run_cfg["max_epochs"]),
        batch_size=cfg.batch_size,
        shuffle=True,
        callbacks=callbacks,
        verbose=0,
    )
    history_df = pd.DataFrame(history.history)
    history_df.insert(0, "epoch", np.arange(1, len(history_df) + 1))
    save_csv(history_df, run_dir / "training_history.csv")

    inputs = [data.alphaearth, data.hydrology, data.dk_evidence]
    outputs = bundle.analysis_model.predict(inputs, batch_size=cfg.batch_size, verbose=0)
    scores, calibration = score_from_analysis_outputs(
        outputs, variant, data.train_idx, cfg
    )
    long = scores_to_long(scores, calibration, data, variant, seed)
    save_csv(long, run_dir / "transition_scores.csv")
    save_csv(
        pd.DataFrame(
            scores["attention_received"],
            columns=data.years,
        ).assign(sample_id=data.sample_ids)[["sample_id", *data.years]],
        run_dir / "attention_received.csv",
    )
    np.save(run_dir / "latent_states.npy", scores["latent_state"].astype(np.float32))
    np.save(
        run_dir / "alphaearth_reconstruction.npy",
        scores["alphaearth_reconstruction"].astype(np.float32),
    )
    np.save(
        run_dir / "hydrology_reconstruction.npy",
        scores["hydrology_reconstruction"].astype(np.float32),
    )
    bundle.model.save(run_dir / "model.keras")
    write_json(run_dir / "score_calibration.json", calibration.to_dict())
    write_json(
        run_dir / "run_configuration.json",
        {**run_cfg, "variant": variant, "seed": seed},
    )

    metrics: dict[str, Any] = {
        "variant": variant,
        "seed": seed,
        "epochs_completed": len(history_df),
        "best_validation_loss": float(history_df["val_loss"].min()),
        "adaptive_threshold": calibration.threshold,
        "n_candidates_all_sites": int(long["is_candidate"].sum()),
    }
    for label, idx in (
        ("train", data.train_idx),
        ("validation", data.val_idx),
        ("test", data.test_idx),
    ):
        ae_mse = float(
            np.mean(
                (data.alphaearth[idx] - scores["alphaearth_reconstruction"][idx]) ** 2
            )
        )
        hydro_mse = float(
            np.mean(
                (data.hydrology[idx] - scores["hydrology_reconstruction"][idx]) ** 2
            )
        )
        metrics[f"{label}_alphaearth_mse"] = ae_mse
        metrics[f"{label}_hydrology_mse"] = hydro_mse
        metrics[f"{label}_balanced_mse"] = 0.5 * (ae_mse + hydro_mse)
    write_json(run_dir / "run_metrics.json", metrics)
    write_json(completed, {"completed": True, "time_utc": pd.Timestamp.utcnow()})
    return long, metrics, bundle, scores, calibration


def rebuild_analysis_model(model, variant: str, seed: int) -> ModelBundle:
    _, keras, _ = import_tensorflow()
    attention_layer = model.get_layer("dk_guided_temporal_attention")
    analysis = keras.Model(
        model.inputs,
        [
            model.get_layer("alphaearth_reconstruction").output,
            model.get_layer("hydrology_reconstruction").output,
            model.get_layer("alphaearth_branch").output,
            model.get_layer("hydrological_encoder").output,
            model.get_layer("latent_state").output,
            attention_layer.output[1],
        ],
    )
    return ModelBundle(model=model, analysis_model=analysis, variant=variant, seed=seed)


def load_completed_run(
    variant: str,
    seed: int,
    data: ModelData,
    cfg: PipelineConfig,
    paths: Paths,
) -> tuple[pd.DataFrame, dict[str, Any], ModelBundle, dict[str, np.ndarray], ScoreCalibration] | None:
    run_dir = paths.runs / variant / f"seed_{seed:04d}"
    required = [
        run_dir / "COMPLETED.json",
        run_dir / "model.keras",
        run_dir / "transition_scores.csv",
        run_dir / "run_metrics.json",
        run_dir / "score_calibration.json",
    ]
    if not all(p.exists() for p in required):
        return None
    _, keras, _ = import_tensorflow()
    register_custom_layers()
    model = keras.models.load_model(run_dir / "model.keras")
    bundle = rebuild_analysis_model(model, variant, seed)
    outputs = bundle.analysis_model.predict(
        [data.alphaearth, data.hydrology, data.dk_evidence],
        batch_size=cfg.batch_size,
        verbose=0,
    )
    c = json.loads((run_dir / "score_calibration.json").read_text(encoding="utf-8"))
    calibration = ScoreCalibration(
        tuple(c["ae_change_median_iqr"]),
        tuple(c["hydrology_change_median_iqr"]),
        tuple(c["attention_change_median_iqr"]),
        float(c["adaptive_threshold"]),
    )
    scores, _ = score_from_analysis_outputs(
        outputs, variant, data.train_idx, cfg, calibration=calibration
    )
    return (
        pd.read_csv(run_dir / "transition_scores.csv"),
        json.loads((run_dir / "run_metrics.json").read_text(encoding="utf-8")),
        bundle,
        scores,
        calibration,
    )


# =============================================================================
# PERSISTENT SYNTHETIC DETECTION TESTS
# =============================================================================


def binary_metrics(labels: np.ndarray, scores: np.ndarray, threshold: float) -> dict[str, float]:
    from sklearn.metrics import (
        average_precision_score,
        balanced_accuracy_score,
        f1_score,
        precision_score,
        recall_score,
        roc_auc_score,
    )

    labels = labels.astype(int).ravel()
    scores = scores.astype(float).ravel()
    pred = (scores >= threshold).astype(int)
    tn = int(np.sum((labels == 0) & (pred == 0)))
    fp = int(np.sum((labels == 0) & (pred == 1)))
    fn = int(np.sum((labels == 1) & (pred == 0)))
    tp = int(np.sum((labels == 1) & (pred == 1)))
    result = {
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "tp": tp,
        "precision": float(precision_score(labels, pred, zero_division=0)),
        "recall": float(recall_score(labels, pred, zero_division=0)),
        "f1": float(f1_score(labels, pred, zero_division=0)),
        "balanced_accuracy": float(balanced_accuracy_score(labels, pred)),
    }
    if len(np.unique(labels)) == 2:
        result["roc_auc"] = float(roc_auc_score(labels, scores))
        result["pr_auc"] = float(average_precision_score(labels, scores))
    else:
        result["roc_auc"] = math.nan
        result["pr_auc"] = math.nan
    return result


def inject_persistent_hydrology_shift(
    x: np.ndarray,
    fraction: float,
    magnitude: float,
    min_affected_years: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    """Add a persistent standardized step; return transition labels and manifest."""
    rng = np.random.default_rng(seed)
    n, t, f = x.shape
    n_inject = max(1, int(round(fraction * n)))
    chosen = rng.choice(n, n_inject, replace=False)
    latest_start_state = t - min_affected_years
    if latest_start_state < 1:
        raise ValueError("Sequence is too short for the requested persistent injection.")
    perturbed = x.copy()
    labels = np.zeros((n, t - 1), dtype=int)
    rows = []
    for i in chosen:
        start_state = int(rng.integers(1, latest_start_state + 1))
        direction = rng.normal(size=f)
        direction /= max(np.linalg.norm(direction), 1e-12)
        perturbed[i, start_state:, :] += magnitude * direction
        labels[i, start_state - 1] = 1
        rows.append(
            {
                "test_position": int(i),
                "transition_index": start_state - 1,
                "magnitude": magnitude,
                "affected_year_count": t - start_state,
            }
        )
    return perturbed, labels, pd.DataFrame(rows)


def synthetic_detection_evaluation(
    bundle: ModelBundle,
    calibration: ScoreCalibration,
    data: ModelData,
    cfg: PipelineConfig,
    seed: int,
    paths: Paths,
) -> pd.DataFrame:
    rows = []
    test_ae = data.alphaearth[data.test_idx]
    test_hydro = data.hydrology[data.test_idx]
    test_dk = data.dk_evidence[data.test_idx]
    for mag_idx, magnitude in enumerate(cfg.synthetic_magnitudes):
        injected, labels, manifest = inject_persistent_hydrology_shift(
            test_hydro,
            cfg.synthetic_fraction,
            magnitude,
            cfg.synthetic_min_affected_years,
            seed + 10000 + mag_idx,
        )
        outputs = bundle.analysis_model.predict(
            [test_ae, injected, test_dk], batch_size=cfg.batch_size, verbose=0
        )
        scores, _ = score_from_analysis_outputs(
            outputs,
            bundle.variant,
            np.arange(len(test_ae)),
            cfg,
            calibration=calibration,
        )
        metrics = binary_metrics(
            labels, scores["adaptive_shift_score"], calibration.threshold
        )
        metrics.update(
            {
                "variant": bundle.variant,
                "seed": seed,
                "magnitude": magnitude,
                "n_test_sites": len(test_ae),
                "n_injected": len(manifest),
                "injection_type": "persistent_hydrology_step",
                "threshold_source": "unmodified_training_groups",
            }
        )
        rows.append(metrics)
        manifest["seed"] = seed
        manifest["sample_id"] = manifest["test_position"].map(
            lambda i: int(data.sample_ids[data.test_idx[int(i)]])
        )
        save_csv(
            manifest,
            paths.runs
            / bundle.variant
            / f"seed_{seed:04d}"
            / f"synthetic_injections_magnitude_{magnitude:g}.csv",
        )
    result = pd.DataFrame(rows)
    save_csv(
        result,
        paths.runs / bundle.variant / f"seed_{seed:04d}" / "synthetic_metrics.csv",
    )
    return result


# =============================================================================
# ENSEMBLE CONSENSUS, PERSISTENCE, AND NON-CAUSAL DK CONTEXT
# =============================================================================


def aggregate_variant_scores(frames: Sequence[pd.DataFrame], cfg: PipelineConfig) -> pd.DataFrame:
    combined = pd.concat(frames, ignore_index=True)
    group_cols = ["sample_id", "year", "split", "variant"]
    aggregated = (
        combined.groupby(group_cols, as_index=False)
        .agg(
            median_adaptive_score=("adaptive_score", "median"),
            mean_adaptive_score=("adaptive_score", "mean"),
            score_std=("adaptive_score", "std"),
            median_threshold=("threshold", "median"),
            selection_frequency=("is_candidate", "mean"),
            median_alphaearth_change=("alphaearth_change", "median"),
            median_hydrology_change=("hydrology_change", "median"),
            median_latent_change=("latent_change", "median"),
            median_attention_change=("attention_change", "median"),
            median_persistence=("persistence_score", "median"),
            n_runs=("seed", "nunique"),
        )
        .sort_values(["sample_id", "year"])
    )
    aggregated["is_stable_candidate"] = (
        aggregated["selection_frequency"] >= cfg.ensemble_selection_frequency
    )
    aggregated["is_persistence_assessable"] = aggregated["median_persistence"].notna()
    aggregated["is_persistent"] = (
        aggregated["median_persistence"] >= cfg.persistence_threshold
    )
    aggregated["is_confirmed_regime_shift"] = (
        aggregated["is_stable_candidate"] & aggregated["is_persistent"]
    )
    return aggregated


def add_dk_context(
    ensemble: pd.DataFrame, annual_events: pd.DataFrame
) -> pd.DataFrame:
    context = annual_events[
        [
            "sample_id",
            "year",
            "event_id",
            "dk_evidence_score",
            "is_dragon_king_supported",
        ]
    ].copy()
    same = context.rename(
        columns={
            "event_id": "same_year_event_id",
            "dk_evidence_score": "same_year_dk_evidence",
            "is_dragon_king_supported": "same_year_supported_dk",
        }
    )
    prior = context.copy()
    prior["year"] = prior["year"] + 1
    prior = prior.rename(
        columns={
            "event_id": "prior_year_event_id",
            "dk_evidence_score": "prior_year_dk_evidence",
            "is_dragon_king_supported": "prior_year_supported_dk",
        }
    )
    out = ensemble.merge(same, on=["sample_id", "year"], how="left")
    out = out.merge(prior, on=["sample_id", "year"], how="left")
    for col in ("same_year_dk_evidence", "prior_year_dk_evidence"):
        out[col] = out[col].fillna(0.0)
    for col in ("same_year_supported_dk", "prior_year_supported_dk"):
        out[col] = out[col].fillna(False)
    out["max_same_or_prior_dk_evidence"] = out[
        ["same_year_dk_evidence", "prior_year_dk_evidence"]
    ].max(axis=1)
    out["dk_context_label"] = np.select(
        [out["same_year_supported_dk"], out["prior_year_supported_dk"]],
        [
            "supported DK event in same year (association only)",
            "supported DK event in prior year (association only)",
        ],
        default="no supported DK event; tail evidence is context only",
    )
    out["causal_attribution_supported"] = False
    return out


def save_ensemble_outputs(
    main: pd.DataFrame,
    coordinates: pd.DataFrame,
    cfg: PipelineConfig,
    paths: Paths,
) -> pd.DataFrame:
    full = main.merge(coordinates, on="sample_id", how="left")
    save_csv(full, paths.ensemble / "adaptive_shift_scores.csv")

    wide_latent = full.pivot(index="sample_id", columns="year", values="median_latent_change").reset_index()
    save_csv(wide_latent, paths.ensemble / "latent_temporal_change.csv")
    wide_attention = full.pivot(index="sample_id", columns="year", values="median_attention_change").reset_index()
    save_csv(wide_attention, paths.ensemble / "attention_temporal_change.csv")
    wide_hydro = full.pivot(index="sample_id", columns="year", values="median_hydrology_change").reset_index()
    save_csv(wide_hydro, paths.ensemble / "change_magnitude.csv")

    candidates = full[full["is_stable_candidate"]].copy()
    confirmed = full[full["is_confirmed_regime_shift"]].copy()
    save_csv(candidates, paths.ensemble / "model_regime_shifts.csv")
    save_csv(candidates, paths.ensemble / "dk_model_regime_shifts.csv")
    save_csv(confirmed, paths.ensemble / "regime_shift_events.csv")
    save_csv(confirmed, paths.ensemble / "regime_shifts.csv")

    year_grid = pd.DataFrame({"year": list(range(cfg.start_year + 1, cfg.end_year + 1))})
    by_year = (
        full.groupby("year", as_index=False)
        .agg(
            n_stable_candidates=("is_stable_candidate", "sum"),
            n_persistence_confirmed=("is_confirmed_regime_shift", "sum"),
            mean_selection_frequency=("selection_frequency", "mean"),
        )
    )
    by_year = year_grid.merge(by_year, on="year", how="left").fillna(0)
    by_year["candidate_percentage"] = 100 * by_year["n_stable_candidates"] / cfg.n_sites
    by_year["confirmed_percentage"] = 100 * by_year["n_persistence_confirmed"] / cfg.n_sites
    save_csv(by_year, paths.ensemble / "regime_shifts_by_year.csv")
    save_csv(by_year, paths.ensemble / "dk_regime_shifts_by_year.csv")
    save_csv(by_year, paths.ensemble / "regime_shift_by_year.csv")

    if not candidates.empty:
        spatial = (
            candidates.groupby("year", as_index=False)
            .agg(
                n_shifts=("sample_id", "count"),
                mean_longitude=("longitude", "mean"),
                mean_latitude=("latitude", "mean"),
                mean_selection_frequency=("selection_frequency", "mean"),
                mean_persistence=("median_persistence", "mean"),
            )
        )
    else:
        spatial = pd.DataFrame(
            columns=[
                "year",
                "n_shifts",
                "mean_longitude",
                "mean_latitude",
                "mean_selection_frequency",
                "mean_persistence",
            ]
        )
    save_csv(spatial, paths.ensemble / "spatial_regime_shift_summary.csv")

    diagnostics = (
        full.groupby("sample_id", as_index=False)
        .agg(
            max_adaptive_score=("median_adaptive_score", "max"),
            max_selection_frequency=("selection_frequency", "max"),
            max_hydrology_change=("median_hydrology_change", "max"),
            max_alphaearth_change=("median_alphaearth_change", "max"),
            median_persistence=("median_persistence", "median"),
        )
    )
    save_csv(diagnostics, paths.ensemble / "model_diagnostics.csv")
    save_csv(
        candidates[
            ["sample_id", "year", "dk_context_label", "causal_attribution_supported"]
        ]
        .groupby(["dk_context_label", "causal_attribution_supported"], as_index=False)
        .size()
        .rename(columns={"size": "n_shifts"}),
        paths.ensemble / "dk_attribution_summary.csv",
    )
    return full


# =============================================================================
# AUDIT-CORRECTED EVALUATION
# =============================================================================


def exact_sequence_duplication_audit(
    data: ModelData, split_manifest: pd.DataFrame
) -> pd.DataFrame:
    x = np.concatenate([data.alphaearth_raw, data.hydrology_raw], axis=-1)
    sequence_hashes = [
        hashlib.sha256(np.round(row, 6).astype("<f4").tobytes()).hexdigest()[:20]
        for row in x
    ]
    frame = split_manifest[["sample_id", "rain_signature_id", "split"]].copy()
    frame["full_sequence_hash"] = sequence_hashes
    train_rain_signatures = set(
        frame.loc[frame["split"] == "train", "rain_signature_id"]
    )
    train_hashes = set(frame.loc[frame["split"] == "train", "full_sequence_hash"])
    frame["rainfall_signature_present_in_training"] = frame[
        "rain_signature_id"
    ].isin(train_rain_signatures) & (frame["split"] != "train")
    frame["exact_full_sequence_present_in_training"] = frame["full_sequence_hash"].isin(
        train_hashes
    ) & (frame["split"] != "train")
    return frame


def variant_reconstruction_table(run_metrics: Sequence[dict[str, Any]]) -> pd.DataFrame:
    frame = pd.DataFrame(run_metrics)
    metrics = [
        "train_alphaearth_mse",
        "train_hydrology_mse",
        "validation_alphaearth_mse",
        "validation_hydrology_mse",
        "test_alphaearth_mse",
        "test_hydrology_mse",
        "test_balanced_mse",
    ]
    rows = []
    for variant, group in frame.groupby("variant"):
        for metric in metrics:
            rows.append(
                {
                    "variant": variant,
                    "metric": metric,
                    "mean": group[metric].mean(),
                    "std_across_seeds": group[metric].std(),
                    "median": group[metric].median(),
                    "n_seeds": group["seed"].nunique(),
                }
            )
    return pd.DataFrame(rows)


def run_stability_tables(
    main_runs: Sequence[pd.DataFrame], cfg: PipelineConfig
) -> tuple[pd.DataFrame, pd.DataFrame]:
    combined = pd.concat(main_runs, ignore_index=True)
    by_year = (
        combined.groupby(["seed", "year"], as_index=False)["is_candidate"]
        .sum()
        .rename(columns={"is_candidate": "n_candidates"})
    )
    totals = by_year.groupby("seed", as_index=False)["n_candidates"].sum()
    dominant = by_year.loc[by_year.groupby("seed")["n_candidates"].idxmax()][
        ["seed", "year", "n_candidates"]
    ].rename(columns={"year": "dominant_year", "n_candidates": "dominant_year_count"})
    seed_sets = {
        int(seed): set(
            zip(
                group.loc[group["is_candidate"], "sample_id"].astype(int),
                group.loc[group["is_candidate"], "year"].astype(int),
            )
        )
        for seed, group in combined.groupby("seed")
    }
    jaccards = []
    seeds = sorted(seed_sets)
    for i, a in enumerate(seeds):
        for b in seeds[i + 1 :]:
            union = seed_sets[a] | seed_sets[b]
            jaccards.append(
                len(seed_sets[a] & seed_sets[b]) / len(union) if union else 1.0
            )
    summary = pd.DataFrame(
        [
            {
                "n_seeds": len(seeds),
                "mean_total_candidates": totals["n_candidates"].mean(),
                "std_total_candidates": totals["n_candidates"].std(),
                "candidate_count_cv": totals["n_candidates"].std()
                / max(totals["n_candidates"].mean(), 1e-12),
                "mean_pairwise_jaccard": float(np.mean(jaccards)) if jaccards else 1.0,
                "min_pairwise_jaccard": float(np.min(jaccards)) if jaccards else 1.0,
                "ensemble_frequency_required": cfg.ensemble_selection_frequency,
                "most_common_dominant_year": int(dominant["dominant_year"].mode().iloc[0]),
                "dominant_year_agreement_fraction": float(
                    dominant["dominant_year"].value_counts(normalize=True).iloc[0]
                ),
            }
        ]
    )
    by_year = by_year.merge(dominant, on="seed", how="left")
    return by_year, summary


def cluster_bootstrap_ci(
    full: pd.DataFrame,
    split_manifest: pd.DataFrame,
    cfg: PipelineConfig,
) -> pd.DataFrame:
    merged = full.merge(
        split_manifest[["sample_id", "rain_signature_id"]],
        on="sample_id",
        how="left",
    )
    groups = sorted(merged["rain_signature_id"].dropna().unique())
    rng = np.random.default_rng(cfg.split_seed + 500)
    stats_rows = []
    for _ in range(cfg.bootstrap_replicates):
        sampled = rng.choice(groups, size=len(groups), replace=True)
        pieces = [merged[merged["rain_signature_id"] == g] for g in sampled]
        boot = pd.concat(pieces, ignore_index=True)
        stats_rows.append(
            {
                "stable_candidates": float(boot["is_stable_candidate"].sum()),
                "persistence_confirmed": float(boot["is_confirmed_regime_shift"].sum()),
                "mean_selection_frequency": float(boot["selection_frequency"].mean()),
                "mean_persistence": float(boot["median_persistence"].mean()),
            }
        )
    boot = pd.DataFrame(stats_rows)
    rows = []
    for col in boot.columns:
        rows.append(
            {
                "statistic": col,
                "bootstrap_mean": boot[col].mean(),
                "ci_lower_2.5%": boot[col].quantile(0.025),
                "ci_upper_97.5%": boot[col].quantile(0.975),
                "resampling_unit": "independent_rainfall_signature",
            }
        )
    return pd.DataFrame(rows)


def spearman_cross_evidence(
    full: pd.DataFrame, split_manifest: pd.DataFrame
) -> pd.DataFrame:
    from scipy.stats import spearmanr

    # Collapse raster replicas before calculating cross-evidence associations.
    grouped = full.merge(
        split_manifest[["sample_id", "rain_signature_id"]],
        on="sample_id",
        how="left",
    ).groupby(["rain_signature_id", "year"], as_index=False).agg(
        median_adaptive_score=("median_adaptive_score", "mean"),
        median_alphaearth_change=("median_alphaearth_change", "mean"),
        median_hydrology_change=("median_hydrology_change", "mean"),
        median_attention_change=("median_attention_change", "mean"),
        max_same_or_prior_dk_evidence=("max_same_or_prior_dk_evidence", "mean"),
        selection_frequency=("selection_frequency", "mean"),
    )
    signals = {
        "median_adaptive_score": grouped["median_adaptive_score"],
        "median_alphaearth_change": grouped["median_alphaearth_change"],
        "median_hydrology_change": grouped["median_hydrology_change"],
        "median_attention_change": grouped["median_attention_change"],
        "dk_evidence_same_or_prior": grouped["max_same_or_prior_dk_evidence"],
        "selection_frequency": grouped["selection_frequency"],
    }
    rows = []
    names = list(signals)
    for i, a in enumerate(names):
        for b in names[i + 1 :]:
            rho, p = spearmanr(signals[a], signals[b], nan_policy="omit")
            rows.append(
                {
                    "signal_A": a,
                    "signal_B": b,
                    "spearman_rho": rho,
                    "p_value_unadjusted": p,
                    "n_independent_group_years": len(grouped),
                    "interpretation": "de-duplicated association only; not causal attribution",
                }
            )
    return pd.DataFrame(rows)


def dk_group_permutation_test(
    full: pd.DataFrame,
    split_manifest: pd.DataFrame,
    cfg: PipelineConfig,
) -> pd.DataFrame:
    from scipy.stats import spearmanr

    df = full.merge(
        split_manifest[["sample_id", "rain_signature_id"]],
        on="sample_id",
        how="left",
    )
    grouped = (
        df.groupby(["rain_signature_id", "year"], as_index=False)
        .agg(
            score=("median_adaptive_score", "mean"),
            dk=("max_same_or_prior_dk_evidence", "mean"),
        )
        .sort_values(["rain_signature_id", "year"])
    )
    observed = float(spearmanr(grouped["score"], grouped["dk"]).statistic)
    pivot = grouped.pivot(index="rain_signature_id", columns="year", values="dk")
    score_pivot = grouped.pivot(index="rain_signature_id", columns="year", values="score")
    common = pivot.index.intersection(score_pivot.index)
    pivot = pivot.loc[common]
    score_pivot = score_pivot.loc[common]
    rng = np.random.default_rng(cfg.split_seed + 900)
    null = []
    for _ in range(cfg.moran_permutations):
        perm = pivot.to_numpy()[rng.permutation(len(pivot))]
        null.append(float(spearmanr(score_pivot.to_numpy().ravel(), perm.ravel()).statistic))
    p = (1 + np.sum(np.abs(null) >= abs(observed))) / (1 + len(null))
    return pd.DataFrame(
        [
            {
                "spearman_rho": observed,
                "group_permutation_p_value": p,
                "n_independent_rainfall_signatures": len(common),
                "n_permutations": len(null),
                "causal_attribution_supported": False,
            }
        ]
    )


def morans_i_grouped(
    full: pd.DataFrame,
    split_manifest: pd.DataFrame,
    cfg: PipelineConfig,
) -> pd.DataFrame:
    from scipy.spatial import cKDTree

    df = full.merge(
        split_manifest[["sample_id", "rain_signature_id"]],
        on="sample_id",
        how="left",
    )
    rng = np.random.default_rng(cfg.split_seed + 700)
    rows = []

    def compute(values: np.ndarray, coords: np.ndarray, k: int) -> float:
        n = len(values)
        dx = values - values.mean()
        _, neighbors = cKDTree(coords).query(coords, k=k + 1)
        neighbors = neighbors[:, 1:]
        numerator = sum(dx[i] * dx[j] for i in range(n) for j in neighbors[i])
        denominator = float(np.sum(dx**2))
        return float((n / (n * k)) * numerator / denominator) if denominator else 0.0

    for year, year_df in df.groupby("year"):
        independent = (
            year_df.groupby("rain_signature_id", as_index=False)
            .agg(
                value=("median_adaptive_score", "mean"),
                longitude=("longitude", "mean"),
                latitude=("latitude", "mean"),
            )
            .dropna()
        )
        n = len(independent)
        k = min(cfg.moran_k_neighbors, n - 1)
        values = independent["value"].to_numpy(float)
        coords = independent[["longitude", "latitude"]].to_numpy(float)
        observed = compute(values, coords, k)
        null = np.array(
            [compute(rng.permutation(values), coords, k) for _ in range(cfg.moran_permutations)]
        )
        p = (1 + np.sum(np.abs(null) >= abs(observed))) / (1 + len(null))
        rows.append(
            {
                "year": int(year),
                "morans_i": observed,
                "expected_i_under_null": -1 / (n - 1),
                "p_value_permutation": p,
                "n_independent_groups": n,
                "k_neighbors": k,
                "permutation_unit": "rainfall_signature_group",
            }
        )
    return pd.DataFrame(rows)


def threshold_sensitivity(
    full: pd.DataFrame, split_manifest: pd.DataFrame
) -> pd.DataFrame:
    merged = full.merge(
        split_manifest[["sample_id", "rain_signature_id"]],
        on="sample_id",
        how="left",
    )
    independent = merged.groupby(["rain_signature_id", "year"], as_index=False).agg(
        score=("median_adaptive_score", "mean"),
        split=("split", "first"),
    )
    calibration_values = independent.loc[
        independent["split"] == "train", "score"
    ].to_numpy(float)
    independent_values = independent["score"].to_numpy(float)
    nominal_values = full["median_adaptive_score"].to_numpy(float)
    rows = []
    for percentile in np.arange(80.0, 100.0, 0.5):
        threshold = float(np.percentile(calibration_values, percentile))
        rows.append(
            {
                "percentile": percentile,
                "threshold": threshold,
                "n_detected": int(np.sum(independent_values >= threshold)),
                "n_detected_independent_group_years": int(
                    np.sum(independent_values >= threshold)
                ),
                "n_detected_nominal_site_years": int(
                    np.sum(nominal_values >= threshold)
                ),
                "threshold_calibration_scope": "training independent rainfall-signature group-years",
            }
        )
    return pd.DataFrame(rows)


def ablation_table(
    variant_ensembles: dict[str, pd.DataFrame],
    reconstruction: pd.DataFrame,
) -> pd.DataFrame:
    rows = []
    for variant, frame in variant_ensembles.items():
        test_mse = reconstruction[
            (reconstruction["variant"] == variant)
            & (reconstruction["metric"] == "test_balanced_mse")
        ]
        rows.append(
            {
                "variant": variant,
                "n_stable_candidates": int(frame["is_stable_candidate"].sum()),
                "n_persistence_confirmed": int(frame["is_confirmed_regime_shift"].sum()),
                "mean_selection_frequency": frame["selection_frequency"].mean(),
                "median_persistence": frame["median_persistence"].median(),
                "mean_test_balanced_mse": float(test_mse["mean"].iloc[0])
                if len(test_mse)
                else math.nan,
            }
        )
    return pd.DataFrame(rows)


def build_evaluation_tables(
    full: pd.DataFrame,
    variant_ensembles: dict[str, pd.DataFrame],
    all_metrics: Sequence[dict[str, Any]],
    main_runs: Sequence[pd.DataFrame],
    synthetic: pd.DataFrame,
    split_manifest: pd.DataFrame,
    data: ModelData,
    fits: Sequence[DKMetricFit],
    cfg: PipelineConfig,
    paths: Paths,
) -> dict[str, pd.DataFrame]:
    reconstruction = variant_reconstruction_table(all_metrics)
    save_csv(reconstruction, paths.evaluation / "01_reconstruction_metrics.csv")
    save_csv(reconstruction, paths.evaluation / "eval_reconstruction_metrics.csv")

    cpe = synthetic.groupby("magnitude", as_index=False).agg(
        recall=("recall", "mean"),
        precision=("precision", "mean"),
        f1=("f1", "mean"),
        balanced_accuracy=("balanced_accuracy", "mean"),
        roc_auc=("roc_auc", "mean"),
        pr_auc=("pr_auc", "mean"),
        n_seeds=("seed", "nunique"),
    )
    save_csv(cpe, paths.evaluation / "02_change_point_error.csv")
    save_csv(cpe, paths.evaluation / "eval_detection_power_curve.csv")
    save_csv(synthetic, paths.evaluation / "eval_synthetic_injection.csv")

    bootstrap = cluster_bootstrap_ci(full, split_manifest, cfg)
    save_csv(bootstrap, paths.evaluation / "03_bootstrap_ci.csv")
    spearman = spearman_cross_evidence(full, split_manifest)
    save_csv(spearman, paths.evaluation / "04_spearman_cross_evidence.csv")

    fit_df = pd.DataFrame(
        [
            {
                "metric": f.metric,
                "status": f.status,
                "threshold": f.threshold,
                "n_exceedances": f.n_exceedances,
                "gpd_shape": f.shape,
                "gpd_scale": f.scale,
                "bootstrap_ks_p_value": f.bootstrap_ks_p_value,
                "gpd_model_adequate": f.model_adequate,
                "n_supported_dragon_kings": sum(f.candidate_supported),
            }
            for f in fits
        ]
    )
    save_csv(fit_df, paths.evaluation / "05_dk_tail_validation.csv")

    persistence = full[
        [
            "sample_id",
            "year",
            "median_persistence",
            "is_persistence_assessable",
            "is_persistent",
            "is_stable_candidate",
            "is_confirmed_regime_shift",
        ]
    ].copy()
    save_csv(persistence, paths.evaluation / "06_persistence_scores.csv")
    moran = morans_i_grouped(full, split_manifest, cfg)
    save_csv(moran, paths.evaluation / "07_morans_i.csv")
    ablation = ablation_table(variant_ensembles, reconstruction)
    save_csv(ablation, paths.evaluation / "08_ablation_study.csv")

    stability_by_year, stability_summary = run_stability_tables(main_runs, cfg)
    save_csv(stability_by_year, paths.evaluation / "eval_stability_by_year.csv")
    save_csv(stability_summary, paths.evaluation / "eval_stability_summary.csv")
    sensitivity = threshold_sensitivity(full, split_manifest)
    save_csv(sensitivity, paths.evaluation / "eval_threshold_sensitivity.csv")
    save_csv(sensitivity, paths.evaluation / "threshold_sensitivity.csv")
    duplication = exact_sequence_duplication_audit(data, split_manifest)
    save_csv(duplication, paths.evaluation / "spatial_duplication_audit.csv")
    dk_permutation = dk_group_permutation_test(full, split_manifest, cfg)
    save_csv(dk_permutation, paths.evaluation / "dk_evidence_association_test.csv")

    return {
        "reconstruction": reconstruction,
        "synthetic_power": cpe,
        "bootstrap": bootstrap,
        "spearman": spearman,
        "dk_fit": fit_df,
        "persistence": persistence,
        "moran": moran,
        "ablation": ablation,
        "stability_by_year": stability_by_year,
        "stability_summary": stability_summary,
        "threshold_sensitivity": sensitivity,
        "duplication": duplication,
        "dk_permutation": dk_permutation,
    }


# =============================================================================
# TABLE COMPATIBILITY, FIGURES, MAP, AND REPORTS
# =============================================================================


def create_comparison_outputs(
    variant_ensembles: dict[str, pd.DataFrame],
    full: pd.DataFrame,
    cfg: PipelineConfig,
    paths: Paths,
) -> None:
    rows = []
    by_year_frames = []
    for variant, frame in variant_ensembles.items():
        rows.append(
            {
                "model": variant,
                "stable_candidates": int(frame["is_stable_candidate"].sum()),
                "persistence_confirmed": int(frame["is_confirmed_regime_shift"].sum()),
                "median_selection_frequency": frame["selection_frequency"].median(),
                "median_persistence": frame["median_persistence"].median(),
            }
        )
        y = frame.groupby("year", as_index=False).agg(
            n_stable_candidates=("is_stable_candidate", "sum"),
            n_persistence_confirmed=("is_confirmed_regime_shift", "sum"),
        )
        y["model"] = variant
        by_year_frames.append(y)
    comparison = pd.DataFrame(rows)
    by_year = pd.concat(by_year_frames, ignore_index=True)
    save_csv(comparison, paths.ensemble / "comparison_report.csv")
    save_csv(by_year, paths.ensemble / "comparison_by_year.csv")

    no_dk = variant_ensembles["fused_no_dk"]
    with_dk = variant_ensembles["fused_dk"]
    agreement = no_dk.merge(
        with_dk,
        on=["sample_id", "year"],
        suffixes=("_no_dk", "_dk"),
    )
    overlap = int(
        (agreement["is_stable_candidate_no_dk"] & agreement["is_stable_candidate_dk"]).sum()
    )
    union = int(
        (agreement["is_stable_candidate_no_dk"] | agreement["is_stable_candidate_dk"]).sum()
    )
    context = pd.DataFrame(
        [
            {
                "fused_no_dk_candidates": int(no_dk["is_stable_candidate"].sum()),
                "fused_dk_candidates": int(with_dk["is_stable_candidate"].sum()),
                "candidate_overlap": overlap,
                "candidate_jaccard": overlap / union if union else 1.0,
                "fused_dk_candidates_with_supported_same_or_prior_dk": int(
                    (
                        full["is_stable_candidate"]
                        & (full["same_year_supported_dk"] | full["prior_year_supported_dk"])
                    ).sum()
                ),
                "causal_attribution_supported": False,
            }
        ]
    )
    save_csv(context, paths.ensemble / "agreement_and_dk_context.csv")
    text = [
        "MODEL COMPARISON — DAILY-EVENT AUDIT-CORRECTED RUN FAMILY",
        "=========================================================",
        comparison.to_string(index=False),
        "",
        context.to_string(index=False),
        "",
        "DK values are statistical context and attention priors only; causal attribution is not claimed.",
    ]
    (paths.reports / "comparison_report.txt").write_text("\n".join(text), encoding="utf-8")


def save_compatibility_tables(
    full: pd.DataFrame,
    annual_original: pd.DataFrame,
    evaluation: dict[str, pd.DataFrame],
    representative_scores: dict[str, np.ndarray],
    data: ModelData,
    cfg: PipelineConfig,
    paths: Paths,
) -> None:
    by_year = pd.read_csv(paths.ensemble / "regime_shifts_by_year.csv")
    temporal_stats = full.groupby("year", as_index=False).agg(
        mean_adaptive_score=("median_adaptive_score", "mean"),
        median_adaptive_score=("median_adaptive_score", "median"),
        score_std=("median_adaptive_score", "std"),
        mean_selection_frequency=("selection_frequency", "mean"),
        n_stable_candidates=("is_stable_candidate", "sum"),
        n_confirmed=("is_confirmed_regime_shift", "sum"),
    )
    save_csv(temporal_stats, paths.ensemble / "temporal_shift_statistics.csv")
    save_csv(by_year, paths.ensemble / "temporal_shifts_by_year.csv")

    documentation = pd.DataFrame(
        [
            {
                "component": "Daily rainfall events",
                "method": f">={cfg.wet_day_threshold_mm} mm/day; separated by >={cfg.min_interevent_dry_days} dry day",
                "status": "assumption; edit PipelineConfig if required",
            },
            {
                "component": "Dragon-King evidence",
                "method": "event total, duration, peak; POT-GPD; top-3 exclusion; bootstrap GOF gate",
                "status": "association prior only, never causal attribution",
            },
            {
                "component": "Data split",
                "method": "grouped by complete daily CHIRPS signature",
                "status": "prevents exact raster replicas crossing splits",
            },
            {
                "component": "Detection",
                "method": "30-seed consensus plus next-year persistence",
                "status": "candidate and confirmed outputs are separate",
            },
        ]
    )
    save_csv(documentation, paths.ensemble / "baseline_vs_model_documentation.csv")

    attention = pd.DataFrame(
        representative_scores["attention_received"], columns=data.years
    )
    attention.insert(0, "sample_id", data.sample_ids)
    save_csv(attention, paths.ensemble / "attention_weights.csv")

    # Training-fitted PCA; coordinates are descriptive and do not enter detection.
    from sklearn.decomposition import PCA

    ae_flat = data.alphaearth.reshape(-1, data.alphaearth.shape[-1])
    train_rows = np.concatenate(
        [
            np.arange(i * len(data.years), (i + 1) * len(data.years))
            for i in data.train_idx
        ]
    )
    pca = PCA(n_components=2, random_state=cfg.split_seed).fit(ae_flat[train_rows])
    coords = pca.transform(ae_flat)
    pca_df = pd.DataFrame(
        {
            "sample_id": np.repeat(data.sample_ids, len(data.years)),
            "year": np.tile(data.years, len(data.sample_ids)),
            "PC1": coords[:, 0],
            "PC2": coords[:, 1],
            "PC1_variance_ratio": pca.explained_variance_ratio_[0],
            "PC2_variance_ratio": pca.explained_variance_ratio_[1],
        }
    )
    save_csv(pca_df, paths.ensemble / "alphaearth_pca_coordinates.csv")

    mag5 = evaluation["synthetic_power"]
    mag5 = mag5.iloc[(mag5["magnitude"] - 5.0).abs().argsort()[:1]].copy()
    mag5["evaluation_type"] = "persistent synthetic proxy; not real ground truth"
    save_csv(mag5, paths.ensemble / "model_metrics.csv")


def _savefig(fig, path: Path) -> None:
    fig.tight_layout()
    fig.savefig(path, dpi=250, bbox_inches="tight")
    import matplotlib.pyplot as plt

    plt.close(fig)


def _plot_boundary(ax, boundary_path: Path) -> None:
    try:
        obj = json.loads(boundary_path.read_text(encoding="utf-8"))
        for feature in obj.get("features", []):
            geom = feature.get("geometry", {})
            coords = geom.get("coordinates", [])
            polygons = coords if geom.get("type") == "MultiPolygon" else [coords]
            for polygon in polygons:
                if not polygon:
                    continue
                ring = np.asarray(polygon[0])
                ax.plot(ring[:, 0], ring[:, 1], color="black", linewidth=1.2)
    except Exception:
        pass


def plot_dragon_king_outputs(
    events: pd.DataFrame,
    fits: Sequence[DKMetricFit],
    cfg: PipelineConfig,
    paths: Paths,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from scipy import stats

    primary = next(f for f in fits if f.metric == "event_total_mm")
    independent = events[
        (events["split"] == "train") & events["is_independent_representative"]
    ]
    x = np.sort(independent[primary.metric].dropna().to_numpy(float))

    fig, ax = plt.subplots(figsize=(10, 7))
    ax.hist(x, bins=45, color="steelblue", alpha=0.7)
    ax.axvline(primary.threshold, color="gray", linestyle="--", label="POT threshold")
    supported = events[
        events["is_independent_representative"] & events["is_dragon_king_supported"]
    ]
    for value in supported["event_total_mm"].unique():
        ax.axvline(value, color="red", linewidth=1.8)
    ax.set_xlabel("Rainfall-event total (mm)")
    ax.set_ylabel("Independent-event count")
    ax.set_title("Rainfall-Event Distribution and Supported Dragon-King Candidates")
    ax.legend()
    _savefig(fig, paths.figures / "dragon_king.png")

    fig, ax = plt.subplots(figsize=(10, 7))
    if np.isfinite(primary.shape):
        exc = np.sort(x[x > primary.threshold] - primary.threshold)[::-1]
        empirical = np.arange(1, len(exc) + 1) / (len(exc) + 1)
        fitted = stats.genpareto.sf(exc, primary.shape, loc=0, scale=primary.scale)
        ax.semilogy(exc + primary.threshold, empirical, "o", alpha=0.55, label="Empirical independent exceedances")
        ax.semilogy(exc + primary.threshold, fitted, "k-", linewidth=2, label="Fitted GPD tail")
    ax.axvline(primary.threshold, linestyle="--", color="gray", label="POT threshold")
    ax.set_xlabel("Rainfall-event total (mm)")
    ax.set_ylabel("Survival probability (log scale)")
    ax.set_title("Dragon-King Test on Independent Daily-Rainfall Events")
    ax.legend()
    _savefig(fig, paths.figures / "dragon_king_diagnostic.png")

    fig, ax = plt.subplots(figsize=(7, 7))
    if np.isfinite(primary.shape):
        exc = np.sort(x[x > primary.threshold] - primary.threshold)
        if len(exc) > cfg.dk_top_k:
            exc = exc[: -cfg.dk_top_k]
        probs = (np.arange(1, len(exc) + 1) - 0.5) / len(exc)
        theory = stats.genpareto.ppf(probs, primary.shape, loc=0, scale=primary.scale)
        ax.scatter(theory, exc, s=14, alpha=0.6)
        limit = max(float(np.max(theory)), float(np.max(exc))) if len(exc) else 1.0
        ax.plot([0, limit], [0, limit], "r--", label="Perfect fit")
    ax.set_xlabel("GPD-fitted quantile")
    ax.set_ylabel("Empirical quantile")
    ax.set_title(
        f"Event-Total GPD Q-Q Plot\nBootstrap KS p={primary.bootstrap_ks_p_value:.4g}"
    )
    ax.legend()
    _savefig(fig, paths.figures / "dk_tail_qq_plot.png")

    stability = pd.read_csv(paths.dk / "dk_tail_threshold_stability.csv")
    fig, axes = plt.subplots(len(cfg.dk_metrics), 2, figsize=(13, 4 * len(cfg.dk_metrics)))
    for row, metric in enumerate(cfg.dk_metrics):
        part = stability[stability["metric"] == metric]
        axes[row, 0].plot(part["percentile"], part["gpd_shape"], marker="o")
        axes[row, 0].axvline(cfg.dk_threshold_percentile, color="gray", linestyle="--")
        axes[row, 0].set_title(f"{metric}: GPD shape")
        axes[row, 1].plot(part["percentile"], part["gpd_scale"], marker="o", color="darkorange")
        axes[row, 1].axvline(cfg.dk_threshold_percentile, color="gray", linestyle="--")
        axes[row, 1].set_title(f"{metric}: GPD scale")
        for ax in axes[row]:
            ax.set_xlabel("Threshold percentile")
    _savefig(fig, paths.figures / "dk_tail_threshold_stability.png")


def plot_all_outputs(
    full: pd.DataFrame,
    variant_ensembles: dict[str, pd.DataFrame],
    evaluation: dict[str, pd.DataFrame],
    synthetic: pd.DataFrame,
    events: pd.DataFrame,
    fits: Sequence[DKMetricFit],
    annual_original: pd.DataFrame,
    coordinates: pd.DataFrame,
    data: ModelData,
    representative_scores: dict[str, np.ndarray],
    representative_history: pd.DataFrame,
    cfg: PipelineConfig,
    paths: Paths,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.decomposition import PCA

    plot_dragon_king_outputs(events, fits, cfg, paths)
    by_year = pd.read_csv(paths.ensemble / "regime_shifts_by_year.csv")
    candidates = full[full["is_stable_candidate"]]
    confirmed = full[full["is_confirmed_regime_shift"]]

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(by_year["year"] - 0.18, by_year["n_stable_candidates"], width=0.36, label="Stable candidates")
    ax.bar(by_year["year"] + 0.18, by_year["n_persistence_confirmed"], width=0.36, label="Persistence-confirmed")
    ax.set_xlabel("Transition year"); ax.set_ylabel("Count"); ax.legend()
    ax.set_title("Hydrological Regime-Shift Candidates by Year")
    _savefig(fig, paths.figures / "fig_shifts_by_year.png")

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.hist(full["median_adaptive_score"], bins=50, color="steelblue")
    stable_min = candidates["median_adaptive_score"].min() if len(candidates) else math.nan
    if np.isfinite(stable_min):
        ax.axvline(stable_min, linestyle="--", color="red", label="Lowest stable-candidate score")
    ax.set_xlabel("Median adaptive shift score"); ax.set_ylabel("Frequency"); ax.legend()
    ax.set_title("Distribution of Ensemble Adaptive Shift Scores")
    _savefig(fig, paths.figures / "fig_score_distribution.png")

    fig, ax = plt.subplots(figsize=(9, 9))
    _plot_boundary(ax, paths.raw / "kodagu_boundary.geojson")
    ax.scatter(coordinates["longitude"], coordinates["latitude"], s=8, c="lightgray", label="All sites")
    if len(candidates):
        ax.scatter(candidates["longitude"], candidates["latitude"], s=25, c="orange", label="Stable candidates")
    if len(confirmed):
        ax.scatter(confirmed["longitude"], confirmed["latitude"], s=40, c="red", marker="*", label="Confirmed")
    ax.set_xlabel("Longitude"); ax.set_ylabel("Latitude"); ax.legend()
    ax.set_title("Spatial Distribution of Audit-Corrected Regime Shifts")
    _savefig(fig, paths.figures / "fig_spatial_distribution.png")
    # Canonical aliases from the saved project, produced from the same data.
    for alias in ("spatial_shifts.png", "kodagu_regime_shift_map.png"):
        fig, ax = plt.subplots(figsize=(9, 9))
        _plot_boundary(ax, paths.raw / "kodagu_boundary.geojson")
        ax.scatter(coordinates["longitude"], coordinates["latitude"], s=8, c="lightgray")
        if len(candidates): ax.scatter(candidates["longitude"], candidates["latitude"], s=24, c="orange")
        if len(confirmed): ax.scatter(confirmed["longitude"], confirmed["latitude"], s=42, c="red", marker="*")
        ax.set_xlabel("Longitude"); ax.set_ylabel("Latitude")
        ax.set_title("Spatial Distribution of Detected Shifts")
        _savefig(fig, paths.figures / alias)

    fig, ax = plt.subplots(figsize=(11, 6))
    for variant, frame in variant_ensembles.items():
        count = frame.groupby("year")["is_stable_candidate"].sum()
        ax.plot(count.index, count.values, marker="o", label=variant)
    ax.set_xlabel("Transition year"); ax.set_ylabel("Stable candidates"); ax.legend()
    ax.set_title("Regime Shifts by Year — Model Variants")
    _savefig(fig, paths.figures / "comparison_shifts_by_year.png")
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(by_year["year"], by_year["n_stable_candidates"], color="steelblue")
    ax.set_xlabel("Year"); ax.set_ylabel("Stable candidates"); ax.set_title("Regime Shifts by Year")
    _savefig(fig, paths.figures / "shifts_by_year.png")

    power = evaluation["synthetic_power"]
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(power["magnitude"], power["recall"], marker="o", label="Recall")
    ax.plot(power["magnitude"], power["precision"], marker="s", label="Precision")
    ax.plot(power["magnitude"], power["f1"], marker="^", label="F1")
    ax.set_xlabel("Persistent injected magnitude (training-scaled units)")
    ax.set_ylabel("Score"); ax.set_ylim(0, 1.05); ax.legend()
    ax.set_title("Detection Power — Persistent Hydrological Shifts")
    _savefig(fig, paths.figures / "eval_detection_power_curve.png")

    sens = evaluation["threshold_sensitivity"]
    for name in ("eval_threshold_sensitivity.png", "threshold_sensitivity.png"):
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(sens["percentile"], sens["n_detected"], marker="o", markersize=3)
        ax.axvline(cfg.adaptive_threshold_percentile, color="gray", linestyle="--", label="Run threshold")
        ax.set_xlabel("Percentile threshold"); ax.set_ylabel("Detected transitions"); ax.legend()
        ax.set_title("Threshold Sensitivity")
        _savefig(fig, paths.figures / name)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(representative_history["epoch"], representative_history["loss"], label="Training loss")
    if "val_loss" in representative_history:
        ax.plot(representative_history["epoch"], representative_history["val_loss"], label="Validation loss")
    ax.set_yscale("log"); ax.set_xlabel("Epoch"); ax.set_ylabel("Balanced reconstruction loss"); ax.legend()
    ax.set_title("Training Curve — Representative Fused-DK Run")
    _savefig(fig, paths.figures / "eval_training_curve.png")

    latent_change_vectors = np.diff(representative_scores["latent_state"], axis=1)
    flat_latent = latent_change_vectors.reshape(-1, latent_change_vectors.shape[-1])
    pca_latent = PCA(n_components=2, random_state=cfg.split_seed).fit_transform(flat_latent)
    candidate_mask = full.sort_values(["sample_id", "year"])["is_stable_candidate"].to_numpy(bool)
    fig, ax = plt.subplots(figsize=(9, 7))
    ax.scatter(pca_latent[~candidate_mask, 0], pca_latent[~candidate_mask, 1], s=8, alpha=0.3, label="Other transitions")
    if candidate_mask.any():
        ax.scatter(pca_latent[candidate_mask, 0], pca_latent[candidate_mask, 1], s=24, color="orange", label="Stable candidates")
    ax.set_xlabel("PC1"); ax.set_ylabel("PC2"); ax.legend(); ax.set_title("Latent-Change PCA")
    _savefig(fig, paths.figures / "eval_latent_pca.png")

    pca_df = pd.read_csv(paths.ensemble / "alphaearth_pca_coordinates.csv")
    fig, ax = plt.subplots(figsize=(9, 7))
    ax.scatter(pca_df["PC1"], pca_df["PC2"], s=8, alpha=0.35)
    ax.set_xlabel("PC1"); ax.set_ylabel("PC2"); ax.set_title("AlphaEarth PCA (Training-Fitted)")
    _savefig(fig, paths.figures / "alphaearth_pca.png")

    attention = representative_scores["attention_received"]
    order = np.argsort(attention.mean(axis=1))
    fig, ax = plt.subplots(figsize=(10, 10))
    im = ax.imshow(attention[order], aspect="auto", cmap="viridis")
    ax.set_xticks(np.arange(len(data.years)), data.years)
    ax.set_xlabel("Key year"); ax.set_ylabel("Sample (sorted)")
    ax.set_title("Temporal Attention Received per Site-Year")
    fig.colorbar(im, ax=ax, label="Attention weight")
    _savefig(fig, paths.figures / "temporal_attention_heatmap.png")

    mean_attn = attention.mean(axis=0); std_attn = attention.std(axis=0)
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(data.years, mean_attn, marker="o")
    ax.fill_between(data.years, mean_attn - std_attn, mean_attn + std_attn, alpha=0.2)
    ax.set_xlabel("Year"); ax.set_ylabel("Mean attention received")
    ax.set_title("Learned Temporal Attention Over Time")
    _savefig(fig, paths.figures / "temporal_attention_over_time.png")

    fig, ax = plt.subplots(figsize=(13, 6))
    for year, part in full.groupby("year"):
        x = year + np.linspace(-0.12, 0.12, len(part))
        color = np.where(part["is_stable_candidate"], "tab:red", "lightgray")
        ax.scatter(x, part["median_adaptive_score"], c=color, s=8, alpha=0.6)
    ax.set_xlabel("Transition year"); ax.set_ylabel("Median adaptive score")
    ax.set_title("Temporal Regime-Shift Timeline")
    _savefig(fig, paths.figures / "regime_shift_timeline.png")

    annual_mean = annual_original.groupby("year", as_index=False)[list(cfg.original_hydrology_features)].mean()
    fig, ax = plt.subplots(figsize=(11, 6))
    ax.plot(annual_mean["year"], annual_mean["rain_annual_mm"], marker="o", label="Annual rain")
    ax.plot(annual_mean["year"], annual_mean["rain_monsoon_mm"], marker="s", label="Monsoon rain")
    ax.plot(annual_mean["year"], annual_mean["et_annual_mm"], marker="^", label="ET")
    ax2 = ax.twinx()
    ax2.plot(annual_mean["year"], annual_mean["sm_surface"], "--", color="green", label="Surface SM")
    ax2.plot(annual_mean["year"], annual_mean["sm_rootzone"], ":", color="darkgreen", label="Root-zone SM")
    ax.set_xlabel("Year"); ax.set_ylabel("mm"); ax2.set_ylabel("Soil moisture")
    ax.set_title("Mean Hydro-Climatic Features by Year")
    ax.legend(loc="upper left"); ax2.legend(loc="upper right")
    _savefig(fig, paths.figures / "feature_trends.png")

    ablation = evaluation["ablation"]
    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(len(ablation)); width = 0.36
    ax.bar(x - width / 2, ablation["n_stable_candidates"], width, label="Stable candidates")
    ax.bar(x + width / 2, ablation["n_persistence_confirmed"], width, label="Confirmed")
    ax.set_xticks(x, ablation["variant"], rotation=20); ax.set_ylabel("Count"); ax.legend()
    ax.set_title("Key Metric Comparison")
    _savefig(fig, paths.figures / "metric_comparison.png")

    moran = evaluation["moran"]
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(moran["year"].astype(str), moran["morans_i"], color="tab:red")
    ax.set_xlabel("Year"); ax.set_ylabel("Moran's I")
    ax.set_title("Spatial Autocorrelation by Year — Independent Groups")
    _savefig(fig, paths.figures / "morans_i.png")

    assessable = full["median_persistence"].dropna()
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.hist(assessable, bins=30, color="mediumpurple", alpha=0.8)
    ax.axvline(cfg.persistence_threshold, color="green", linestyle="--", label="Persistence threshold")
    ax.axvline(0, color="red", linestyle="--", label="Reversion boundary")
    ax.set_xlabel("Persistence score"); ax.set_ylabel("Transitions"); ax.legend()
    ax.set_title("Persistence of Detected Regime Shifts")
    _savefig(fig, paths.figures / "persistence.png")

    ae_err = np.mean((data.alphaearth - representative_scores["alphaearth_reconstruction"]) ** 2, axis=(1, 2))
    hydro_err = np.mean((data.hydrology - representative_scores["hydrology_reconstruction"]) ** 2, axis=(1, 2))
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.hist(ae_err, bins=30, alpha=0.55, label="AlphaEarth branch")
    ax.hist(hydro_err, bins=30, alpha=0.55, label="Hydrology branch")
    ax.set_xlabel("Per-site reconstruction MSE"); ax.set_ylabel("Count"); ax.legend()
    ax.set_title("Branch-Balanced Reconstruction Error")
    _savefig(fig, paths.figures / "reconstruction_error.png")

    target_magnitude = float(
        synthetic.loc[(synthetic["magnitude"] - 5.0).abs().idxmin(), "magnitude"]
    )
    mag5 = synthetic[np.isclose(synthetic["magnitude"], target_magnitude)]
    tn, fp, fn, tp = [int(mag5[c].sum()) for c in ("tn", "fp", "fn", "tp")]
    fig, ax = plt.subplots(figsize=(6, 5))
    image = ax.imshow([[tn, fp], [fn, tp]], cmap="Blues")
    for i, row in enumerate([[tn, fp], [fn, tp]]):
        for j, value in enumerate(row): ax.text(j, i, str(value), ha="center", va="center")
    ax.set_xticks([0, 1], ["No shift", "Shift"]); ax.set_yticks([0, 1], ["No shift", "Injected shift"])
    ax.set_xlabel("Predicted"); ax.set_ylabel("Synthetic proxy label")
    ax.set_title("Confusion Matrix — Persistent Synthetic Shifts")
    fig.colorbar(image, ax=ax)
    _savefig(fig, paths.figures / "confusion_matrix.png")


def make_interactive_map(
    full: pd.DataFrame,
    coordinates: pd.DataFrame,
    paths: Paths,
) -> None:
    try:
        import folium
    except ImportError:
        return
    center = [coordinates["latitude"].mean(), coordinates["longitude"].mean()]
    m = folium.Map(location=center, zoom_start=9, tiles="CartoDB positron")
    for _, row in full[full["is_stable_candidate"]].iterrows():
        color = "red" if row["is_confirmed_regime_shift"] else "orange"
        folium.CircleMarker(
            [row["latitude"], row["longitude"]],
            radius=4 + 4 * row["selection_frequency"],
            color=color,
            fill=True,
            fill_opacity=0.7,
            popup=(
                f"Site {int(row['sample_id'])}; year {int(row['year'])}; "
                f"frequency {row['selection_frequency']:.2f}; "
                f"persistence {row['median_persistence']:.3f}; "
                f"{row['dk_context_label']}"
            ),
        ).add_to(m)
    m.save(paths.reports / "kodagu_regime_shift_interactive_map.html")


def save_final_reports(
    full: pd.DataFrame,
    evaluation: dict[str, pd.DataFrame],
    cfg: PipelineConfig,
    run_cfg: dict[str, Any],
    cfg_hash: str,
    paths: Paths,
) -> None:
    stable = int(full["is_stable_candidate"].sum())
    confirmed = int(full["is_confirmed_regime_shift"].sum())
    assessable = full[full["is_persistence_assessable"]]
    supported_context = int(
        (
            full["is_stable_candidate"]
            & (full["same_year_supported_dk"] | full["prior_year_supported_dk"])
        ).sum()
    )
    mag5 = evaluation["synthetic_power"].iloc[
        (evaluation["synthetic_power"]["magnitude"] - 5.0).abs().argsort()[:1]
    ]
    f1_mag5 = float(mag5["f1"].iloc[0]) if len(mag5) else math.nan
    model_status = (
        "synthetic proxy performance remains modest; do not claim operational validation"
        if not np.isfinite(f1_mag5) or f1_mag5 < 0.5
        else "synthetic proxy performance passed the assumed F1 >= 0.5 reporting gate; real validation is still absent"
    )
    duplication = evaluation["duplication"]
    rainfall_leakage_count = int(
        duplication["rainfall_signature_present_in_training"].sum()
    )
    full_sequence_leakage_count = int(
        duplication["exact_full_sequence_present_in_training"].sum()
    )
    dk_test = evaluation["dk_permutation"].iloc[0]

    summary = pd.DataFrame(
        {
            "Metric": [
                "Study Region",
                "Historical Period",
                "Nominal Spatial Samples",
                "Independent Rainfall Signatures",
                "Validation/Test Rainfall Signatures Duplicated in Training",
                "Exact Validation/Test Sequences Duplicated in Training",
                "Compatible Random Seeds",
                "Stable Regime-Shift Candidates",
                "Persistence-Confirmed Regime Shifts",
                "Assessable Transition Median Persistence",
                "Stable Candidates with Supported Same/Prior-Year DK Context",
                "DK Causal Attribution Claimed",
                "Synthetic F1 at Magnitude 5",
                "Model Validation Status",
                "Run Family",
                "Configuration Hash",
            ],
            "Value": [
                cfg.study_region,
                f"{cfg.start_year}-{cfg.end_year}",
                cfg.n_sites,
                duplication["rain_signature_id"].nunique(),
                rainfall_leakage_count,
                full_sequence_leakage_count,
                len(run_cfg["run_seeds"]),
                stable,
                confirmed,
                assessable["median_persistence"].median(),
                supported_context,
                False,
                f1_mag5,
                model_status,
                run_cfg["run_family"],
                cfg_hash,
            ],
        }
    )
    save_csv(summary, paths.ensemble / "final_project_results_summary.csv")

    config_text = [
        "KODAGU DAILY-EVENT DK-GUIDED TEMPORAL ATTENTION MODEL",
        "=====================================================",
        f"Run family: {run_cfg['run_family']}",
        f"Configuration hash: {cfg_hash}",
        f"Years/sites: {cfg.start_year}-{cfg.end_year} / {cfg.n_sites}",
        f"Model seeds: {run_cfg['run_seeds']}",
        f"d_model/heads/ff: {cfg.d_model}/{cfg.n_heads}/{cfg.ff_dim}",
        f"Branch reconstruction weights: AlphaEarth={cfg.alphaearth_branch_weight}, hydrology={cfg.hydrology_branch_weight}",
        f"Adaptive score: {cfg.latent_score_weight} balanced latent + {cfg.attention_score_weight} attention",
        f"Threshold: training-only {cfg.adaptive_threshold_percentile}th percentile",
        f"Consensus frequency: {cfg.ensemble_selection_frequency}",
        f"Persistence threshold: {cfg.persistence_threshold}",
        "Causal DK attribution: prohibited by design",
    ]
    (paths.reports / "model_configuration.txt").write_text(
        "\n".join(config_text), encoding="utf-8"
    )
    (paths.reports / "dk_model_configuration.txt").write_text(
        "\n".join(config_text), encoding="utf-8"
    )
    (paths.ensemble / "adaptive_threshold.txt").write_text(
        "Per-seed training thresholds are stored in 06_runs/*/seed_*/score_calibration.json.\n",
        encoding="utf-8",
    )

    report = [
        "KODAGU HYDROLOGICAL REGIME-SHIFT ANALYSIS REPORT",
        "================================================",
        "",
        "Methodological scope",
        "Daily CHIRPS rainfall was separated into events. Event total, duration, and peak intensity were tested using training-only, spatially de-duplicated POT-GPD models. Validated evidence entered temporal attention as a prior. Annual AlphaEarth embeddings and the original five annual hydrological variables were retained in separate, equally weighted branches.",
        "",
        "Audit controls",
        f"Rainfall-signature leakage after grouped splitting: {rainfall_leakage_count}.",
        f"Exact full-sequence leakage after grouped splitting: {full_sequence_leakage_count}.",
        "Scaling and score calibration were fitted on training groups only.",
        "AlphaEarth and hydrology received equal reconstruction and change-score weights.",
        f"Compatible-run consensus used {len(run_cfg['run_seeds'])} seeds and a selection frequency of {cfg.ensemble_selection_frequency:.2f}.",
        f"Persistence confirmation required a score of at least {cfg.persistence_threshold:.2f}; final-year candidates were not automatically confirmed.",
        "",
        "Results",
        f"Stable candidates: {stable}.",
        f"Persistence-confirmed shifts: {confirmed}.",
        f"Synthetic proxy F1 at magnitude 5: {f1_mag5:.3f}.",
        f"Status: {model_status}.",
        f"DK-score association rho={float(dk_test['spearman_rho']):.3f}, group-permutation p={float(dk_test['group_permutation_p_value']):.4f}.",
        "",
        "Interpretation",
        "Dragon-King evidence is statistical event context, not proof that an event caused a regime shift. Historical detections are model candidates until validated against independent hydrological or landslide observations.",
    ]
    (paths.reports / "analysis_report.txt").write_text("\n".join(report), encoding="utf-8")
    advanced = [
        "ADVANCED EVALUATION REPORT",
        "==========================",
        "",
        "1. Reconstruction metrics",
        evaluation["reconstruction"].to_string(index=False),
        "",
        "2. Persistent synthetic detection power",
        evaluation["synthetic_power"].to_string(index=False),
        "",
        "3. Rainfall-signature cluster bootstrap",
        evaluation["bootstrap"].to_string(index=False),
        "",
        "4. Cross-evidence associations",
        evaluation["spearman"].to_string(index=False),
        "",
        "5. Dragon-King tail validation",
        evaluation["dk_fit"].to_string(index=False),
        "",
        "6. Persistence",
        evaluation["persistence"].describe(include="all").to_string(),
        "",
        "7. Grouped Moran's I",
        evaluation["moran"].to_string(index=False),
        "",
        "8. Ablation study",
        evaluation["ablation"].to_string(index=False),
        "",
        "All synthetic results are proxy evaluations; no real-event ground truth was used.",
    ]
    (paths.evaluation / "advanced_evaluation_report.txt").write_text(
        "\n".join(advanced), encoding="utf-8"
    )
    write_json(
        paths.reports / "run_metadata.json",
        {
            "run_family": run_cfg["run_family"],
            "pipeline_version": PIPELINE_VERSION,
            "pipeline_source_sha256": run_cfg["pipeline_source_sha256"],
            "software_versions": run_cfg["software_versions"],
            "configuration_hash": cfg_hash,
            "created_utc": pd.Timestamp.utcnow(),
            "python": sys.version,
            "platform": platform.platform(),
            "working_directory": str(paths.run_root),
            "scientific_status": "exploratory; no causal DK attribution; no external ground truth",
        },
    )


def save_run_manifest(paths: Paths, cfg_dict_: dict[str, Any], cfg_hash: str) -> None:
    manifest_path = paths.manifest / "run_manifest.json"
    current = {
        "configuration_hash": cfg_hash,
        "configuration": cfg_dict_,
        "method_identity": f"{cfg_dict_['run_family']}__{cfg_hash}",
        "incompatible_runs_must_remain_separate": True,
    }
    if manifest_path.exists():
        old = json.loads(manifest_path.read_text(encoding="utf-8"))
        if old.get("configuration_hash") != cfg_hash:
            raise RuntimeError("Existing run directory contains an incompatible manifest.")
    write_json(manifest_path, current)
    write_json(paths.manifest / "assumptions_and_user_actions.json", cfg_dict_)
    write_json(
        paths.manifest / "data_source_provenance.json",
        {
            "Kodagu_boundary": {
                "earth_engine_asset": cfg_dict_["gaul_asset"],
                "source": "FAO GAUL 2015 level-2 administrative boundary",
            },
            "daily_rainfall": {
                "earth_engine_asset": cfg_dict_["chirps_asset"],
                "catalog_url": "https://developers.google.com/earth-engine/datasets/catalog/UCSB-CHG_CHIRPS_DAILY",
            },
            "soil_moisture": {
                "earth_engine_asset": cfg_dict_["smap_asset"],
                "catalog_url": "https://developers.google.com/earth-engine/datasets/catalog/NASA_SMAP_SPL4SMGP_008",
            },
            "annual_embedding": {
                "source": "user-supplied validated CSV",
                "input_identity": cfg_dict_["uploaded_inputs"]["alphaearth"],
                "original_earth_engine_catalog": "https://developers.google.com/earth-engine/datasets/catalog/GOOGLE_SATELLITE_EMBEDDING_V1_ANNUAL",
            },
            "annual_hydrological_state": {
                "source": "user-supplied validated CSV",
                "input_identity": cfg_dict_["uploaded_inputs"]["hydrology"],
            },
            "sample_coordinates": {
                "source": "user-supplied validated CSV",
                "input_identity": cfg_dict_["uploaded_inputs"]["coordinates"],
            },
            "earth_engine_authentication": "https://developers.google.com/earth-engine/guides/auth",
        },
    )


def inventory_outputs(paths: Paths) -> pd.DataFrame:
    rows = []
    for path in sorted(paths.run_root.rglob("*")):
        if path.is_file():
            rows.append(
                {
                    "relative_path": str(path.relative_to(paths.run_root)),
                    "size_bytes": path.stat().st_size,
                    "sha256": sha256_file(path),
                }
            )
    frame = pd.DataFrame(rows)
    save_csv(frame, paths.manifest / "output_inventory_and_checksums.csv")
    return frame


def stage_rank(stage: str) -> int:
    return {
        "extract": 1,
        "events": 2,
        "dk": 3,
        "features": 4,
        "train": 5,
        "evaluate": 6,
        "report": 7,
        "all": 99,
    }[stage]


def load_cached_raw(cfg: PipelineConfig, paths: Paths):
    required = {
        "rain": paths.raw / "rainfall_daily_2017_2024.parquet",
        "smap": paths.raw / "smap_daily_processed.parquet",
        "ae": paths.raw / "kodagu_alphaearth_2017_2024.csv",
        "annual": paths.raw / "kodagu_hydrological_features_2017_2024.csv",
        "coords": paths.raw / "sample_coordinates.csv",
    }
    missing = [str(p) for p in required.values() if not p.exists()]
    if missing:
        raise FileNotFoundError(
            "Cached inputs are missing. Run --stage extract first:\n" + "\n".join(missing)
        )
    alpha, annual, coordinates, _ = validate_uploaded_annual_inputs(
        {
            "alphaearth": required["ae"],
            "hydrology": required["annual"],
            "coordinates": required["coords"],
        },
        cfg,
    )
    return (
        pd.read_parquet(required["rain"]),
        pd.read_parquet(required["smap"]),
        alpha,
        annual,
        coordinates,
    )


def self_test() -> None:
    # Twenty-four synthetic sites form twelve independent rainfall-signature
    # groups (two exact raster replicas per group), which is enough to exercise
    # the production split guard requiring at least ten independent groups.
    cfg = PipelineConfig(n_sites=24, dk_monte_carlo=200, dk_gof_bootstrap=30)
    dates = pd.date_range("2017-01-01", periods=40, freq="D")
    rows = []
    for s in range(cfg.n_sites):
        base = np.zeros(len(dates))
        base[[1, 2, 5, 10, 11, 12, 25]] = [2, 3, 5, 1, 2, 10 + s, 4]
        # Create exact replica pairs to test grouped splitting.
        if s % 2 == 1:
            base = np.zeros(len(dates))
            base[[1, 2, 5, 10, 11, 12, 25]] = [2, 3, 5, 1, 2, 10 + s - 1, 4]
        for d, value in zip(dates, base):
            rows.append({"sample_id": s, "date": d, "rain_mm_day": value})
    rain = pd.DataFrame(rows)
    signatures = rainfall_signature_ids(rain, cfg)
    coordinates = pd.DataFrame(
        {
            "sample_id": np.arange(cfg.n_sites),
            "longitude": np.linspace(75.4, 76.1, cfg.n_sites),
            "latitude": np.linspace(12.0, 12.8, cfg.n_sites),
        }
    )
    split = grouped_split_manifest(signatures, coordinates, cfg)
    assert split.groupby("rain_signature_id")["split"].nunique().max() == 1
    events, members = extract_rainfall_events(rain, signatures, cfg)
    assert {"event_total_mm", "duration_days", "peak_intensity_mm_day"}.issubset(events)
    assert len(events) > 0 and len(members) > 0
    x = np.zeros((8, 8, 10), dtype=np.float32)
    injected, labels, manifest = inject_persistent_hydrology_shift(x, 0.25, 3.0, 2, 42)
    assert labels.sum() == len(manifest) and not np.array_equal(x, injected)
    print("SELF-TEST PASSED: event extraction, replica grouping, split isolation, and persistent injection.")


def main() -> None:
    args = parse_args()
    if args.self_test:
        self_test()
        return

    cfg = PipelineConfig()
    input_paths = resolve_uploaded_inputs(args)
    _, _, _, input_validation = validate_uploaded_annual_inputs(input_paths, cfg)
    if args.validate_inputs:
        print(
            json.dumps(
                {
                    **input_validation,
                    "inputs": {
                        key: {
                            "path": str(path),
                            "size_bytes": path.stat().st_size,
                            "sha256": sha256_file(path),
                        }
                        for key, path in input_paths.items()
                    },
                },
                indent=2,
            )
        )
        return

    run_cfg = config_dict(cfg, args.quick_test)
    # Content hashes, rather than machine-specific absolute paths, define the
    # compatible input family. Any changed input produces a new run directory.
    run_cfg["uploaded_inputs"] = {
        key: {
            "canonical_filename": {
                "alphaearth": DEFAULT_ALPHAEARTH_FILENAME,
                "hydrology": DEFAULT_HYDROLOGY_FILENAME,
                "coordinates": DEFAULT_COORDINATES_FILENAME,
            }[key],
            "size_bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }
        for key, path in input_paths.items()
    }
    cfg_hash = config_hash(run_cfg)
    paths = Paths.create(args.project_root, run_cfg["run_family"], cfg_hash)
    save_run_manifest(paths, run_cfg, cfg_hash)
    logger = setup_logging(paths.manifest / "pipeline.log")
    logger.info("Run root: %s", paths.run_root)
    logger.info("Configuration hash: %s", cfg_hash)
    rank = stage_rank(args.stage)

    # -------- 1. Uploaded annual tables + required daily EE extraction --------
    if rank == 1 or args.stage == "all":
        ae, annual_original, coordinates = stage_uploaded_annual_inputs(
            input_paths, cfg, paths, logger
        )
        ee = init_earth_engine(args.ee_project_id, logger)
        geom, sites = get_kodagu_boundary_and_uploaded_sites(
            ee, coordinates, cfg, paths, logger
        )
        rain = extract_daily_rainfall(
            ee, geom, sites, cfg, paths, args.refresh_data, logger
        )
        smap = extract_daily_smap(
            ee, geom, sites, cfg, paths, args.refresh_data, logger
        )
        reconcile_daily_sources_with_uploaded_annual(
            rain, smap, annual_original, cfg, paths
        )
        if rank == 1:
            inventory_outputs(paths)
            logger.info("Extraction stage completed.")
            return
    else:
        rain, smap, ae, annual_original, coordinates = load_cached_raw(cfg, paths)
        rain["date"] = pd.to_datetime(rain["date"])
        smap["date"] = pd.to_datetime(smap["date"])

    # ---------------- 2. Event extraction and grouped split ----------------
    signatures = rainfall_signature_ids(rain, cfg)
    split_manifest = grouped_split_manifest(signatures, coordinates, cfg)
    save_csv(split_manifest, paths.manifest / "split_manifest.csv")
    events, membership = extract_rainfall_events(rain, signatures, cfg)
    save_event_outputs(events, membership, split_manifest, cfg, paths)
    events = events.merge(
        split_manifest[["sample_id", "split"]], on="sample_id", how="left"
    )
    if rank == 2:
        inventory_outputs(paths)
        logger.info("Rainfall-event stage completed.")
        return

    # ---------------- 3. Dragon-King evidence ----------------
    events, fits = apply_dragon_king(events, split_manifest, cfg, paths, logger)
    if rank == 3:
        plot_dragon_king_outputs(events, fits, cfg, paths)
        inventory_outputs(paths)
        logger.info("Dragon-King stage completed.")
        return

    # ---------------- 4. Antecedent SM + model table ----------------
    event_sm = add_antecedent_soil_moisture(events, smap, cfg, paths)
    annual_events = select_annual_event_features(
        event_sm, annual_original, cfg, paths
    )
    model_table = build_model_feature_table(
        annual_original, annual_events, split_manifest, cfg, paths
    )
    data = prepare_model_data(model_table, cfg, paths)
    if rank == 4:
        inventory_outputs(paths)
        logger.info("Feature-construction stage completed.")
        return

    # ---------------- 5. Separate compatible model runs ----------------
    run_seeds = [int(x) for x in run_cfg["run_seeds"]]
    variant_runs: dict[str, list[pd.DataFrame]] = {
        v: [] for v in cfg.model_variants
    }
    all_metrics: list[dict[str, Any]] = []
    synthetic_frames: list[pd.DataFrame] = []
    representative_scores: dict[str, np.ndarray] | None = None
    representative_history: pd.DataFrame | None = None

    for variant in cfg.model_variants:
        for seed in run_seeds:
            cached = load_completed_run(variant, seed, data, cfg, paths)
            if cached is None:
                result = train_one_run(
                    variant, seed, data, cfg, run_cfg, paths, logger
                )
            else:
                logger.info("Reusing completed %s seed %d", variant, seed)
                result = cached
            long, metrics, bundle, scores, calibration = result
            variant_runs[variant].append(long)
            all_metrics.append(metrics)
            if variant == "fused_dk":
                synthetic_frames.append(
                    synthetic_detection_evaluation(
                        bundle, calibration, data, cfg, seed, paths
                    )
                )
                if seed == run_seeds[0]:
                    representative_scores = {k: np.asarray(v) for k, v in scores.items()}
                    representative_history = pd.read_csv(
                        paths.runs / variant / f"seed_{seed:04d}" / "training_history.csv"
                    )
            _, keras, _ = import_tensorflow()
            keras.backend.clear_session()

    registry = pd.DataFrame(all_metrics)
    save_csv(registry, paths.manifest / "run_registry.csv")
    synthetic = pd.concat(synthetic_frames, ignore_index=True)
    variant_ensembles = {
        variant: aggregate_variant_scores(frames, cfg)
        for variant, frames in variant_runs.items()
    }
    main_ensemble = add_dk_context(variant_ensembles["fused_dk"], annual_events)
    full = save_ensemble_outputs(main_ensemble, coordinates, cfg, paths)
    create_comparison_outputs(variant_ensembles, full, cfg, paths)

    # Copy canonical representative model/history names without mixing seeds.
    import shutil

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
        inventory_outputs(paths)
        logger.info("Training and ensemble stage completed.")
        return

    assert representative_scores is not None and representative_history is not None

    # ---------------- 6. Evaluation ----------------
    evaluation = build_evaluation_tables(
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
    save_compatibility_tables(
        full,
        annual_original,
        evaluation,
        representative_scores,
        data,
        cfg,
        paths,
    )
    if rank == 6:
        inventory_outputs(paths)
        logger.info("Evaluation stage completed.")
        return

    # ---------------- 7. Figures, map, and reports ----------------
    plot_all_outputs(
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
        make_interactive_map(full, coordinates, paths)
    save_final_reports(full, evaluation, cfg, run_cfg, cfg_hash, paths)
    inventory_outputs(paths)
    logger.info("Complete pipeline finished: %s", paths.run_root)


if __name__ == "__main__":
    main()
