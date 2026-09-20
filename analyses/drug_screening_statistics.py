"""Run blocked inference across completed drug-screening biological replicates.

The six input directories must be outputs from ``drug_screening_single_run.py``
for two species and BR1-BR3. Per-well values are retained for display, but the
inferential observations are treatment means calculated within each biological
replicate. Each eligible species/compound/metric family is analysed as
``metric ~ condition + biological_replicate``. Tukey all-pair comparisons use
the residual mean square and degrees of freedom from that blocked model.

Raw AUC is the inferential endpoint; relative AUC is presentation-only. Failed
Gompertz fits are never replaced by zero. Combination regimens that changed
between runs are exported and plotted descriptively rather than pooled into an
invalid common-dose test.
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from textwrap import fill
from typing import Dict, Iterable, List, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
ANALYSES_DIR = PROJECT_ROOT / "analyses"
for directory in [SRC_DIR, ANALYSES_DIR]:
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))

from growth_analysis import DISPLAY_SPECIES, GrowthDataValidationError  # noqa: E402
from drug_screening_single_run import condition_color_map, condition_order, expected_time_points  # noqa: E402


EXPECTED_SPECIES = ["E_coli", "B_subtilis"]
EXPECTED_REPLICATES = ["BR1", "BR2", "BR3"]
EXPECTED_TECHNICAL_WELLS = 3
MAX_TIME_MIN = 1045
TIMEPOINTS = expected_time_points(MAX_TIME_MIN)
SCHIFF_COMPOUNDS = ["C1", "C2", "C3", "C4"]
SCHIFF_CONCENTRATIONS = [6.25, 12.5, 25.0, 50.0, 100.0]
PENAG_CONCENTRATIONS = [12.5, 25.0, 50.0, 100.0]
RAW_AUC = "AUC_0_17h25min_OD_h"
REL_AUC = "relative_AUC_percent"
KZ = "Kz_OD600_per_h"
TLAG = "TLag_h"
METRICS = {
    "AUC_0_1045_min": {
        "technical": RAW_AUC,
        "summary": f"{RAW_AUC}_mean",
        "plot": REL_AUC,
        "label": "Relative AUC 0-17 h 25 min (% of vehicle)",
        "inferential_label": "AUC_0_1045_min",
    },
    "Kz": {
        "technical": KZ,
        "summary": f"{KZ}_mean",
        "plot": KZ,
        "label": "Growth rate, Kz (OD600·h⁻¹)",
        "inferential_label": "Kz",
    },
    "TLag": {
        "technical": TLAG,
        "summary": f"{TLAG}_mean",
        "plot": TLAG,
        "label": "Lag time, TLag (h)",
        "inferential_label": "TLag",
    },
}
METRIC_DISPLAY_TITLES = {
    "AUC_0_1045_min": "Relative AUC",
    "Kz": "Growth rate",
    "TLag": "Lag time",
}
TECHNICAL_REQUIRED = {
    "run_id",
    "biological_replicate",
    "species",
    "well",
    "technical_replicate",
    "treatment_type",
    "compound",
    "compound_concentration_uM",
    "ampicillin_concentration_ug_mL",
    "condition_id",
    "condition",
    RAW_AUC,
    REL_AUC,
    KZ,
    TLAG,
    "gompertz_R2",
    "gompertz_fit_status",
    "gompertz_fit_warning",
}
SUMMARY_REQUIRED = {
    "run_id",
    "biological_replicate",
    "species",
    "treatment_type",
    "compound",
    "compound_concentration_uM",
    "ampicillin_concentration_ug_mL",
    "condition_id",
    "condition",
    "n_technical_wells",
    f"{RAW_AUC}_mean",
    f"{REL_AUC}_mean",
    f"{KZ}_mean",
    f"{TLAG}_mean",
    "n_valid_gompertz_fits",
    "n_failed_gompertz_fits",
}
GROWTH_REQUIRED = {
    "run_id",
    "biological_replicate",
    "species",
    "well",
    "time_min",
    "time_h",
    "raw_OD600",
    "corrected_OD600",
    "condition_id",
}
RUN_REQUIRED = {
    "run_id",
    "species",
    "biological_replicate",
    "analysis_endpoint_min",
    "number_of_retained_time_points",
    "number_of_processed_nonblank_measurements",
}
FAMILY_ORDER = {
    "schiff": ["vehicle_1pct_DMSO"],
    "penag": ["vehicle_1pct_DMSO"],
}
for compound in SCHIFF_COMPOUNDS:
    FAMILY_ORDER[f"schiff_{compound}"] = ["vehicle_1pct_DMSO"] + [
        f"{compound}_{concentration:g}uM" for concentration in SCHIFF_CONCENTRATIONS
    ]
FAMILY_ORDER["penag_PenAg"] = ["vehicle_1pct_DMSO"] + [
    f"PenAg_{concentration:g}uM" for concentration in PENAG_CONCENTRATIONS
]


class DrugScreeningStatisticsError(GrowthDataValidationError):
    """Raised when run outputs are malformed or cannot support the analysis."""


@dataclass(frozen=True)
class FamilySpec:
    """Definition of one treatment family used for blocked inference."""
    species: str
    compound: str
    family_type: str
    condition_ids: List[str]

    @property
    def family_id(self) -> str:
        return f"{self.species}_{self.compound}"


@dataclass(frozen=True)
class AnovaResult:
    """Blocked-ANOVA components used for reporting and Tukey comparisons."""
    species: str
    compound: str
    family_type: str
    metric: str
    number_of_conditions: int
    number_of_biological_replicates: int
    total_model_observations: int
    condition_SS: float
    block_SS: float
    residual_SS: float
    condition_df: int
    block_df: int
    residual_df: int
    condition_MS: float
    block_MS: float
    residual_MS: float
    condition_F: float
    condition_p_value: float
    block_F: float
    block_p_value: float
    partial_eta_squared: float
    model_status: str

    def to_dict(self) -> dict:
        return self.__dict__.copy()


def parse_args() -> argparse.Namespace:
    """Parse six run directories and the statistical output directory."""
    parser = argparse.ArgumentParser(
        description="Run randomized-complete-block statistics on completed drug-screening outputs."
    )
    parser.add_argument("--e-coli-br1", required=True)
    parser.add_argument("--e-coli-br2", required=True)
    parser.add_argument("--e-coli-br3", required=True)
    parser.add_argument("--b-subtilis-br1", required=True)
    parser.add_argument("--b-subtilis-br2", required=True)
    parser.add_argument("--b-subtilis-br3", required=True)
    parser.add_argument(
        "--output-dir",
        default=str(PROJECT_ROOT / "results" / "drug_screening" / "statistics"),
        help="Output directory for statistical tables and figures.",
    )
    return parser.parse_args()


def significance_label(p_value: float) -> str:
    if pd.isna(p_value) or p_value >= 0.05:
        return "n.s."
    if p_value < 0.0001:
        return "****"
    if p_value < 0.001:
        return "***"
    if p_value < 0.01:
        return "**"
    return "*"


def _format_errors(errors: Sequence[str]) -> str:
    return "Drug-screening statistics validation failed:\n- " + "\n- ".join(errors)


def _hash_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _as_float(value: object) -> float:
    if pd.isna(value):
        return math.nan
    return float(value)


def _display_condition(row: pd.Series) -> str:
    if row["condition_id"] == "vehicle_1pct_DMSO":
        return "1% DMSO"
    if row["treatment_type"] == "ampicillin_control":
        return f"{_as_float(row['ampicillin_concentration_ug_mL']):g} ug/mL amp"
    if row["treatment_type"] == "combination":
        return (
            f"{_as_float(row['compound_concentration_uM']):g} uM "
            f"+ {_as_float(row['ampicillin_concentration_ug_mL']):g} ug/mL amp"
        )
    if row["compound"] == "DMSO":
        return "1% DMSO"
    return f"{row['compound']} {_as_float(row['compound_concentration_uM']):g} uM"


def _condition_x_label(condition_id: str, definitions: pd.DataFrame) -> str:
    row = definitions[definitions["condition_id"].eq(condition_id)].iloc[0]
    if condition_id == "vehicle_1pct_DMSO":
        return "1% DMSO"
    if row["treatment_type"] == "ampicillin_control":
        return f"{_as_float(row['ampicillin_concentration_ug_mL']):g}"
    return f"{_as_float(row['compound_concentration_uM']):g}"


def build_input_map(args: argparse.Namespace) -> Dict[Tuple[str, str], Path]:
    return {
        ("E_coli", "BR1"): Path(args.e_coli_br1),
        ("E_coli", "BR2"): Path(args.e_coli_br2),
        ("E_coli", "BR3"): Path(args.e_coli_br3),
        ("B_subtilis", "BR1"): Path(args.b_subtilis_br1),
        ("B_subtilis", "BR2"): Path(args.b_subtilis_br2),
        ("B_subtilis", "BR3"): Path(args.b_subtilis_br3),
    }


def read_run_directory(path: Path, expected_species: str, expected_replicate: str) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, Dict[str, str]]:
    required = {
        "growth": path / "blank_corrected_growth_data.csv",
        "technical": path / "technical_well_growth_metrics.csv",
        "summary": path / "biological_replicate_summary.csv",
        "run": path / "run_summary.csv",
    }
    missing = [str(file_path) for file_path in required.values() if not file_path.exists()]
    if missing:
        raise FileNotFoundError("Missing required run output file(s): " + ", ".join(missing))
    before_hashes = {name: _hash_file(file_path) for name, file_path in required.items()}
    growth = pd.read_csv(required["growth"])
    technical = pd.read_csv(required["technical"])
    summary = pd.read_csv(required["summary"])
    run_summary = pd.read_csv(required["run"])
    for table, required_columns, table_name in [
        (growth, GROWTH_REQUIRED, "blank_corrected_growth_data.csv"),
        (technical, TECHNICAL_REQUIRED, "technical_well_growth_metrics.csv"),
        (summary, SUMMARY_REQUIRED, "biological_replicate_summary.csv"),
        (run_summary, RUN_REQUIRED, "run_summary.csv"),
    ]:
        missing_columns = required_columns.difference(table.columns)
        if missing_columns:
            raise DrugScreeningStatisticsError(
                _format_errors([f"{path / table_name} is missing columns: {sorted(missing_columns)}"])
            )
    for table in [growth, technical, summary, run_summary]:
        species_values = sorted(table["species"].dropna().astype(str).unique().tolist())
        replicate_values = sorted(table["biological_replicate"].dropna().astype(str).unique().tolist())
        if species_values != [expected_species] or replicate_values != [expected_replicate]:
            raise DrugScreeningStatisticsError(
                _format_errors(
                    [
                        f"{path} expected {expected_species} {expected_replicate}, "
                        f"found species={species_values}, biological_replicate={replicate_values}."
                    ]
                )
            )
    return growth, technical, summary, run_summary, before_hashes


def read_all_runs(input_dirs: Dict[Tuple[str, str], Path]) -> dict:
    growth_frames = []
    technical_frames = []
    summary_frames = []
    run_frames = []
    hashes: Dict[Path, str] = {}
    for (species, replicate), path in input_dirs.items():
        growth, technical, summary, run_summary, before_hashes = read_run_directory(path, species, replicate)
        growth_frames.append(growth.assign(source_run_dir=str(path)))
        technical_frames.append(technical.assign(source_run_dir=str(path)))
        summary_frames.append(summary.assign(source_run_dir=str(path)))
        run_frames.append(run_summary.assign(source_run_dir=str(path)))
        for name, digest in before_hashes.items():
            file_path = {
                "growth": path / "blank_corrected_growth_data.csv",
                "technical": path / "technical_well_growth_metrics.csv",
                "summary": path / "biological_replicate_summary.csv",
                "run": path / "run_summary.csv",
            }[name]
            hashes[file_path] = digest
    return {
        "growth": pd.concat(growth_frames, ignore_index=True),
        "technical": pd.concat(technical_frames, ignore_index=True),
        "summary": pd.concat(summary_frames, ignore_index=True),
        "runs": pd.concat(run_frames, ignore_index=True),
        "hashes": hashes,
    }


def validate_loaded_data(growth: pd.DataFrame, technical: pd.DataFrame, summary: pd.DataFrame, runs: pd.DataFrame) -> List[str]:
    errors: List[str] = []
    warnings: List[str] = []
    for frame in [growth, technical, summary, runs]:
        frame["species"] = frame["species"].astype(str)
        frame["biological_replicate"] = frame["biological_replicate"].astype(str)
    for numeric in ["time_min", "time_h", "raw_OD600", "corrected_OD600"]:
        if numeric in growth.columns:
            growth[numeric] = pd.to_numeric(growth[numeric], errors="coerce")
    for numeric in [RAW_AUC, REL_AUC, KZ, TLAG, "gompertz_R2"]:
        technical[numeric] = pd.to_numeric(technical[numeric], errors="coerce")
    for numeric in [f"{RAW_AUC}_mean", f"{REL_AUC}_mean", f"{KZ}_mean", f"{TLAG}_mean", "n_technical_wells"]:
        summary[numeric] = pd.to_numeric(summary[numeric], errors="coerce")

    for species in EXPECTED_SPECIES:
        observed_reps = sorted(runs.loc[runs["species"].eq(species), "biological_replicate"].unique().tolist())
        if observed_reps != EXPECTED_REPLICATES:
            errors.append(f"{species} must contain BR1, BR2 and BR3 exactly once; found {observed_reps}.")
        for replicate in EXPECTED_REPLICATES:
            run_rows = runs[(runs["species"].eq(species)) & (runs["biological_replicate"].eq(replicate))]
            if len(run_rows) != 1:
                errors.append(f"{species} {replicate} must have exactly one run_summary row; found {len(run_rows)}.")
                continue
            row = run_rows.iloc[0]
            if int(row["analysis_endpoint_min"]) != MAX_TIME_MIN:
                errors.append(f"{species} {replicate} AUC endpoint must be {MAX_TIME_MIN} min.")
            if int(row["number_of_retained_time_points"]) != len(TIMEPOINTS):
                errors.append(f"{species} {replicate} must retain {len(TIMEPOINTS)} time points.")

            gr = growth[(growth["species"].eq(species)) & (growth["biological_replicate"].eq(replicate))]
            tech = technical[(technical["species"].eq(species)) & (technical["biological_replicate"].eq(replicate))]
            summ = summary[(summary["species"].eq(species)) & (summary["biological_replicate"].eq(replicate))]
            if len(tech) != 93:
                errors.append(f"{species} {replicate} must contain 93 technical nonblank wells; found {len(tech)}.")
            if len(gr) != 93 * len(TIMEPOINTS):
                errors.append(f"{species} {replicate} must contain {93 * len(TIMEPOINTS)} growth rows; found {len(gr)}.")
            for well, well_data in gr.groupby("well", sort=False):
                observed_times = well_data.sort_values("time_min")["time_min"].astype(int).tolist()
                if observed_times != TIMEPOINTS:
                    errors.append(f"{species} {replicate} well {well} does not contain the expected retained time points.")
            counts = tech.groupby("condition_id")["well"].nunique()
            bad_counts = counts[counts.ne(EXPECTED_TECHNICAL_WELLS)]
            if not bad_counts.empty:
                errors.append(f"{species} {replicate} technical-well counts are not three per condition: {bad_counts.to_dict()}.")
            summary_bad = summ[summ["n_technical_wells"].ne(EXPECTED_TECHNICAL_WELLS)]
            if not summary_bad.empty:
                errors.append(f"{species} {replicate} summary rows do not all report three technical wells.")
            vehicle = tech[tech["condition_id"].eq("vehicle_1pct_DMSO")]
            if not math.isclose(float(vehicle[REL_AUC].mean()), 100.0, rel_tol=1e-9, abs_tol=1e-9):
                errors.append(f"{species} {replicate} vehicle relative-AUC mean is not 100%.")

    duplicate_tech = technical.duplicated(["species", "biological_replicate", "well"], keep=False)
    if duplicate_tech.any():
        errors.append("Duplicate technical-well metric rows are present.")
    duplicate_growth = growth.duplicated(["species", "biological_replicate", "well", "time_min"], keep=False)
    if duplicate_growth.any():
        errors.append("Duplicate growth measurements are present for the same well and time.")
    definitions = technical.drop_duplicates(
        ["species", "condition_id", "treatment_type", "compound", "compound_concentration_uM", "ampicillin_concentration_ug_mL"]
    )
    inconsistent = definitions.groupby(["species", "condition_id"], dropna=False).size().loc[lambda values: values.gt(1)]
    if not inconsistent.empty:
        errors.append(f"Condition IDs map to inconsistent treatment definitions: {inconsistent.to_dict()}.")

    compare_summary_to_technical(technical, summary, errors)
    if errors:
        raise DrugScreeningStatisticsError(_format_errors(errors))
    if (technical[RAW_AUC] < 0).any():
        warnings.append("Negative raw AUC values are present and retained unchanged.")
    if (technical[REL_AUC] < 0).any() or (technical[REL_AUC] > 100).any():
        warnings.append("Relative AUC values outside 0-100% are present and retained unchanged.")
    return warnings


def compare_summary_to_technical(technical: pd.DataFrame, summary: pd.DataFrame, errors: List[str]) -> None:
    checks = [
        (RAW_AUC, f"{RAW_AUC}_mean"),
        (REL_AUC, f"{REL_AUC}_mean"),
        (KZ, f"{KZ}_mean"),
        (TLAG, f"{TLAG}_mean"),
    ]
    grouped = technical.groupby(["species", "biological_replicate", "condition_id"], dropna=False)
    for (species, replicate, condition_id), tech_group in grouped:
        summary_row = summary[
            summary["species"].eq(species)
            & summary["biological_replicate"].eq(replicate)
            & summary["condition_id"].eq(condition_id)
        ]
        if len(summary_row) != 1:
            errors.append(f"Missing or duplicate summary row for {species} {replicate} {condition_id}.")
            continue
        summary_row = summary_row.iloc[0]
        for technical_column, summary_column in checks:
            expected = float(pd.to_numeric(tech_group[technical_column], errors="coerce").mean())
            observed = float(summary_row[summary_column])
            if not math.isclose(expected, observed, rel_tol=1e-8, abs_tol=1e-8):
                errors.append(
                    f"Summary {summary_column} does not match technical mean for "
                    f"{species} {replicate} {condition_id}: {observed} vs {expected}."
                )


def build_family_specs() -> List[FamilySpec]:
    specs: List[FamilySpec] = []
    for species in EXPECTED_SPECIES:
        for compound in SCHIFF_COMPOUNDS:
            specs.append(FamilySpec(species, compound, "schiff", FAMILY_ORDER[f"schiff_{compound}"]))
        specs.append(FamilySpec(species, "PenAg", "penag", FAMILY_ORDER["penag_PenAg"]))
    return specs


def prepare_biological_replicate_metrics(summary: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "run_id",
        "biological_replicate",
        "species",
        "treatment_type",
        "compound",
        "compound_concentration_uM",
        "ampicillin_concentration_ug_mL",
        "condition_id",
        "condition",
        "n_technical_wells",
        f"{RAW_AUC}_mean",
        f"{REL_AUC}_mean",
        f"{KZ}_mean",
        f"{TLAG}_mean",
        "n_valid_gompertz_fits",
        "n_failed_gompertz_fits",
    ]
    result = summary[columns].copy()
    result = result.rename(
        columns={
            f"{RAW_AUC}_mean": "raw_AUC_mean",
            f"{REL_AUC}_mean": "relative_AUC_mean",
            f"{KZ}_mean": "Kz_mean",
            f"{TLAG}_mean": "TLag_mean",
        }
    )
    result["eligible_for_AUC_inference"] = False
    result["eligible_for_Kz_inference"] = False
    result["eligible_for_TLag_inference"] = False
    return result


def annotate_biological_metric_eligibility(br_metrics: pd.DataFrame, anova: pd.DataFrame) -> pd.DataFrame:
    result = br_metrics.copy()
    if anova.empty:
        return result
    for row in anova.itertuples(index=False):
        family = FamilySpec(row.species, row.compound, row.family_type, FAMILY_ORDER[f"{row.family_type}_{row.compound}"])
        column = {
            "AUC_0_1045_min": "eligible_for_AUC_inference",
            "Kz": "eligible_for_Kz_inference",
            "TLag": "eligible_for_TLag_inference",
        }[row.metric]
        mask = result["species"].eq(row.species) & result["condition_id"].isin(family.condition_ids)
        result.loc[mask, column] = True
    return result


def metric_column(metric: str) -> str:
    if metric == "AUC_0_1045_min":
        return "raw_AUC_mean"
    if metric == "Kz":
        return "Kz_mean"
    if metric == "TLag":
        return "TLag_mean"
    raise KeyError(metric)


def family_dataset(br_metrics: pd.DataFrame, family: FamilySpec, metric: str) -> pd.DataFrame:
    value_column = metric_column(metric)
    data = br_metrics[
        br_metrics["species"].eq(family.species)
        & br_metrics["condition_id"].isin(family.condition_ids)
    ].copy()
    data["condition_id"] = pd.Categorical(data["condition_id"], categories=family.condition_ids, ordered=True)
    data["biological_replicate"] = pd.Categorical(data["biological_replicate"], categories=EXPECTED_REPLICATES, ordered=True)
    data["value"] = pd.to_numeric(data[value_column], errors="coerce")
    return data.sort_values(["condition_id", "biological_replicate"], kind="stable").reset_index(drop=True)


def family_eligibility(br_metrics: pd.DataFrame, family: FamilySpec, metric: str) -> Tuple[bool, str, pd.DataFrame]:
    data = family_dataset(br_metrics, family, metric)
    value_column = metric_column(metric)
    expected_rows = len(family.condition_ids) * len(EXPECTED_REPLICATES)
    if len(data) != expected_rows:
        return False, f"Incomplete block family: expected {expected_rows} condition-by-BR means, found {len(data)}.", data
    counts = data.groupby(["condition_id", "biological_replicate"], observed=False).size()
    if not counts.eq(1).all():
        return False, "Every condition-by-biological-replicate cell must contain one BR mean.", data
    if data["value"].isna().any():
        return False, f"{value_column} contains missing or non-estimable values.", data
    if metric in {"Kz", "TLag"}:
        bad_fits = data[pd.to_numeric(data["n_valid_gompertz_fits"], errors="coerce").lt(EXPECTED_TECHNICAL_WELLS)]
        if not bad_fits.empty:
            return False, "At least one condition has fewer than three valid, interpretable Gompertz fits.", data
    return True, "eligible", data


def randomized_block_anova(data: pd.DataFrame, family: FamilySpec, metric: str) -> AnovaResult:
    """Fit condition and biological-replicate block effects to BR means."""
    t = len(family.condition_ids)
    b = len(EXPECTED_REPLICATES)
    values = data["value"].astype(float)
    grand_mean = float(values.mean())
    condition_means = data.groupby("condition_id", observed=False)["value"].mean().reindex(family.condition_ids)
    block_means = data.groupby("biological_replicate", observed=False)["value"].mean().reindex(EXPECTED_REPLICATES)
    total_ss = float(((values - grand_mean) ** 2).sum())
    condition_ss = float(b * ((condition_means - grand_mean) ** 2).sum())
    block_ss = float(t * ((block_means - grand_mean) ** 2).sum())
    residual_ss = float(total_ss - condition_ss - block_ss)
    if abs(residual_ss) < 1e-12:
        residual_ss = 0.0
    condition_df = t - 1
    block_df = b - 1
    residual_df = condition_df * block_df
    condition_ms = condition_ss / condition_df
    block_ms = block_ss / block_df
    residual_ms = residual_ss / residual_df
    condition_f = condition_ms / residual_ms if residual_ms > 0 else math.inf
    block_f = block_ms / residual_ms if residual_ms > 0 else math.inf
    condition_p = float(stats.f.sf(condition_f, condition_df, residual_df))
    block_p = float(stats.f.sf(block_f, block_df, residual_df))
    eta = condition_ss / (condition_ss + residual_ss) if (condition_ss + residual_ss) else math.nan
    return AnovaResult(
        species=family.species,
        compound=family.compound,
        family_type=family.family_type,
        metric=metric,
        number_of_conditions=t,
        number_of_biological_replicates=b,
        total_model_observations=len(data),
        condition_SS=condition_ss,
        block_SS=block_ss,
        residual_SS=residual_ss,
        condition_df=condition_df,
        block_df=block_df,
        residual_df=residual_df,
        condition_MS=condition_ms,
        block_MS=block_ms,
        residual_MS=residual_ms,
        condition_F=condition_f,
        condition_p_value=condition_p,
        block_F=block_f,
        block_p_value=block_p,
        partial_eta_squared=eta,
        model_status="fitted",
    )


def tukey_from_block(data: pd.DataFrame, anova: AnovaResult, family: FamilySpec) -> pd.DataFrame:
    """Calculate all-pair Tukey tests from the blocked-model residual."""
    means = data.groupby("condition_id", observed=False)["value"].mean().reindex(family.condition_ids)
    t = anova.number_of_conditions
    b = anova.number_of_biological_replicates
    mse = anova.residual_MS
    tukey_se = math.sqrt(mse / b) if mse >= 0 else math.nan
    q_crit = float(stats.studentized_range.ppf(0.95, t, anova.residual_df))
    rows = []
    for first, second in itertools.combinations(family.condition_ids, 2):
        mean_1 = float(means.loc[first])
        mean_2 = float(means.loc[second])
        diff = mean_1 - mean_2
        q = abs(diff) / tukey_se if tukey_se and tukey_se > 0 else math.inf
        p = float(stats.studentized_range.sf(q, t, anova.residual_df))
        half_width = q_crit * tukey_se if np.isfinite(tukey_se) else math.nan
        rows.append(
            {
                "species": family.species,
                "compound": family.compound,
                "family_type": family.family_type,
                "metric": anova.metric,
                "condition_1": first,
                "condition_2": second,
                "mean_condition_1": mean_1,
                "mean_condition_2": mean_2,
                "mean_difference_condition_1_minus_condition_2": diff,
                "Tukey_standard_error": tukey_se,
                "q_statistic": float(q),
                "residual_df": int(anova.residual_df),
                "number_of_groups": int(t),
                "Tukey_adjusted_p_value": p,
                "simultaneous_95CI_lower": float(diff - half_width),
                "simultaneous_95CI_upper": float(diff + half_width),
                "significance_label": significance_label(p),
                "statistically_significant": bool(p < 0.05),
            }
        )
    return pd.DataFrame.from_records(rows)


def run_inferential_models(br_metrics: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, List[str]]:
    anova_rows = []
    tukey_frames = []
    letter_frames = []
    exclusions = []
    diagnostics = []
    for family in build_family_specs():
        for metric in METRICS:
            eligible, reason, data = family_eligibility(br_metrics, family, metric)
            if not eligible:
                exclusions.append(
                    {
                        "species": family.species,
                        "compound": family.compound,
                        "family_type": family.family_type,
                        "metric": metric,
                        "condition_id": "",
                        "exclusion_reason": reason,
                    }
                )
                continue
            anova = randomized_block_anova(data, family, metric)
            anova_rows.append(anova.to_dict())
            tukey = tukey_from_block(data, anova, family)
            tukey_frames.append(tukey)
            letters = compact_letter_display(tukey, family.condition_ids)
            validate_compact_letters(tukey, letters)
            letter_frames.append(
                letters.assign(
                    species=family.species,
                    compound=family.compound,
                    family_type=family.family_type,
                    metric=metric,
                )
            )
            diagnostics.append(model_diagnostics(data, anova, family, metric))

    anova_df = pd.DataFrame.from_records(anova_rows)
    tukey_df = pd.concat(tukey_frames, ignore_index=True) if tukey_frames else pd.DataFrame()
    letters_df = pd.concat(letter_frames, ignore_index=True) if letter_frames else pd.DataFrame()
    diagnostics_df = pd.concat(diagnostics, ignore_index=True) if diagnostics else pd.DataFrame()
    return anova_df, tukey_df, letters_df, diagnostics_df, exclusions


def compact_letter_display(tukey: pd.DataFrame, condition_ids: Sequence[str]) -> pd.DataFrame:
    significant = {
        frozenset((row.condition_1, row.condition_2))
        for row in tukey.itertuples(index=False)
        if bool(row.statistically_significant)
    }
    nonsig_pairs = {
        frozenset((row.condition_1, row.condition_2))
        for row in tukey.itertuples(index=False)
        if not bool(row.statistically_significant)
    }
    all_cliques: List[Tuple[str, ...]] = []
    for size in range(len(condition_ids), 0, -1):
        for combo in itertools.combinations(condition_ids, size):
            pairs = [frozenset(pair) for pair in itertools.combinations(combo, 2)]
            if not any(pair in significant for pair in pairs):
                all_cliques.append(combo)

    uncovered_pairs = set(nonsig_pairs)
    uncovered_singles = set(condition_ids)
    selected: List[Tuple[str, ...]] = []
    for clique in all_cliques:
        clique_pairs = {frozenset(pair) for pair in itertools.combinations(clique, 2)}
        if clique_pairs & uncovered_pairs:
            selected.append(clique)
            uncovered_pairs -= clique_pairs
            uncovered_singles -= set(clique)
    for condition_id in list(uncovered_singles):
        selected.append((condition_id,))

    alphabet = list("abcdefghijklmnopqrstuvwxyz")
    letter_map = {condition_id: "" for condition_id in condition_ids}
    for index, clique in enumerate(selected):
        letter = alphabet[index] if index < len(alphabet) else f"L{index + 1}"
        for condition_id in clique:
            letter_map[condition_id] += letter
    return pd.DataFrame(
        {
            "condition_id": list(condition_ids),
            "compact_letter": [letter_map[condition_id] for condition_id in condition_ids],
        }
    )


def validate_compact_letters(tukey: pd.DataFrame, letters: pd.DataFrame) -> None:
    lookup = letters.set_index("condition_id")["compact_letter"].to_dict()
    errors = []
    for row in tukey.itertuples(index=False):
        shared = bool(set(lookup[row.condition_1]) & set(lookup[row.condition_2]))
        if bool(row.statistically_significant) and shared:
            errors.append(f"{row.condition_1} and {row.condition_2} are significant but share a compact letter.")
        if not bool(row.statistically_significant) and not shared:
            errors.append(f"{row.condition_1} and {row.condition_2} are nonsignificant but share no compact letter.")
    if errors:
        raise DrugScreeningStatisticsError(_format_errors(errors))


def model_diagnostics(data: pd.DataFrame, anova: AnovaResult, family: FamilySpec, metric: str) -> pd.DataFrame:
    y = data["value"].to_numpy(dtype=float)
    condition_dummies = pd.get_dummies(data["condition_id"].astype(str), drop_first=True)
    block_dummies = pd.get_dummies(data["biological_replicate"].astype(str), drop_first=True)
    x = pd.concat([pd.Series(1.0, index=data.index, name="Intercept"), condition_dummies, block_dummies], axis=1).to_numpy(dtype=float)
    beta = np.linalg.lstsq(x, y, rcond=None)[0]
    fitted = x @ beta
    residual = y - fitted
    xtx_inv = np.linalg.pinv(x.T @ x)
    hat = np.einsum("ij,jk,ik->i", x, xtx_inv, x)
    p = x.shape[1]
    mse = anova.residual_MS
    cooks = (residual**2 / (p * mse)) * (hat / ((1 - hat) ** 2)) if mse > 0 else np.full_like(residual, np.nan)
    threshold = 4 / max(len(data) - p, 1)
    return pd.DataFrame(
        {
            "species": family.species,
            "compound": family.compound,
            "family_type": family.family_type,
            "metric": metric,
            "condition_id": data["condition_id"].astype(str).to_numpy(),
            "biological_replicate": data["biological_replicate"].astype(str).to_numpy(),
            "fitted_value": fitted,
            "residual": residual,
            "hat_value": hat,
            "cooks_distance": cooks,
            "unusual_or_influential": np.abs(residual) > 2 * np.std(residual, ddof=1),
            "cooks_threshold": threshold,
            "model_status": "fitted",
            "residual_count": len(residual),
            "residual_mean": float(np.mean(residual)),
            "residual_sample_SD": float(np.std(residual, ddof=1)),
            "maximum_absolute_residual": float(np.max(np.abs(residual))),
        }
    )


def plot_model_diagnostics(diagnostics: pd.DataFrame, output_dir: Path) -> List[Path]:
    diag_dir = output_dir / "model_diagnostics"
    diag_dir.mkdir(parents=True, exist_ok=True)
    paths: List[Path] = []
    if diagnostics.empty:
        return paths
    for (species, compound, metric), group in diagnostics.groupby(["species", "compound", "metric"], sort=False):
        fig, axes = plt.subplots(1, 2, figsize=(9, 4))
        axes[0].scatter(group["fitted_value"], group["residual"], color="#4c78a8", s=22, alpha=0.85)
        axes[0].axhline(0, color="black", linewidth=0.8, linestyle="--")
        axes[0].set_xlabel("Fitted value")
        axes[0].set_ylabel("Residual")
        axes[0].set_title("Residuals vs fitted")
        ordered = np.sort(group["residual"].to_numpy(dtype=float))
        probs = (np.arange(1, len(ordered) + 1) - 0.5) / len(ordered)
        theoretical = stats.norm.ppf(probs)
        axes[1].scatter(theoretical, ordered, color="#f58518", s=22, alpha=0.85)
        q_min = min(theoretical.min(), ordered.min())
        q_max = max(theoretical.max(), ordered.max())
        axes[1].plot([q_min, q_max], [q_min, q_max], color="black", linewidth=0.8, linestyle="--")
        axes[1].set_xlabel("Theoretical normal quantile")
        axes[1].set_ylabel("Observed residual")
        axes[1].set_title("Residual Q-Q")
        fig.suptitle(f"{DISPLAY_SPECIES.get(species, species)} {compound} {METRIC_DISPLAY_TITLES.get(metric, metric)}")
        fig.tight_layout()
        safe = f"{species}_{compound}_{metric}".replace("/", "_")
        png = diag_dir / f"{safe}_diagnostics.png"
        pdf = diag_dir / f"{safe}_diagnostics.pdf"
        fig.savefig(png, dpi=300)
        fig.savefig(pdf)
        plt.close(fig)
        paths.extend([png, pdf])
    return paths


def condition_definitions(technical: pd.DataFrame) -> pd.DataFrame:
    return technical.drop_duplicates(
        ["species", "treatment_type", "compound", "compound_concentration_uM", "ampicillin_concentration_ug_mL", "condition_id", "condition"]
    ).copy()


def technical_plot_values(technical: pd.DataFrame, metric: str) -> pd.DataFrame:
    column = METRICS[metric]["plot"]
    data = technical.copy()
    data["plot_value"] = pd.to_numeric(data[column], errors="coerce")
    if metric in {"Kz", "TLag"}:
        data = data[data["gompertz_fit_status"].astype(str).str.startswith("success")].copy()
    return data


def summary_plot_values(br_metrics: pd.DataFrame, metric: str) -> pd.DataFrame:
    if metric == "AUC_0_1045_min":
        column = "relative_AUC_mean"
    elif metric == "Kz":
        column = "Kz_mean"
    else:
        column = "TLag_mean"
    data = br_metrics.copy()
    data["plot_value"] = pd.to_numeric(data[column], errors="coerce")
    return data


def condition_palette(condition_ids: Sequence[str]) -> Dict[str, tuple]:
    fake_rows = []
    for condition_id in condition_ids:
        if condition_id == "vehicle_1pct_DMSO":
            fake_rows.append(
                {
                    "condition_id": condition_id,
                    "treatment_type": "vehicle_control",
                    "compound": "DMSO",
                    "compound_concentration_uM": math.nan,
                    "ampicillin_concentration_ug_mL": math.nan,
                }
            )
        elif condition_id.startswith("PenAg"):
            conc = float(condition_id.split("_")[1].replace("uM", ""))
            fake_rows.append(
                {
                    "condition_id": condition_id,
                    "treatment_type": "penag_monotherapy",
                    "compound": "PenAg",
                    "compound_concentration_uM": conc,
                    "ampicillin_concentration_ug_mL": math.nan,
                }
            )
        else:
            compound, concentration = condition_id.split("_", 1)
            fake_rows.append(
                {
                    "condition_id": condition_id,
                    "treatment_type": "compound_monotherapy",
                    "compound": compound,
                    "compound_concentration_uM": float(concentration.replace("uM", "")),
                    "ampicillin_concentration_ug_mL": math.nan,
                }
            )
    colors = condition_color_map(pd.DataFrame(fake_rows))
    return {condition_id: colors[condition_id] for condition_id in condition_ids}


def add_grouped_dot_panel(
    ax: plt.Axes,
    technical: pd.DataFrame,
    br_metrics: pd.DataFrame,
    letters: pd.DataFrame,
    family: FamilySpec,
    metric: str,
    show_reference: bool = False,
) -> None:
    tech = technical_plot_values(technical, metric)
    br = summary_plot_values(br_metrics, metric)
    tech = tech[tech["species"].eq(family.species) & tech["condition_id"].isin(family.condition_ids)].copy()
    br = br[br["species"].eq(family.species) & br["condition_id"].isin(family.condition_ids)].copy()
    definitions = condition_definitions(technical)
    palette = condition_palette(family.condition_ids)
    x_lookup = {condition_id: index for index, condition_id in enumerate(family.condition_ids)}
    rng = np.random.default_rng(25)
    for condition_id in family.condition_ids:
        values = tech[tech["condition_id"].eq(condition_id)]["plot_value"].dropna().to_numpy(dtype=float)
        if len(values):
            jitter = rng.uniform(-0.11, 0.11, len(values))
            ax.scatter(
                np.full(len(values), x_lookup[condition_id]) + jitter,
                values,
                color=palette[condition_id],
                edgecolor="none",
                s=18,
                alpha=0.65,
                zorder=2,
            )
        br_values = br[br["condition_id"].eq(condition_id)]["plot_value"].dropna().to_numpy(dtype=float)
        if len(br_values):
            mean = float(np.mean(br_values))
            sd = float(np.std(br_values, ddof=1)) if len(br_values) > 1 else 0.0
            x = x_lookup[condition_id]
            ax.errorbar(x, mean, yerr=sd, color="black", capsize=4, linewidth=1.0, zorder=4)
            ax.hlines(mean, x - 0.18, x + 0.18, color="black", linewidth=1.5, zorder=5)
    if not letters.empty:
        letter_lookup = {
            row.condition_id: row.compact_letter
            for row in letters[
                letters["species"].eq(family.species)
                & letters["compound"].eq(family.compound)
                & letters["metric"].eq(metric)
            ].itertuples(index=False)
        }
        y_values = pd.concat([tech["plot_value"], br["plot_value"]], ignore_index=True).dropna()
        if not y_values.empty:
            y_min = float(y_values.min())
            y_max = float(y_values.max())
            span = max(y_max - y_min, abs(y_max) * 0.1, 1e-6)
            for condition_id, letter in letter_lookup.items():
                ax.text(x_lookup[condition_id], y_max + span * 0.07, letter, ha="center", va="bottom", fontsize=9)
    if show_reference:
        ax.axhline(100, color="dimgray", linewidth=0.8, linestyle="--", zorder=1)
    y_values = pd.concat([tech["plot_value"], br["plot_value"]], ignore_index=True).dropna()
    if not y_values.empty:
        y_min = float(y_values.min())
        y_max = float(y_values.max())
        span = max(y_max - y_min, abs(y_max) * 0.1, 1e-6)
        ax.set_ylim(y_min - span * 0.12, y_max + span * 0.24)
    ax.set_xticks(range(len(family.condition_ids)))
    ax.set_xticklabels([_condition_x_label(condition_id, definitions) for condition_id in family.condition_ids], rotation=45, ha="right")
    ax.grid(axis="y", alpha=0.22)


def harmonize_y_limits(axes: Sequence[plt.Axes]) -> None:
    bottoms = []
    tops = []
    for ax in axes:
        bottom, top = ax.get_ylim()
        bottoms.append(bottom)
        tops.append(top)
    if not bottoms:
        return
    common_bottom = min(bottoms)
    common_top = max(tops)
    for ax in axes:
        ax.set_ylim(common_bottom, common_top)


def plot_schiff_figures(technical: pd.DataFrame, br_metrics: pd.DataFrame, letters: pd.DataFrame, output_dir: Path) -> List[Path]:
    paths: List[Path] = []
    specs = [spec for spec in build_family_specs() if spec.family_type == "schiff"]
    for metric, stem, ylabel in [
        ("AUC_0_1045_min", "schiff_base_relative_auc_statistics", METRICS["AUC_0_1045_min"]["label"]),
        ("Kz", "schiff_base_growth_rate_statistics", METRICS["Kz"]["label"]),
        ("TLag", "schiff_base_lag_time_statistics", METRICS["TLag"]["label"]),
    ]:
        fig, axes = plt.subplots(2, 4, figsize=(17, 9), sharey=False)
        for ax, family in zip(axes.flatten(), specs):
            add_grouped_dot_panel(
                ax,
                technical,
                br_metrics,
                letters,
                family,
                metric,
                show_reference=metric == "AUC_0_1045_min",
            )
            ax.set_title(f"{DISPLAY_SPECIES.get(family.species, family.species)} {family.compound}")
            ax.set_xlabel("Concentration (uM)")
            ax.set_ylabel(ylabel)
        harmonize_y_limits(axes.flatten())
        if metric == "AUC_0_1045_min":
            letter_note = "Compact letters are derived from raw-AUC randomized-block Tukey comparisons."
        else:
            letter_note = "Compact letters are derived from randomized-block Tukey comparisons for the plotted metric."
        note = (
            "Small points are technical wells. Inference uses n=3 biological replicates after within-run averaging; "
            "overall mean and SD are calculated from the three biological-replicate means. "
            f"{letter_note}"
        )
        fig.text(0.5, 0.02, fill(note, 140), ha="center", fontsize=9)
        fig.tight_layout(rect=[0, 0.06, 1, 0.98])
        png = output_dir / f"{stem}.png"
        pdf = output_dir / f"{stem}.pdf"
        fig.savefig(png, dpi=300)
        fig.savefig(pdf)
        plt.close(fig)
        paths.extend([png, pdf])
    return paths


def plot_penag_figure(technical: pd.DataFrame, br_metrics: pd.DataFrame, letters: pd.DataFrame, output_dir: Path) -> List[Path]:
    fig, axes = plt.subplots(2, 3, figsize=(14, 8.5), sharey=False)
    paths: List[Path] = []
    for row_index, species in enumerate(["E_coli", "B_subtilis"]):
        family = FamilySpec(species, "PenAg", "penag", FAMILY_ORDER["penag_PenAg"])
        for col_index, metric in enumerate(["AUC_0_1045_min", "Kz", "TLag"]):
            ax = axes[row_index, col_index]
            add_grouped_dot_panel(
                ax,
                technical,
                br_metrics,
                letters,
                family,
                metric,
                show_reference=metric == "AUC_0_1045_min",
            )
            ax.set_title(f"{DISPLAY_SPECIES.get(species, species)} {METRIC_DISPLAY_TITLES[metric]}")
            ax.set_xlabel("Concentration (uM)")
            ax.set_ylabel(METRICS[metric]["label"])
    for col_index in range(3):
        harmonize_y_limits([axes[0, col_index], axes[1, col_index]])
    note = (
        "Small points are technical wells. Inference uses n=3 biological replicates after within-run averaging; "
        "overall mean and SD are calculated from biological-replicate means. Relative-AUC letters use raw-AUC inference; "
        "growth-rate and lag-time letters use their respective blocked models."
    )
    fig.text(0.5, 0.02, fill(note, 130), ha="center", fontsize=9)
    fig.tight_layout(rect=[0, 0.07, 1, 0.98])
    png = output_dir / "penag_growth_metrics_statistics.png"
    pdf = output_dir / "penag_growth_metrics_statistics.pdf"
    fig.savefig(png, dpi=300)
    fig.savefig(pdf)
    plt.close(fig)
    paths.extend([png, pdf])
    return paths


def compound_selection_heatmap(br_metrics: pd.DataFrame, output_dir: Path) -> Tuple[pd.DataFrame, List[Path]]:
    rows = []
    for species in EXPECTED_SPECIES:
        for compound in SCHIFF_COMPOUNDS + ["PenAg"]:
            for concentration in [12.5, 25.0, 50.0, 100.0]:
                subset = br_metrics[
                    br_metrics["species"].eq(species)
                    & br_metrics["compound"].eq(compound)
                    & pd.to_numeric(br_metrics["compound_concentration_uM"], errors="coerce").eq(concentration)
                    & br_metrics["treatment_type"].isin(["compound_monotherapy", "penag_monotherapy"])
                ]
                rows.append(
                    {
                        "species": species,
                        "compound": compound,
                        "compound_concentration_uM": concentration,
                        "mean_relative_AUC_percent": float(subset["relative_AUC_mean"].mean()),
                        "n_biological_replicates": int(subset["relative_AUC_mean"].count()),
                    }
                )
    table = pd.DataFrame.from_records(rows)
    fig, axes = plt.subplots(1, 2, figsize=(11, 5), sharex=True, sharey=True)
    vmin = float(table["mean_relative_AUC_percent"].min())
    vmax = float(table["mean_relative_AUC_percent"].max())
    for ax, species in zip(axes, EXPECTED_SPECIES):
        pivot = table[table["species"].eq(species)].pivot(
            index="compound", columns="compound_concentration_uM", values="mean_relative_AUC_percent"
        ).reindex(SCHIFF_COMPOUNDS + ["PenAg"])
        im = ax.imshow(pivot.to_numpy(dtype=float), cmap="viridis", vmin=vmin, vmax=vmax, aspect="auto")
        ax.set_title(DISPLAY_SPECIES.get(species, species))
        ax.set_xticks(range(len(pivot.columns)))
        ax.set_xticklabels([f"{value:g}" for value in pivot.columns])
        ax.set_yticks(range(len(pivot.index)))
        ax.set_yticklabels(pivot.index)
        ax.set_xlabel("Concentration (uM)")
        for i, compound in enumerate(pivot.index):
            for j, concentration in enumerate(pivot.columns):
                value = pivot.loc[compound, concentration]
                ax.text(j, i, f"{value:.0f}", ha="center", va="center", color="white" if value < (vmin + vmax) / 2 else "black", fontsize=8)
    cbar = fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.9)
    cbar.set_label("Relative AUC (% of vehicle)")
    fig.suptitle("Compound-selection relative AUC heatmap")
    png = output_dir / "compound_selection_relative_auc_heatmap.png"
    pdf = output_dir / "compound_selection_relative_auc_heatmap.pdf"
    fig.savefig(png, dpi=300)
    fig.savefig(pdf)
    plt.close(fig)
    return table, [png, pdf]


def fit_availability_summary(technical: pd.DataFrame, br_metrics: pd.DataFrame) -> pd.DataFrame:
    records = []
    for keys, group in technical.groupby(["species", "biological_replicate", "condition_id", "condition"], dropna=False):
        species, replicate, condition_id, condition = keys
        status = group["gompertz_fit_status"].astype(str)
        records.append(
            {
                "species": species,
                "biological_replicate": replicate,
                "condition_id": condition_id,
                "condition": condition,
                "n_technical_wells": int(group["well"].nunique()),
                "n_success_or_warning_fits": int(status.str.startswith("success").sum()),
                "n_failed_fits": int((~status.str.startswith("success")).sum()),
                "n_Kz_values": int(pd.to_numeric(group[KZ], errors="coerce").count()),
                "n_TLag_values": int(pd.to_numeric(group[TLAG], errors="coerce").count()),
            }
        )
    return pd.DataFrame.from_records(records)


def od600_linearity_qc(growth: pd.DataFrame) -> pd.DataFrame:
    records = []
    for keys, group in growth.groupby(["run_id", "species", "biological_replicate", "condition_id", "condition", "well"], dropna=False):
        run_id, species, replicate, condition_id, condition, well = keys
        raw = pd.to_numeric(group["raw_OD600"], errors="coerce")
        corrected = pd.to_numeric(group["corrected_OD600"], errors="coerce")
        reached = raw >= 1.0
        records.append(
            {
                "run_id": run_id,
                "species": species,
                "biological_replicate": replicate,
                "condition_id": condition_id,
                "condition": condition,
                "well": well,
                "raw_OD600_threshold_reached": bool(reached.any()),
                "n_raw_measurements_ge_1": int(reached.sum()),
                "proportion_raw_measurements_ge_1": float(reached.mean()),
                "maximum_raw_OD600": float(raw.max()),
                "first_time_min_raw_OD600_ge_1": float(group.loc[reached, "time_min"].min()) if reached.any() else math.nan,
                "n_corrected_measurements_ge_1": int((corrected >= 1.0).sum()),
                "maximum_corrected_OD600": float(corrected.max()),
                "action": "flagged_preserved_not_excluded",
            }
        )
    return pd.DataFrame.from_records(records)


def combination_condition_sets(technical: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    def valid_metric_values(subset: pd.DataFrame, metric: str, column: str) -> pd.Series:
        metric_subset = subset
        if metric in {"Kz", "TLag"}:
            metric_subset = subset[subset["gompertz_fit_status"].astype(str).str.startswith("success")]
        return pd.to_numeric(metric_subset[column], errors="coerce")

    def valid_metric_mean(subset: pd.DataFrame, metric: str, column: str) -> float:
        values = valid_metric_values(subset, metric, column)
        return float(values.mean())

    descriptive_rows = []
    delta_rows = []
    exclusion_rows = []
    for species in EXPECTED_SPECIES:
        for replicate in EXPECTED_REPLICATES:
            run = technical[technical["species"].eq(species) & technical["biological_replicate"].eq(replicate)]
            vehicle = run[run["condition_id"].eq("vehicle_1pct_DMSO")]
            for compound in SCHIFF_COMPOUNDS:
                combo = run[(run["treatment_type"].eq("combination")) & (run["compound"].eq(compound))]
                if combo.empty:
                    exclusion_rows.append({"species": species, "biological_replicate": replicate, "compound": compound, "reason": "Missing combination condition."})
                    continue
                combo_def = combo.iloc[0]
                compound_conc = float(combo_def["compound_concentration_uM"])
                amp_conc = float(combo_def["ampicillin_concentration_ug_mL"])
                compound_alone = run[
                    run["treatment_type"].eq("compound_monotherapy")
                    & run["compound"].eq(compound)
                    & pd.to_numeric(run["compound_concentration_uM"], errors="coerce").eq(compound_conc)
                ]
                amp_alone = run[
                    run["treatment_type"].eq("ampicillin_control")
                    & pd.to_numeric(run["ampicillin_concentration_ug_mL"], errors="coerce").eq(amp_conc)
                ]
                parts = {
                    "vehicle": vehicle,
                    "compound_alone": compound_alone,
                    "ampicillin_alone": amp_alone,
                    "combination": combo,
                }
                for label, subset in parts.items():
                    if subset["well"].nunique() != EXPECTED_TECHNICAL_WELLS:
                        exclusion_rows.append(
                            {
                                "species": species,
                                "biological_replicate": replicate,
                                "compound": compound,
                                "reason": f"Missing matched {label} condition for {compound_conc:g} uM compound and {amp_conc:g} ug/mL ampicillin.",
                            }
                        )
                if any(subset["well"].nunique() != EXPECTED_TECHNICAL_WELLS for subset in parts.values()):
                    continue
                for label, subset in parts.items():
                    for metric, column in [("relative_AUC", REL_AUC), ("raw_AUC", RAW_AUC), ("Kz", KZ), ("TLag", TLAG)]:
                        values = valid_metric_values(subset, metric, column)
                        descriptive_rows.append(
                            {
                                "species": species,
                                "biological_replicate": replicate,
                                "compound": compound,
                                "compound_concentration_uM": compound_conc,
                                "ampicillin_concentration_ug_mL": amp_conc,
                                "comparison_condition": label,
                                "metric": metric,
                                "n_technical_wells": int(values.count()),
                                "technical_mean": float(values.mean()),
                                "technical_SD": float(values.std(ddof=1)),
                                "condition_id": subset["condition_id"].iloc[0],
                            }
                        )
                mean_combo = valid_metric_mean(combo, "relative_AUC", REL_AUC)
                mean_amp = valid_metric_mean(amp_alone, "relative_AUC", REL_AUC)
                raw_combo = valid_metric_mean(combo, "raw_AUC", RAW_AUC)
                raw_amp = valid_metric_mean(amp_alone, "raw_AUC", RAW_AUC)
                kz_combo = valid_metric_mean(combo, "Kz", KZ)
                kz_amp = valid_metric_mean(amp_alone, "Kz", KZ)
                tlag_combo = valid_metric_mean(combo, "TLag", TLAG)
                tlag_amp = valid_metric_mean(amp_alone, "TLag", TLAG)
                delta = mean_combo - mean_amp
                if delta > 0:
                    interpretation = "possible attenuation of ampicillin activity"
                elif delta < 0:
                    interpretation = "greater growth inhibition than ampicillin alone"
                else:
                    interpretation = "similar to ampicillin alone"
                delta_rows.append(
                    {
                        "species": species,
                        "biological_replicate": replicate,
                        "compound": compound,
                        "compound_concentration_uM": compound_conc,
                        "ampicillin_concentration_ug_mL": amp_conc,
                        "delta_relative_AUC_vs_ampicillin": delta,
                        "delta_raw_AUC_vs_ampicillin": raw_combo - raw_amp,
                        "delta_Kz_vs_ampicillin": kz_combo - kz_amp,
                        "delta_TLag_vs_ampicillin": tlag_combo - tlag_amp,
                        "interpretation": interpretation,
                    }
                )
    return (
        pd.DataFrame.from_records(descriptive_rows),
        pd.DataFrame.from_records(delta_rows),
        pd.DataFrame.from_records(exclusion_rows),
    )


def plot_combination_metric(descriptive: pd.DataFrame, technical: pd.DataFrame, metric: str, species: str, output_dir: Path) -> List[Path]:
    plot_data = descriptive[(descriptive["species"].eq(species)) & (descriptive["metric"].eq(metric))].copy()
    labels = ["vehicle", "compound_alone", "ampicillin_alone", "combination"]
    colors = {
        "vehicle": "#4c78a8",
        "compound_alone": "#f58518",
        "ampicillin_alone": "#54a24b",
        "combination": "#b279a2",
    }
    metric_source = {"relative_AUC": REL_AUC, "raw_AUC": RAW_AUC, "Kz": KZ, "TLag": TLAG}[metric]
    fig, axes = plt.subplots(3, 4, figsize=(16, 10), sharey=False)
    rng = np.random.default_rng(1045)
    for row_index, replicate in enumerate(EXPECTED_REPLICATES):
        for col_index, compound in enumerate(SCHIFF_COMPOUNDS):
            ax = axes[row_index, col_index]
            panel = plot_data[plot_data["biological_replicate"].eq(replicate) & plot_data["compound"].eq(compound)]
            for x, label in enumerate(labels):
                row = panel[panel["comparison_condition"].eq(label)]
                if row.empty:
                    ax.text(x, 0.5, "Not estimable", rotation=90, ha="center", va="center", fontsize=7)
                    continue
                row = row.iloc[0]
                source_values = technical[
                    technical["species"].eq(species)
                    & technical["biological_replicate"].eq(replicate)
                    & technical["condition_id"].eq(row["condition_id"])
                ].copy()
                if metric in {"Kz", "TLag"}:
                    source_values = source_values[
                        source_values["gompertz_fit_status"].astype(str).str.startswith("success")
                    ].copy()
                values = pd.to_numeric(source_values[metric_source], errors="coerce").dropna().to_numpy(dtype=float)
                if len(values):
                    jitter = rng.uniform(-0.08, 0.08, len(values))
                    ax.scatter(
                        np.full(len(values), x) + jitter,
                        values,
                        color=colors[label],
                        edgecolor="none",
                        s=20,
                        alpha=0.75,
                        zorder=3,
                    )
                ax.errorbar(x, row["technical_mean"], yerr=row["technical_SD"], color="black", capsize=4, linewidth=1)
                ax.hlines(row["technical_mean"], x - 0.18, x + 0.18, color="black", linewidth=1.5)
                ax.text(x, row["technical_mean"], f"n={int(row['n_technical_wells'])}", fontsize=7, ha="center", va="bottom")
            subtitle = ""
            if not panel.empty:
                first = panel.iloc[0]
                subtitle = f"{first['compound_concentration_uM']:g} uM + {first['ampicillin_concentration_ug_mL']:g} ug/mL amp"
            ax.set_title(f"{replicate} {compound}\n{subtitle}", fontsize=9)
            ax.set_xticks(range(len(labels)))
            ax.set_xticklabels(["vehicle", "compound", "amp", "combo"], rotation=45, ha="right")
            ax.grid(axis="y", alpha=0.22)
            ax.set_ylabel(
                {
                    "relative_AUC": "Relative AUC (% of vehicle)",
                    "raw_AUC": "Raw AUC (OD600·h)",
                    "Kz": "Growth rate, Kz (OD600·h⁻¹)",
                    "TLag": "Lag time, TLag (h)",
                }[metric]
            )
    note = (
        "Run-level descriptive summaries only. Points show individual technical wells; black mean lines and error bars show the technical-well mean +/- SD "
        "within one biological run; they do not represent biological-replicate variation. No p-values are calculated."
    )
    fig.text(0.5, 0.02, fill(note, 130), ha="center", fontsize=9)
    fig.tight_layout(rect=[0, 0.06, 1, 0.98])
    stem = {
        "relative_AUC": "combination_treatment_relative_auc_run_level",
        "Kz": "combination_treatment_growth_rate_run_level",
        "TLag": "combination_treatment_lag_time_run_level",
    }[metric]
    png = output_dir / f"{stem}_{species}.png"
    pdf = output_dir / f"{stem}_{species}.pdf"
    fig.savefig(png, dpi=300)
    fig.savefig(pdf)
    plt.close(fig)
    return [png, pdf]


def plot_combination_delta(delta: pd.DataFrame, output_dir: Path) -> List[Path]:
    colors = {"C1": "#4c78a8", "C2": "#f58518", "C3": "#54a24b", "C4": "#b279a2"}
    fig, axes = plt.subplots(2, 3, figsize=(13, 7), sharey=True)
    for row_index, species in enumerate(["E_coli", "B_subtilis"]):
        for col_index, replicate in enumerate(EXPECTED_REPLICATES):
            ax = axes[row_index, col_index]
            panel = delta[delta["species"].eq(species) & delta["biological_replicate"].eq(replicate)]
            for x, compound in enumerate(SCHIFF_COMPOUNDS):
                row = panel[panel["compound"].eq(compound)]
                if row.empty:
                    continue
                row = row.iloc[0]
                ax.scatter(x, row["delta_relative_AUC_vs_ampicillin"], color=colors[compound], s=46)
            ax.axhline(0, color="black", linestyle="--", linewidth=0.8)
            subtitle = ""
            if not panel.empty:
                first = panel.iloc[0]
                subtitle = f"{first['compound_concentration_uM']:g} uM + {first['ampicillin_concentration_ug_mL']:g} ug/mL amp"
            ax.set_title(f"{DISPLAY_SPECIES.get(species, species)} {replicate}\n{subtitle}", fontsize=9)
            ax.set_xticks(range(len(SCHIFF_COMPOUNDS)))
            ax.set_xticklabels(SCHIFF_COMPOUNDS)
            ax.set_ylabel("")
            ax.grid(axis="y", alpha=0.22)
    note = (
        "Positive values mean the combination allowed more growth than matched ampicillin alone, consistent with possible "
        "attenuation of ampicillin activity. Negative values indicate greater growth inhibition. Descriptive only."
    )
    fig.text(0.015, 0.5, "Combination - ampicillin relative AUC (percentage points)", rotation=90, va="center", fontsize=11)
    fig.text(0.5, 0.02, fill(note, 130), ha="center", fontsize=9)
    fig.tight_layout(rect=[0.06, 0.08, 1, 0.98])
    png = output_dir / "combination_minus_ampicillin_relative_auc.png"
    pdf = output_dir / "combination_minus_ampicillin_relative_auc.pdf"
    fig.savefig(png, dpi=300)
    fig.savefig(pdf)
    plt.close(fig)
    return [png, pdf]


def write_outputs(
    output_dir: Path,
    technical: pd.DataFrame,
    br_metrics: pd.DataFrame,
    anova: pd.DataFrame,
    tukey: pd.DataFrame,
    letters: pd.DataFrame,
    heatmap_summary: pd.DataFrame,
    fit_summary: pd.DataFrame,
    od_qc: pd.DataFrame,
    exclusions: pd.DataFrame,
    combination_descriptive: pd.DataFrame,
    combination_delta: pd.DataFrame,
    diagnostics: pd.DataFrame,
) -> List[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = [
        output_dir / "combined_technical_well_metrics.csv",
        output_dir / "combined_biological_replicate_metrics.csv",
        output_dir / "randomised_block_anova_results.csv",
        output_dir / "tukey_block_pairwise_results.csv",
        output_dir / "compact_letter_display.csv",
        output_dir / "compound_selection_relative_auc_summary.csv",
        output_dir / "fit_availability_summary.csv",
        output_dir / "od600_linearity_qc_summary.csv",
        output_dir / "statistical_analysis_exclusions.csv",
        output_dir / "combination_treatment_descriptive_metrics.csv",
        output_dir / "combination_treatment_delta_vs_ampicillin.csv",
        output_dir / "statistical_model_diagnostics_summary.csv",
    ]
    technical.to_csv(paths[0], index=False)
    br_metrics.to_csv(paths[1], index=False)
    anova.to_csv(paths[2], index=False)
    tukey.to_csv(paths[3], index=False)
    letters.to_csv(paths[4], index=False)
    heatmap_summary.to_csv(paths[5], index=False)
    fit_summary.to_csv(paths[6], index=False)
    od_qc.to_csv(paths[7], index=False)
    exclusions.to_csv(paths[8], index=False)
    combination_descriptive.to_csv(paths[9], index=False)
    combination_delta.to_csv(paths[10], index=False)
    diagnostics.to_csv(paths[11], index=False)
    return paths


def build_exclusion_table(base_exclusions: List[dict], technical: pd.DataFrame, combination_exclusions: pd.DataFrame) -> pd.DataFrame:
    records = list(base_exclusions)
    for _, row in technical[technical["treatment_type"].eq("ampicillin_control")].drop_duplicates(
        ["species", "biological_replicate", "condition_id"]
    ).iterrows():
        records.append(
            {
                "species": row["species"],
                "compound": row["compound"],
                "family_type": "control",
                "metric": "all",
                "condition_id": row["condition_id"],
                "exclusion_reason": "Ampicillin control is not part of the monotherapy concentration-response models.",
            }
        )
    for _, row in technical[technical["treatment_type"].eq("combination")].drop_duplicates(
        ["species", "biological_replicate", "condition_id"]
    ).iterrows():
        replicate_count = int(
            technical[
                technical["species"].eq(row["species"])
                & technical["condition_id"].eq(row["condition_id"])
            ]["biological_replicate"].nunique()
        )
        reason = (
            "Combination regimen differs across BR1-BR3; analysed descriptively by run. "
            f"This exact regimen is present in {replicate_count} biological replicate(s)."
        )
        records.append(
            {
                "species": row["species"],
                "compound": row["compound"],
                "family_type": "combination",
                "metric": "all",
                "condition_id": row["condition_id"],
                "exclusion_reason": reason,
            }
        )
    if not combination_exclusions.empty:
        for row in combination_exclusions.itertuples(index=False):
            records.append(
                {
                    "species": row.species,
                    "compound": row.compound,
                    "family_type": "combination",
                    "metric": "all",
                    "condition_id": "",
                    "exclusion_reason": f"{row.biological_replicate}: {row.reason}",
                }
            )
    failed = technical[~technical["gompertz_fit_status"].astype(str).str.startswith("success")]
    for row in failed.itertuples(index=False):
        records.append(
            {
                "species": row.species,
                "compound": row.compound,
                "family_type": row.treatment_type,
                "metric": "Kz/TLag",
                "condition_id": row.condition_id,
                "exclusion_reason": f"Failed or non-interpretable Gompertz fit in well {row.well}: {row.gompertz_fit_status}.",
            }
        )
    return pd.DataFrame.from_records(records).drop_duplicates().reset_index(drop=True)


def verify_source_hashes(hashes: Dict[Path, str]) -> None:
    changed = [str(path) for path, digest in hashes.items() if _hash_file(path) != digest]
    if changed:
        raise DrugScreeningStatisticsError(_format_errors(["Source CSV files changed during analysis: " + ", ".join(changed)]))


def run_analysis(input_dirs: Dict[Tuple[str, str], Path], output_dir: Path | str) -> dict:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    loaded = read_all_runs(input_dirs)
    growth = loaded["growth"]
    technical = loaded["technical"]
    summary = loaded["summary"]
    runs = loaded["runs"]
    validation_warnings = validate_loaded_data(growth, technical, summary, runs)
    br_metrics = prepare_biological_replicate_metrics(summary)
    anova, tukey, letters, diagnostics, model_exclusions = run_inferential_models(br_metrics)
    br_metrics = annotate_biological_metric_eligibility(br_metrics, anova)
    fit_summary = fit_availability_summary(technical, br_metrics)
    od_qc = od600_linearity_qc(growth)
    combination_descriptive, combination_delta, combination_exclusions = combination_condition_sets(technical)
    exclusions = build_exclusion_table(model_exclusions, technical, combination_exclusions)
    heatmap_summary, heatmap_paths = compound_selection_heatmap(br_metrics, output_dir)
    table_paths = write_outputs(
        output_dir,
        technical,
        br_metrics,
        anova,
        tukey,
        letters,
        heatmap_summary,
        fit_summary,
        od_qc,
        exclusions,
        combination_descriptive,
        combination_delta,
        diagnostics,
    )
    figure_paths: List[Path] = []
    figure_paths.extend(plot_schiff_figures(technical, br_metrics, letters, output_dir))
    figure_paths.extend(plot_penag_figure(technical, br_metrics, letters, output_dir))
    figure_paths.extend(heatmap_paths)
    for species in EXPECTED_SPECIES:
        figure_paths.extend(plot_combination_metric(combination_descriptive, technical, "relative_AUC", species, output_dir))
        figure_paths.extend(plot_combination_metric(combination_descriptive, technical, "Kz", species, output_dir))
        figure_paths.extend(plot_combination_metric(combination_descriptive, technical, "TLag", species, output_dir))
    figure_paths.extend(plot_combination_delta(combination_delta, output_dir))
    figure_paths.extend(plot_model_diagnostics(diagnostics, output_dir))
    verify_source_hashes(loaded["hashes"])
    return {
        "growth": growth,
        "technical": technical,
        "summary": summary,
        "runs": runs,
        "br_metrics": br_metrics,
        "anova": anova,
        "tukey": tukey,
        "letters": letters,
        "diagnostics": diagnostics,
        "fit_summary": fit_summary,
        "od_qc": od_qc,
        "exclusions": exclusions,
        "combination_descriptive": combination_descriptive,
        "combination_delta": combination_delta,
        "table_paths": table_paths,
        "figure_paths": figure_paths,
        "validation_warnings": validation_warnings,
        "output_dir": output_dir,
    }


def main() -> int:
    """Command-line entry point for the complete drug-screening statistics stage."""
    args = parse_args()
    result = run_analysis(build_input_map(args), Path(args.output_dir))
    tukey = result["tukey"]
    anova = result["anova"]
    od_qc = result["od_qc"]
    print("Drug-screening statistical analysis summary")
    print(f"- output folder: {result['output_dir']}")
    print(f"- technical-well metric rows: {len(result['technical'])}")
    print(f"- biological-replicate metric rows: {len(result['br_metrics'])}")
    print(f"- ANOVA models fitted: {len(anova)}")
    print(f"- residual df values: {sorted(anova['residual_df'].drop_duplicates().astype(int).tolist())}")
    print(f"- Tukey comparisons: {len(tukey)}")
    print(f"- significant Tukey comparisons: {int(tukey['statistically_significant'].sum())}")
    print(f"- nonsignificant Tukey comparisons: {int((~tukey['statistically_significant']).sum())}")
    print(f"- combination delta rows: {len(result['combination_delta'])}")
    print(f"- raw OD600>=1 flagged wells: {int(od_qc['raw_OD600_threshold_reached'].sum())}")
    if result["validation_warnings"]:
        print("Warnings")
        for warning in result["validation_warnings"]:
            print(f"- {warning}")
    print("Saved outputs")
    for path in result["table_paths"] + result["figure_paths"]:
        print(f"- {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
