"""Process one species-specific drug-screening biological replicate.

The workbook importer standardizes annotated treatment labels, restricts the
analysis to the declared endpoint, validates the expected plate structure, and
subtracts the time-matched mean of blank wells G4-G6. It then calculates AUC
from observed corrected OD600 and fits the shared modified Gompertz model to
each technical well before averaging technical wells within the run.

This is the first stage of the drug-screening workflow. It does not pool
biological replicates or perform statistical tests. The raw workbook is read
only; all derived CSV and figure files are written beneath the selected output
directory.
"""

from __future__ import annotations

import argparse
import math
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from growth_analysis import (  # noqa: E402
    DISPLAY_SPECIES,
    GrowthDataValidationError,
    apply_time_matched_blank_correction,
    calculate_trapezoidal_auc,
    construct_well,
    fit_one_growth_curve,
    modified_gompertz,
    parse_time_label,
)


ACCEPTED_SPECIES = {"E_coli", "B_subtilis"}
ACCEPTED_BIOLOGICAL_REPLICATES = {"BR1", "BR2", "BR3"}
BLANK_WELLS = ["G4", "G5", "G6"]
DEFAULT_MAX_TIME_MIN = 1045
DEFAULT_LARGE_CHANGE_THRESHOLD = 0.2
EXPECTED_TOTAL_WELLS = 96
EXPECTED_BLANK_WELLS = 3
EXPECTED_NONBLANK_WELLS = 93
EXPECTED_TECHNICAL_WELLS_PER_CONDITION = 3
EXPECTED_AMPICILLIN_CONTROL_COUNT = 2

STANDARD_COLUMNS = [
    "run_id",
    "biological_replicate",
    "species",
    "well",
    "technical_replicate",
    "original_content",
    "original_group",
    "treatment_type",
    "compound",
    "compound_concentration_uM",
    "ampicillin_concentration_ug_mL",
    "condition_id",
]
QC_FLAG_COLUMNS = [
    "flag_type",
    "run_id",
    "biological_replicate",
    "species",
    "well",
    "condition_id",
    "time_min",
    "time_h",
    "raw_OD600",
    "corrected_OD600",
    "details",
]


def parse_args() -> argparse.Namespace:
    """Parse the raw workbook, run identity, endpoint, and output options."""
    parser = argparse.ArgumentParser(
        description=(
            "Process one drug-screening workbook through blank correction, QC, "
            "growth curves, AUC, and modified-Gompertz per-well metrics."
        )
    )
    parser.add_argument("--input", required=True, help="Path to one raw .xlsx workbook.")
    parser.add_argument("--species", required=True, choices=sorted(ACCEPTED_SPECIES))
    parser.add_argument("--biological-replicate", required=True, choices=sorted(ACCEPTED_BIOLOGICAL_REPLICATES))
    parser.add_argument(
        "--max-time-min",
        type=int,
        default=DEFAULT_MAX_TIME_MIN,
        help="Inclusive analysis endpoint in minutes. Default: 1045.",
    )
    parser.add_argument(
        "--output-dir",
        default=str(PROJECT_ROOT / "results" / "drug_screening"),
        help="Parent output directory. Species and run ID child folders are created automatically.",
    )
    parser.add_argument(
        "--large-change-threshold",
        type=float,
        default=DEFAULT_LARGE_CHANGE_THRESHOLD,
        help="Absolute OD600 change between consecutive readings flagged for QC.",
    )
    return parser.parse_args()


def build_run_id(species: str, biological_replicate: str) -> str:
    """Construct a stable run identifier from user-supplied labels."""

    if species not in ACCEPTED_SPECIES:
        raise GrowthDataValidationError(f"Unsupported species {species!r}; use E_coli or B_subtilis.")
    if biological_replicate not in ACCEPTED_BIOLOGICAL_REPLICATES:
        raise GrowthDataValidationError(
            f"Unsupported biological replicate {biological_replicate!r}; use BR1, BR2, or BR3."
        )
    return f"{species}_{biological_replicate}"


def expected_time_points(max_time_min: int = DEFAULT_MAX_TIME_MIN) -> List[int]:
    """Return expected 5-minute time points from 0 through max_time_min."""

    if max_time_min < 0 or max_time_min % 5 != 0:
        raise GrowthDataValidationError("--max-time-min must be a nonnegative multiple of 5 minutes.")
    return list(range(0, max_time_min + 1, 5))


def _clean_text(value: object) -> str:
    if pd.isna(value):
        return ""
    return str(value).strip()


def _normalise_units(value: object) -> str:
    text = _clean_text(value)
    text = (
        text.replace("\ufffd", "u")
        .replace("micro", "u")
        .replace("MICRO", "u")
        .replace("µ", "u")
        .replace("μ", "u")
    )
    return re.sub(r"\s+", "", text).lower()


def _format_number(value: float) -> str:
    return f"{float(value):g}"


def _parse_uM(group: object) -> float:
    match = re.search(r"(\d+(?:\.\d+)?)u?m", _normalise_units(group))
    if not match:
        raise GrowthDataValidationError(f"Could not parse compound concentration from Group={group!r}.")
    return float(match.group(1))


def _parse_ampicillin(group: object) -> float:
    match = re.search(r"(\d+(?:\.\d+)?)u?g/ml", _normalise_units(group))
    if not match:
        raise GrowthDataValidationError(f"Could not parse ampicillin concentration from Group={group!r}.")
    return float(match.group(1))


def standardize_drug_treatment(content: object, group: object) -> dict:
    """Parse workbook annotations into stable drug-treatment metadata fields."""

    original_content = _clean_text(content)
    original_group = _clean_text(group)
    content_norm = re.sub(r"\s+", "", original_content).lower()
    group_norm = _normalise_units(original_group)

    if content_norm == "blank" and group_norm == "blank":
        return {
            "treatment_type": "blank",
            "compound": "Blank",
            "compound_concentration_uM": math.nan,
            "ampicillin_concentration_ug_mL": math.nan,
            "condition_id": "blank",
            "condition_label": "Blank",
            "dmso_percent": math.nan,
        }

    if content_norm == "vehiclecontrol" and group_norm == "1%dmso":
        return {
            "treatment_type": "vehicle_control",
            "compound": "DMSO",
            "compound_concentration_uM": math.nan,
            "ampicillin_concentration_ug_mL": math.nan,
            "condition_id": "vehicle_1pct_DMSO",
            "condition_label": "Vehicle Control (1% DMSO)",
            "dmso_percent": 1.0,
        }

    if content_norm == "positivecontrol(ampicillin)":
        amp = _parse_ampicillin(original_group)
        return {
            "treatment_type": "ampicillin_control",
            "compound": "Ampicillin",
            "compound_concentration_uM": math.nan,
            "ampicillin_concentration_ug_mL": amp,
            "condition_id": f"ampicillin_{_format_number(amp)}ug_mL",
            "condition_label": f"Ampicillin {_format_number(amp)} ug/mL",
            "dmso_percent": math.nan,
        }

    combination_match = re.fullmatch(r"(c[1-4])\+ampicillin", content_norm)
    if combination_match:
        compound = combination_match.group(1).upper()
        compound_conc = _parse_uM(original_group)
        amp = _parse_ampicillin(original_group)
        return {
            "treatment_type": "combination",
            "compound": compound,
            "compound_concentration_uM": compound_conc,
            "ampicillin_concentration_ug_mL": amp,
            "condition_id": (
                f"{compound}_{_format_number(compound_conc)}uM_"
                f"ampicillin_{_format_number(amp)}ug_mL"
            ),
            "condition_label": (
                f"{compound} {_format_number(compound_conc)} uM + "
                f"ampicillin {_format_number(amp)} ug/mL"
            ),
            "dmso_percent": math.nan,
        }

    if re.fullmatch(r"c[1-4]", content_norm):
        compound = content_norm.upper()
        concentration = _parse_uM(original_group)
        return {
            "treatment_type": "compound_monotherapy",
            "compound": compound,
            "compound_concentration_uM": concentration,
            "ampicillin_concentration_ug_mL": math.nan,
            "condition_id": f"{compound}_{_format_number(concentration)}uM",
            "condition_label": f"{compound} {_format_number(concentration)} uM",
            "dmso_percent": math.nan,
        }

    if content_norm == "penag":
        concentration = _parse_uM(original_group)
        return {
            "treatment_type": "penag_monotherapy",
            "compound": "PenAg",
            "compound_concentration_uM": concentration,
            "ampicillin_concentration_ug_mL": math.nan,
            "condition_id": f"PenAg_{_format_number(concentration)}uM",
            "condition_label": f"PenAg {_format_number(concentration)} uM",
            "dmso_percent": math.nan,
        }

    raise GrowthDataValidationError(
        f"Unrecognised treatment annotation: Content={original_content!r}, Group={original_group!r}."
    )


def _find_measurement_sheet(sheets: Dict[str, pd.DataFrame]) -> Tuple[str, pd.DataFrame, int]:
    candidates = []
    for sheet_name, sheet in sheets.items():
        first_column = sheet.iloc[:, 0].astype(str)
        header_rows = first_column[first_column.str.contains("Well", case=False, na=False)].index.tolist()
        for header_row in header_rows:
            if sheet.shape[1] >= 5 and _clean_text(sheet.iloc[header_row + 1, 2]).lower() == "time":
                candidates.append((sheet_name, sheet, int(header_row)))
    if len(candidates) != 1:
        raise GrowthDataValidationError(
            f"Expected exactly one BMG measurement sheet, found {len(candidates)} candidate(s)."
        )
    return candidates[0]


def read_instrument_metadata(sheet: pd.DataFrame, header_row: int) -> dict:
    """Read simple metadata key/value strings above the BMG measurement table."""

    metadata = {}
    for row_number in range(min(header_row, 8)):
        value = _clean_text(sheet.iloc[row_number, 0])
        if not value:
            continue
        if ":" in value:
            key, item = value.split(":", 1)
            metadata[key.strip()] = item.strip()
        else:
            metadata[f"metadata_row_{row_number + 1}"] = value
    return metadata


def import_drug_screening_workbook(
    input_path: Path | str,
    species: str,
    biological_replicate: str,
    max_time_min: int = DEFAULT_MAX_TIME_MIN,
) -> Tuple[pd.DataFrame, dict]:
    """Import one annotated BMG Excel workbook into trimmed long-form data."""

    input_path = Path(input_path)
    if not input_path.exists():
        raise FileNotFoundError(f"Workbook does not exist: {input_path}")

    run_id = build_run_id(species, biological_replicate)
    expected_times = expected_time_points(max_time_min)
    sheets = pd.read_excel(input_path, sheet_name=None, header=None, dtype=object)
    sheet_name, sheet, header_row = _find_measurement_sheet(sheets)
    metadata = read_instrument_metadata(sheet, header_row)
    metadata["selected_sheet"] = sheet_name

    time_row = header_row + 1
    parsed_time_columns = []
    for column_index in range(4, sheet.shape[1]):
        label = sheet.iloc[time_row, column_index]
        if not _clean_text(label):
            continue
        try:
            time_min = parse_time_label(label)
        except ValueError:
            continue
        if time_min <= max_time_min:
            parsed_time_columns.append((column_index, int(time_min)))

    observed_times = [time for _, time in parsed_time_columns]
    if observed_times != expected_times:
        raise GrowthDataValidationError(
            "Retained time points must be 0 to "
            f"{max_time_min} minutes in 5-minute intervals; found {len(observed_times)} time point(s)."
        )

    data_wide = sheet.iloc[time_row + 1 :, :].dropna(how="all").copy()
    data_wide = data_wide[data_wide.iloc[:, 0].map(_clean_text).ne("")].copy()
    records = []
    for _, row in data_wide.iterrows():
        well = construct_well(row.iloc[0], row.iloc[1])
        treatment = standardize_drug_treatment(row.iloc[2], row.iloc[3])
        for column_index, time_min in parsed_time_columns:
            raw_value = row.iloc[column_index]
            records.append(
                {
                    "run_id": run_id,
                    "experiment_id": run_id,
                    "biological_replicate": biological_replicate,
                    "species": "Blank" if treatment["treatment_type"] == "blank" else species,
                    "well": well,
                    "technical_replicate": math.nan,
                    "original_content": _clean_text(row.iloc[2]),
                    "original_group": _clean_text(row.iloc[3]),
                    "condition_label": treatment["condition_label"],
                    "condition": treatment["condition_label"],
                    "dmso_percent": treatment["dmso_percent"],
                    "treatment_type": treatment["treatment_type"],
                    "compound": treatment["compound"],
                    "compound_concentration_uM": treatment["compound_concentration_uM"],
                    "ampicillin_concentration_ug_mL": treatment["ampicillin_concentration_ug_mL"],
                    "condition_id": treatment["condition_id"],
                    "time_min": int(time_min),
                    "time_h": time_min / 60.0,
                    "od600_raw": raw_value,
                    "od600": pd.to_numeric(pd.Series([raw_value]), errors="coerce").iloc[0],
                }
            )

    data = pd.DataFrame.from_records(records)
    data["well"] = data["well"].astype(str).str.upper()
    sample_wells = (
        data[data["treatment_type"].ne("blank")]
        .drop_duplicates(["condition_id", "well"])
        .sort_values(["condition_id", "well"], kind="stable")
        .copy()
    )
    sample_wells["technical_replicate"] = sample_wells.groupby("condition_id").cumcount() + 1
    replicate_lookup = sample_wells.set_index("well")["technical_replicate"].to_dict()
    sample_mask = data["treatment_type"].ne("blank")
    data.loc[sample_mask, "technical_replicate"] = data.loc[sample_mask, "well"].map(replicate_lookup)

    sort_columns = ["treatment_type", "condition_id", "technical_replicate", "well", "time_min"]
    return data.sort_values(sort_columns, kind="stable").reset_index(drop=True), metadata


def validate_drug_screening_measurements(data: pd.DataFrame, max_time_min: int = DEFAULT_MAX_TIME_MIN) -> None:
    """Validate wells, controls, time points, and OD after endpoint trimming."""

    expected_times = expected_time_points(max_time_min)
    errors: List[str] = []
    required_columns = {
        *STANDARD_COLUMNS,
        "condition_label",
        "time_min",
        "time_h",
        "od600",
    }
    missing = required_columns.difference(data.columns)
    if missing:
        errors.append(f"Missing required columns: {sorted(missing)}")
        raise GrowthDataValidationError("Drug-screening validation failed:\n- " + "\n- ".join(errors))

    total_wells = int(data["well"].nunique())
    blank_wells = sorted(data.loc[data["treatment_type"].eq("blank"), "well"].drop_duplicates().tolist())
    nonblank_wells = int(data.loc[data["treatment_type"].ne("blank"), "well"].nunique())
    if total_wells != EXPECTED_TOTAL_WELLS:
        errors.append(f"Expected 96 total wells, found {total_wells}.")
    if blank_wells != BLANK_WELLS:
        errors.append(f"Expected blank wells {BLANK_WELLS}, found {blank_wells}.")
    if nonblank_wells != EXPECTED_NONBLANK_WELLS:
        errors.append(f"Expected 93 nonblank wells, found {nonblank_wells}.")

    duplicate_rows = data.duplicated(["well", "time_min"], keep=False)
    if duplicate_rows.any():
        errors.append("Duplicate measurements found for the same well and time.")

    numeric_od = pd.to_numeric(data["od600"], errors="coerce")
    if numeric_od.isna().any():
        errors.append("OD600 values must be numeric and present.")

    for well, well_data in data.groupby("well", sort=True):
        observed = sorted(pd.to_numeric(well_data["time_min"], errors="coerce").dropna().astype(int).tolist())
        if observed != expected_times:
            errors.append(f"Well {well} does not contain the complete retained time series.")

    technical_counts = (
        data[data["treatment_type"].ne("blank")]
        .drop_duplicates(["condition_id", "well"])
        .groupby("condition_id")["well"]
        .nunique()
    )
    bad_counts = technical_counts[technical_counts.ne(EXPECTED_TECHNICAL_WELLS_PER_CONDITION)]
    if not bad_counts.empty:
        errors.append(f"Expected three technical wells per condition; found {bad_counts.to_dict()}.")

    required_controls = {"vehicle_1pct_DMSO"}
    observed_controls = set(data["condition_id"].drop_duplicates())
    missing_controls = sorted(required_controls.difference(observed_controls))
    if missing_controls:
        errors.append(f"Missing required controls: {missing_controls}.")
    ampicillin_controls = data[data["treatment_type"].eq("ampicillin_control")]
    ampicillin_control_count = int(ampicillin_controls["condition_id"].nunique())
    if ampicillin_control_count != EXPECTED_AMPICILLIN_CONTROL_COUNT:
        errors.append(
            "Expected two positive-control ampicillin concentrations, "
            f"found {ampicillin_control_count}."
        )

    if errors:
        raise GrowthDataValidationError("Drug-screening validation failed:\n- " + "\n- ".join(errors))


def prepare_blank_outputs(blank_correction) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Rename generic blank-correction outputs for the drug-screening exports."""

    blank_qc = blank_correction.blank_qc_summary.rename(
        columns={
            "starting_OD": "starting_OD600",
            "ending_OD": "ending_OD600",
            "mean_OD": "mean_OD600",
            "SD_OD": "SD_OD600",
            "minimum_OD": "minimum_OD600",
            "maximum_OD": "maximum_OD600",
        }
    )
    blank_qc = blank_qc[
        [
            "well",
            "number_of_measurements",
            "starting_OD600",
            "ending_OD600",
            "mean_OD600",
            "SD_OD600",
            "minimum_OD600",
            "maximum_OD600",
            "change_from_start_to_end",
        ]
    ]
    blank_trace = blank_correction.blank_trace_used.rename(
        columns={
            "G4_raw_OD": "G4_raw_OD600",
            "G5_raw_OD": "G5_raw_OD600",
            "G6_raw_OD": "G6_raw_OD600",
            "mean_blank_OD_used": "mean_blank_OD600_used",
        }
    )
    return blank_qc, blank_trace


def validate_corrected_drug_screening_data(
    corrected_data: pd.DataFrame,
    max_time_min: int = DEFAULT_MAX_TIME_MIN,
) -> None:
    """Validate the corrected nonblank measurement table before metrics."""

    expected_times = expected_time_points(max_time_min)
    errors = []
    if corrected_data["well"].isin(BLANK_WELLS).any() or corrected_data["treatment_type"].eq("blank").any():
        errors.append("Blank wells must be excluded from corrected sample data.")
    if corrected_data["well"].nunique() != EXPECTED_NONBLANK_WELLS:
        errors.append(f"Expected 93 nonblank wells, found {corrected_data['well'].nunique()}.")
    expected_rows = EXPECTED_NONBLANK_WELLS * len(expected_times)
    if len(corrected_data) != expected_rows:
        errors.append(f"Expected {expected_rows} processed nonblank measurements, found {len(corrected_data)}.")
    for column in ["raw_OD600", "blank_OD600", "corrected_OD600", "time_h"]:
        if pd.to_numeric(corrected_data[column], errors="coerce").isna().any():
            errors.append(f"{column} values must be numeric and present.")
    for well, well_data in corrected_data.groupby("well", sort=True):
        observed = well_data.sort_values("time_min")["time_min"].astype(int).tolist()
        if observed != expected_times:
            errors.append(f"Well {well} does not contain the complete retained corrected time series.")
    if errors:
        raise GrowthDataValidationError("Corrected drug-screening validation failed:\n- " + "\n- ".join(errors))


def calculate_technical_well_metrics(
    corrected_data: pd.DataFrame,
    run_id: str,
    biological_replicate: str,
    max_time_min: int = DEFAULT_MAX_TIME_MIN,
) -> pd.DataFrame:
    """Calculate AUC, vehicle-relative AUC, and Gompertz fits per nonblank well."""

    validate_corrected_drug_screening_data(corrected_data, max_time_min=max_time_min)
    original_raw = corrected_data["raw_OD600"].copy(deep=True)
    original_corrected = corrected_data["corrected_OD600"].copy(deep=True)
    end_h = max_time_min / 60.0
    records = []

    for well, well_data in corrected_data.groupby("well", sort=True):
        well_data = well_data.sort_values("time_h")
        first = well_data.iloc[0]
        auc = calculate_trapezoidal_auc(well_data, 0.0, end_h, "time_h", "corrected_OD600")
        fit = fit_one_growth_curve(well_data, "time_h", "corrected_OD600")
        status = str(fit["gompertz_fit_status"])
        fit_warning = ""
        if status.startswith("success_with_warning:"):
            fit_warning = status.split(":", 1)[1].strip()
        elif status.startswith("fit_failed:"):
            fit_warning = status.split(":", 1)[1].strip()
        records.append(
            {
                "run_id": run_id,
                "biological_replicate": biological_replicate,
                "species": first["species"],
                "well": well,
                "technical_replicate": int(first["technical_replicate"]),
                "treatment_type": first["treatment_type"],
                "compound": first["compound"],
                "compound_concentration_uM": first["compound_concentration_uM"],
                "ampicillin_concentration_ug_mL": first["ampicillin_concentration_ug_mL"],
                "condition_id": first["condition_id"],
                "condition": first["condition"],
                "AUC_0_17h25min_OD_h": auc,
                "A": fit["A_OD600"],
                "Kz": fit["Kz_OD600_per_h"],
                "TLag": fit["TLag_h"],
                "A_OD600": fit["A_OD600"],
                "Kz_OD600_per_h": fit["Kz_OD600_per_h"],
                "TLag_h": fit["TLag_h"],
                "fit_R2": fit["gompertz_R2"],
                "fit_status": status,
                "fit_warning": fit_warning,
                "gompertz_R2": fit["gompertz_R2"],
                "gompertz_fit_status": status,
                "gompertz_fit_warning": fit_warning,
            }
        )

    pd.testing.assert_series_equal(corrected_data["raw_OD600"], original_raw)
    pd.testing.assert_series_equal(corrected_data["corrected_OD600"], original_corrected)

    metrics = pd.DataFrame.from_records(records)
    vehicle_auc = metrics.loc[metrics["treatment_type"].eq("vehicle_control"), "AUC_0_17h25min_OD_h"]
    if len(vehicle_auc) != EXPECTED_TECHNICAL_WELLS_PER_CONDITION:
        raise GrowthDataValidationError(
            f"Expected three vehicle-control wells for relative AUC, found {len(vehicle_auc)}."
        )
    vehicle_mean = float(vehicle_auc.mean())
    if vehicle_mean == 0 or math.isnan(vehicle_mean):
        raise GrowthDataValidationError("Vehicle-control mean AUC is unavailable or zero.")
    metrics["relative_AUC_percent"] = 100.0 * metrics["AUC_0_17h25min_OD_h"] / vehicle_mean
    return metrics.sort_values(["treatment_type", "compound", "condition_id", "well"], kind="stable").reset_index(drop=True)


def summarise_drug_screening_biological_replicate(metrics: pd.DataFrame) -> pd.DataFrame:
    """Average technical wells into one biological-replicate row per treatment."""

    records = []
    group_columns = [
        "run_id",
        "biological_replicate",
        "species",
        "treatment_type",
        "compound",
        "compound_concentration_uM",
        "ampicillin_concentration_ug_mL",
        "condition_id",
        "condition",
    ]
    for keys, group in metrics.groupby(group_columns, dropna=False, sort=False):
        record = dict(zip(group_columns, keys))
        record["n_technical_wells"] = int(group["well"].nunique())
        for metric in [
            "AUC_0_17h25min_OD_h",
            "relative_AUC_percent",
            "A",
            "Kz",
            "TLag",
            "A_OD600",
            "Kz_OD600_per_h",
            "TLag_h",
        ]:
            valid_values = group.loc[
                group["gompertz_fit_status"].astype(str).str.startswith("success")
                if metric in {"A", "Kz", "TLag", "A_OD600", "Kz_OD600_per_h", "TLag_h"}
                else group.index,
                metric,
            ]
            record[f"{metric}_mean"] = valid_values.mean()
            record[f"{metric}_SD_technical"] = valid_values.std(ddof=1)
        fit_success = group["gompertz_fit_status"].astype(str).str.startswith("success")
        record["n_valid_gompertz_fits"] = int(fit_success.sum())
        record["n_failed_gompertz_fits"] = int((~fit_success).sum())
        records.append(record)
    return pd.DataFrame.from_records(records)


def _append_flag(flags: List[dict], flag_type: str, row: Optional[pd.Series] = None, **details) -> None:
    base = {
        "flag_type": flag_type,
        "run_id": details.pop("run_id", ""),
        "biological_replicate": details.pop("biological_replicate", ""),
        "species": details.pop("species", ""),
        "well": details.pop("well", ""),
        "condition_id": details.pop("condition_id", ""),
        "time_min": details.pop("time_min", math.nan),
        "time_h": details.pop("time_h", math.nan),
        "raw_OD600": details.pop("raw_OD600", math.nan),
        "corrected_OD600": details.pop("corrected_OD600", math.nan),
        "details": details.pop("details", ""),
    }
    if row is not None:
        for column in base:
            if column in row.index and column not in {"flag_type", "details"}:
                base[column] = row[column]
        if "od600" in row.index:
            base["raw_OD600"] = row["od600"]
    base.update(details)
    flags.append(base)


def build_quality_control_flags(
    measurement_data: pd.DataFrame,
    corrected_data: pd.DataFrame,
    metrics: pd.DataFrame,
    max_time_min: int = DEFAULT_MAX_TIME_MIN,
    large_change_threshold: float = DEFAULT_LARGE_CHANGE_THRESHOLD,
) -> pd.DataFrame:
    """Create explicit QC flags while retaining all measurements."""

    expected_times = expected_time_points(max_time_min)
    flags: List[dict] = []

    numeric_raw = pd.to_numeric(measurement_data["od600"], errors="coerce")
    for _, row in measurement_data.loc[numeric_raw.isna()].iterrows():
        _append_flag(flags, "missing_or_nonnumeric_measurement", row, details="Raw OD600 is missing or nonnumeric.")

    duplicate_mask = measurement_data.duplicated(["well", "time_min"], keep=False)
    for _, row in measurement_data.loc[duplicate_mask].iterrows():
        _append_flag(flags, "duplicate_measurement", row, details="Duplicate well/time measurement.")

    for well, well_data in measurement_data.groupby("well", sort=True):
        observed = sorted(well_data["time_min"].dropna().astype(int).tolist())
        if observed != expected_times:
            first = well_data.iloc[0]
            _append_flag(
                flags,
                "incomplete_time_series",
                first,
                details=f"Expected {len(expected_times)} time points, found {len(observed)}.",
            )

    for _, row in corrected_data.loc[corrected_data["raw_OD600"].ge(1.0)].iterrows():
        _append_flag(flags, "raw_OD600_greater_or_equal_1", row, details="Raw OD600 is at or above 1.0.")
    for _, row in corrected_data.loc[corrected_data["corrected_OD600"].ge(1.0)].iterrows():
        _append_flag(
            flags,
            "corrected_OD600_greater_or_equal_1",
            row,
            details="Blank-corrected OD600 is at or above 1.0.",
        )
    for _, row in corrected_data.loc[corrected_data["corrected_OD600"].lt(0.0)].iterrows():
        _append_flag(flags, "negative_corrected_OD600", row, details="Negative corrected value retained.")

    ordered = corrected_data.sort_values(["well", "time_min"], kind="stable").copy()
    ordered["previous_raw_OD600"] = ordered.groupby("well")["raw_OD600"].shift(1)
    ordered["previous_corrected_OD600"] = ordered.groupby("well")["corrected_OD600"].shift(1)
    ordered["delta_raw_OD600"] = ordered["raw_OD600"] - ordered["previous_raw_OD600"]
    ordered["delta_corrected_OD600"] = ordered["corrected_OD600"] - ordered["previous_corrected_OD600"]
    for _, row in ordered.loc[ordered["delta_raw_OD600"].abs().ge(large_change_threshold)].iterrows():
        _append_flag(
            flags,
            "large_raw_consecutive_change",
            row,
            details=f"Absolute raw OD600 change >= {large_change_threshold:g}.",
        )
    for _, row in ordered.loc[ordered["delta_corrected_OD600"].abs().ge(large_change_threshold)].iterrows():
        _append_flag(
            flags,
            "large_corrected_consecutive_change",
            row,
            details=f"Absolute corrected OD600 change >= {large_change_threshold:g}.",
        )

    required_controls = {"vehicle_1pct_DMSO"}
    missing_controls = sorted(required_controls.difference(set(corrected_data["condition_id"].drop_duplicates())))
    for condition_id in missing_controls:
        _append_flag(flags, "missing_control", condition_id=condition_id, details="Required control condition missing.")
    ampicillin_control_count = int(
        corrected_data.loc[
            corrected_data["treatment_type"].eq("ampicillin_control"),
            "condition_id",
        ].nunique()
    )
    if ampicillin_control_count != EXPECTED_AMPICILLIN_CONTROL_COUNT:
        _append_flag(
            flags,
            "missing_control",
            details=(
                "Expected two positive-control ampicillin concentrations, "
                f"found {ampicillin_control_count}."
            ),
        )

    fit_success = metrics["gompertz_fit_status"].astype(str).str.startswith("success")
    for _, row in metrics.loc[~fit_success].iterrows():
        _append_flag(
            flags,
            "failed_model_fit",
            row,
            details=str(row["gompertz_fit_status"]),
        )
    for _, row in metrics.loc[fit_success & metrics["gompertz_R2"].lt(0.8)].iterrows():
        _append_flag(
            flags,
            "poor_quality_model_fit",
            row,
            details=f"Gompertz R2 < 0.8 ({row['gompertz_R2']:.4g}).",
        )

    if not flags:
        return pd.DataFrame(columns=QC_FLAG_COLUMNS)

    return pd.DataFrame.from_records(flags).reindex(columns=QC_FLAG_COLUMNS).sort_values(
        ["flag_type", "condition_id", "well", "time_min"],
        kind="stable",
    ).reset_index(drop=True)


def add_qc_flag_columns(corrected_data: pd.DataFrame, metrics: pd.DataFrame, flags: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Attach semicolon-separated QC flag labels to exported measurement and metric tables."""

    corrected = corrected_data.copy()
    metric_table = metrics.copy()
    measurement_flags = flags[flags["time_min"].notna() & flags["well"].astype(str).ne("")]
    if measurement_flags.empty:
        corrected["applicable_QC_flags"] = ""
    else:
        collapsed = (
            measurement_flags.groupby(["well", "time_min"])["flag_type"]
            .apply(lambda values: ";".join(sorted(set(values))))
            .rename("applicable_QC_flags")
            .reset_index()
        )
        corrected = corrected.merge(collapsed, on=["well", "time_min"], how="left")
        corrected["applicable_QC_flags"] = corrected["applicable_QC_flags"].fillna("")

    if flags.empty:
        metric_table["applicable_QC_flags"] = ""
    else:
        well_flags = (
            flags[flags["well"].astype(str).ne("")]
            .groupby("well")["flag_type"]
            .apply(lambda values: ";".join(sorted(set(values))))
            .rename("applicable_QC_flags")
            .reset_index()
        )
        metric_table = metric_table.merge(well_flags, on="well", how="left")
        metric_table["applicable_QC_flags"] = metric_table["applicable_QC_flags"].fillna("")
    return corrected, metric_table


def _condition_sort_key(row: pd.Series) -> Tuple[int, str, float, float, str]:
    type_order = {
        "vehicle_control": 0,
        "ampicillin_control": 1,
        "compound_monotherapy": 2,
        "penag_monotherapy": 3,
        "combination": 4,
    }
    concentration = row["compound_concentration_uM"]
    amp = row["ampicillin_concentration_ug_mL"]
    return (
        type_order.get(row["treatment_type"], 99),
        str(row["compound"]),
        float(concentration) if pd.notna(concentration) else -1.0,
        float(amp) if pd.notna(amp) else -1.0,
        str(row["condition_id"]),
    )


def condition_order(data: pd.DataFrame) -> List[str]:
    conditions = data.drop_duplicates("condition_id").copy()
    conditions["_sort_key"] = conditions.apply(_condition_sort_key, axis=1)
    return conditions.sort_values("_sort_key", kind="stable")["condition_id"].tolist()


def condition_color_map(data: pd.DataFrame) -> Dict[str, object]:
    order = condition_order(data)
    palettes = [plt.get_cmap("tab20").colors, plt.get_cmap("tab20b").colors, plt.get_cmap("tab20c").colors]
    colors = [color for palette in palettes for color in palette]
    return {condition_id: colors[index % len(colors)] for index, condition_id in enumerate(order)}


def _panel_groups(data: pd.DataFrame) -> List[Tuple[str, pd.DataFrame]]:
    panels = [
        ("Controls", data[data["treatment_type"].isin(["vehicle_control", "ampicillin_control"])]),
        ("C1", data[(data["compound"].eq("C1")) & data["treatment_type"].eq("compound_monotherapy")]),
        ("C2", data[(data["compound"].eq("C2")) & data["treatment_type"].eq("compound_monotherapy")]),
        ("C3", data[(data["compound"].eq("C3")) & data["treatment_type"].eq("compound_monotherapy")]),
        ("C4", data[(data["compound"].eq("C4")) & data["treatment_type"].eq("compound_monotherapy")]),
        ("PenAg", data[data["compound"].eq("PenAg")]),
        ("Combination treatments", data[data["treatment_type"].eq("combination")]),
    ]
    return [(title, panel) for title, panel in panels if not panel.empty]


def _set_corrected_limits(ax: plt.Axes, panel: pd.DataFrame) -> None:
    y_min = float(panel["corrected_OD600"].min())
    y_max = float(panel["corrected_OD600"].max())
    span = max(y_max - y_min, 0.05)
    bottom = y_min - span * 0.05 if y_min < 0 else 0
    ax.set_ylim(bottom=bottom, top=y_max + span * 0.08)


def plot_blank_well_qc(
    measurement_data: pd.DataFrame,
    blank_trace: pd.DataFrame,
    output_dir: Path,
) -> List[Path]:
    blank_data = measurement_data[measurement_data["well"].isin(BLANK_WELLS)].copy()
    fig, ax = plt.subplots(figsize=(9, 5))
    for well in BLANK_WELLS:
        well_data = blank_data[blank_data["well"].eq(well)].sort_values("time_min")
        ax.plot(well_data["time_h"], well_data["od600"], linewidth=1.4, label=f"{well} raw OD600")
    ax.plot(
        blank_trace["time_h"],
        blank_trace["mean_blank_OD600_used"],
        color="black",
        linewidth=2.2,
        label="Mean blank OD600 used",
    )
    ax.set_xlabel("Time (h)")
    ax.set_ylabel("OD600")
    ax.set_title("Blank well QC")
    ax.grid(alpha=0.25)
    ax.legend(frameon=False, ncol=2)
    fig.tight_layout()
    png = output_dir / "blank_well_qc.png"
    pdf = output_dir / "blank_well_qc.pdf"
    fig.savefig(png, dpi=300)
    fig.savefig(pdf)
    plt.close(fig)
    return [png, pdf]


def plot_individual_growth_curves(corrected_data: pd.DataFrame, output_dir: Path) -> List[Path]:
    colors = condition_color_map(corrected_data)
    panels = _panel_groups(corrected_data)
    fig, axes = plt.subplots(3, 3, figsize=(16, 11), sharex=True)
    axes_flat = axes.flatten()
    for ax, (title, panel) in zip(axes_flat, panels):
        for (condition_id, well), curve in panel.groupby(["condition_id", "well"], sort=False):
            label = curve["condition"].iloc[0]
            ax.plot(
                curve["time_h"],
                curve["corrected_OD600"],
                color=colors[condition_id],
                linewidth=0.9,
                alpha=0.55,
                label=label,
            )
        handles, labels = ax.get_legend_handles_labels()
        unique = dict(zip(labels, handles))
        ax.legend(unique.values(), unique.keys(), frameon=False, fontsize=7)
        ax.set_title(title)
        ax.set_xlabel("Time (h)")
        ax.set_ylabel("Blank-corrected OD600")
        ax.grid(alpha=0.25)
        _set_corrected_limits(ax, panel)
    for ax in axes_flat[len(panels) :]:
        ax.set_visible(False)
    species = DISPLAY_SPECIES.get(corrected_data["species"].iloc[0], corrected_data["species"].iloc[0])
    run_id = corrected_data["run_id"].iloc[0]
    fig.suptitle(f"Individual blank-corrected growth curves: {species} {run_id}", y=0.995)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    png = output_dir / "individual_blank_corrected_growth_curves.png"
    pdf = output_dir / "individual_blank_corrected_growth_curves.pdf"
    fig.savefig(png, dpi=300)
    fig.savefig(pdf)
    plt.close(fig)
    return [png, pdf]


def plot_mean_sd_growth_curves(corrected_data: pd.DataFrame, output_dir: Path) -> List[Path]:
    colors = condition_color_map(corrected_data)
    panels = _panel_groups(corrected_data)
    summary = (
        corrected_data.groupby(["condition_id", "condition", "time_min", "time_h"], as_index=False, sort=False)
        .agg(mean_corrected_OD600=("corrected_OD600", "mean"), sd_corrected_OD600=("corrected_OD600", "std"))
    )
    fig, axes = plt.subplots(3, 3, figsize=(16, 11), sharex=True)
    axes_flat = axes.flatten()
    for ax, (title, panel) in zip(axes_flat, panels):
        panel_summary = summary[summary["condition_id"].isin(panel["condition_id"].drop_duplicates())]
        for condition_id, curve in panel_summary.groupby("condition_id", sort=False):
            curve = curve.sort_values("time_h")
            color = colors[condition_id]
            x = curve["time_h"].to_numpy()
            mean = curve["mean_corrected_OD600"].to_numpy()
            sd = curve["sd_corrected_OD600"].fillna(0).to_numpy()
            ax.plot(x, mean, color=color, linewidth=1.7, label=curve["condition"].iloc[0])
            ax.fill_between(x, mean - sd, mean + sd, color=color, alpha=0.16, linewidth=0)
        ax.legend(frameon=False, fontsize=7)
        ax.set_title(title)
        ax.set_xlabel("Time (h)")
        ax.set_ylabel("Blank-corrected OD600")
        ax.grid(alpha=0.25)
        _set_corrected_limits(ax, panel)
    for ax in axes_flat[len(panels) :]:
        ax.set_visible(False)
    species = DISPLAY_SPECIES.get(corrected_data["species"].iloc[0], corrected_data["species"].iloc[0])
    run_id = corrected_data["run_id"].iloc[0]
    fig.suptitle(f"Mean +/- SD blank-corrected growth curves: {species} {run_id}", y=0.995)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    png = output_dir / "mean_sd_blank_corrected_growth_curves.png"
    pdf = output_dir / "mean_sd_blank_corrected_growth_curves.pdf"
    fig.savefig(png, dpi=300)
    fig.savefig(pdf)
    plt.close(fig)
    return [png, pdf]


def plot_gompertz_fits(corrected_data: pd.DataFrame, metrics: pd.DataFrame, output_dir: Path) -> List[Path]:
    colors = condition_color_map(corrected_data)
    panels = _panel_groups(corrected_data)
    fig, axes = plt.subplots(3, 3, figsize=(16, 11), sharex=True)
    axes_flat = axes.flatten()
    failed = []
    for ax, (title, panel) in zip(axes_flat, panels):
        for (condition_id, well), curve in panel.groupby(["condition_id", "well"], sort=False):
            curve = curve.sort_values("time_h")
            baseline = curve["corrected_OD600"].iloc[0]
            w = curve["corrected_OD600"] - baseline
            color = colors[condition_id]
            ax.scatter(curve["time_h"], w, s=6, color=color, alpha=0.35)
            fit_row = metrics[metrics["well"].eq(well)].iloc[0]
            if str(fit_row["gompertz_fit_status"]).startswith("success"):
                t_fit = np.linspace(0.0, float(curve["time_h"].max()), 300)
                y_fit = modified_gompertz(t_fit, fit_row["A_OD600"], fit_row["Kz_OD600_per_h"], fit_row["TLag_h"])
                ax.plot(t_fit, y_fit, color="black", linewidth=0.9, alpha=0.55)
            else:
                failed.append(well)
        ax.set_title(title)
        ax.set_xlabel("Time (h)")
        ax.set_ylabel("Change in blank-corrected OD600, W(t)")
        ax.grid(alpha=0.25)
    for ax in axes_flat[len(panels) :]:
        ax.set_visible(False)
    species = DISPLAY_SPECIES.get(corrected_data["species"].iloc[0], corrected_data["species"].iloc[0])
    title = f"Modified Gompertz fits: {species} {corrected_data['run_id'].iloc[0]}"
    if failed:
        title += f" | Failed fits: {len(set(failed))}"
    fig.suptitle(title, y=0.995)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    png = output_dir / "gompertz_fits.png"
    pdf = output_dir / "gompertz_fits.pdf"
    fig.savefig(png, dpi=300)
    fig.savefig(pdf)
    plt.close(fig)
    return [png, pdf]


def write_outputs(
    output_dir: Path,
    source_path: Path,
    measurement_data: pd.DataFrame,
    blank_qc: pd.DataFrame,
    blank_trace: pd.DataFrame,
    corrected_data: pd.DataFrame,
    metrics: pd.DataFrame,
    summary: pd.DataFrame,
    flags: pd.DataFrame,
    metadata: dict,
    max_time_min: int,
) -> List[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    successful_fits = int(metrics["gompertz_fit_status"].astype(str).str.startswith("success").sum())
    failed_fits = int(len(metrics) - successful_fits)
    nonblank_wells = corrected_data["well"].nunique()
    run_summary = pd.DataFrame(
        [
            {
                "source_filename": source_path.name,
                "run_id": corrected_data["run_id"].iloc[0],
                "species": corrected_data["species"].iloc[0],
                "biological_replicate": corrected_data["biological_replicate"].iloc[0],
                "analysis_endpoint_min": max_time_min,
                "analysis_endpoint_h": max_time_min / 60.0,
                "total_number_of_wells": int(measurement_data["well"].nunique()),
                "number_of_blank_wells": int(measurement_data.loc[measurement_data["treatment_type"].eq("blank"), "well"].nunique()),
                "number_of_nonblank_wells": int(nonblank_wells),
                "number_of_retained_time_points": int(corrected_data["time_min"].nunique()),
                "number_of_processed_nonblank_measurements": int(len(corrected_data)),
                "number_of_QC_flags": int(len(flags)),
                "number_of_successful_fits": successful_fits,
                "number_of_failed_fits": failed_fits,
                "selected_sheet": metadata.get("selected_sheet", ""),
                "instrument_test_id": metadata.get("Test ID", ""),
                "instrument_date": metadata.get("Date", ""),
                "instrument_time": metadata.get("Time", ""),
                "instrument_id1": metadata.get("ID1", ""),
            }
        ]
    )

    paths = {
        "run_summary": output_dir / "run_summary.csv",
        "blank_qc": output_dir / "blank_qc_summary.csv",
        "blank_trace": output_dir / "blank_trace_used.csv",
        "flags": output_dir / "quality_control_flags.csv",
        "corrected": output_dir / "blank_corrected_growth_data.csv",
        "metrics": output_dir / "technical_well_growth_metrics.csv",
        "summary": output_dir / "biological_replicate_summary.csv",
    }
    run_summary.to_csv(paths["run_summary"], index=False)
    blank_qc.to_csv(paths["blank_qc"], index=False)
    blank_trace.to_csv(paths["blank_trace"], index=False)
    flags.to_csv(paths["flags"], index=False)
    corrected_data.to_csv(paths["corrected"], index=False)
    metrics.to_csv(paths["metrics"], index=False)
    summary.to_csv(paths["summary"], index=False)
    return list(paths.values())


def run_analysis(
    input_path: Path | str,
    species: str,
    biological_replicate: str,
    max_time_min: int = DEFAULT_MAX_TIME_MIN,
    output_parent: Path | str = PROJECT_ROOT / "results" / "drug_screening",
    large_change_threshold: float = DEFAULT_LARGE_CHANGE_THRESHOLD,
) -> dict:
    """Run the full single-workbook drug-screening workflow."""

    input_path = Path(input_path)
    run_id = build_run_id(species, biological_replicate)
    output_dir = Path(output_parent) / species / run_id

    measurement_data, metadata = import_drug_screening_workbook(
        input_path,
        species=species,
        biological_replicate=biological_replicate,
        max_time_min=max_time_min,
    )
    validate_drug_screening_measurements(measurement_data, max_time_min=max_time_min)

    blank_correction = apply_time_matched_blank_correction(
        measurement_data=measurement_data,
        blank_well_ids=BLANK_WELLS,
        time_column="time_min",
        raw_od_column="od600",
    )
    corrected_data = blank_correction.corrected_data.copy()
    validate_corrected_drug_screening_data(corrected_data, max_time_min=max_time_min)
    blank_qc, blank_trace = prepare_blank_outputs(blank_correction)

    metrics = calculate_technical_well_metrics(
        corrected_data,
        run_id=run_id,
        biological_replicate=biological_replicate,
        max_time_min=max_time_min,
    )
    summary = summarise_drug_screening_biological_replicate(metrics)
    flags = build_quality_control_flags(
        measurement_data,
        corrected_data,
        metrics,
        max_time_min=max_time_min,
        large_change_threshold=large_change_threshold,
    )
    corrected_data, metrics = add_qc_flag_columns(corrected_data, metrics, flags)
    summary = summarise_drug_screening_biological_replicate(metrics)

    table_paths = write_outputs(
        output_dir,
        input_path,
        measurement_data,
        blank_qc,
        blank_trace,
        corrected_data,
        metrics,
        summary,
        flags,
        metadata,
        max_time_min=max_time_min,
    )
    figure_paths = []
    figure_paths.extend(plot_blank_well_qc(measurement_data, blank_trace, output_dir))
    figure_paths.extend(plot_individual_growth_curves(corrected_data, output_dir))
    figure_paths.extend(plot_mean_sd_growth_curves(corrected_data, output_dir))
    figure_paths.extend(plot_gompertz_fits(corrected_data, metrics, output_dir))

    return {
        "run_id": run_id,
        "output_dir": output_dir,
        "measurement_data": measurement_data,
        "corrected_data": corrected_data,
        "metrics": metrics,
        "summary": summary,
        "flags": flags,
        "blank_trace": blank_trace,
        "table_paths": table_paths,
        "figure_paths": figure_paths,
    }


def main() -> int:
    """Command-line entry point for a single drug-screening run."""
    args = parse_args()
    result = run_analysis(
        input_path=args.input,
        species=args.species,
        biological_replicate=args.biological_replicate,
        max_time_min=args.max_time_min,
        output_parent=args.output_dir,
        large_change_threshold=args.large_change_threshold,
    )
    corrected = result["corrected_data"]
    metrics = result["metrics"]
    summary = result["summary"]
    flags = result["flags"]
    successful_fits = int(metrics["gompertz_fit_status"].astype(str).str.startswith("success").sum())
    failed_fits = int(len(metrics) - successful_fits)

    print("Drug-screening single-run validation summary")
    print(f"- run_id: {result['run_id']}")
    print(f"- output folder: {result['output_dir']}")
    print(f"- total wells: {result['measurement_data']['well'].nunique()}")
    print(f"- blank wells: {result['measurement_data'].loc[result['measurement_data']['treatment_type'].eq('blank'), 'well'].nunique()}")
    print(f"- nonblank wells: {corrected['well'].nunique()}")
    print(f"- retained time points per well: {corrected['time_min'].nunique()}")
    print(f"- processed nonblank measurements: {len(corrected)}")
    print(f"- technical-well metrics rows: {len(metrics)}")
    print(f"- biological-replicate summary rows: {len(summary)}")
    print(f"- QC flags: {len(flags)}")
    print(f"- successful Gompertz fits: {successful_fits}")
    print(f"- failed Gompertz fits: {failed_fits}")
    vehicle_mean = summary.loc[summary["condition_id"].eq("vehicle_1pct_DMSO"), "relative_AUC_percent_mean"]
    if not vehicle_mean.empty:
        print(f"- vehicle-control mean relative AUC: {vehicle_mean.iloc[0]:.6g}%")
    print("")
    print("Saved outputs")
    for path in result["table_paths"] + result["figure_paths"]:
        print(f"- {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
