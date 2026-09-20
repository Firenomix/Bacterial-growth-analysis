"""Reusable numerical and validation routines for bacterial growth analyses.

This module contains experiment-independent operations shared by the command-line
workflows in :mod:`analyses`: parsing plate-reader exports, validating plate and
time-course structure, applying time-matched blank correction, integrating
observed blank-corrected OD600, fitting the modified Gompertz model, and drawing
quality-control growth curves.

The functions do not select experimental groups or perform inferential
statistics. Raw measurements are copied into derived tables and are never
modified in place; negative blank-corrected values are intentionally retained.
"""

from __future__ import annotations

import math
import re
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Sequence, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import OptimizeWarning, curve_fit


EXPECTED_TIME_MIN = list(range(0, 1021, 5))
EXPECTED_DMSO_PERCENT = [0.0, 0.25, 0.5, 1.0, 2.0, 4.0]
EXPECTED_SPECIES = {"E_coli", "B_subtilis"}
DMSO_ONLY_TIME_MIN = list(range(0, 996, 5))
DMSO_ONLY_DMSO_PERCENT = [0.0, 0.25, 0.5, 1.0, 2.0]
DMSO_ONLY_REPLICATES_BY_DMSO = {
    0.0: 5,
    0.25: 3,
    0.5: 3,
    1.0: 3,
    2.0: 3,
}
DISPLAY_SPECIES = {
    "E_coli": "E. coli",
    "B_subtilis": "B. subtilis",
}
DEFAULT_A2_EXCLUSION_REASON = "Abnormal upward blank drift compared with A1 and A3."
OUTPUT_COLUMNS = [
    "experiment_id",
    "well",
    "species",
    "condition_label",
    "dmso_percent",
    "technical_replicate",
    "time_min",
    "od600",
    "original_content",
    "original_group",
]

SPECIES_MAP = {
    "E. Coli": "E_coli",
    "B. Subtilis": "B_subtilis",
}


class GrowthDataValidationError(ValueError):
    """Raised when imported growth data fails experiment-design validation."""


@dataclass(frozen=True)
class ValidationSummary:
    """Compact validation summary for saving or printing."""

    unique_wells: int
    measurements_per_well: int
    time_min_start: int
    time_min_end: int
    time_step_min: int
    species_count: int
    dmso_condition_count_per_species: dict
    replicate_count_per_species_condition: dict
    od600_missing_count: int
    od600_non_numeric_count: int

    def to_records(self) -> List[dict]:
        records = [
            {"check": "unique_wells", "value": self.unique_wells},
            {"check": "measurements_per_well", "value": self.measurements_per_well},
            {"check": "time_min_start", "value": self.time_min_start},
            {"check": "time_min_end", "value": self.time_min_end},
            {"check": "time_step_min", "value": self.time_step_min},
            {"check": "species_count", "value": self.species_count},
            {"check": "od600_missing_count", "value": self.od600_missing_count},
            {"check": "od600_non_numeric_count", "value": self.od600_non_numeric_count},
        ]
        for species, count in self.dmso_condition_count_per_species.items():
            records.append(
                {
                    "check": "dmso_condition_count",
                    "species": species,
                    "value": count,
                }
            )
        for key, count in self.replicate_count_per_species_condition.items():
            species, dmso_percent = key
            records.append(
                {
                    "check": "technical_replicate_count",
                    "species": species,
                    "dmso_percent": dmso_percent,
                    "value": count,
                }
            )
        return records


@dataclass(frozen=True)
class BlankCorrectionResult:
    """Outputs from time-matched blank correction."""

    corrected_data: pd.DataFrame
    blank_qc_summary: pd.DataFrame
    blank_trace_used: pd.DataFrame


def parse_time_label(label: object) -> int:
    """Parse BMG time labels such as ``0 h`` or ``1 h 5 min`` into minutes."""

    text = str(label).strip()
    hours_match = re.search(r"(\d+)\s*h", text, flags=re.IGNORECASE)
    minutes_match = re.search(r"(\d+)\s*min", text, flags=re.IGNORECASE)

    if not hours_match and not minutes_match:
        raise ValueError(f"Could not parse time label: {label!r}")

    hours = int(hours_match.group(1)) if hours_match else 0
    minutes = int(minutes_match.group(1)) if minutes_match else 0
    return hours * 60 + minutes


def construct_well(well_row: object, well_col: object) -> str:
    """Construct a plate well identifier, for example ``B`` + ``2`` -> ``B2``."""

    row = str(well_row).strip().upper()
    col_text = str(well_col).strip()
    if not row or not col_text:
        raise ValueError(f"Missing well row or column: row={well_row!r}, col={well_col!r}")
    return f"{row}{int(float(col_text))}"


def parse_dmso_percent(content: object) -> float:
    """Extract numeric DMSO percent, treating Control as 0%."""

    text = str(content).strip()
    if text == "Control":
        return 0.0
    match = re.fullmatch(r"DMSO\s+(\d+(?:\.\d+)?)%", text)
    if not match:
        raise ValueError(f"Could not parse DMSO treatment label: {content!r}")
    return float(match.group(1))


def standardize_species(group: object) -> str:
    """Standardize original Group labels to analysis-safe species names."""

    text = str(group).strip()
    try:
        return SPECIES_MAP[text]
    except KeyError as exc:
        raise ValueError(f"Unknown Group label: {group!r}") from exc


def import_bmg_clariostar_csv(
    input_path: Union[str, Path],
    experiment_id: Optional[str] = None,
    include_blank_rows: bool = False,
) -> pd.DataFrame:
    """Import an annotated BMG CLARIOstar CSV export into tidy long-form data."""

    input_path = Path(input_path)
    experiment_id = experiment_id or input_path.stem

    raw = pd.read_csv(input_path, header=None, dtype=str, keep_default_na=False, engine="python")
    if raw.shape[0] < 13:
        raise ValueError(f"Expected at least 13 logical rows, found {raw.shape[0]}")

    time_labels = raw.iloc[11, 4:].tolist()
    time_min = [parse_time_label(label) for label in time_labels if str(label).strip()]
    measurement_count = len(time_min)
    data_wide = raw.iloc[12:, : 4 + measurement_count].copy()

    data_wide.columns = ["well_row", "well_col", "original_content", "original_group"] + [
        f"t_{minute}" for minute in time_min
    ]
    data_wide = data_wide[data_wide["well_row"].astype(str).str.strip() != ""].copy()
    data_wide["well"] = [
        construct_well(row, col)
        for row, col in zip(data_wide["well_row"], data_wide["well_col"])
    ]
    data_wide["well_col_numeric"] = pd.to_numeric(data_wide["well_col"], errors="raise")
    data_wide["condition_label"] = data_wide["original_content"].astype(str).str.strip()
    data_wide["is_blank"] = (
        data_wide["condition_label"].eq("Blank")
        & data_wide["original_group"].astype(str).str.strip().eq("Blank")
    )

    blank_wide = data_wide[data_wide["is_blank"]].copy()
    blank_wide["species"] = "Blank"
    blank_wide["dmso_percent"] = math.nan
    blank_wide["technical_replicate"] = math.nan

    sample_wide = data_wide[~data_wide["is_blank"]].copy()
    sample_wide["species"] = sample_wide["original_group"].map(standardize_species)
    sample_wide["dmso_percent"] = sample_wide["condition_label"].map(parse_dmso_percent)

    sample_wide = sample_wide.sort_values(
        ["species", "dmso_percent", "well_row", "well_col_numeric"],
        kind="stable",
    ).copy()
    sample_wide["technical_replicate"] = (
        sample_wide.groupby(["species", "dmso_percent"]).cumcount() + 1
    )
    data_wide = pd.concat([blank_wide, sample_wide], ignore_index=True, sort=False)
    if not include_blank_rows:
        data_wide = data_wide[~data_wide["is_blank"]].copy()

    measurement_columns = [f"t_{minute}" for minute in time_min]
    tidy = data_wide.melt(
        id_vars=[
            "well",
            "species",
            "condition_label",
            "dmso_percent",
            "technical_replicate",
            "original_content",
            "original_group",
        ],
        value_vars=measurement_columns,
        var_name="time_label_internal",
        value_name="od600_raw",
    )
    tidy["time_min"] = tidy["time_label_internal"].str.replace(r"^t_", "", regex=True).astype(int)
    tidy["od600"] = pd.to_numeric(tidy["od600_raw"].astype(str).str.strip(), errors="coerce")
    tidy.insert(0, "experiment_id", experiment_id)

    if include_blank_rows:
        tidy = tidy[OUTPUT_COLUMNS].sort_values(["well", "time_min"], kind="stable")
    else:
        tidy = tidy[OUTPUT_COLUMNS].sort_values(
            ["species", "dmso_percent", "technical_replicate", "time_min"],
            kind="stable",
        )
    return tidy.reset_index(drop=True)


def _format_errors(errors: Sequence[str]) -> str:
    return "Growth data validation failed:\n- " + "\n- ".join(errors)


def _expected_replicate_count(
    dmso_percent: float,
    expected_replicates: Union[int, dict],
) -> int:
    if isinstance(expected_replicates, dict):
        return int(expected_replicates[float(dmso_percent)])
    return int(expected_replicates)


def validate_growth_data(
    data: pd.DataFrame,
    expected_unique_wells: int = 60,
    expected_time_min: Optional[Sequence[int]] = None,
    expected_dmso_percent: Optional[Sequence[float]] = None,
    expected_replicates: Union[int, dict] = 5,
) -> ValidationSummary:
    """Validate tidy growth data against the expected DMSO-tolerance design."""

    expected_time_min = list(expected_time_min or EXPECTED_TIME_MIN)
    expected_dmso_percent = [float(value) for value in (expected_dmso_percent or EXPECTED_DMSO_PERCENT)]
    errors: List[str] = []
    required_columns = set(OUTPUT_COLUMNS)
    missing_columns = required_columns.difference(data.columns)
    if missing_columns:
        errors.append(f"Missing required columns: {sorted(missing_columns)}")
        raise GrowthDataValidationError(_format_errors(errors))

    unique_wells = int(data["well"].nunique())
    if unique_wells != expected_unique_wells:
        errors.append(f"Expected {expected_unique_wells} unique wells, found {unique_wells}")

    counts_per_well = data.groupby("well")["time_min"].count()
    unique_counts = sorted(counts_per_well.unique().tolist())
    measurements_per_well = unique_counts[0] if len(unique_counts) == 1 else -1
    expected_measurements_per_well = len(expected_time_min)
    if unique_counts != [expected_measurements_per_well]:
        errors.append(
            f"Expected {expected_measurements_per_well} measurements per well, found counts {unique_counts}"
        )

    observed_times = sorted(data["time_min"].drop_duplicates().astype(int).tolist())
    if observed_times != expected_time_min:
        errors.append(
            "Expected time values "
            f"from {expected_time_min[0]} to {expected_time_min[-1]} minutes "
            "in 5-minute intervals"
        )
    time_min_start = observed_times[0] if observed_times else -1
    time_min_end = observed_times[-1] if observed_times else -1
    time_step_min = (
        observed_times[1] - observed_times[0]
        if len(observed_times) > 1
        else -1
    )

    species = set(data["species"].drop_duplicates().tolist())
    if species != EXPECTED_SPECIES:
        errors.append(f"Expected species {sorted(EXPECTED_SPECIES)}, found {sorted(species)}")

    dmso_counts = (
        data.drop_duplicates(["species", "dmso_percent"])
        .groupby("species")["dmso_percent"]
        .nunique()
        .to_dict()
    )
    for species_name in EXPECTED_SPECIES:
        count = int(dmso_counts.get(species_name, 0))
        if count != len(expected_dmso_percent):
            errors.append(
                f"Expected {len(expected_dmso_percent)} DMSO concentrations for {species_name}, found {count}"
            )

    observed_concentrations = sorted(data["dmso_percent"].drop_duplicates().astype(float).tolist())
    if observed_concentrations != expected_dmso_percent:
        errors.append(
            "Expected DMSO concentrations "
            f"{expected_dmso_percent}, found {observed_concentrations}"
        )

    replicate_counts_series = (
        data.drop_duplicates(["species", "dmso_percent", "technical_replicate"])
        .groupby(["species", "dmso_percent"])["technical_replicate"]
        .nunique()
    )
    replicate_counts = {
        (species_name, float(dmso_percent)): int(count)
        for (species_name, dmso_percent), count in replicate_counts_series.items()
    }
    expected_pairs = {
        (species_name, dmso_percent)
        for species_name in EXPECTED_SPECIES
        for dmso_percent in expected_dmso_percent
    }
    for key in sorted(expected_pairs):
        count = replicate_counts.get(key, 0)
        expected_count = _expected_replicate_count(key[1], expected_replicates)
        if count != expected_count:
            errors.append(
                f"Expected {expected_count} technical replicates for {key[0]} at {key[1]}% DMSO, found {count}"
            )

    od600_missing_count = int(data["od600"].isna().sum())
    od600_non_numeric_count = od600_missing_count
    if od600_missing_count:
        errors.append(f"Expected all OD600 values to be numeric and present, found {od600_missing_count} missing/non-numeric")

    summary = ValidationSummary(
        unique_wells=unique_wells,
        measurements_per_well=int(measurements_per_well),
        time_min_start=int(time_min_start),
        time_min_end=int(time_min_end),
        time_step_min=int(time_step_min),
        species_count=len(species),
        dmso_condition_count_per_species={key: int(value) for key, value in dmso_counts.items()},
        replicate_count_per_species_condition=replicate_counts,
        od600_missing_count=od600_missing_count,
        od600_non_numeric_count=od600_non_numeric_count,
    )

    if errors:
        raise GrowthDataValidationError(_format_errors(errors))
    return summary


def flag_quality_issues(data: pd.DataFrame, large_change_threshold: float = 0.2) -> pd.DataFrame:
    """Flag OD600 saturation-range values and large consecutive-reading changes."""

    ordered = data.sort_values(["well", "time_min"], kind="stable").copy()
    ordered["previous_od600"] = ordered.groupby("well")["od600"].shift(1)
    ordered["delta_od600"] = ordered["od600"] - ordered["previous_od600"]

    high_od = ordered[ordered["od600"] >= 1.0].copy()
    high_od["flag_type"] = "od600_greater_or_equal_1"

    large_change = ordered[ordered["delta_od600"].abs() >= large_change_threshold].copy()
    large_change["flag_type"] = "large_consecutive_change"

    columns = [
        "flag_type",
        "experiment_id",
        "well",
        "species",
        "condition_label",
        "dmso_percent",
        "technical_replicate",
        "time_min",
        "od600",
        "previous_od600",
        "delta_od600",
        "original_content",
        "original_group",
    ]
    flags = pd.concat([high_od[columns], large_change[columns]], ignore_index=True)
    return flags.sort_values(["flag_type", "species", "dmso_percent", "well", "time_min"]).reset_index(drop=True)


def apply_time_matched_blank_correction(
    measurement_data: pd.DataFrame,
    blank_well_ids: Sequence[str],
    excluded_blank_well_ids: Optional[Sequence[str]] = None,
    time_column: str = "time_min",
    raw_od_column: str = "od600",
    excluded_blank_reasons: Optional[dict] = None,
) -> BlankCorrectionResult:
    """Subtract the time-matched mean blank trace from bacterial sample wells."""

    excluded_blank_well_ids = excluded_blank_well_ids or []
    excluded_blank_reasons = excluded_blank_reasons or {}
    blank_well_ids = [str(well).strip().upper() for well in blank_well_ids]
    excluded_blank_well_ids = [str(well).strip().upper() for well in excluded_blank_well_ids]
    blank_wells_all = sorted(set(blank_well_ids).union(excluded_blank_well_ids))

    errors: List[str] = []
    required_columns = {"well", time_column, raw_od_column}
    missing_columns = required_columns.difference(measurement_data.columns)
    if missing_columns:
        errors.append(f"Missing required columns for blank correction: {sorted(missing_columns)}")
    if errors:
        raise GrowthDataValidationError(_format_errors(errors))

    data = measurement_data.copy()
    data["well"] = data["well"].astype(str).str.strip().str.upper()
    data[time_column] = pd.to_numeric(data[time_column], errors="raise")

    for well in excluded_blank_well_ids:
        reason = str(excluded_blank_reasons.get(well, "")).strip()
        if not reason:
            errors.append(f"Excluded blank well {well} must have a documented exclusion reason")

    if not blank_well_ids:
        errors.append("At least one valid blank well is required for correction")

    blank_label_mask = pd.Series(False, index=data.index)
    if {"original_content", "original_group"}.issubset(data.columns):
        blank_label_mask = (
            data["original_content"].astype(str).str.strip().eq("Blank")
            & data["original_group"].astype(str).str.strip().eq("Blank")
        )

    blank_rows = data[data["well"].isin(blank_wells_all)].copy()
    if blank_rows.empty and blank_label_mask.any():
        blank_rows = data[blank_label_mask].copy()

    observed_blank_wells = set(blank_rows["well"].drop_duplicates().tolist())
    for well in blank_wells_all:
        if well not in observed_blank_wells:
            errors.append(f"Missing required blank well {well}")

    if {"original_content", "original_group"}.issubset(data.columns):
        mislabeled = blank_rows[~blank_label_mask.loc[blank_rows.index]]
        if not mislabeled.empty:
            wells = sorted(mislabeled["well"].drop_duplicates().tolist())
            errors.append(
                "Blank wells must be labelled Content = 'Blank' and Group = 'Blank'; "
                f"mismatched wells: {wells}"
            )

    duplicate_blank = blank_rows.duplicated(["well", time_column], keep=False)
    if duplicate_blank.any():
        duplicate_pairs = (
            blank_rows.loc[duplicate_blank, ["well", time_column]]
            .drop_duplicates()
            .sort_values(["well", time_column])
        )
        errors.append(
            "Duplicate blank measurements for the same well and time: "
            f"{duplicate_pairs.to_dict(orient='records')}"
        )

    blank_rows[raw_od_column] = pd.to_numeric(blank_rows[raw_od_column], errors="coerce")
    if blank_rows[raw_od_column].isna().any():
        errors.append("Blank OD values must be numeric and present")

    used_blank_rows = blank_rows[blank_rows["well"].isin(blank_well_ids)].copy()
    if used_blank_rows.empty:
        errors.append("At least one valid blank well is required for correction")

    if errors:
        raise GrowthDataValidationError(_format_errors(errors))

    sample_mask = ~data["well"].isin(blank_wells_all)
    if blank_label_mask.any():
        sample_mask = sample_mask & ~blank_label_mask
    sample_rows = data[sample_mask].copy()
    sample_row_count = len(sample_rows)

    sample_rows[raw_od_column] = pd.to_numeric(sample_rows[raw_od_column], errors="coerce")
    if sample_rows[raw_od_column].isna().any():
        raise GrowthDataValidationError(
            _format_errors(["Sample OD values must be numeric and present"])
        )

    sample_times = sorted(sample_rows[time_column].drop_duplicates().tolist())
    for well in blank_wells_all:
        well_times = sorted(blank_rows.loc[blank_rows["well"].eq(well), time_column].tolist())
        if well_times != sample_times:
            raise GrowthDataValidationError(
                _format_errors(
                    [
                        "Blank and sample time points must match; "
                        f"well {well} does not match sample time points"
                    ]
                )
            )

    blank_trace_wide = used_blank_rows.pivot(
        index=time_column,
        columns="well",
        values=raw_od_column,
    ).sort_index()
    blank_trace_wide["mean_blank_OD_used"] = blank_trace_wide[blank_well_ids].mean(axis=1)
    blank_trace_used = blank_trace_wide.reset_index()
    blank_trace_used["time_h"] = blank_trace_used[time_column] / 60.0
    rename_map = {well: f"{well}_raw_OD" for well in blank_well_ids}
    blank_trace_used = blank_trace_used.rename(columns=rename_map)
    blank_trace_columns = [time_column, "time_h"] + [
        f"{well}_raw_OD" for well in blank_well_ids
    ] + ["mean_blank_OD_used"]
    blank_trace_used = blank_trace_used[blank_trace_columns]

    blank_qc_records = []
    for well in blank_wells_all:
        well_data = blank_rows[blank_rows["well"].eq(well)].sort_values(time_column)
        values = well_data[raw_od_column]
        included = well in blank_well_ids
        exclusion_reason = "" if included else excluded_blank_reasons.get(well, "")
        blank_qc_records.append(
            {
                "well": well,
                "included_in_correction": included,
                "exclusion_reason": exclusion_reason,
                "number_of_measurements": int(values.count()),
                "starting_OD": values.iloc[0],
                "ending_OD": values.iloc[-1],
                "mean_OD": values.mean(),
                "SD_OD": values.std(),
                "minimum_OD": values.min(),
                "maximum_OD": values.max(),
                "change_from_start_to_end": values.iloc[-1] - values.iloc[0],
            }
        )
    blank_qc_summary = pd.DataFrame(blank_qc_records).sort_values("well").reset_index(drop=True)

    correction_lookup = blank_trace_wide["mean_blank_OD_used"].rename("blank_OD600").reset_index()
    corrected = sample_rows.merge(correction_lookup, on=time_column, how="left", validate="many_to_one")
    if len(corrected) != sample_row_count:
        raise GrowthDataValidationError(_format_errors(["No sample rows may be lost during blank correction"]))
    if corrected["blank_OD600"].isna().any():
        raise GrowthDataValidationError(
            _format_errors(["Every sample row must receive a time-matched blank value"])
        )

    corrected["time_h"] = corrected[time_column] / 60.0
    corrected["raw_OD600"] = corrected[raw_od_column]
    corrected["corrected_OD600"] = corrected["raw_OD600"] - corrected["blank_OD600"]
    corrected["condition"] = corrected["condition_label"]

    corrected_columns = [
        "well",
        "species",
        "condition",
        "dmso_percent",
        time_column,
        "time_h",
        "raw_OD600",
        "blank_OD600",
        "corrected_OD600",
    ]
    optional_columns = [
        "experiment_id",
        "run_id",
        "biological_replicate",
        "technical_replicate",
        "original_content",
        "original_group",
        "treatment_type",
        "compound",
        "compound_concentration_uM",
        "ampicillin_concentration_ug_mL",
        "condition_id",
    ]
    corrected_columns.extend([column for column in optional_columns if column in corrected.columns])
    corrected = corrected[corrected_columns].sort_values(
        ["species", "dmso_percent", "technical_replicate", time_column],
        kind="stable",
    )

    if corrected["well"].isin(blank_wells_all).any():
        raise GrowthDataValidationError(
            _format_errors(["Blank wells must not be included in corrected sample data"])
        )

    return BlankCorrectionResult(
        corrected_data=corrected.reset_index(drop=True),
        blank_qc_summary=blank_qc_summary,
        blank_trace_used=blank_trace_used.reset_index(drop=True),
    )


def validate_corrected_growth_data(
    data: pd.DataFrame,
    expected_unique_wells: int = 60,
    expected_time_min: Optional[Sequence[int]] = None,
    expected_dmso_percent: Optional[Sequence[float]] = None,
    expected_replicates: Union[int, dict] = 5,
) -> None:
    """Validate blank-corrected bacterial sample data before quantification."""

    expected_time_min = list(expected_time_min or EXPECTED_TIME_MIN)
    expected_dmso_percent = [float(value) for value in (expected_dmso_percent or EXPECTED_DMSO_PERCENT)]
    errors: List[str] = []
    required_columns = {
        "well",
        "species",
        "condition",
        "dmso_percent",
        "technical_replicate",
        "time_min",
        "time_h",
        "raw_OD600",
        "blank_OD600",
        "corrected_OD600",
    }
    missing_columns = required_columns.difference(data.columns)
    if missing_columns:
        errors.append(f"Missing required corrected-data columns: {sorted(missing_columns)}")
        raise GrowthDataValidationError(_format_errors(errors))

    if data["well"].isin(["A1", "A2", "A3"]).any() or data["species"].eq("Blank").any():
        errors.append("Blank wells must be absent from corrected bacterial sample data")

    unique_wells = int(data["well"].nunique())
    if unique_wells != expected_unique_wells:
        errors.append(f"Expected {expected_unique_wells} bacterial sample wells, found {unique_wells}")

    data = data.copy()
    data["time_h"] = pd.to_numeric(data["time_h"], errors="coerce")
    data["corrected_OD600"] = pd.to_numeric(data["corrected_OD600"], errors="coerce")
    if data["time_h"].isna().any():
        errors.append("time_h values must be numeric")
    if data["corrected_OD600"].isna().any():
        errors.append("corrected_OD600 values must be numeric and present")

    expected_time_h = [minute / 60.0 for minute in expected_time_min]
    for well, well_data in data.groupby("well", sort=True):
        observed = well_data["time_h"].tolist()
        if len(observed) != len(set(observed)):
            errors.append(f"Well {well} has duplicated time_h values")
        if observed != sorted(observed):
            errors.append(f"Well {well} time_h values are not ordered")
        if sorted(observed) != expected_time_h:
            errors.append(f"Well {well} does not contain the expected 0-17 h time points")

    replicate_counts = (
        data.drop_duplicates(["species", "dmso_percent", "well"])
        .groupby(["species", "dmso_percent"])["well"]
        .nunique()
    )
    expected_pairs = {
        (species_name, dmso_percent)
        for species_name in EXPECTED_SPECIES
        for dmso_percent in expected_dmso_percent
    }
    for key in sorted(expected_pairs):
        count = int(replicate_counts.get(key, 0))
        expected_count = _expected_replicate_count(key[1], expected_replicates)
        if count != expected_count:
            errors.append(
                f"Expected {expected_count} technical wells for {key[0]} at {key[1]}% DMSO, found {count}"
            )

    if errors:
        raise GrowthDataValidationError(_format_errors(errors))


def calculate_trapezoidal_auc(
    curve_data: pd.DataFrame,
    start_h: float,
    end_h: float,
    time_column: str = "time_h",
    response_column: str = "corrected_OD600",
) -> float:
    """Calculate trapezoidal AUC over an exact closed time window."""

    curve = curve_data.sort_values(time_column).copy()
    time = pd.to_numeric(curve[time_column], errors="coerce")
    response = pd.to_numeric(curve[response_column], errors="coerce")
    if time.isna().any() or response.isna().any():
        raise GrowthDataValidationError(
            _format_errors([f"{time_column} and {response_column} must be numeric for AUC"])
        )
    if time.duplicated().any():
        raise GrowthDataValidationError(_format_errors([f"Duplicate {time_column} values in curve"]))
    if start_h not in set(time.tolist()) or end_h not in set(time.tolist()):
        raise GrowthDataValidationError(
            _format_errors([f"Required AUC boundaries {start_h:g} h and {end_h:g} h are missing"])
        )

    window = curve[(time >= start_h) & (time <= end_h)].copy()
    x = pd.to_numeric(window[time_column], errors="raise").to_numpy(dtype=float)
    y = pd.to_numeric(window[response_column], errors="raise").to_numpy(dtype=float)
    if hasattr(np, "trapezoid"):
        return float(np.trapezoid(y, x))
    return float(np.trapz(y, x))


def calculate_w_response(
    curve_data: pd.DataFrame,
    time_column: str = "time_h",
    response_column: str = "corrected_OD600",
) -> pd.DataFrame:
    """Calculate W(t) as corrected OD600 at t minus corrected OD600 at t=0."""

    curve = curve_data.sort_values(time_column).copy()
    if curve.empty:
        raise GrowthDataValidationError(_format_errors(["Cannot calculate W(t) for an empty curve"]))
    baseline = pd.to_numeric(curve.iloc[0][response_column], errors="raise")
    curve["W"] = pd.to_numeric(curve[response_column], errors="raise") - baseline
    return curve


def modified_gompertz(time_h: np.ndarray, A: float, Kz: float, TLag: float) -> np.ndarray:
    """Modified Gompertz model: A*exp(-exp((e*Kz/A)*(TLag-t)+1))."""

    time_h = np.asarray(time_h, dtype=float)
    return A * np.exp(-np.exp(((math.e * Kz / A) * (TLag - time_h)) + 1.0))


def fit_one_growth_curve(
    curve_data: pd.DataFrame,
    time_column: str = "time_h",
    response_column: str = "corrected_OD600",
) -> dict:
    """Fit one corrected growth curve to the exact modified Gompertz equation."""

    try:
        curve = calculate_w_response(curve_data, time_column, response_column)
        time = pd.to_numeric(curve[time_column], errors="raise").to_numpy(dtype=float)
        response = pd.to_numeric(curve["W"], errors="raise").to_numpy(dtype=float)
        if len(time) < 4:
            raise GrowthDataValidationError("At least four time points are required for Gompertz fitting")

        response_range = float(np.nanmax(response) - np.nanmin(response))
        if response_range <= 0:
            raise GrowthDataValidationError("Growth response has no positive dynamic range")
        max_response = float(np.nanmax(response))
        A_start = max(max_response, response_range, 1e-3)
        slopes = np.diff(response) / np.diff(time)
        positive_slopes = slopes[np.isfinite(slopes) & (slopes > 0)]
        Kz_start = float(np.nanmax(positive_slopes)) if len(positive_slopes) else 0.05
        threshold = 0.05 * A_start
        above_threshold = np.where(response >= threshold)[0]
        TLag_start = float(time[above_threshold[0]]) if len(above_threshold) else 1.0
        TLag_start = min(max(TLag_start, 0.0), 17.0)

        upper_A = max(A_start * 10.0, 5.0)
        upper_Kz = max(Kz_start * 20.0, 5.0)
        with warnings.catch_warnings(record=True) as caught_warnings:
            warnings.simplefilter("always", OptimizeWarning)
            params, _ = curve_fit(
                modified_gompertz,
                time,
                response,
                p0=[A_start, Kz_start, TLag_start],
                bounds=([1e-9, 1e-9, 0.0], [upper_A, upper_Kz, 17.0]),
                maxfev=20000,
            )

        fitted = modified_gompertz(time, *params)
        residual_sum_squares = float(np.sum((response - fitted) ** 2))
        total_sum_squares = float(np.sum((response - np.mean(response)) ** 2))
        r2 = math.nan if total_sum_squares == 0 else 1.0 - residual_sum_squares / total_sum_squares
        warning_messages = [str(warning.message) for warning in caught_warnings]
        status = "success" if not warning_messages else "success_with_warning: " + "; ".join(warning_messages)
        return {
            "Kz_OD600_per_h": float(params[1]),
            "TLag_h": float(params[2]),
            "A_OD600": float(params[0]),
            "gompertz_R2": float(r2),
            "gompertz_fit_status": status,
        }
    except Exception as exc:
        return {
            "Kz_OD600_per_h": math.nan,
            "TLag_h": math.nan,
            "A_OD600": math.nan,
            "gompertz_R2": math.nan,
            "gompertz_fit_status": f"fit_failed: {exc}",
        }


def fit_all_wells(
    corrected_data: pd.DataFrame,
    biological_replicate: str,
    time_column: str = "time_h",
    response_column: str = "corrected_OD600",
    auc_windows: Optional[Sequence[tuple]] = None,
    expected_unique_wells: int = 60,
    expected_time_min: Optional[Sequence[int]] = None,
    expected_dmso_percent: Optional[Sequence[float]] = None,
    expected_replicates: Union[int, dict] = 5,
) -> pd.DataFrame:
    """Calculate AUCs and Gompertz fits independently for every bacterial well."""

    auc_windows = list(auc_windows or [("0_17h", 0.0, 17.0), ("0_12h", 0.0, 12.0)])
    validate_corrected_growth_data(
        corrected_data,
        expected_unique_wells=expected_unique_wells,
        expected_time_min=expected_time_min,
        expected_dmso_percent=expected_dmso_percent,
        expected_replicates=expected_replicates,
    )
    original_raw = corrected_data["raw_OD600"].copy(deep=True)
    original_corrected = corrected_data["corrected_OD600"].copy(deep=True)
    records = []

    for well, well_data in corrected_data.groupby("well", sort=True):
        well_data = well_data.sort_values(time_column)
        first = well_data.iloc[0]
        auc_values = {}
        for window_label, start_h, end_h in auc_windows:
            auc_values[f"AUC_{window_label}_OD_h"] = calculate_trapezoidal_auc(
                well_data,
                start_h,
                end_h,
                time_column,
                response_column,
            )
        fit = fit_one_growth_curve(well_data, time_column, response_column)
        records.append(
            {
                "biological_replicate": biological_replicate,
                "well": well,
                "species": first["species"],
                "condition": first["condition"],
                "DMSO_percent": float(first["dmso_percent"]),
                "n_timepoints": int(well_data[time_column].nunique()),
                **auc_values,
                **fit,
            }
        )

    pd.testing.assert_series_equal(corrected_data["raw_OD600"], original_raw)
    pd.testing.assert_series_equal(corrected_data["corrected_OD600"], original_corrected)
    metrics = pd.DataFrame.from_records(records)
    metrics = add_relative_auc(metrics)
    auc_output_columns = []
    for window_label, _, _ in auc_windows:
        auc_output_columns.extend(
            [
                f"AUC_{window_label}_OD_h",
                f"relative_AUC_{window_label}_percent",
            ]
        )
    return metrics[
        [
            "biological_replicate",
            "well",
            "species",
            "condition",
            "DMSO_percent",
            "n_timepoints",
            *auc_output_columns,
            "Kz_OD600_per_h",
            "TLag_h",
            "A_OD600",
            "gompertz_R2",
            "gompertz_fit_status",
        ]
    ].sort_values(["species", "DMSO_percent", "well"], kind="stable").reset_index(drop=True)


def add_relative_auc(metrics: pd.DataFrame) -> pd.DataFrame:
    """Normalise AUCs to the matching species, biological replicate and AUC window."""

    metrics = metrics.copy()
    auc_columns = [
        column
        for column in metrics.columns
        if column.startswith("AUC_") and column.endswith("_OD_h")
    ]
    for auc_column in auc_columns:
        relative_column = "relative_" + auc_column.removesuffix("_OD_h") + "_percent"
        control_means = (
            metrics[metrics["DMSO_percent"].eq(0.0)]
            .groupby(["biological_replicate", "species"])[auc_column]
            .mean()
            .rename("control_mean")
            .reset_index()
        )
        metrics = metrics.merge(control_means, on=["biological_replicate", "species"], how="left")
        metrics[relative_column] = 100.0 * metrics[auc_column] / metrics["control_mean"]
        metrics = metrics.drop(columns=["control_mean"])
    return metrics


def summarise_technical_wells(metrics: pd.DataFrame) -> pd.DataFrame:
    """Average five technical wells into one biological-replicate result per group."""

    records = []
    auc_columns = [
        column
        for column in metrics.columns
        if column.startswith("AUC_") and column.endswith("_OD_h")
    ]
    for (biological_replicate, species, dmso_percent), group in metrics.groupby(
        ["biological_replicate", "species", "DMSO_percent"],
        sort=True,
    ):
        valid_fit_mask = (
            group["gompertz_fit_status"].astype(str).str.startswith("success")
            & group[["Kz_OD600_per_h", "TLag_h", "A_OD600", "gompertz_R2"]].notna().all(axis=1)
        )
        valid_fits = group[valid_fit_mask]
        if valid_fits.empty:
            warnings.warn(
                f"No valid Gompertz fits for {biological_replicate} {species} {dmso_percent:g}% DMSO",
                RuntimeWarning,
            )
        condition_values = group["condition"].drop_duplicates().tolist()
        condition = condition_values[0] if condition_values else ""
        record = {
            "biological_replicate": biological_replicate,
            "species": species,
            "condition": condition,
            "DMSO_percent": float(dmso_percent),
            "n_technical_wells": int(group["well"].nunique()),
        }
        for auc_column in auc_columns:
            relative_column = "relative_" + auc_column.removesuffix("_OD_h") + "_percent"
            record[f"{auc_column}_mean"] = group[auc_column].mean()
            record[f"{auc_column}_SD_technical"] = group[auc_column].std(ddof=1)
            if relative_column in group.columns:
                record[f"{relative_column}_mean"] = group[relative_column].mean()
                record[f"{relative_column}_SD_technical"] = group[relative_column].std(ddof=1)
        record.update(
            {
                "Kz_OD600_per_h_mean": valid_fits["Kz_OD600_per_h"].mean(),
                "Kz_OD600_per_h_SD_technical": valid_fits["Kz_OD600_per_h"].std(ddof=1),
                "TLag_h_mean": valid_fits["TLag_h"].mean(),
                "TLag_h_SD_technical": valid_fits["TLag_h"].std(ddof=1),
                "A_OD600_mean": valid_fits["A_OD600"].mean(),
                "A_OD600_SD_technical": valid_fits["A_OD600"].std(ddof=1),
                "gompertz_R2_mean": valid_fits["gompertz_R2"].mean(),
                "n_valid_gompertz_fits": int(valid_fit_mask.sum()),
                "n_failed_gompertz_fits": int((~valid_fit_mask).sum()),
            }
        )
        records.append(record)

    return pd.DataFrame.from_records(records).sort_values(
        ["species", "DMSO_percent"],
        kind="stable",
    ).reset_index(drop=True)


def save_quality_summary(
    summary: ValidationSummary,
    flags: pd.DataFrame,
    output_dir: Union[str, Path],
    large_change_threshold: float,
) -> Path:
    """Save a compact CSV QC summary with validation and flag counts."""

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    records = summary.to_records()
    records.extend(
        [
            {
                "check": "od600_greater_or_equal_1_count",
                "value": int((flags["flag_type"] == "od600_greater_or_equal_1").sum()),
            },
            {
                "check": "large_consecutive_change_count",
                "value": int((flags["flag_type"] == "large_consecutive_change").sum()),
                "threshold": large_change_threshold,
            },
        ]
    )
    summary_path = output_dir / "quality_control_summary.csv"
    pd.DataFrame(records).to_csv(summary_path, index=False)
    return summary_path


def _condition_color_map(data: pd.DataFrame) -> dict:
    concentrations = sorted(data["dmso_percent"].drop_duplicates().tolist())
    cmap = plt.get_cmap("viridis", len(concentrations))
    return {concentration: cmap(index) for index, concentration in enumerate(concentrations)}


def _fixed_condition_color_map(dmso_order: Optional[Sequence[float]] = None) -> dict:
    dmso_order = [float(value) for value in (dmso_order or EXPECTED_DMSO_PERCENT)]
    cmap = plt.get_cmap("viridis", len(dmso_order))
    return {concentration: cmap(index) for index, concentration in enumerate(dmso_order)}


def _set_corrected_y_limits(ax: plt.Axes, data: pd.DataFrame) -> None:
    y_min = float(data["corrected_OD600"].min())
    y_max = float(data["corrected_OD600"].max())
    padding = max((y_max - y_min) * 0.05, 0.01)
    bottom = y_min - padding if y_min < 0 else 0
    ax.set_ylim(bottom=bottom, top=y_max + padding)


def plot_blank_well_qc(
    measurement_data: pd.DataFrame,
    blank_trace_used: pd.DataFrame,
    blank_qc_summary: pd.DataFrame,
    output_dir: Union[str, Path],
    time_column: str = "time_min",
    raw_od_column: str = "od600",
) -> Path:
    """Save a QC plot of included and excluded blank wells."""

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    data = measurement_data.copy()
    data["well"] = data["well"].astype(str).str.strip().str.upper()
    blank_wells = blank_qc_summary["well"].tolist()
    blank_data = data[data["well"].isin(blank_wells)].copy()
    blank_data["time_h"] = blank_data[time_column] / 60.0

    included_lookup = blank_qc_summary.set_index("well")["included_in_correction"].to_dict()
    reason_lookup = blank_qc_summary.set_index("well")["exclusion_reason"].to_dict()
    fig, ax = plt.subplots(figsize=(10, 6))
    for well in blank_wells:
        well_data = blank_data[blank_data["well"].eq(well)].sort_values(time_column)
        included = bool(included_lookup[well])
        if included:
            label = f"{well} included"
            ax.plot(
                well_data["time_h"],
                well_data[raw_od_column],
                linewidth=1.8,
                label=label,
            )
        else:
            reason = reason_lookup[well]
            label = f"{well} excluded: {reason}"
            ax.plot(
                well_data["time_h"],
                well_data[raw_od_column],
                color="crimson",
                linestyle="--",
                linewidth=1.8,
                label=label,
            )

    ax.plot(
        blank_trace_used["time_h"],
        blank_trace_used["mean_blank_OD_used"],
        color="black",
        linewidth=2.4,
        label="Mean blank used for correction",
    )
    ax.set_xlabel("Time (h)")
    ax.set_ylabel("OD600")
    ax.set_title("Blank well QC")
    ax.grid(alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    path = output_dir / "blank_well_qc.png"
    fig.savefig(path, dpi=300)
    plt.close(fig)
    return path


def plot_individual_raw_growth_curves(data: pd.DataFrame, output_dir: Union[str, Path]) -> List[Path]:
    """Save individual technical-replicate growth-curve plots separated by species."""

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    paths: List[Path] = []
    color_map = _condition_color_map(data)

    for species, species_data in data.groupby("species", sort=True):
        fig, ax = plt.subplots(figsize=(10, 6))
        for (dmso_percent, well), well_data in species_data.groupby(["dmso_percent", "well"], sort=True):
            ax.plot(
                well_data["time_min"],
                well_data["od600"],
                color=color_map[dmso_percent],
                alpha=0.45,
                linewidth=1.0,
            )
        handles = [
            plt.Line2D([0], [0], color=color_map[dmso], linewidth=2, label=f"{dmso:g}%")
            for dmso in sorted(species_data["dmso_percent"].drop_duplicates())
        ]
        ax.legend(handles=handles, title="DMSO", frameon=False, ncol=3)
        ax.set_title(f"Individual raw growth curves: {species}")
        ax.set_xlabel("Time (min)")
        ax.set_ylabel("OD600")
        ax.set_xlim(0, 1020)
        ax.grid(alpha=0.25)
        fig.tight_layout()
        path = output_dir / f"individual_raw_growth_curves_{species}.png"
        fig.savefig(path, dpi=300)
        plt.close(fig)
        paths.append(path)

    return paths


def plot_individual_blank_corrected_growth_curves(
    data: pd.DataFrame,
    output_dir: Union[str, Path],
    dmso_order: Optional[Sequence[float]] = None,
) -> List[Path]:
    """Save individual blank-corrected technical-replicate plots by species."""

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    paths: List[Path] = []
    dmso_order = [float(value) for value in (dmso_order or EXPECTED_DMSO_PERCENT)]
    color_map = _fixed_condition_color_map(dmso_order)

    for species in sorted(data["species"].drop_duplicates()):
        species_data = data[data["species"].eq(species)]
        fig, ax = plt.subplots(figsize=(10, 6))
        for (dmso_percent, well), well_data in species_data.groupby(["dmso_percent", "well"], sort=True):
            ax.plot(
                well_data["time_h"],
                well_data["corrected_OD600"],
                color=color_map[float(dmso_percent)],
                alpha=0.45,
                linewidth=1.0,
            )
        handles = [
            plt.Line2D([0], [0], color=color_map[dmso], linewidth=2, label=f"{dmso:g}%")
            for dmso in dmso_order
        ]
        ax.legend(handles=handles, title="DMSO", frameon=False, ncol=3)
        ax.set_title(f"Individual blank-corrected growth curves: {DISPLAY_SPECIES.get(species, species)}")
        ax.set_xlabel("Time (h)")
        ax.set_ylabel("Blank-corrected OD600")
        ax.set_xlim(0, 17)
        _set_corrected_y_limits(ax, species_data)
        ax.grid(alpha=0.25)
        fig.tight_layout()
        path = output_dir / f"individual_blank_corrected_growth_curves_{species}.png"
        fig.savefig(path, dpi=300)
        plt.close(fig)
        paths.append(path)

    return paths


def plot_mean_sd_growth_curves(data: pd.DataFrame, output_dir: Union[str, Path]) -> List[Path]:
    """Save mean +/- SD raw growth-curve plots separated by species."""

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    paths: List[Path] = []
    color_map = _condition_color_map(data)

    summary = (
        data.groupby(["species", "dmso_percent", "condition_label", "time_min"], as_index=False)
        .agg(mean_od600=("od600", "mean"), sd_od600=("od600", "std"))
    )

    for species, species_data in summary.groupby("species", sort=True):
        fig, ax = plt.subplots(figsize=(10, 6))
        for dmso_percent, curve in species_data.groupby("dmso_percent", sort=True):
            curve = curve.sort_values("time_min")
            color = color_map[dmso_percent]
            x = curve["time_min"].to_numpy()
            mean = curve["mean_od600"].to_numpy()
            sd = curve["sd_od600"].fillna(0).to_numpy()
            ax.plot(x, mean, color=color, linewidth=2, label=f"{dmso_percent:g}%")
            ax.fill_between(x, mean - sd, mean + sd, color=color, alpha=0.18, linewidth=0)
        ax.legend(title="DMSO", frameon=False, ncol=3)
        ax.set_title(f"Mean +/- SD raw growth curves: {species}")
        ax.set_xlabel("Time (min)")
        ax.set_ylabel("OD600")
        ax.set_xlim(0, 1020)
        ax.grid(alpha=0.25)
        fig.tight_layout()
        path = output_dir / f"mean_sd_growth_curves_{species}.png"
        fig.savefig(path, dpi=300)
        plt.close(fig)
        paths.append(path)

    return paths


def plot_mean_sd_blank_corrected_growth_curves(
    data: pd.DataFrame,
    output_dir: Union[str, Path],
    dmso_order: Optional[Sequence[float]] = None,
) -> List[Path]:
    """Save mean +/- SD blank-corrected growth-curve plots by species."""

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    paths: List[Path] = []
    dmso_order = [float(value) for value in (dmso_order or EXPECTED_DMSO_PERCENT)]
    color_map = _fixed_condition_color_map(dmso_order)

    summary = (
        data.groupby(["species", "dmso_percent", "condition", "time_min", "time_h"], as_index=False)
        .agg(mean_corrected_OD600=("corrected_OD600", "mean"), sd_corrected_OD600=("corrected_OD600", "std"))
    )

    for species in sorted(summary["species"].drop_duplicates()):
        species_data = summary[summary["species"].eq(species)]
        fig, ax = plt.subplots(figsize=(10, 6))
        for dmso_percent, curve in species_data.groupby("dmso_percent", sort=True):
            curve = curve.sort_values("time_min")
            color = color_map[float(dmso_percent)]
            x = curve["time_h"].to_numpy()
            mean = curve["mean_corrected_OD600"].to_numpy()
            sd = curve["sd_corrected_OD600"].fillna(0).to_numpy()
            ax.plot(x, mean, color=color, linewidth=2, label=f"{dmso_percent:g}%")
            ax.fill_between(x, mean - sd, mean + sd, color=color, alpha=0.18, linewidth=0)
        handles = [
            plt.Line2D([0], [0], color=color_map[dmso], linewidth=2, label=f"{dmso:g}%")
            for dmso in dmso_order
        ]
        ax.legend(handles=handles, title="DMSO", frameon=False, ncol=3)
        ax.set_title(f"Mean +/- SD blank-corrected growth curves: {DISPLAY_SPECIES.get(species, species)}")
        ax.set_xlabel("Time (h)")
        ax.set_ylabel("Blank-corrected OD600")
        ax.set_xlim(0, 17)
        _set_corrected_y_limits(ax, data[data["species"].eq(species)])
        ax.grid(alpha=0.25)
        fig.tight_layout()
        path = output_dir / f"mean_sd_blank_corrected_growth_curves_{species}.png"
        fig.savefig(path, dpi=300)
        plt.close(fig)
        paths.append(path)

    return paths
