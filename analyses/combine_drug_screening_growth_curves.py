"""Combine drug-screening growth curves descriptively across BR1-BR3.

Inputs are the three ``blank_corrected_growth_data.csv`` files created by
``drug_screening_single_run.py`` for one species. Technical wells are first
averaged within each biological replicate; matched treatment curves are then
summarized across biological-replicate means. Regimens that differ between
runs remain separate and are shown as run-level curves.

This module performs no inferential statistics and never pools technical wells
across biological replicates. Its CSV outputs retain the intermediate
biological-replicate curves so the hierarchy is auditable.
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path
from textwrap import fill
from typing import Dict, List, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
ANALYSES_DIR = PROJECT_ROOT / "analyses"
for directory in [SRC_DIR, ANALYSES_DIR]:
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))

from growth_analysis import DISPLAY_SPECIES, GrowthDataValidationError  # noqa: E402
from drug_screening_single_run import (  # noqa: E402
    ACCEPTED_BIOLOGICAL_REPLICATES,
    ACCEPTED_SPECIES,
    DEFAULT_MAX_TIME_MIN,
    condition_color_map,
    condition_order,
    expected_time_points,
)


EXPECTED_COMBINATION_COMPOUNDS = ["C1", "C2", "C3", "C4"]
EXPECTED_TECHNICAL_WELLS_PER_CONDITION = 3
TREATMENT_FIELDS = [
    "treatment_type",
    "compound",
    "compound_concentration_uM",
    "ampicillin_concentration_ug_mL",
    "condition_id",
]
CONDITION_FIELDS = TREATMENT_FIELDS + ["condition"]
REQUIRED_COLUMNS = {
    "species",
    "biological_replicate",
    "well",
    "technical_replicate",
    "time_min",
    "time_h",
    "corrected_OD600",
    *CONDITION_FIELDS,
}
FIGURE_NOTE_MATCHED = (
    "Mean ± SD, n=3 independent biological replicates; each biological replicate "
    "is the mean of three technical wells."
)
FIGURE_NOTE_COMBINATION = (
    "Lines show the mean of three technical wells within each biological replicate; "
    "shaded bands show the within-run technical-well SD. Positive-control ampicillin "
    "traces are shown for reference. BR1 used a different compound-ampicillin "
    "regimen from BR2 and BR3, so the runs were not averaged."
)


def parse_args() -> argparse.Namespace:
    """Parse three per-run CSV inputs and the requested species/output path."""
    parser = argparse.ArgumentParser(
        description=(
            "Combine single-run drug-screening blank-corrected growth data into "
            "descriptive BR-level growth-curve summaries. No inferential tests are run."
        )
    )
    parser.add_argument("--input-br1", required=True, help="BR1 blank_corrected_growth_data.csv")
    parser.add_argument("--input-br2", required=True, help="BR2 blank_corrected_growth_data.csv")
    parser.add_argument("--input-br3", required=True, help="BR3 blank_corrected_growth_data.csv")
    parser.add_argument("--species", required=True, choices=sorted(ACCEPTED_SPECIES))
    parser.add_argument(
        "--output-dir",
        default=None,
        help=(
            "Optional output directory. Default: "
            "results/drug_screening/<species>/combined_biological_replicates/"
        ),
    )
    parser.add_argument(
        "--max-time-min",
        type=int,
        default=DEFAULT_MAX_TIME_MIN,
        help="Inclusive expected endpoint in minutes. Default: 1045.",
    )
    return parser.parse_args()


def _format_errors(errors: Sequence[str]) -> str:
    return "Combined drug-screening validation failed:\n- " + "\n- ".join(errors)


def read_blank_corrected_inputs(
    input_paths: Dict[str, Path | str],
    requested_species: str,
    max_time_min: int = DEFAULT_MAX_TIME_MIN,
) -> pd.DataFrame:
    """Read BR1-BR3 blank-corrected outputs without modifying them."""

    if requested_species not in ACCEPTED_SPECIES:
        raise GrowthDataValidationError(f"Unsupported species {requested_species!r}.")
    frames = []
    for expected_replicate, path in input_paths.items():
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Input CSV does not exist: {path}")
        frame = pd.read_csv(path)
        frame["source_file"] = str(path)
        frame["input_argument_replicate"] = expected_replicate
        frames.append(frame)
    combined = pd.concat(frames, ignore_index=True)
    validate_combined_inputs(combined, requested_species=requested_species, max_time_min=max_time_min)
    return combined


def validate_combined_inputs(
    data: pd.DataFrame,
    requested_species: str,
    max_time_min: int = DEFAULT_MAX_TIME_MIN,
) -> None:
    """Validate BR1-BR3 single-run corrected growth files before combining."""

    errors: List[str] = []
    missing = REQUIRED_COLUMNS.difference(data.columns)
    if missing:
        errors.append(f"Missing required columns: {sorted(missing)}")
        raise GrowthDataValidationError(_format_errors(errors))

    cleaned = data.copy()
    cleaned["time_min"] = pd.to_numeric(cleaned["time_min"], errors="coerce")
    cleaned["time_h"] = pd.to_numeric(cleaned["time_h"], errors="coerce")
    cleaned["corrected_OD600"] = pd.to_numeric(cleaned["corrected_OD600"], errors="coerce")
    if cleaned[["time_min", "time_h", "corrected_OD600"]].isna().any().any():
        errors.append("time_min, time_h and corrected_OD600 must be numeric and present.")

    observed_species = sorted(cleaned["species"].drop_duplicates().astype(str).tolist())
    if observed_species != [requested_species]:
        errors.append(f"All inputs must belong to requested species {requested_species}; found {observed_species}.")

    if "source_file" in cleaned.columns:
        by_source_replicates = cleaned.groupby("source_file")["biological_replicate"].nunique()
        if not by_source_replicates.eq(1).all():
            errors.append("Each input file must contain exactly one biological_replicate value.")

    observed_replicates = sorted(cleaned["biological_replicate"].drop_duplicates().astype(str).tolist())
    expected_replicates = sorted(ACCEPTED_BIOLOGICAL_REPLICATES)
    if observed_replicates != expected_replicates:
        errors.append(f"BR1, BR2 and BR3 must each be present exactly once; found {observed_replicates}.")

    if "source_file" in cleaned.columns:
        source_count_by_replicate = cleaned.drop_duplicates(["source_file", "biological_replicate"]).groupby(
            "biological_replicate"
        )["source_file"].nunique()
        if not source_count_by_replicate.reindex(expected_replicates, fill_value=0).eq(1).all():
            errors.append("Each of BR1, BR2 and BR3 must come from exactly one input file.")

    expected_times = expected_time_points(max_time_min)
    for replicate, replicate_data in cleaned.groupby("biological_replicate", sort=True):
        observed_times = sorted(replicate_data["time_min"].dropna().astype(int).drop_duplicates().tolist())
        if observed_times != expected_times:
            errors.append(
                f"{replicate} must contain the exact 0 to {max_time_min} minute time series; "
                f"found {len(observed_times)} unique time points."
            )
        duplicate_measurements = replicate_data.duplicated(["well", "time_min"], keep=False)
        if duplicate_measurements.any():
            errors.append(f"{replicate} contains duplicate measurements for the same well and time point.")

        condition_counts = (
            replicate_data.drop_duplicates(["condition_id", "well"])
            .groupby("condition_id")["well"]
            .nunique()
        )
        bad_counts = condition_counts[condition_counts.ne(EXPECTED_TECHNICAL_WELLS_PER_CONDITION)]
        if not bad_counts.empty:
            errors.append(
                f"{replicate} must contain three technical wells per condition; "
                f"found {bad_counts.to_dict()}."
            )

        for well, well_data in replicate_data.groupby("well", sort=True):
            well_times = well_data.sort_values("time_min")["time_min"].astype(int).tolist()
            if well_times != expected_times:
                errors.append(f"{replicate} well {well} does not contain one measurement at every time point.")

    condition_definitions = cleaned.drop_duplicates(TREATMENT_FIELDS)
    inconsistent_condition_ids = (
        condition_definitions.groupby("condition_id", dropna=False)
        .size()
        .loc[lambda values: values.gt(1)]
    )
    if not inconsistent_condition_ids.empty:
        errors.append(
            "condition_id must not map to multiple treatment definitions: "
            f"{inconsistent_condition_ids.to_dict()}."
        )

    if errors:
        raise GrowthDataValidationError(_format_errors(errors))


def build_condition_matching_summary(data: pd.DataFrame) -> pd.DataFrame:
    """Summarise condition presence and eligibility for matched BR mean +/- SD curves."""

    records = []
    definitions = data.drop_duplicates(CONDITION_FIELDS).copy()
    for _, condition_row in definitions.sort_values(CONDITION_FIELDS, kind="stable").iterrows():
        condition_id = condition_row["condition_id"]
        subset = data[data["condition_id"].eq(condition_id)]
        present = {
            replicate: bool(subset["biological_replicate"].eq(replicate).any())
            for replicate in ["BR1", "BR2", "BR3"]
        }
        n_replicates = int(sum(present.values()))
        is_combination = condition_row["treatment_type"] == "combination"
        eligible = (not is_combination) and n_replicates == 3
        if eligible:
            reason = ""
        elif is_combination:
            reason = (
                "Combination regimen is not identical across BR1-BR3; "
                "compound and ampicillin concentrations are kept as separate conditions."
            )
        else:
            missing = [replicate for replicate, is_present in present.items() if not is_present]
            reason = f"Missing biological replicate(s): {', '.join(missing)}."
        records.append(
            {
                "species": condition_row["species"] if "species" in condition_row else subset["species"].iloc[0],
                **{field: condition_row[field] for field in CONDITION_FIELDS},
                "present_in_BR1": present["BR1"],
                "present_in_BR2": present["BR2"],
                "present_in_BR3": present["BR3"],
                "number_of_biological_replicates": n_replicates,
                "eligible_for_combined_mean_SD": eligible,
                "exclusion_reason": reason,
            }
        )
    return pd.DataFrame.from_records(records)


def matched_condition_ids(condition_matching: pd.DataFrame) -> List[str]:
    """Return condition IDs eligible for matched BR mean +/- SD curves."""

    matched = condition_matching[condition_matching["eligible_for_combined_mean_SD"]].copy()
    if matched.empty:
        raise GrowthDataValidationError(
            _format_errors(
                [
                    "No matched conditions are present in BR1, BR2 and BR3."
                ]
            )
        )
    return matched["condition_id"].tolist()


def calculate_biological_replicate_mean_curves(
    data: pd.DataFrame,
    matched_ids: Sequence[str],
) -> pd.DataFrame:
    """Average technical-well OD600 at each time within each biological run."""

    matched = data[data["condition_id"].isin(matched_ids)].copy()
    group_columns = ["species", "biological_replicate", *CONDITION_FIELDS, "time_min", "time_h"]
    result = (
        matched.groupby(group_columns, dropna=False, as_index=False, sort=False)
        .agg(
            number_of_technical_wells=("well", "nunique"),
            biological_replicate_mean_corrected_OD600=("corrected_OD600", "mean"),
            technical_well_SD_corrected_OD600=("corrected_OD600", lambda values: values.std(ddof=1)),
        )
    )
    bad = result[result["number_of_technical_wells"].ne(EXPECTED_TECHNICAL_WELLS_PER_CONDITION)]
    if not bad.empty:
        raise GrowthDataValidationError(
            _format_errors(["Matched BR mean curves must use three technical wells per BR/condition/time."])
        )
    return result


def calculate_combined_mean_sd_curves(
    br_mean_curves: pd.DataFrame,
    max_time_min: int = DEFAULT_MAX_TIME_MIN,
) -> pd.DataFrame:
    """Calculate matched-run means and between-BR sample SD from BR curves."""

    id_columns = ["species", *CONDITION_FIELDS, "time_min", "time_h"]
    pivot = br_mean_curves.pivot(
        index=id_columns,
        columns="biological_replicate",
        values="biological_replicate_mean_corrected_OD600",
    ).reset_index()
    pivot.columns.name = None
    for replicate in ["BR1", "BR2", "BR3"]:
        if replicate not in pivot.columns:
            raise GrowthDataValidationError(_format_errors([f"Missing {replicate} BR mean column."]))
    values = pivot[["BR1", "BR2", "BR3"]]
    pivot = pivot.rename(
        columns={
            "BR1": "BR1_mean_corrected_OD600",
            "BR2": "BR2_mean_corrected_OD600",
            "BR3": "BR3_mean_corrected_OD600",
        }
    )
    pivot["combined_mean_corrected_OD600"] = values.mean(axis=1)
    pivot["combined_SD_corrected_OD600"] = values.std(axis=1, ddof=1)
    pivot["number_of_biological_replicates"] = values.notna().sum(axis=1).astype(int)
    expected_condition_count = int(br_mean_curves["condition_id"].nunique())
    expected_rows = expected_condition_count * len(expected_time_points(max_time_min))
    if len(pivot) != expected_rows:
        raise GrowthDataValidationError(
            _format_errors([f"Expected {expected_rows} combined matched rows, found {len(pivot)}."])
        )
    if not pivot["number_of_biological_replicates"].eq(3).all():
        raise GrowthDataValidationError(_format_errors(["Every combined row must contain three BR means."]))
    return pivot.sort_values(["condition_id", "time_min"], kind="stable").reset_index(drop=True)


def calculate_combination_run_level_curves(
    data: pd.DataFrame,
    max_time_min: int = DEFAULT_MAX_TIME_MIN,
) -> pd.DataFrame:
    """Calculate run-level mean +/- technical SD for combinations and positive controls."""

    run_level = data[data["treatment_type"].isin(["combination", "ampicillin_control"])].copy()
    group_columns = [
        "species",
        "biological_replicate",
        "treatment_type",
        "compound",
        "compound_concentration_uM",
        "ampicillin_concentration_ug_mL",
        "condition_id",
        "condition",
        "time_min",
        "time_h",
    ]
    result = (
        run_level.groupby(group_columns, dropna=False, as_index=False, sort=False)
        .agg(
            number_of_technical_wells=("well", "nunique"),
            run_mean_corrected_OD600=("corrected_OD600", "mean"),
            within_run_technical_SD_corrected_OD600=("corrected_OD600", lambda values: values.std(ddof=1)),
        )
    )
    expected_groups = run_level.drop_duplicates(["biological_replicate", "condition_id"]).shape[0]
    expected_rows = expected_groups * len(expected_time_points(max_time_min))
    if len(result) != expected_rows:
        raise GrowthDataValidationError(
            _format_errors([f"Expected {expected_rows} run-level combination/control rows, found {len(result)}."])
        )
    if not result["number_of_technical_wells"].eq(EXPECTED_TECHNICAL_WELLS_PER_CONDITION).all():
        raise GrowthDataValidationError(
            _format_errors(["Run-level combination/control curves must use three technical wells per BR/condition/time."])
        )
    return result.sort_values(
        ["treatment_type", "compound", "condition_id", "biological_replicate", "time_min"],
        kind="stable",
    ).reset_index(drop=True)


def _display_condition(row: pd.Series) -> str:
    if row["treatment_type"] == "vehicle_control":
        return "1% DMSO vehicle"
    if row["treatment_type"] == "ampicillin_control":
        return f"Ampicillin {row['ampicillin_concentration_ug_mL']:g} µg/mL"
    if row["treatment_type"] in {"compound_monotherapy", "penag_monotherapy"}:
        return f"{row['compound']} {row['compound_concentration_uM']:g} µM"
    return str(row.get("condition", row["condition_id"]))


def _matched_panel_groups(combined: pd.DataFrame) -> List[Tuple[str, pd.DataFrame]]:
    panels = [
        ("Controls", combined[combined["treatment_type"].isin(["vehicle_control", "ampicillin_control"])]),
        ("C1", combined[(combined["compound"].eq("C1")) & combined["treatment_type"].eq("compound_monotherapy")]),
        ("C2", combined[(combined["compound"].eq("C2")) & combined["treatment_type"].eq("compound_monotherapy")]),
        ("C3", combined[(combined["compound"].eq("C3")) & combined["treatment_type"].eq("compound_monotherapy")]),
        ("C4", combined[(combined["compound"].eq("C4")) & combined["treatment_type"].eq("compound_monotherapy")]),
        ("PenAg", combined[combined["compound"].eq("PenAg")]),
    ]
    return panels


def _set_common_y_limits(axes: Sequence[plt.Axes], values: pd.Series) -> None:
    y_min = float(values.min())
    y_max = float(values.max())
    span = max(y_max - y_min, 0.05)
    bottom = y_min - span * 0.05 if y_min < 0 else 0
    top = y_max + span * 0.08
    for ax in axes:
        ax.set_ylim(bottom, top)


def plot_matched_combined_curves(combined: pd.DataFrame, output_dir: Path, species: str) -> List[Path]:
    """Plot matched-condition combined BR mean +/- between-BR SD curves."""

    plot_source = combined.copy()
    colors = condition_color_map(plot_source)
    order = condition_order(plot_source)
    panels = _matched_panel_groups(plot_source)
    fig, axes = plt.subplots(2, 3, figsize=(16, 9), sharex=True)
    axes_flat = axes.flatten()
    for ax, (title, panel) in zip(axes_flat, panels):
        panel_ids = [condition_id for condition_id in order if condition_id in set(panel["condition_id"])]
        for condition_id in panel_ids:
            curve = panel[panel["condition_id"].eq(condition_id)].sort_values("time_h")
            if curve.empty:
                continue
            x = curve["time_h"].to_numpy(dtype=float)
            mean = curve["combined_mean_corrected_OD600"].to_numpy(dtype=float)
            sd = curve["combined_SD_corrected_OD600"].fillna(0).to_numpy(dtype=float)
            label = _display_condition(curve.iloc[0])
            ax.plot(x, mean, color=colors[condition_id], linewidth=1.8, label=label)
            ax.fill_between(x, mean - sd, mean + sd, color=colors[condition_id], alpha=0.18, linewidth=0)
        ax.set_title(title)
        ax.set_xlabel("Time (h)")
        ax.set_ylabel("Blank-corrected OD600")
        ax.grid(axis="y", alpha=0.25)
        ax.legend(frameon=False, fontsize=7)
    y_values = pd.concat(
        [
            combined["combined_mean_corrected_OD600"],
            combined["combined_mean_corrected_OD600"] - combined["combined_SD_corrected_OD600"].fillna(0),
            combined["combined_mean_corrected_OD600"] + combined["combined_SD_corrected_OD600"].fillna(0),
        ],
        ignore_index=True,
    )
    _set_common_y_limits(axes_flat, y_values)
    species_label = DISPLAY_SPECIES.get(species, species)
    fig.suptitle(f"Combined mean ± SD blank-corrected growth curves: {species_label}", y=0.985)
    fig.text(0.5, 0.025, fill(FIGURE_NOTE_MATCHED, width=125), ha="center", fontsize=9)
    fig.tight_layout(rect=[0, 0.07, 1, 0.96])
    png = output_dir / f"combined_mean_sd_blank_corrected_growth_curves_{species}.png"
    pdf = output_dir / f"combined_mean_sd_blank_corrected_growth_curves_{species}.pdf"
    fig.savefig(png, dpi=300)
    fig.savefig(pdf)
    plt.close(fig)
    return [png, pdf]


def _combination_label(row: pd.Series) -> str:
    if row["treatment_type"] == "ampicillin_control":
        return f"{row['biological_replicate']}: ampicillin {row['ampicillin_concentration_ug_mL']:g} ug/mL"
    return (
        f"{row['biological_replicate']}: {row['compound_concentration_uM']:g} µM "
        f"{row['compound']} + {row['ampicillin_concentration_ug_mL']:g} µg/mL ampicillin"
    )


def plot_combination_run_level_curves(combination: pd.DataFrame, output_dir: Path, species: str) -> List[Path]:
    """Plot unmatched combination regimens with positive-control ampicillin references."""

    replicate_styles = {
        "BR1": {"color": "#1f77b4", "linestyle": "-"},
        "BR2": {"color": "#d62728", "linestyle": "--"},
        "BR3": {"color": "#2ca02c", "linestyle": ":"},
    }
    fig, axes = plt.subplots(2, 3, figsize=(16, 9), sharex=True, sharey=True)
    axes_flat = axes.flatten()
    panels = [("Positive controls", combination[combination["treatment_type"].eq("ampicillin_control")])]
    panels.extend(
        (
            compound,
            combination[(combination["compound"].eq(compound)) & combination["treatment_type"].eq("combination")],
        )
        for compound in EXPECTED_COMBINATION_COMPOUNDS
    )
    for ax, (title, panel) in zip(axes_flat, panels):
        for replicate in ["BR1", "BR2", "BR3"]:
            replicate_panel = panel[panel["biological_replicate"].eq(replicate)]
            for condition_id, curve in replicate_panel.groupby("condition_id", sort=False):
                curve = curve.sort_values("time_h")
                if curve.empty:
                    continue
                style = replicate_styles[replicate].copy()
                if title == "Positive controls" and condition_id.endswith("8ug_mL"):
                    style["linestyle"] = "-."
                x = curve["time_h"].to_numpy(dtype=float)
                mean = curve["run_mean_corrected_OD600"].to_numpy(dtype=float)
                sd = curve["within_run_technical_SD_corrected_OD600"].fillna(0).to_numpy(dtype=float)
                label = _combination_label(curve.iloc[0])
                ax.plot(x, mean, linewidth=1.8, label=label, **style)
                ax.fill_between(x, mean - sd, mean + sd, color=style["color"], alpha=0.14, linewidth=0)
        ax.set_title(title)
        ax.set_xlabel("Time (h)")
        ax.set_ylabel("Blank-corrected OD600")
        ax.grid(axis="y", alpha=0.25)
        ax.legend(frameon=False, fontsize=7)
    for ax in axes_flat[len(panels):]:
        ax.axis("off")
    y_values = pd.concat(
        [
            combination["run_mean_corrected_OD600"],
            combination["run_mean_corrected_OD600"] - combination["within_run_technical_SD_corrected_OD600"].fillna(0),
            combination["run_mean_corrected_OD600"] + combination["within_run_technical_SD_corrected_OD600"].fillna(0),
        ],
        ignore_index=True,
    )
    _set_common_y_limits(axes_flat, y_values)
    species_label = DISPLAY_SPECIES.get(species, species)
    fig.suptitle(f"Run-level combination-treatment growth curves: {species_label}", y=0.985)
    fig.text(0.5, 0.025, fill(FIGURE_NOTE_COMBINATION, width=125), ha="center", fontsize=9)
    fig.tight_layout(rect=[0, 0.09, 1, 0.96])
    png = output_dir / f"combination_treatment_run_level_growth_curves_{species}.png"
    pdf = output_dir / f"combination_treatment_run_level_growth_curves_{species}.pdf"
    fig.savefig(png, dpi=300)
    fig.savefig(pdf)
    plt.close(fig)
    return [png, pdf]


def write_outputs(
    output_dir: Path,
    br_mean_curves: pd.DataFrame,
    combined_curves: pd.DataFrame,
    combination_curves: pd.DataFrame,
    condition_matching: pd.DataFrame,
) -> List[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = [
        output_dir / "biological_replicate_mean_growth_curves.csv",
        output_dir / "combined_mean_sd_growth_curves.csv",
        output_dir / "combination_treatment_run_level_growth_curves.csv",
        output_dir / "condition_matching_summary.csv",
    ]
    br_mean_curves.to_csv(paths[0], index=False)
    combined_curves.to_csv(paths[1], index=False)
    combination_curves.to_csv(paths[2], index=False)
    condition_matching.to_csv(paths[3], index=False)
    return paths


def run_combined_analysis(
    input_br1: Path | str,
    input_br2: Path | str,
    input_br3: Path | str,
    species: str,
    output_dir: Path | str | None = None,
    max_time_min: int = DEFAULT_MAX_TIME_MIN,
) -> dict:
    """Run the descriptive combined-growth analysis for one species."""

    if output_dir is None:
        output_dir = PROJECT_ROOT / "results" / "drug_screening" / species / "combined_biological_replicates"
    output_dir = Path(output_dir)
    data = read_blank_corrected_inputs(
        {"BR1": input_br1, "BR2": input_br2, "BR3": input_br3},
        requested_species=species,
        max_time_min=max_time_min,
    )
    validate_combined_inputs(data, requested_species=species, max_time_min=max_time_min)
    condition_matching = build_condition_matching_summary(data)
    matched_ids = matched_condition_ids(condition_matching)
    br_mean_curves = calculate_biological_replicate_mean_curves(data, matched_ids)
    combined_curves = calculate_combined_mean_sd_curves(br_mean_curves, max_time_min=max_time_min)
    combination_curves = calculate_combination_run_level_curves(data, max_time_min=max_time_min)
    table_paths = write_outputs(
        output_dir,
        br_mean_curves,
        combined_curves,
        combination_curves,
        condition_matching,
    )
    figure_paths = []
    figure_paths.extend(plot_matched_combined_curves(combined_curves, output_dir, species))
    figure_paths.extend(plot_combination_run_level_curves(combination_curves, output_dir, species))
    return {
        "species": species,
        "output_dir": output_dir,
        "data": data,
        "condition_matching": condition_matching,
        "biological_replicate_mean_curves": br_mean_curves,
        "combined_curves": combined_curves,
        "combination_curves": combination_curves,
        "table_paths": table_paths,
        "figure_paths": figure_paths,
    }


def main() -> int:
    """Command-line entry point for descriptive cross-run growth curves."""
    args = parse_args()
    result = run_combined_analysis(
        input_br1=args.input_br1,
        input_br2=args.input_br2,
        input_br3=args.input_br3,
        species=args.species,
        output_dir=args.output_dir,
        max_time_min=args.max_time_min,
    )
    matching = result["condition_matching"]
    matched_count = int(matching["eligible_for_combined_mean_SD"].sum())
    unmatched_count = int((~matching["eligible_for_combined_mean_SD"]).sum())
    print("Combined drug-screening growth-curve summary")
    print(f"- species: {args.species}")
    print(f"- output folder: {result['output_dir']}")
    print(f"- input rows: {len(result['data'])}")
    print(f"- biological replicates: {sorted(result['data']['biological_replicate'].unique().tolist())}")
    print(f"- matched conditions: {matched_count}")
    print(f"- unmatched/ineligible conditions: {unmatched_count}")
    print(f"- BR mean curve rows: {len(result['biological_replicate_mean_curves'])}")
    print(f"- combined matched rows: {len(result['combined_curves'])}")
    print(f"- combination/positive-control run-level rows: {len(result['combination_curves'])}")
    print("")
    print("Saved outputs")
    for path in result["table_paths"] + result["figure_paths"]:
        print(f"- {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
