"""Generate thesis-ready drug-screening monotherapy figures and audit tables.

This presentation layer reads the same six validated single-run directories as
``drug_screening_statistics.py`` and reuses its blocked-model implementation.
Growth curves are summarized hierarchically through biological-replicate means.
Metric panels display technical wells and BR summaries but infer only from the
BR means. The full Tukey family is calculated from the blocked-model residual;
the figures display the pre-specified vehicle comparisons.

No raw measurements, per-run metrics, or statistical result tables from the
main statistics stage are modified by this script.
"""

from __future__ import annotations

import argparse
import math
import sys
from dataclasses import dataclass
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

from growth_analysis import DISPLAY_SPECIES  # noqa: E402
from drug_screening_statistics import (  # noqa: E402
    EXPECTED_REPLICATES,
    EXPECTED_SPECIES,
    EXPECTED_TECHNICAL_WELLS,
    FAMILY_ORDER,
    KZ,
    MAX_TIME_MIN,
    RAW_AUC,
    REL_AUC,
    SCHIFF_COMPOUNDS,
    SCHIFF_CONCENTRATIONS,
    TLAG,
    DrugScreeningStatisticsError,
    FamilySpec,
    _format_errors,
    family_eligibility,
    family_dataset,
    prepare_biological_replicate_metrics,
    read_all_runs,
    randomized_block_anova,
    significance_label,
    tukey_from_block,
    validate_loaded_data,
    verify_source_hashes,
)


VEHICLE = "vehicle_1pct_DMSO"
PENAG_AUC_CONDITIONS = [VEHICLE, "PenAg_12.5uM", "PenAg_25uM", "PenAg_50uM", "PenAg_100uM"]
PENAG_TLAG_CONDITIONS = [VEHICLE, "PenAg_12.5uM", "PenAg_25uM"]
SCHIFF_AUC_CONDITIONS = [VEHICLE] + [
    f"{compound}_{concentration:g}uM"
    for compound in SCHIFF_COMPOUNDS
    for concentration in SCHIFF_CONCENTRATIONS
]
COMPOUND_COLORS = {
    "DMSO": "#374151",
    "C1": "#1f4e79",
    "C2": "#b45309",
    "C3": "#166534",
    "C4": "#6d3978",
    "PenAg_12.5uM": "#1e40af",
    "PenAg_25uM": "#9a3412",
    "PenAg_50uM": "#166534",
    "PenAg_100uM": "#6b21a8",
}


@dataclass(frozen=True)
class ThesisFamily:
    """Conditions and metrics included in one thesis model/figure family."""
    figure_family: str
    family: FamilySpec
    metric: str
    inference_metric: str
    plotted_metric: str
    planned_vehicle_comparisons: Tuple[str, ...]
    bracket_rule: str


def parse_args() -> argparse.Namespace:
    """Parse six single-run directories and the thesis-figure output path."""
    parser = argparse.ArgumentParser(
        description="Create thesis figures for drug-screening Schiff-base and PenAg monotherapy results."
    )
    parser.add_argument("--e-coli-br1", required=True)
    parser.add_argument("--e-coli-br2", required=True)
    parser.add_argument("--e-coli-br3", required=True)
    parser.add_argument("--b-subtilis-br1", required=True)
    parser.add_argument("--b-subtilis-br2", required=True)
    parser.add_argument("--b-subtilis-br3", required=True)
    parser.add_argument(
        "--output-dir",
        default=PROJECT_ROOT / "results" / "drug_screening" / "statistics" / "thesis_figures",
        help="Directory for thesis figure outputs.",
    )
    return parser.parse_args()


def build_input_map(args: argparse.Namespace) -> Dict[Tuple[str, str], Path]:
    return {
        ("E_coli", "BR1"): Path(args.e_coli_br1),
        ("E_coli", "BR2"): Path(args.e_coli_br2),
        ("E_coli", "BR3"): Path(args.e_coli_br3),
        ("B_subtilis", "BR1"): Path(args.b_subtilis_br1),
        ("B_subtilis", "BR2"): Path(args.b_subtilis_br2),
        ("B_subtilis", "BR3"): Path(args.b_subtilis_br3),
    }


def thesis_families() -> List[ThesisFamily]:
    families: List[ThesisFamily] = []
    for species in EXPECTED_SPECIES:
        for compound in SCHIFF_COMPOUNDS:
            condition_ids = tuple(FAMILY_ORDER[f"schiff_{compound}"])
            families.append(
                ThesisFamily(
                    figure_family="schiff_relative_auc",
                    family=FamilySpec(species, compound, "schiff", condition_ids),
                    metric="AUC_0_1045_min",
                    inference_metric="raw_AUC_0_1045_min",
                    plotted_metric="relative_AUC_percent",
                    planned_vehicle_comparisons=condition_ids[1:],
                    bracket_rule="significant_vehicle_comparisons_only",
                )
            )
        families.append(
            ThesisFamily(
                figure_family="penag_relative_auc",
                family=FamilySpec(species, "PenAg", "penag_auc", tuple(PENAG_AUC_CONDITIONS)),
                metric="AUC_0_1045_min",
                inference_metric="raw_AUC_0_1045_min",
                plotted_metric="relative_AUC_percent",
                planned_vehicle_comparisons=tuple(PENAG_AUC_CONDITIONS[1:]),
                bracket_rule="all_vehicle_comparisons",
            )
        )
        families.append(
            ThesisFamily(
                figure_family="penag_lag_time",
                family=FamilySpec(species, "PenAg", "penag_tlag", tuple(PENAG_TLAG_CONDITIONS)),
                metric="TLag",
                inference_metric="TLag_h",
                plotted_metric="TLag_h",
                planned_vehicle_comparisons=tuple(PENAG_TLAG_CONDITIONS[1:]),
                bracket_rule="all_vehicle_comparisons",
            )
        )
    return families


def condition_label(condition_id: str) -> str:
    if condition_id == VEHICLE:
        return "1% DMSO"
    if condition_id.startswith("PenAg"):
        return condition_id.split("_", 1)[1].replace("uM", " µM")
    compound, concentration = condition_id.split("_", 1)
    return f"{compound}\n{concentration.replace('uM', ' µM')}"


def condition_compound(condition_id: str) -> str:
    if condition_id == VEHICLE:
        return "DMSO"
    return condition_id.split("_", 1)[0]


def condition_concentration(condition_id: str) -> float:
    if condition_id == VEHICLE:
        return math.nan
    return float(condition_id.split("_", 1)[1].replace("uM", ""))


def condition_color(condition_id: str) -> str:
    if condition_id.startswith("PenAg"):
        return COMPOUND_COLORS[condition_id]
    return COMPOUND_COLORS[condition_compound(condition_id)]


def validate_thesis_inputs(growth: pd.DataFrame, technical: pd.DataFrame, summary: pd.DataFrame, runs: pd.DataFrame) -> List[str]:
    warnings = validate_loaded_data(growth, technical, summary, runs)
    errors: List[str] = []
    for species in EXPECTED_SPECIES:
        reference_times = None
        for replicate in EXPECTED_REPLICATES:
            run_growth = growth[growth["species"].eq(species) & growth["biological_replicate"].eq(replicate)]
            observed_times = tuple(sorted(pd.to_numeric(run_growth["time_min"], errors="coerce").dropna().unique().astype(int).tolist()))
            if observed_times[0] != 0 or observed_times[-1] != MAX_TIME_MIN or len(observed_times) != 210:
                errors.append(f"{species} {replicate} growth data must contain 210 time points from 0 to {MAX_TIME_MIN} min.")
            if reference_times is None:
                reference_times = observed_times
            elif observed_times != reference_times:
                errors.append(f"{species} time points differ between biological replicates.")
    if technical["treatment_type"].eq("combination").any():
        pass
    if errors:
        raise DrugScreeningStatisticsError(_format_errors(errors))
    return warnings


def run_thesis_models(br_metrics: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Run declared blocked models and retain the complete Tukey families."""
    anova_rows = []
    tukey_frames = []
    nonestimable_rows = []
    for spec in thesis_families():
        eligible, reason, data = family_eligibility(br_metrics, spec.family, spec.metric)
        if not eligible:
            for condition_id in spec.planned_vehicle_comparisons:
                nonestimable_rows.append(
                    {
                        "species": spec.family.species,
                        "compound": spec.family.compound,
                        "metric": spec.metric,
                        "model_family": spec.figure_family,
                        "vehicle_condition": VEHICLE,
                        "treatment_condition": condition_id,
                        "treatment_concentration": condition_concentration(condition_id),
                        "estimability_status": "not_estimable",
                        "reason": reason,
                        "drawn_on_figure": False,
                    }
                )
            continue
        anova = randomized_block_anova(data, spec.family, spec.metric)
        anova_record = anova.to_dict()
        anova_record["figure_family"] = spec.figure_family
        anova_record["inference_metric"] = spec.inference_metric
        anova_record["plotted_metric"] = spec.plotted_metric
        anova_rows.append(anova_record)
        tukey = tukey_from_block(data, anova, spec.family)
        tukey["figure_family"] = spec.figure_family
        tukey["inference_metric"] = spec.inference_metric
        tukey["plotted_metric"] = spec.plotted_metric
        tukey_frames.append(tukey)
    anova_df = pd.DataFrame.from_records(anova_rows)
    tukey_df = pd.concat(tukey_frames, ignore_index=True) if tukey_frames else pd.DataFrame()
    vehicle = filter_vehicle_comparisons(tukey_df, nonestimable_rows)
    return anova_df, tukey_df, vehicle


def _comparison_label(adjusted_p_value: float) -> str:
    if pd.isna(adjusted_p_value) or adjusted_p_value >= 0.05:
        return "n.s."
    return significance_label(adjusted_p_value)


def filter_vehicle_comparisons(tukey: pd.DataFrame, nonestimable_rows: Sequence[dict] | None = None) -> pd.DataFrame:
    rows = []
    nonestimable_rows = list(nonestimable_rows or [])
    for spec in thesis_families():
        panel = tukey[
            tukey["species"].eq(spec.family.species)
            & tukey["compound"].eq(spec.family.compound)
            & tukey["metric"].eq(spec.metric)
            & tukey["figure_family"].eq(spec.figure_family)
        ]
        for condition_id in spec.planned_vehicle_comparisons:
            match = panel[
                (
                    panel["condition_1"].eq(VEHICLE)
                    & panel["condition_2"].eq(condition_id)
                )
                | (
                    panel["condition_2"].eq(VEHICLE)
                    & panel["condition_1"].eq(condition_id)
                )
            ]
            if match.empty:
                continue
            row = match.iloc[0]
            vehicle_first = row["condition_1"] == VEHICLE
            vehicle_mean = float(row["mean_condition_1"] if vehicle_first else row["mean_condition_2"])
            treatment_mean = float(row["mean_condition_2"] if vehicle_first else row["mean_condition_1"])
            label = _comparison_label(float(row["Tukey_adjusted_p_value"]))
            is_sig = label != "n.s."
            drawn = is_sig if spec.figure_family == "schiff_relative_auc" else True
            rows.append(
                {
                    "species": spec.family.species,
                    "compound": spec.family.compound,
                    "metric": spec.metric,
                    "model_family": spec.figure_family,
                    "vehicle_condition": VEHICLE,
                    "treatment_condition": condition_id,
                    "treatment_concentration": condition_concentration(condition_id),
                    "vehicle_mean": vehicle_mean,
                    "treatment_mean": treatment_mean,
                    "signed_mean_difference_treatment_minus_vehicle": treatment_mean - vehicle_mean,
                    "Tukey_standard_error": row["Tukey_standard_error"],
                    "q_statistic": row["q_statistic"],
                    "Tukey_adjusted_p_value": row["Tukey_adjusted_p_value"],
                    "simultaneous_95CI_lower": row["simultaneous_95CI_lower"],
                    "simultaneous_95CI_upper": row["simultaneous_95CI_upper"],
                    "significance_label": label,
                    "estimability_status": "estimable",
                    "reason": "",
                    "drawn_on_figure": bool(drawn),
                }
            )
    rows.extend(nonestimable_rows)
    return pd.DataFrame.from_records(rows)


def thesis_metric_technical_data(technical: pd.DataFrame) -> pd.DataFrame:
    rows = []
    figure_specs = [
        ("schiff_relative_auc", "relative_AUC_percent", REL_AUC, SCHIFF_AUC_CONDITIONS),
        ("penag_relative_auc", "relative_AUC_percent", REL_AUC, PENAG_AUC_CONDITIONS),
        ("penag_lag_time", "TLag_h", TLAG, PENAG_TLAG_CONDITIONS),
    ]
    for figure_family, metric, column, condition_ids in figure_specs:
        subset = technical[technical["condition_id"].isin(condition_ids)].copy()
        if figure_family == "penag_lag_time":
            subset = subset[subset["gompertz_fit_status"].astype(str).str.startswith("success")].copy()
        for row in subset.itertuples(index=False):
            rows.append(
                {
                    "figure_family": figure_family,
                    "metric": metric,
                    "species": row.species,
                    "biological_replicate": row.biological_replicate,
                    "run_id": row.run_id,
                    "well": row.well,
                    "technical_replicate": row.technical_replicate,
                    "condition_id": row.condition_id,
                    "condition": row.condition,
                    "treatment_type": row.treatment_type,
                    "compound": row.compound,
                    "compound_concentration_uM": row.compound_concentration_uM,
                    "ampicillin_concentration_ug_mL": row.ampicillin_concentration_ug_mL,
                    "plot_value": getattr(row, column),
                    "raw_AUC_0_1045_min": getattr(row, RAW_AUC),
                    "relative_AUC_percent": getattr(row, REL_AUC),
                    "TLag_h": getattr(row, TLAG),
                    "gompertz_fit_status": row.gompertz_fit_status,
                }
            )
    return pd.DataFrame.from_records(rows)


def thesis_metric_br_data(br_metrics: pd.DataFrame) -> pd.DataFrame:
    rows = []
    figure_specs = [
        ("schiff_relative_auc", "relative_AUC_percent", "relative_AUC_mean", SCHIFF_AUC_CONDITIONS, "raw_AUC_mean"),
        ("penag_relative_auc", "relative_AUC_percent", "relative_AUC_mean", PENAG_AUC_CONDITIONS, "raw_AUC_mean"),
        ("penag_lag_time", "TLag_h", "TLag_mean", PENAG_TLAG_CONDITIONS, "TLag_mean"),
    ]
    for figure_family, metric, plot_column, condition_ids, inference_column in figure_specs:
        subset = br_metrics[br_metrics["condition_id"].isin(condition_ids)].copy()
        if figure_family == "penag_lag_time":
            subset = subset[pd.to_numeric(subset["n_valid_gompertz_fits"], errors="coerce").eq(EXPECTED_TECHNICAL_WELLS)].copy()
        for row in subset.itertuples(index=False):
            rows.append(
                {
                    "figure_family": figure_family,
                    "metric": metric,
                    "species": row.species,
                    "biological_replicate": row.biological_replicate,
                    "run_id": row.run_id,
                    "condition_id": row.condition_id,
                    "condition": row.condition,
                    "treatment_type": row.treatment_type,
                    "compound": row.compound,
                    "compound_concentration_uM": row.compound_concentration_uM,
                    "ampicillin_concentration_ug_mL": row.ampicillin_concentration_ug_mL,
                    "n_technical_wells": row.n_technical_wells,
                    "n_valid_gompertz_fits": row.n_valid_gompertz_fits,
                    "plot_value": getattr(row, plot_column),
                    "inference_value": getattr(row, inference_column),
                }
            )
    return pd.DataFrame.from_records(rows)


def summarize_growth_curves(growth: pd.DataFrame, condition_ids: Sequence[str]) -> pd.DataFrame:
    data = growth[growth["condition_id"].isin(condition_ids)].copy()
    run_means = (
        data.groupby(["species", "biological_replicate", "condition_id", "time_min", "time_h"], dropna=False)["corrected_OD600"]
        .mean()
        .rename("biological_replicate_mean_OD600")
        .reset_index()
    )
    summary = (
        run_means.groupby(["species", "condition_id", "time_min", "time_h"], dropna=False)["biological_replicate_mean_OD600"]
        .agg(combined_mean_OD600="mean", combined_SD_OD600=lambda values: values.std(ddof=1), n_biological_replicates="count")
        .reset_index()
    )
    return summary


def _plot_growth_curve_figure(
    summary: pd.DataFrame,
    species: str,
    condition_ids: Sequence[str],
    output_dir: Path,
    stem: str,
    title: str,
) -> List[Path]:
    fig, ax = plt.subplots(figsize=(8, 5))
    for condition_id in condition_ids:
        curve = summary[summary["species"].eq(species) & summary["condition_id"].eq(condition_id)].sort_values("time_min")
        if curve.empty:
            continue
        x = curve["time_h"].to_numpy(dtype=float)
        y = curve["combined_mean_OD600"].to_numpy(dtype=float)
        sd = curve["combined_SD_OD600"].to_numpy(dtype=float)
        color = condition_color(condition_id)
        ax.plot(x, y, color=color, linewidth=1.8, label=condition_label(condition_id).replace("\n", " "))
        ax.fill_between(x, y - sd, y + sd, color=color, alpha=0.16, linewidth=0)
    y_values = summary[summary["species"].eq(species)]["combined_mean_OD600"].dropna()
    if not y_values.empty:
        lower = min(float(y_values.min()), 0.0)
        upper = float(y_values.max())
        span = max(upper - lower, 0.05)
        ax.set_ylim(lower - span * 0.08, upper + span * 0.12)
    ax.set_xlim(0, MAX_TIME_MIN / 60)
    ax.set_xlabel("Time (h)")
    ax.set_ylabel("Blank-corrected OD600")
    ax.set_title(title)
    ax.legend(frameon=False, fontsize=8)
    ax.grid(axis="y", alpha=0.22)
    note = "Lines show unsmoothed means of BR1-BR3 biological-replicate means; ribbons show biological SD across BR means."
    fig.text(0.5, 0.02, fill(note, 100), ha="center", fontsize=8)
    fig.tight_layout(rect=[0, 0.06, 1, 1])
    png = output_dir / f"{stem}.png"
    pdf = output_dir / f"{stem}.pdf"
    fig.savefig(png, dpi=300)
    fig.savefig(pdf)
    plt.close(fig)
    return [png, pdf]


def add_vehicle_brackets(
    ax: plt.Axes,
    comparisons: pd.DataFrame,
    x_lookup: Dict[str, int],
    y_top: float,
    y_bottom: float,
    draw_only_significant: bool,
) -> None:
    panel = comparisons[comparisons["drawn_on_figure"].eq(True)].copy()
    if draw_only_significant:
        panel = panel[panel["significance_label"].ne("n.s.")].copy()
    if panel.empty:
        return
    panel["x_position"] = panel["treatment_condition"].map(x_lookup)
    panel = panel.sort_values("x_position", kind="stable")
    span = max(y_top - y_bottom, abs(y_top) * 0.1, 1e-6)
    bracket_height = span * 0.025
    step = span * 0.07
    start_y = y_top + span * 0.06
    for level, row in enumerate(panel.itertuples(index=False)):
        if pd.isna(row.x_position):
            continue
        x1 = x_lookup[VEHICLE]
        x2 = int(row.x_position)
        y = start_y + level * step
        label = str(row.significance_label)
        ax.plot([x1, x1, x2, x2], [y, y + bracket_height, y + bracket_height, y], color="black", linewidth=0.65, clip_on=False)
        ax.text(
            (x1 + x2) / 2,
            y + bracket_height,
            label,
            ha="center",
            va="bottom",
            fontsize=7 if label == "n.s." else 8.5,
            color="dimgray" if label == "n.s." else "black",
            clip_on=False,
        )
    ax.set_ylim(top=start_y + len(panel) * step + bracket_height * 3)


def _metric_arrays(metric_data: pd.DataFrame, condition_id: str) -> np.ndarray:
    return pd.to_numeric(metric_data[metric_data["condition_id"].eq(condition_id)]["plot_value"], errors="coerce").dropna().to_numpy(dtype=float)


def _plot_metric_axis(
    ax: plt.Axes,
    technical: pd.DataFrame,
    br_data: pd.DataFrame,
    comparisons: pd.DataFrame,
    condition_ids: Sequence[str],
    ylabel: str,
    xlabel: str,
    show_reference: bool,
    draw_only_significant: bool,
) -> None:
    x_lookup = {condition_id: index for index, condition_id in enumerate(condition_ids)}
    rng = np.random.default_rng(1045)
    for condition_id in condition_ids:
        x = x_lookup[condition_id]
        values = _metric_arrays(technical, condition_id)
        if len(values):
            jitter = rng.uniform(-0.13, 0.13, len(values))
            ax.scatter(
                np.full(len(values), x) + jitter,
                values,
                color=condition_color(condition_id),
                edgecolor="#111827",
                linewidth=0.25,
                s=24,
                alpha=0.9,
                zorder=3,
            )
        br_values = _metric_arrays(br_data, condition_id)
        if len(br_values):
            mean = float(np.mean(br_values))
            sd = float(np.std(br_values, ddof=1)) if len(br_values) > 1 else 0.0
            ax.errorbar(x, mean, yerr=sd, color="black", capsize=4, linewidth=1.0, zorder=4)
            ax.hlines(mean, x - 0.18, x + 0.18, color="black", linewidth=1.5, zorder=5)
    y_values = pd.concat([technical["plot_value"], br_data["plot_value"]], ignore_index=True)
    y_values = pd.to_numeric(y_values, errors="coerce").dropna()
    if not y_values.empty:
        y_min = float(y_values.min())
        y_max = float(y_values.max())
        span = max(y_max - y_min, abs(y_max) * 0.1, 1e-6)
        ax.set_ylim(y_min - span * 0.14, y_max + span * 0.22)
        add_vehicle_brackets(ax, comparisons, x_lookup, y_max, y_min, draw_only_significant)
    if show_reference:
        ax.axhline(100, color="dimgray", linewidth=0.8, linestyle="--", zorder=1)
    ax.set_xticks(range(len(condition_ids)))
    ax.set_xticklabels([condition_label(condition_id) for condition_id in condition_ids], rotation=45, ha="right")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(axis="y", alpha=0.22)


def harmonize_axis_limits(axes: Sequence[plt.Axes]) -> None:
    bottoms = [ax.get_ylim()[0] for ax in axes]
    tops = [ax.get_ylim()[1] for ax in axes]
    if bottoms and tops:
        for ax in axes:
            ax.set_ylim(min(bottoms), max(tops))


def plot_schiff_auc_figures(technical: pd.DataFrame, br_data: pd.DataFrame, comparisons: pd.DataFrame, output_dir: Path) -> List[Path]:
    paths: List[Path] = []
    axes = []
    figs = []
    for species in EXPECTED_SPECIES:
        fig, ax = plt.subplots(figsize=(15, 5.5))
        tech = technical[technical["figure_family"].eq("schiff_relative_auc") & technical["species"].eq(species)]
        br = br_data[br_data["figure_family"].eq("schiff_relative_auc") & br_data["species"].eq(species)]
        comp = comparisons[comparisons["model_family"].eq("schiff_relative_auc") & comparisons["species"].eq(species)]
        _plot_metric_axis(
            ax,
            tech,
            br,
            comp,
            SCHIFF_AUC_CONDITIONS,
            "Relative AUC₀–17 h 25 min (% of vehicle)",
            "Treatment",
            show_reference=True,
            draw_only_significant=True,
        )
        for separator in [5.5, 10.5, 15.5]:
            ax.axvline(separator, color="lightgray", linewidth=0.8, zorder=0)
        ax.set_title(f"{DISPLAY_SPECIES.get(species, species)} Schiff-base relative AUC")
        note = (
            "Relative AUC is shown for presentation, whereas statistical inference was performed on raw AUC using "
            "biological-replicate means. Tukey-adjusted comparisons were calculated from the randomized-block model; "
            "only significant comparisons with the 1% DMSO vehicle are annotated. Unmarked eligible comparisons were "
            "not significant after adjustment."
        )
        fig.text(0.5, 0.02, fill(note, 155), ha="center", fontsize=8)
        fig.tight_layout(rect=[0, 0.08, 1, 1])
        figs.append((fig, species))
        axes.append(ax)
    harmonize_axis_limits(axes)
    for fig, species in figs:
        stem = f"schiff_base_relative_auc_all_concentrations_{species}"
        png = output_dir / f"{stem}.png"
        pdf = output_dir / f"{stem}.pdf"
        fig.savefig(png, dpi=300)
        fig.savefig(pdf)
        plt.close(fig)
        paths.extend([png, pdf])
    return paths


def plot_penag_metric_figures(
    technical: pd.DataFrame,
    br_data: pd.DataFrame,
    comparisons: pd.DataFrame,
    output_dir: Path,
    figure_family: str,
    condition_ids: Sequence[str],
    ylabel: str,
    xlabel: str,
    stem_metric: str,
    display_metric: str,
    show_reference: bool,
) -> List[Path]:
    paths: List[Path] = []
    axes = []
    figs = []
    for species in EXPECTED_SPECIES:
        fig, ax = plt.subplots(figsize=(6.8, 5.2))
        tech = technical[technical["figure_family"].eq(figure_family) & technical["species"].eq(species)]
        br = br_data[br_data["figure_family"].eq(figure_family) & br_data["species"].eq(species)]
        comp = comparisons[comparisons["model_family"].eq(figure_family) & comparisons["species"].eq(species)]
        _plot_metric_axis(
            ax,
            tech,
            br,
            comp,
            condition_ids,
            ylabel,
            xlabel,
            show_reference=show_reference,
            draw_only_significant=False,
        )
        ax.set_title(f"{DISPLAY_SPECIES.get(species, species)} PenAg {display_metric}")
        note = (
            "Tukey-adjusted comparisons were calculated from the randomized-block model; only vehicle-versus-treatment "
            "comparisons are shown. Relative AUC is plotted for presentation when applicable."
        )
        fig.text(0.5, 0.02, fill(note, 90), ha="center", fontsize=8)
        fig.tight_layout(rect=[0, 0.1, 1, 1])
        figs.append((fig, species))
        axes.append(ax)
    harmonize_axis_limits(axes)
    for fig, species in figs:
        stem = f"penag_{stem_metric}_{species}"
        png = output_dir / f"{stem}.png"
        pdf = output_dir / f"{stem}.pdf"
        fig.savefig(png, dpi=300)
        fig.savefig(pdf)
        plt.close(fig)
        paths.extend([png, pdf])
    return paths


def plot_growth_curve_figures(growth: pd.DataFrame, output_dir: Path) -> Tuple[pd.DataFrame, List[Path]]:
    schiff_condition_ids = [VEHICLE] + [f"{compound}_100uM" for compound in SCHIFF_COMPOUNDS]
    penag_condition_ids = PENAG_AUC_CONDITIONS
    schiff_summary = summarize_growth_curves(growth, schiff_condition_ids)
    penag_summary = summarize_growth_curves(growth, penag_condition_ids)
    paths: List[Path] = []
    for species in EXPECTED_SPECIES:
        paths.extend(
            _plot_growth_curve_figure(
                schiff_summary,
                species,
                schiff_condition_ids,
                output_dir,
                f"schiff_base_100uM_growth_curves_{species}",
                f"{DISPLAY_SPECIES.get(species, species)} Schiff-base highest tested concentration",
            )
        )
        paths.extend(
            _plot_growth_curve_figure(
                penag_summary,
                species,
                penag_condition_ids,
                output_dir,
                f"penag_growth_curves_all_concentrations_{species}",
                f"{DISPLAY_SPECIES.get(species, species)} PenAg growth curves",
            )
        )
    growth_summary = pd.concat(
        [
            schiff_summary.assign(figure_family="schiff_100uM_growth_curves"),
            penag_summary.assign(figure_family="penag_growth_curves"),
        ],
        ignore_index=True,
    )
    return growth_summary, paths


def penag_lag_fit_availability(technical: pd.DataFrame) -> pd.DataFrame:
    data = technical[technical["condition_id"].isin(PENAG_TLAG_CONDITIONS)].copy()
    data = data[data["species"].isin(EXPECTED_SPECIES)].copy()
    return data[
        [
            "species",
            "biological_replicate",
            "run_id",
            "well",
            "technical_replicate",
            "condition_id",
            "condition",
            "compound_concentration_uM",
            TLAG,
            "gompertz_R2",
            "gompertz_fit_status",
            "gompertz_fit_warning",
        ]
    ].copy()


def build_manifest(
    figure_paths: Sequence[Path],
    technical: pd.DataFrame,
    br_data: pd.DataFrame,
    comparisons: pd.DataFrame,
    growth: pd.DataFrame,
) -> pd.DataFrame:
    records = []
    for png in [path for path in figure_paths if path.suffix.lower() == ".png"]:
        stem = png.stem
        if stem.startswith("schiff_base_100uM_growth_curves"):
            species = stem.rsplit("_", 2)[-2] + "_" + stem.rsplit("_", 2)[-1] if stem.endswith("B_subtilis") else "E_coli"
            family = "schiff_100uM_growth_curves"
            conditions = [VEHICLE] + [f"{compound}_100uM" for compound in SCHIFF_COMPOUNDS]
            metric = "corrected_OD600"
            inference = "none"
            rule = "no significance brackets on growth curves"
        elif stem.startswith("penag_growth_curves"):
            species = "B_subtilis" if stem.endswith("B_subtilis") else "E_coli"
            family = "penag_growth_curves"
            conditions = PENAG_AUC_CONDITIONS
            metric = "corrected_OD600"
            inference = "none"
            rule = "no significance brackets on growth curves"
        elif stem.startswith("schiff_base_relative_auc"):
            species = "B_subtilis" if stem.endswith("B_subtilis") else "E_coli"
            family = "schiff_relative_auc"
            conditions = SCHIFF_AUC_CONDITIONS
            metric = "relative_AUC_percent"
            inference = "raw_AUC_0_1045_min"
            rule = "only significant vehicle comparisons drawn"
        elif stem.startswith("penag_relative_auc"):
            species = "B_subtilis" if stem.endswith("B_subtilis") else "E_coli"
            family = "penag_relative_auc"
            conditions = PENAG_AUC_CONDITIONS
            metric = "relative_AUC_percent"
            inference = "raw_AUC_0_1045_min"
            rule = "all vehicle comparisons drawn"
        else:
            species = "B_subtilis" if stem.endswith("B_subtilis") else "E_coli"
            family = "penag_lag_time"
            conditions = PENAG_TLAG_CONDITIONS
            metric = "TLag_h"
            inference = "TLag_h"
            rule = "all vehicle comparisons drawn"
        if family in {"schiff_100uM_growth_curves", "penag_growth_curves"}:
            tech_count = int(
                growth[
                    growth["species"].eq(species)
                    & growth["condition_id"].isin(conditions)
                ].shape[0]
            )
            br_count = int(
                growth[
                    growth["species"].eq(species)
                    & growth["condition_id"].isin(conditions)
                ]["biological_replicate"].nunique()
            )
        else:
            tech_count = int(
                technical[
                    technical["figure_family"].eq(family)
                    & technical["species"].eq(species)
                ]["well"].count()
            )
            br_count = int(
                br_data[
                    br_data["figure_family"].eq(family)
                    & br_data["species"].eq(species)
                ]["biological_replicate"].nunique()
            )
        records.append(
            {
                "figure_png": str(png),
                "figure_pdf": str(png.with_suffix(".pdf")),
                "species": species,
                "included_conditions": ";".join(conditions),
                "plotted_metric": metric,
                "inference_metric": inference,
                "statistical_family": family,
                "number_of_technical_points": tech_count,
                "number_of_biological_replicates": br_count,
                "bracket_display_rule": rule,
                "vehicle_comparisons_drawn": int(
                    comparisons[
                        comparisons["model_family"].eq(family)
                        & comparisons["species"].eq(species)
                        & comparisons["drawn_on_figure"].eq(True)
                    ].shape[0]
                ),
            }
        )
    return pd.DataFrame.from_records(records)


def write_outputs(
    output_dir: Path,
    technical: pd.DataFrame,
    br_data: pd.DataFrame,
    anova: pd.DataFrame,
    vehicle_comparisons: pd.DataFrame,
    lag_fit: pd.DataFrame,
    manifest: pd.DataFrame,
) -> List[Path]:
    paths = [
        output_dir / "thesis_figure_technical_well_data.csv",
        output_dir / "thesis_figure_biological_replicate_data.csv",
        output_dir / "randomised_block_anova_thesis_models.csv",
        output_dir / "thesis_figure_vehicle_tukey_comparisons.csv",
        output_dir / "penag_lag_fit_availability.csv",
        output_dir / "thesis_figure_manifest.csv",
    ]
    technical.to_csv(paths[0], index=False)
    br_data.to_csv(paths[1], index=False)
    anova.to_csv(paths[2], index=False)
    vehicle_comparisons.to_csv(paths[3], index=False)
    lag_fit.to_csv(paths[4], index=False)
    manifest.to_csv(paths[5], index=False)
    return paths


def run_analysis(input_dirs: Dict[Tuple[str, str], Path], output_dir: Path | str) -> dict:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    loaded = read_all_runs(input_dirs)
    growth = loaded["growth"]
    technical = loaded["technical"]
    summary = loaded["summary"]
    runs = loaded["runs"]
    warnings = validate_thesis_inputs(growth, technical, summary, runs)
    br_metrics = prepare_biological_replicate_metrics(summary)
    anova, complete_tukey, vehicle_comparisons = run_thesis_models(br_metrics)
    technical_points = thesis_metric_technical_data(technical)
    br_points = thesis_metric_br_data(br_metrics)
    growth_summary, figure_paths = plot_growth_curve_figures(growth, output_dir)
    figure_paths.extend(plot_schiff_auc_figures(technical_points, br_points, vehicle_comparisons, output_dir))
    figure_paths.extend(
        plot_penag_metric_figures(
            technical_points,
            br_points,
            vehicle_comparisons,
            output_dir,
            "penag_relative_auc",
            PENAG_AUC_CONDITIONS,
            "Relative AUC₀–17 h 25 min (% of vehicle)",
            "PenAg concentration (µM)",
            "relative_auc",
            "Relative AUC",
            show_reference=True,
        )
    )
    figure_paths.extend(
        plot_penag_metric_figures(
            technical_points,
            br_points,
            vehicle_comparisons,
            output_dir,
            "penag_lag_time",
            PENAG_TLAG_CONDITIONS,
            "Lag time, TLag (h)",
            "PenAg concentration (µM)",
            "lag_time",
            "Lag time",
            show_reference=False,
        )
    )
    lag_fit = penag_lag_fit_availability(technical)
    manifest = build_manifest(figure_paths, technical_points, br_points, vehicle_comparisons, growth)
    table_paths = write_outputs(output_dir, technical_points, br_points, anova, vehicle_comparisons, lag_fit, manifest)
    verify_source_hashes(loaded["hashes"])
    return {
        "growth": growth,
        "technical": technical,
        "summary": summary,
        "br_metrics": br_metrics,
        "anova": anova,
        "complete_tukey": complete_tukey,
        "vehicle_comparisons": vehicle_comparisons,
        "technical_points": technical_points,
        "br_points": br_points,
        "growth_summary": growth_summary,
        "lag_fit": lag_fit,
        "manifest": manifest,
        "figure_paths": figure_paths,
        "table_paths": table_paths,
        "warnings": warnings,
        "output_dir": output_dir,
    }


def main() -> int:
    """Command-line entry point for thesis-ready monotherapy figures."""
    args = parse_args()
    result = run_analysis(build_input_map(args), Path(args.output_dir))
    comparisons = result["vehicle_comparisons"]
    print("Drug-screening thesis figure summary")
    print(f"- output folder: {result['output_dir']}")
    print(f"- thesis technical metric rows: {len(result['technical_points'])}")
    print(f"- thesis biological-replicate rows: {len(result['br_points'])}")
    print(f"- thesis ANOVA models fitted: {len(result['anova'])}")
    print(f"- complete all-pair Tukey comparisons calculated: {len(result['complete_tukey'])}")
    print(f"- vehicle comparisons in thesis table: {len(comparisons)}")
    print(f"- vehicle brackets drawn: {int(comparisons['drawn_on_figure'].sum())}")
    print(f"- figure files: {len(result['figure_paths'])}")
    print(f"- table files: {len(result['table_paths'])}")
    if result["warnings"]:
        print("Warnings")
        for warning in result["warnings"]:
            print(f"- {warning}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
