"""Plot matched BR2/BR3 combination-treatment metrics descriptively.

The input is the long-format ``combination_treatment_descriptive_metrics.csv``
written by ``drug_screening_statistics.py``. ``technical_mean`` is already the
within-run mean of three technical wells and is used directly. Repeated
ampicillin-only rows are verified for numerical identity and deduplicated; they
are never averaged when inconsistent.

The script shows BR2 and BR3 values plus their unweighted mean when both are
available. It performs no hypothesis tests, does not reconstruct technical
well observations, and does not modify its input table.
"""

from __future__ import annotations

import argparse
import math
import sys
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from growth_analysis import DISPLAY_SPECIES, GrowthDataValidationError  # noqa: E402


EXPECTED_SPECIES = ("E_coli", "B_subtilis")
EXPECTED_REPLICATES = ("BR2", "BR3")
EXPECTED_COMPOUNDS = ("C1", "C2", "C3", "C4")
EXPECTED_METRICS = ("relative_AUC", "TLag", "Kz")
EXPECTED_TECHNICAL_WELLS = 3
COMPOUND_CONCENTRATION_UM = 25.0
AMPICILLIN_CONCENTRATION_UG_ML = 4.0
TREATMENT_ORDER = ("Ampicillin 4 ug/mL", "C1 + Amp", "C2 + Amp", "C3 + Amp", "C4 + Amp")
TREATMENT_TICK_LABELS = {
    "Ampicillin 4 ug/mL": "Ampicillin\n4 ug/mL",
    "C1 + Amp": "C1 + Amp",
    "C2 + Amp": "C2 + Amp",
    "C3 + Amp": "C3 + Amp",
    "C4 + Amp": "C4 + Amp",
}
REPLICATE_COLORS = {"BR2": "#1f4e79", "BR3": "#8f3f00"}
REPLICATE_OFFSETS = {"BR2": -0.12, "BR3": 0.12}
REQUIRED_COLUMNS = {
    "species",
    "biological_replicate",
    "compound",
    "compound_concentration_uM",
    "ampicillin_concentration_ug_mL",
    "comparison_condition",
    "metric",
    "n_technical_wells",
    "technical_mean",
    "technical_SD",
    "condition_id",
}


class CombinationDescriptivePlotError(GrowthDataValidationError):
    """Raised when combination-treatment descriptive plot inputs are invalid."""


@dataclass(frozen=True)
class MetricSpec:
    """Display title and verified units for one descriptive metric."""
    metric: str
    title: str
    y_label: str


METRIC_SPECS = {
    "relative_AUC": MetricSpec(
        metric="relative_AUC",
        title="Relative AUC",
        y_label="Relative AUC 0-17 h 25 min (% of vehicle)",
    ),
    "TLag": MetricSpec(metric="TLag", title="Lag time", y_label="Lag time, TLag (h)"),
    "Kz": MetricSpec(
        metric="Kz",
        title="Growth rate",
        y_label="Growth rate, Kz (OD$_{600}$·h$^{-1}$)",
    ),
}


def parse_args() -> argparse.Namespace:
    """Parse the long metrics table and output directory."""
    parser = argparse.ArgumentParser(
        description="Plot BR2/BR3 descriptive 25 uM compound + 4 ug/mL ampicillin metrics."
    )
    parser.add_argument(
        "--input",
        default=PROJECT_ROOT
        / "results"
        / "drug_screening"
        / "statistics"
        / "combination_treatment_descriptive_metrics.csv",
        help="Long-format combination-treatment descriptive metrics CSV.",
    )
    parser.add_argument(
        "--output-dir",
        default=PROJECT_ROOT
        / "results"
        / "drug_screening"
        / "statistics"
        / "combination_treatment_descriptive_plots",
        help="Directory for plot-ready tables and figure files.",
    )
    return parser.parse_args()


def _as_numeric(series: pd.Series, column: str) -> pd.Series:
    values = pd.to_numeric(series, errors="coerce")
    invalid = values.isna() & series.notna()
    if invalid.any():
        raise CombinationDescriptivePlotError(f"Column {column!r} contains non-numeric values.")
    return values


def validate_input_columns(data: pd.DataFrame) -> None:
    missing = sorted(REQUIRED_COLUMNS - set(data.columns))
    if missing:
        raise CombinationDescriptivePlotError(f"Missing required columns: {missing}")


def infer_metric_units(plot_data: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Convert only when the stored units are clearly ratios/minutes/per-minute."""
    result = plot_data.copy()
    notes: List[dict] = []

    for metric in EXPECTED_METRICS:
        metric_mask = result["metric"].eq(metric)
        values = pd.to_numeric(result.loc[metric_mask, "technical_mean"], errors="coerce")
        finite = values[np.isfinite(values)]
        conversion = "none"
        source_units = "unknown"
        plotted_units = "unknown"

        if metric == "relative_AUC":
            source_units = "percent"
            plotted_units = "percent"
            if not finite.empty and finite.abs().max() <= 2.0:
                result.loc[metric_mask, "plotted_value"] = values * 100.0
                conversion = "multiplied by 100 because values were stored as ratios"
                source_units = "ratio"
            else:
                result.loc[metric_mask, "plotted_value"] = values
                conversion = "none; values are already percentages"
        elif metric == "TLag":
            source_units = "hours"
            plotted_units = "hours"
            if not finite.empty and finite.median() > 30.0:
                result.loc[metric_mask, "plotted_value"] = values / 60.0
                conversion = "divided by 60 because values appeared to be stored in minutes"
                source_units = "minutes"
            else:
                result.loc[metric_mask, "plotted_value"] = values
                conversion = "none; values are already hours"
        elif metric == "Kz":
            source_units = "OD600 per hour"
            plotted_units = "OD600 per hour"
            result.loc[metric_mask, "plotted_value"] = values
            conversion = "none; existing Gompertz implementation stores Kz as OD600 per hour"

        result.loc[metric_mask, "unit_conversion"] = conversion
        notes.append(
            {
                "metric": metric,
                "source_units_inferred": source_units,
                "plotted_units": plotted_units,
                "conversion_applied": conversion,
                "minimum_source_value": float(finite.min()) if not finite.empty else math.nan,
                "median_source_value": float(finite.median()) if not finite.empty else math.nan,
                "maximum_source_value": float(finite.max()) if not finite.empty else math.nan,
            }
        )

    return result, pd.DataFrame.from_records(notes)


def _deduplicate_ampicillin(group: pd.DataFrame) -> pd.Series:
    values = pd.to_numeric(group["technical_mean"], errors="coerce").to_numpy(dtype=float)
    if not np.allclose(values, values[0], rtol=1e-10, atol=1e-12, equal_nan=True):
        first = group[["compound", "technical_mean"]].to_dict("records")
        raise CombinationDescriptivePlotError(
            "Repeated ampicillin-only values disagree for "
            f"{group['species'].iloc[0]} {group['biological_replicate'].iloc[0]} "
            f"{group['metric'].iloc[0]}: {first}"
        )
    row = group.iloc[0].copy()
    row["compound"] = "Ampicillin"
    row["treatment"] = "Ampicillin 4 ug/mL"
    row["source_rows_deduplicated"] = int(len(group))
    return row


def build_plotting_tables(data: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    validate_input_columns(data)
    work = data.copy(deep=True)
    work["compound_concentration_uM"] = _as_numeric(work["compound_concentration_uM"], "compound_concentration_uM")
    work["ampicillin_concentration_ug_mL"] = _as_numeric(
        work["ampicillin_concentration_ug_mL"], "ampicillin_concentration_ug_mL"
    )
    work["n_technical_wells"] = _as_numeric(work["n_technical_wells"], "n_technical_wells")
    work["technical_mean"] = _as_numeric(work["technical_mean"], "technical_mean")

    base = work[
        work["species"].isin(EXPECTED_SPECIES)
        & work["biological_replicate"].isin(EXPECTED_REPLICATES)
        & work["metric"].isin(EXPECTED_METRICS)
        & work["ampicillin_concentration_ug_mL"].eq(AMPICILLIN_CONCENTRATION_UG_ML)
    ].copy()

    combination = base[
        base["comparison_condition"].eq("combination")
        & base["compound"].isin(EXPECTED_COMPOUNDS)
        & base["compound_concentration_uM"].eq(COMPOUND_CONCENTRATION_UM)
    ].copy()
    combination["treatment"] = combination["compound"].astype(str) + " + Amp"
    combination["source_rows_deduplicated"] = 1

    duplicate_combo = combination.duplicated(["species", "biological_replicate", "metric", "treatment"], keep=False)
    if duplicate_combo.any():
        rows = combination.loc[
            duplicate_combo, ["species", "biological_replicate", "metric", "treatment", "technical_mean"]
        ].to_dict("records")
        raise CombinationDescriptivePlotError(f"Duplicate combination rows after filtering: {rows}")

    amp = base[base["comparison_condition"].eq("ampicillin_alone")].copy()
    amp_groups = ["species", "biological_replicate", "metric"]
    if amp.empty:
        amp_dedup = pd.DataFrame(columns=list(work.columns) + ["treatment", "source_rows_deduplicated"])
    else:
        amp_dedup = pd.DataFrame([_deduplicate_ampicillin(group) for _, group in amp.groupby(amp_groups, sort=False)])

    plot_data = pd.concat([amp_dedup, combination], ignore_index=True, sort=False)
    plot_data = plot_data[
        [
            "species",
            "biological_replicate",
            "treatment",
            "metric",
            "technical_mean",
            "technical_SD",
            "n_technical_wells",
            "compound",
            "compound_concentration_uM",
            "ampicillin_concentration_ug_mL",
            "comparison_condition",
            "condition_id",
            "source_rows_deduplicated",
        ]
    ].copy()

    warnings_table = validation_warnings(plot_data)
    complete = complete_expected_grid(plot_data)
    complete, unit_notes = infer_metric_units(complete)
    wide_qc = build_wide_qc_table(complete)
    return complete, wide_qc, warnings_table, unit_notes


def complete_expected_grid(plot_data: pd.DataFrame) -> pd.DataFrame:
    keys = pd.MultiIndex.from_product(
        [EXPECTED_SPECIES, EXPECTED_REPLICATES, TREATMENT_ORDER, EXPECTED_METRICS],
        names=["species", "biological_replicate", "treatment", "metric"],
    ).to_frame(index=False)
    complete = keys.merge(plot_data, on=["species", "biological_replicate", "treatment", "metric"], how="left")
    complete["missing_value"] = complete["technical_mean"].isna()
    complete["missing_reason"] = np.where(
        complete["missing_value"],
        "Missing expected species x biological replicate x treatment x metric value.",
        "",
    )
    order_lookup = {treatment: index for index, treatment in enumerate(TREATMENT_ORDER)}
    metric_lookup = {metric: index for index, metric in enumerate(EXPECTED_METRICS)}
    complete["_treatment_order"] = complete["treatment"].map(order_lookup)
    complete["_metric_order"] = complete["metric"].map(metric_lookup)
    complete = complete.sort_values(
        ["species", "biological_replicate", "_treatment_order", "_metric_order"]
    ).drop(columns=["_treatment_order", "_metric_order"])
    return complete.reset_index(drop=True)


def validation_warnings(plot_data: pd.DataFrame) -> pd.DataFrame:
    rows: List[dict] = []
    n_bad = plot_data["n_technical_wells"].notna() & plot_data["n_technical_wells"].ne(EXPECTED_TECHNICAL_WELLS)
    for row in plot_data.loc[n_bad].itertuples(index=False):
        rows.append(
            {
                "level": "warning",
                "species": row.species,
                "biological_replicate": row.biological_replicate,
                "treatment": row.treatment,
                "metric": row.metric,
                "message": f"n_technical_wells is {row.n_technical_wells:g}; expected {EXPECTED_TECHNICAL_WELLS}.",
            }
        )
    return pd.DataFrame.from_records(rows, columns=["level", "species", "biological_replicate", "treatment", "metric", "message"])


def missing_value_warnings(plot_data: pd.DataFrame) -> pd.DataFrame:
    rows: List[dict] = []
    missing = plot_data[plot_data["missing_value"]]
    for row in missing.itertuples(index=False):
        rows.append(
            {
                "level": "warning",
                "species": row.species,
                "biological_replicate": row.biological_replicate,
                "treatment": row.treatment,
                "metric": row.metric,
                "message": row.missing_reason,
            }
        )
    return pd.DataFrame.from_records(rows, columns=["level", "species", "biological_replicate", "treatment", "metric", "message"])


def build_wide_qc_table(plot_data: pd.DataFrame) -> pd.DataFrame:
    wide = plot_data.pivot_table(
        index=["species", "biological_replicate", "treatment"],
        columns="metric",
        values="plotted_value",
        aggfunc="first",
        dropna=False,
    ).reset_index()
    wide.columns.name = None
    for metric in EXPECTED_METRICS:
        if metric not in wide.columns:
            wide[metric] = np.nan
    n_technical = plot_data.pivot_table(
        index=["species", "biological_replicate", "treatment"],
        columns="metric",
        values="n_technical_wells",
        aggfunc="first",
        dropna=False,
    ).reset_index()
    n_technical.columns = [
        column if column in {"species", "biological_replicate", "treatment"} else f"n_technical_wells_{column}"
        for column in n_technical.columns
    ]
    wide = wide.merge(n_technical, on=["species", "biological_replicate", "treatment"], how="left")
    metric_missing = wide[list(EXPECTED_METRICS)].isna()
    wide["missing_metrics"] = metric_missing.apply(
        lambda row: ";".join([metric for metric, missing in row.items() if missing]), axis=1
    )
    ordered = ["species", "biological_replicate", "treatment", "relative_AUC", "TLag", "Kz"]
    ordered += [f"n_technical_wells_{metric}" for metric in EXPECTED_METRICS]
    ordered += ["missing_metrics"]
    return wide[ordered].sort_values(["species", "biological_replicate", "treatment"]).reset_index(drop=True)


def mean_line_segments(plot_data: pd.DataFrame, species: str, metric: str) -> pd.DataFrame:
    rows: List[dict] = []
    panel = plot_data[plot_data["species"].eq(species) & plot_data["metric"].eq(metric)].copy()
    for treatment in TREATMENT_ORDER:
        values = panel[panel["treatment"].eq(treatment)].set_index("biological_replicate")["plotted_value"]
        valid = values.reindex(EXPECTED_REPLICATES).dropna()
        if len(valid) == len(EXPECTED_REPLICATES):
            rows.append({"treatment": treatment, "metric": metric, "mean_value": float(valid.mean())})
    return pd.DataFrame.from_records(rows, columns=["treatment", "metric", "mean_value"])


def _panel_ylim(values: pd.Series) -> Tuple[float, float]:
    finite = pd.to_numeric(values, errors="coerce").dropna()
    if finite.empty:
        return 0.0, 1.0
    ymin = float(finite.min())
    ymax = float(finite.max())
    span = ymax - ymin
    if span == 0:
        span = max(abs(ymax), 1.0) * 0.2
    lower = ymin - span * 0.18
    upper = ymax + span * 0.22
    if ymin >= 0:
        lower = min(0.0, lower)
    return lower, upper


def _draw_panel(
    ax: plt.Axes,
    panel: pd.DataFrame,
    metric: str,
    species: str,
    show_ylabel: bool = True,
    panel_title: Optional[str] = None,
    show_xlabel: bool = True,
    panel_label: str = "",
) -> None:
    x_positions = {treatment: index for index, treatment in enumerate(TREATMENT_ORDER)}
    for replicate in EXPECTED_REPLICATES:
        rep_data = panel[panel["biological_replicate"].eq(replicate)]
        xs = [x_positions[treatment] + REPLICATE_OFFSETS[replicate] for treatment in rep_data["treatment"]]
        ax.scatter(
            xs,
            rep_data["plotted_value"],
            color=REPLICATE_COLORS[replicate],
            edgecolor="#111827",
            linewidth=0.45,
            s=54,
            alpha=0.95,
            label=replicate,
            zorder=3,
        )

    means = mean_line_segments(panel, species, metric)
    for row in means.itertuples(index=False):
        x = x_positions[row.treatment]
        ax.hlines(row.mean_value, x - 0.24, x + 0.24, colors="#111827", linewidth=1.6, zorder=4)

    missing = panel[panel["missing_value"]]
    for row in missing.itertuples(index=False):
        ax.text(
            x_positions[row.treatment],
            0.02,
            "missing",
            transform=ax.get_xaxis_transform(),
            ha="center",
            va="bottom",
            fontsize=7,
            color="#6b7280",
            rotation=90,
        )

    ax.set_xticks(range(len(TREATMENT_ORDER)))
    ax.set_xticklabels([TREATMENT_TICK_LABELS[treatment] for treatment in TREATMENT_ORDER], rotation=32, ha="right")
    ax.set_xlim(-0.5, len(TREATMENT_ORDER) - 0.5)
    ax.set_ylim(*_panel_ylim(panel["plotted_value"]))
    ax.grid(axis="y", color="#d1d5db", alpha=0.55, linewidth=0.7)
    ax.set_axisbelow(True)
    title = panel_title if panel_title is not None else f"{DISPLAY_SPECIES.get(species, species)} {METRIC_SPECS[metric].title}"
    ax.set_title(title, fontsize=12)
    if panel_label:
        ax.text(
            -0.11,
            1.05,
            panel_label,
            transform=ax.transAxes,
            ha="left",
            va="bottom",
            fontsize=11,
            fontweight="bold",
            clip_on=False,
        )
    if show_ylabel:
        ax.set_ylabel(METRIC_SPECS[metric].y_label)
    ax.set_xlabel("Treatment" if show_xlabel else "")


def plot_individual_figures(plot_data: pd.DataFrame, output_dir: Path) -> List[Path]:
    paths: List[Path] = []
    for species in EXPECTED_SPECIES:
        for metric in EXPECTED_METRICS:
            fig, ax = plt.subplots(figsize=(6.8, 4.8))
            panel = plot_data[plot_data["species"].eq(species) & plot_data["metric"].eq(metric)].copy()
            _draw_panel(ax, panel, metric, species)
            handles, labels = ax.get_legend_handles_labels()
            ax.legend(handles[:2], labels[:2], title="Biological replicate", frameon=False, loc="best")
            fig.tight_layout()
            stem = f"combination_treatment_descriptive_{metric}_{species}"
            png = output_dir / f"{stem}.png"
            pdf = output_dir / f"{stem}.pdf"
            fig.savefig(png, dpi=300)
            fig.savefig(pdf)
            plt.close(fig)
            paths.extend([png, pdf])
    return paths


def plot_combined_figure(plot_data: pd.DataFrame, output_dir: Path) -> List[Path]:
    fig, axes = plt.subplots(2, 3, figsize=(15.5, 8.8))
    for row_index, species in enumerate(EXPECTED_SPECIES):
        for col_index, metric in enumerate(EXPECTED_METRICS):
            ax = axes[row_index, col_index]
            panel = plot_data[plot_data["species"].eq(species) & plot_data["metric"].eq(metric)].copy()
            _draw_panel(ax, panel, metric, species, show_ylabel=True)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(
        handles[:2],
        labels[:2],
        title="Biological replicate",
        frameon=False,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.955),
        ncol=2,
    )
    fig.suptitle("BR2/BR3 25 uM compound + 4 ug/mL ampicillin descriptive metrics", y=0.995, fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.88])
    png = output_dir / "combination_treatment_descriptive_combined.png"
    pdf = output_dir / "combination_treatment_descriptive_combined.pdf"
    fig.savefig(png, dpi=300)
    fig.savefig(pdf)
    plt.close(fig)
    return [png, pdf]


def plot_two_column_combined_figure(plot_data: pd.DataFrame, output_dir: Path) -> List[Path]:
    species_order = ("B_subtilis", "E_coli")
    metric_order = ("relative_AUC", "Kz", "TLag")
    panel_labels = {
        ("B_subtilis", "relative_AUC"): "(A)",
        ("E_coli", "relative_AUC"): "(B)",
        ("B_subtilis", "Kz"): "(C)",
        ("E_coli", "Kz"): "(D)",
        ("B_subtilis", "TLag"): "(E)",
        ("E_coli", "TLag"): "(F)",
    }

    fig, axes = plt.subplots(3, 2, figsize=(8.3, 11.0), sharex=False)
    for row_index, metric in enumerate(metric_order):
        for col_index, species in enumerate(species_order):
            ax = axes[row_index, col_index]
            panel = plot_data[plot_data["species"].eq(species) & plot_data["metric"].eq(metric)].copy()
            _draw_panel(
                ax,
                panel,
                metric,
                species,
                show_ylabel=True,
                panel_title=METRIC_SPECS[metric].title,
                show_xlabel=False,
                panel_label=panel_labels[(species, metric)],
            )
            ax.tick_params(axis="both", labelsize=9)
            ax.yaxis.label.set_size(10)
            ax.title.set_size(10.5)

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(
        handles[:2],
        labels[:2],
        title="Biological replicate",
        frameon=False,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.945),
        ncol=2,
        fontsize=9.5,
        title_fontsize=9.5,
    )
    fig.suptitle(
        "BR2/BR3 25 uM compound + 4 ug/mL ampicillin descriptive metrics",
        y=0.986,
        fontsize=13.5,
    )
    fig.text(0.305, 0.885, r"$\it{B.\ subtilis}$", ha="center", fontsize=12, fontweight="bold")
    fig.text(0.755, 0.885, r"$\it{E.\ coli}$", ha="center", fontsize=12, fontweight="bold")
    fig.supxlabel("Treatment", y=0.035, fontsize=10.5)
    fig.subplots_adjust(left=0.13, right=0.98, bottom=0.13, top=0.83, hspace=0.68, wspace=0.36)

    png = output_dir / "combination_treatment_descriptive_combined_two_column.png"
    pdf = output_dir / "combination_treatment_descriptive_combined_two_column.pdf"
    fig.savefig(png, dpi=300)
    fig.savefig(pdf)
    plt.close(fig)
    return [png, pdf]


def plot_all(plot_data: pd.DataFrame, output_dir: Path) -> List[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    return (
        plot_individual_figures(plot_data, output_dir)
        + plot_combined_figure(plot_data, output_dir)
        + plot_two_column_combined_figure(plot_data, output_dir)
    )


def run(input_path: Path, output_dir: Path) -> dict:
    data = pd.read_csv(input_path)
    input_before = input_path.read_bytes()
    plot_data, wide_qc, warnings_table, unit_notes = build_plotting_tables(data)
    warnings_table = pd.concat([warnings_table, missing_value_warnings(plot_data)], ignore_index=True)

    output_dir.mkdir(parents=True, exist_ok=True)
    long_path = output_dir / "combination_treatment_descriptive_plot_data_long.csv"
    wide_path = output_dir / "combination_treatment_descriptive_qc_wide.csv"
    warnings_path = output_dir / "combination_treatment_descriptive_warnings.csv"
    unit_notes_path = output_dir / "combination_treatment_descriptive_unit_notes.csv"
    plot_data.to_csv(long_path, index=False)
    wide_qc.to_csv(wide_path, index=False)
    warnings_table.to_csv(warnings_path, index=False)
    unit_notes.to_csv(unit_notes_path, index=False)
    figure_paths = plot_all(plot_data, output_dir)

    if input_before != input_path.read_bytes():
        raise CombinationDescriptivePlotError("Input file changed while generating plots.")

    for warning_row in warnings_table.itertuples(index=False):
        warnings.warn(warning_row.message, RuntimeWarning, stacklevel=2)

    return {
        "plot_data": plot_data,
        "wide_qc": wide_qc,
        "warnings": warnings_table,
        "unit_notes": unit_notes,
        "tables": [long_path, wide_path, warnings_path, unit_notes_path],
        "figures": figure_paths,
    }


def _print_summary(result: dict, output_dir: Path) -> None:
    plot_data = result["plot_data"]
    wide_qc = result["wide_qc"]
    warnings_table = result["warnings"]
    print(f"Output directory: {output_dir}")
    print(f"Long plotting table rows: {len(plot_data)}")
    print(f"Wide QC table rows: {len(wide_qc)}")
    print(f"Missing plotted values: {int(plot_data['missing_value'].sum())}")
    print(f"Warnings: {len(warnings_table)}")
    for row in result["unit_notes"].itertuples(index=False):
        print(f"{row.metric}: {row.conversion_applied}")
    print("Figures written:")
    for path in result["figures"]:
        print(f"- {path}")


def main() -> int:
    """Command-line entry point for combination-treatment descriptive plots."""
    args = parse_args()
    input_path = Path(args.input)
    output_dir = Path(args.output_dir)
    result = run(input_path, output_dir)
    _print_summary(result, output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
