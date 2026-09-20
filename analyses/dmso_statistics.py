"""Run randomized-block statistics across BR1-BR3 DMSO summaries."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from dmso_statistics import (  # noqa: E402
    DESCRIPTIVE_METRICS,
    EXPECTED_BIOLOGICAL_REPLICATES,
    EXPECTED_DMSO_PERCENT,
    EXPECTED_SPECIES,
    INFERENTIAL_METRICS,
    calculate_descriptive_statistics,
    detect_value_warnings,
    read_biological_summary_workbook,
    run_all_randomized_block_anovas,
    run_all_randomized_block_tukey,
    significance_label,
    validate_biological_summary_data,
)


DISPLAY_SPECIES = {
    "E_coli": "E. coli",
    "B_subtilis": "B. subtilis",
}
DISPLAY_SPECIES_ITALIC = {
    "E_coli": r"$\it{E.\ coli}$",
    "B_subtilis": r"$\it{B.\ subtilis}$",
}
MARKERS = {
    "BR1": "o",
    "BR2": "s",
    "BR3": "^",
}
MAIN_FIGURE_METRICS = [
    ("AUC_0_17h_OD_h_mean", "AUC$_{0–17\\ h}$ (OD$_{600}$·h)", "Growth over 0–17 h (AUC)"),
    ("Kz_OD600_per_h_mean", "Growth rate, $K_z$ (OD$_{600}$·h$^{-1}$)", "Growth rate"),
    ("TLag_h_mean", "Lag time, $T_{\\mathrm{Lag}}$ (h)", "Lag time"),
]
EARLY_AUC_METRIC = "AUC_0_12h_OD_h_mean"
EARLY_AUC_YLABEL = "AUC$_{0–12\\ h}$ (OD$_{600}$·h)"
CAPTION_MARKDOWN = (
    "Effect of DMSO concentration on bacterial growth. Growth was evaluated using area under the "
    "curve (AUC) over 0–17 h, modified Gompertz growth rate ($K_z$) and lag time "
    "($T_{\\mathrm{Lag}}$). Each point represents the mean of five technical wells from one "
    "independent biological experiment (n = 3); grey lines connect matched biological experiments "
    "and black lines show the mean ± SD. All metrics were calculated from blank-corrected "
    "OD$_{600}$ measurements. Statistical comparisons were performed using randomized-block "
    "one-way ANOVA followed by Tukey’s multiple-comparisons test. Only comparisons with the 0% "
    "DMSO control are displayed. n.s., p ≥ 0.05; *p < 0.05; **p < 0.01; "
    "***p < 0.001; ****p < 0.0001."
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run randomized-block ANOVA and randomized-block Tukey HSD on "
            "completed DMSO biological-replicate summary workbook."
        )
    )
    parser.add_argument(
        "--input",
        required=True,
        help="Path to the completed biological-replicate summary workbook. The workbook is read-only.",
    )
    parser.add_argument(
        "--sheet-name",
        default=None,
        help="Optional workbook sheet name. If omitted, the script auto-detects the matching sheet.",
    )
    parser.add_argument(
        "--output-dir",
        default=str(PROJECT_ROOT / "results" / "dmso_statistics"),
        help="Directory for statistical output CSVs, Markdown summary, and figures.",
    )
    return parser.parse_args()


def condition_color_map() -> Dict[float, tuple]:
    cmap = plt.get_cmap("viridis", len(EXPECTED_DMSO_PERCENT))
    return {
        concentration: cmap(index)
        for index, concentration in enumerate(EXPECTED_DMSO_PERCENT)
    }


def _format_dmso(value: float) -> str:
    return f"{value:g}%"


def _format_dmso_tick(value: float) -> str:
    return f"{value:g}"


def _format_p(value: float) -> str:
    if pd.isna(value):
        return "NA"
    if value < 0.0001:
        return "<0.0001"
    return f"{value:.4g}"


def _metric_title(metric: str) -> str:
    titles = {
        "AUC_0_17h_OD_h_mean": "AUC 0-17 h",
        "AUC_0_12h_OD_h_mean": "AUC 0-12 h",
        "Kz_OD600_per_h_mean": "Growth rate",
        "TLag_h_mean": "Lag time",
    }
    return titles.get(metric, metric)


def _figure_annotation_label(adjusted_p_value: float) -> str:
    if pd.isna(adjusted_p_value) or adjusted_p_value >= 0.05:
        return "n.s."
    return significance_label(adjusted_p_value)


def _panel_control_pairs(tukey_results: pd.DataFrame, species: str, metric: str) -> pd.DataFrame:
    panel = tukey_results[
        tukey_results["species"].eq(species)
        & tukey_results["metric"].eq(metric)
        & tukey_results["DMSO_group_1"].astype(float).eq(0.0)
        & tukey_results["DMSO_group_2"].astype(float).isin(EXPECTED_DMSO_PERCENT[1:])
    ].copy()
    panel["figure_annotation_label"] = panel["adjusted_p_value"].map(_figure_annotation_label)
    return panel.sort_values(
        ["DMSO_group_1", "DMSO_group_2"],
        kind="stable",
    )


def _add_significance_brackets(
    ax: plt.Axes,
    comparison_pairs: pd.DataFrame,
    x_lookup: Dict[float, int],
    y_top: float,
    y_bottom: float,
) -> None:
    if comparison_pairs.empty:
        return
    span = max(y_top - y_bottom, abs(y_top) * 0.1, 1e-6)
    bracket_height = span * 0.025
    step = span * 0.068
    start_y = y_top + span * 0.06

    for level, row in enumerate(comparison_pairs.itertuples(index=False)):
        group_1 = float(row.DMSO_group_1)
        group_2 = float(row.DMSO_group_2)
        x1 = x_lookup[group_1]
        x2 = x_lookup[group_2]
        y = start_y + level * step
        label = str(row.figure_annotation_label)
        is_not_significant = label == "n.s."
        ax.plot(
            [x1, x1, x2, x2],
            [y, y + bracket_height, y + bracket_height, y],
            color="black",
            linewidth=0.65,
            clip_on=False,
        )
        ax.text(
            (x1 + x2) / 2,
            y + bracket_height,
            label,
            ha="center",
            va="bottom",
            fontsize=7 if is_not_significant else 8.5,
            color="dimgray" if is_not_significant else "black",
            clip_on=False,
        )
    ax.set_ylim(top=start_y + len(comparison_pairs) * step + bracket_height * 3)


def plot_metric_panel(
    ax: plt.Axes,
    data: pd.DataFrame,
    tukey_results: pd.DataFrame,
    species: str,
    metric: str,
    ylabel: str,
    color_map: Dict[float, tuple],
    panel_label: str = "",
    panel_title: str = "",
    show_ylabel: bool = True,
) -> None:
    species_data = data[data["species"].eq(species)]
    x_lookup = {dmso: index for index, dmso in enumerate(EXPECTED_DMSO_PERCENT)}

    for replicate in EXPECTED_BIOLOGICAL_REPLICATES:
        rep_data = species_data[species_data["biological_replicate"].eq(replicate)].sort_values("DMSO_percent")
        x_values = [x_lookup[float(value)] for value in rep_data["DMSO_percent"]]
        ax.plot(
            x_values,
            rep_data[metric],
            color="black",
            alpha=0.25,
            linewidth=0.8,
            zorder=1,
        )
        for x_value, row in zip(x_values, rep_data.itertuples(index=False)):
            dmso = float(row.DMSO_percent)
            ax.scatter(
                x_value,
                getattr(row, metric),
                color=color_map[dmso],
                edgecolor="black",
                linewidth=0.4,
                marker=MARKERS[replicate],
                s=42,
                alpha=0.9,
                zorder=3,
                label=replicate,
            )

    grouped = (
        species_data.groupby("DMSO_percent", as_index=False)[metric]
        .agg(mean="mean", sd=lambda values: values.std(ddof=1))
        .sort_values("DMSO_percent")
    )
    ax.errorbar(
        [x_lookup[float(value)] for value in grouped["DMSO_percent"]],
        grouped["mean"],
        yerr=grouped["sd"],
        color="black",
        marker="_",
        markersize=18,
        linewidth=1.1,
        capsize=4,
        zorder=4,
    )

    y_values = []
    for row in grouped.itertuples(index=False):
        y_values.extend([row.mean + row.sd, row.mean - row.sd])
    y_values.extend(pd.to_numeric(species_data[metric], errors="coerce").tolist())
    finite_y = [float(value) for value in y_values if np.isfinite(value)]
    y_min = min(finite_y)
    y_max = max(finite_y)
    span = max(y_max - y_min, abs(y_max) * 0.1, 1e-6)
    ax.set_ylim(y_min - span * 0.12, y_max + span * 0.22)

    comparison_pairs = _panel_control_pairs(tukey_results, species, metric)
    _add_significance_brackets(ax, comparison_pairs, x_lookup, y_max, y_min)

    title = panel_title or _metric_title(metric)
    ax.set_title(title, fontsize=11, pad=8)
    if panel_label:
        ax.text(
            -0.11,
            1.04,
            panel_label,
            transform=ax.transAxes,
            ha="left",
            va="bottom",
            fontsize=11,
            fontweight="bold",
        )
    ax.set_ylabel(ylabel if show_ylabel else "")
    ax.set_xticks(range(len(EXPECTED_DMSO_PERCENT)))
    ax.set_xticklabels([_format_dmso_tick(value) for value in EXPECTED_DMSO_PERCENT])
    ax.grid(axis="y", alpha=0.25)


def _dedupe_legend(ax: plt.Axes) -> Tuple[List, List[str]]:
    handles, labels = ax.get_legend_handles_labels()
    deduped = {}
    for handle, label in zip(handles, labels):
        if label in EXPECTED_BIOLOGICAL_REPLICATES and label not in deduped:
            deduped[label] = handle
    return list(deduped.values()), list(deduped.keys())


def plot_species_biological_metrics(
    data: pd.DataFrame,
    tukey_results: pd.DataFrame,
    output_dir: Path,
) -> List[Path]:
    paths: List[Path] = []
    colors = condition_color_map()

    for species in EXPECTED_SPECIES:
        fig, axes = plt.subplots(1, 3, figsize=(15, 5.1))
        for panel_label, ax, (metric, ylabel, panel_title) in zip(
            ["(A)", "(B)", "(C)"],
            axes,
            MAIN_FIGURE_METRICS,
        ):
            plot_metric_panel(
                ax,
                data,
                tukey_results,
                species,
                metric,
                ylabel,
                colors,
                panel_label=panel_label,
                panel_title=panel_title,
            )
        handles, labels = _dedupe_legend(axes[0])
        fig.legend(
            handles,
            labels,
            title="Biological replicate",
            loc="upper center",
            bbox_to_anchor=(0.5, 0.84),
            ncol=3,
            frameon=False,
            fontsize=9,
            title_fontsize=9,
        )
        fig.suptitle(
            f"Effect of DMSO concentration on {DISPLAY_SPECIES_ITALIC[species]} growth",
            y=0.97,
            fontsize=13,
        )
        fig.supxlabel("DMSO concentration (% v/v)", y=0.06, fontsize=10)
        fig.subplots_adjust(left=0.07, right=0.99, bottom=0.16, top=0.69, wspace=0.25)
        png_path = output_dir / f"dmso_biological_metrics_{species}.png"
        pdf_path = output_dir / f"dmso_biological_metrics_{species}.pdf"
        fig.savefig(png_path, dpi=600)
        fig.savefig(pdf_path)
        plt.close(fig)
        paths.extend([png_path, pdf_path])

    return paths


def plot_two_column_biological_metrics(
    data: pd.DataFrame,
    tukey_results: pd.DataFrame,
    output_dir: Path,
) -> List[Path]:
    """Create the combined 3-row x 2-column DMSO metrics figure."""
    colors = condition_color_map()
    species_order = ["B_subtilis", "E_coli"]
    panel_labels = {
        ("B_subtilis", "AUC_0_17h_OD_h_mean"): "(A)",
        ("E_coli", "AUC_0_17h_OD_h_mean"): "(B)",
        ("B_subtilis", "Kz_OD600_per_h_mean"): "(C)",
        ("E_coli", "Kz_OD600_per_h_mean"): "(D)",
        ("B_subtilis", "TLag_h_mean"): "(E)",
        ("E_coli", "TLag_h_mean"): "(F)",
    }

    fig, axes = plt.subplots(3, 2, figsize=(8.3, 11.0), sharex=False)
    for row_index, (metric, ylabel, panel_title) in enumerate(MAIN_FIGURE_METRICS):
        for col_index, species in enumerate(species_order):
            ax = axes[row_index, col_index]
            plot_metric_panel(
                ax,
                data,
                tukey_results,
                species,
                metric,
                ylabel,
                colors,
                panel_label=panel_labels[(species, metric)],
                panel_title=panel_title,
            )
            ax.tick_params(axis="both", labelsize=9)
            ax.xaxis.label.set_visible(False)
            ax.yaxis.label.set_size(10)
            ax.title.set_size(10.5)

    handles, labels = _dedupe_legend(axes[0, 0])
    fig.legend(
        handles,
        labels,
        title="Biological replicate",
        loc="upper center",
        bbox_to_anchor=(0.5, 0.945),
        ncol=3,
        frameon=False,
        fontsize=9.5,
        title_fontsize=9.5,
    )
    fig.suptitle(
        "Effect of DMSO concentration on bacterial growth",
        y=0.986,
        fontsize=13.5,
    )
    fig.text(0.305, 0.885, DISPLAY_SPECIES_ITALIC["B_subtilis"], ha="center", fontsize=12, fontweight="bold")
    fig.text(0.755, 0.885, DISPLAY_SPECIES_ITALIC["E_coli"], ha="center", fontsize=12, fontweight="bold")
    fig.supxlabel("DMSO concentration (% v/v)", y=0.025, fontsize=10.5)
    fig.subplots_adjust(left=0.115, right=0.98, bottom=0.075, top=0.83, hspace=0.48, wspace=0.34)

    png_path = output_dir / "dmso_biological_metrics_two_column.png"
    pdf_path = output_dir / "dmso_biological_metrics_two_column.pdf"
    fig.savefig(png_path, dpi=600)
    fig.savefig(pdf_path)
    plt.close(fig)
    return [png_path, pdf_path]


def plot_early_growth_auc(
    data: pd.DataFrame,
    tukey_results: pd.DataFrame,
    output_dir: Path,
) -> List[Path]:
    colors = condition_color_map()
    species_order = ["E_coli", "B_subtilis"]
    fig, axes = plt.subplots(1, 2, figsize=(11, 5.1), sharey=False)
    for panel_label, ax, species in zip(["(A)", "(B)"], axes, species_order):
        plot_metric_panel(
            ax,
            data,
            tukey_results,
            species,
            EARLY_AUC_METRIC,
            "",
            colors,
            panel_label=panel_label,
            panel_title=DISPLAY_SPECIES_ITALIC[species],
            show_ylabel=False,
        )

    handles, labels = _dedupe_legend(axes[0])
    fig.legend(
        handles,
        labels,
        title="Biological replicate",
        loc="upper center",
        bbox_to_anchor=(0.5, 0.84),
        ncol=3,
        frameon=False,
        fontsize=9,
        title_fontsize=9,
    )
    fig.suptitle("Early-growth AUC over 0–12 h", y=0.97, fontsize=13)
    fig.supxlabel("DMSO concentration (% v/v)", y=0.06, fontsize=10)
    fig.supylabel(EARLY_AUC_YLABEL, x=0.02, fontsize=10)
    fig.subplots_adjust(left=0.10, right=0.99, bottom=0.16, top=0.69, wspace=0.22)
    png_path = output_dir / "early_growth_AUC_biological_replicates.png"
    pdf_path = output_dir / "early_growth_AUC_biological_replicates.pdf"
    fig.savefig(png_path, dpi=600)
    fig.savefig(pdf_path)
    plt.close(fig)
    return [png_path, pdf_path]


def write_figure_captions(output_dir: Path) -> Path:
    path = output_dir / "figure_captions.md"
    lines = [
        "# DMSO Figure Captions",
        "",
        CAPTION_MARKDOWN,
        "",
        "Species names are rendered as *E. coli* and *B. subtilis* in the figures.",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def write_markdown_summary(
    output_dir: Path,
    anova_results: pd.DataFrame,
    tukey_results: pd.DataFrame,
    warnings: Sequence[str],
) -> Path:
    significant = tukey_results[tukey_results["statistically_significant"].astype(bool)].copy()
    path = output_dir / "statistical_analysis_summary.md"

    lines = [
        "# DMSO Statistical Analysis Summary",
        "",
        "## Design",
        "",
        "- Analysis: randomized-block one-way ANOVA plus randomized-block Tukey all-pairwise comparisons.",
        "- Block: biological experiment (`BR1`, `BR2`, `BR3`).",
        "- Treatment factor: DMSO concentration (`0%`, `0.25%`, `0.5%`, `1%`, `2%`, `4%`).",
        "- Statistical sample size: `n = 3` independent biological experiments.",
        "- Technical wells were already averaged before this statistical analysis.",
        "- Primary endpoint: `AUC_0_17h_OD_h_mean`.",
        "- Secondary endpoint: `AUC_0_12h_OD_h_mean`.",
        "- Exploratory model-derived endpoints: `Kz_OD600_per_h_mean` and `TLag_h_mean`.",
        "- No Geisser-Greenhouse correction was applied.",
        "- Non-significant results do not demonstrate equivalence.",
        "",
        "## Omnibus ANOVA",
        "",
        "| Species | Metric | F(5,10) | Treatment p | Partial eta squared | Block p |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for row in anova_results.itertuples(index=False):
        lines.append(
            "| "
            f"{DISPLAY_SPECIES.get(row.species, row.species)} | "
            f"{row.metric} | "
            f"{row.treatment_F:.4g} | "
            f"{_format_p(row.treatment_p_value)} | "
            f"{row.partial_eta_squared_treatment:.4g} | "
            f"{_format_p(row.block_p_value)} |"
        )

    lines.extend(["", "## Significant Tukey Comparisons", ""])
    if significant.empty:
        lines.append("No Tukey-adjusted pairwise comparisons were significant at p < 0.05.")
    else:
        lines.extend(
            [
                "| Species | Metric | Comparison | Difference | Adjusted p | 95% simultaneous CI | Stars |",
                "|---|---|---|---:|---:|---|---|",
            ]
        )
        for row in significant.itertuples(index=False):
            comparison = f"{_format_dmso(row.DMSO_group_1)} vs {_format_dmso(row.DMSO_group_2)}"
            ci = f"[{row.simultaneous_95CI_lower:.4g}, {row.simultaneous_95CI_upper:.4g}]"
            lines.append(
                "| "
                f"{DISPLAY_SPECIES.get(row.species, row.species)} | "
                f"{row.metric} | "
                f"{comparison} | "
                f"{row.mean_difference_group_2_minus_group_1:.4g} | "
                f"{_format_p(row.adjusted_p_value)} | "
                f"{ci} | "
                f"{row.significance_label} |"
            )

    lines.extend(
        [
            "",
            "## Limitations and Warnings",
            "",
            "- Conclusions are limited by the small number of biological experiments (`n = 3`).",
            "- Tukey comparisons are reported even when the omnibus ANOVA is not significant and should be interpreted cautiously.",
            "- `A_OD600_mean` is retained descriptively when present, but no inferential test is performed on it.",
        ]
    )
    for warning in warnings:
        lines.append(f"- {warning}")

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def main() -> int:
    args = parse_args()
    input_path = Path(args.input)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    raw_data, sheet_name, workbook_warnings = read_biological_summary_workbook(
        input_path,
        sheet_name=args.sheet_name,
    )
    validated = validate_biological_summary_data(raw_data)
    value_warnings = detect_value_warnings(validated)
    warnings = workbook_warnings + value_warnings

    descriptive = calculate_descriptive_statistics(validated)
    anova_results = run_all_randomized_block_anovas(validated)
    tukey_results = run_all_randomized_block_tukey(validated, anova_results)
    significant_tukey = tukey_results[tukey_results["statistically_significant"].astype(bool)].copy()

    validated_path = output_dir / "validated_biological_replicate_data.csv"
    descriptive_path = output_dir / "descriptive_statistics.csv"
    anova_path = output_dir / "randomized_block_anova_results.csv"
    tukey_path = output_dir / "tukey_all_pairwise_results.csv"
    significant_path = output_dir / "significant_tukey_comparisons.csv"

    validated.to_csv(validated_path, index=False)
    descriptive.to_csv(descriptive_path, index=False)
    anova_results.to_csv(anova_path, index=False)
    tukey_results.to_csv(tukey_path, index=False)
    significant_tukey.to_csv(significant_path, index=False)
    summary_path = write_markdown_summary(output_dir, anova_results, tukey_results, warnings)
    captions_path = write_figure_captions(output_dir)

    figure_paths = []
    figure_paths.extend(plot_species_biological_metrics(validated, tukey_results, output_dir))
    figure_paths.extend(plot_two_column_biological_metrics(validated, tukey_results, output_dir))
    figure_paths.extend(plot_early_growth_auc(validated, tukey_results, output_dir))

    print("DMSO statistical analysis summary")
    print(f"- input workbook: {input_path}")
    print(f"- sheet: {sheet_name}")
    print(f"- output folder: {output_dir}")
    print(f"- validated rows: {len(validated)}")
    print(f"- biological replicates: {sorted(validated['biological_replicate'].unique().tolist())}")
    print(f"- species: {sorted(validated['species'].unique().tolist())}")
    print(f"- DMSO concentrations: {sorted(validated['DMSO_percent'].unique().tolist())}")
    print(f"- ANOVA rows: {len(anova_results)}")
    print(f"- residual df values: {sorted(anova_results['residual_df'].unique().tolist())}")
    print(f"- Tukey comparisons: {len(tukey_results)}")
    print(f"- significant Tukey comparisons: {len(significant_tukey)}")
    if warnings:
        print("Warnings")
        for warning in warnings:
            print(f"- {warning}")
    print("Saved outputs")
    for path in [
        validated_path,
        descriptive_path,
        anova_path,
        tukey_path,
        significant_path,
        summary_path,
        captions_path,
        *figure_paths,
    ]:
        print(f"- {path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
