from pathlib import Path
import importlib.util
import inspect
import math
import os
import sys

import numpy as np
import pandas as pd
import pytest
from scipy import stats

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import dmso_statistics as ds  # noqa: E402


def load_analysis_script():
    script_path = PROJECT_ROOT / "analyses" / "dmso_statistics.py"
    spec = importlib.util.spec_from_file_location("dmso_statistics_analysis_script", script_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def make_biological_summary() -> pd.DataFrame:
    records = []
    treatment_effects = {
        0.0: 0.0,
        0.25: 1.0,
        0.5: 2.0,
        1.0: 3.0,
        2.0: 4.0,
        4.0: 5.0,
    }
    block_effects = {"BR1": -1.0, "BR2": 0.5, "BR3": 0.5}
    residuals = {
        "BR1": [0.2, -0.1, 0.1, -0.2, 0.0, 0.0],
        "BR2": [-0.1, 0.2, -0.2, 0.0, 0.1, 0.0],
        "BR3": [-0.1, -0.1, 0.1, 0.2, -0.1, 0.0],
    }
    species_base = {"B_subtilis": 5.0, "E_coli": 10.0}
    for species in ds.EXPECTED_SPECIES:
        for replicate in ds.EXPECTED_BIOLOGICAL_REPLICATES:
            for index, dmso in enumerate(ds.EXPECTED_DMSO_PERCENT):
                value = species_base[species] + treatment_effects[dmso] + block_effects[replicate] + residuals[replicate][index]
                condition = "Control" if dmso == 0 else f"DMSO {dmso:g}%"
                records.append(
                    {
                        "biological_replicate": replicate,
                        "species": species,
                        "condition": condition,
                        "DMSO_percent": dmso,
                        "n_technical_wells": 5,
                        "AUC_0_17h_OD_h_mean": value,
                        "AUC_0_17h_OD_h_SD_technical": 9999.0,
                        "AUC_0_12h_OD_h_mean": value / 2,
                        "AUC_0_12h_OD_h_SD_technical": 9999.0,
                        "Kz_OD600_per_h_mean": value / 20,
                        "Kz_OD600_per_h_SD_technical": 9999.0,
                        "TLag_h_mean": 8 - value / 10,
                        "TLag_h_SD_technical": 9999.0,
                        "A_OD600_mean": value / 5,
                        "A_OD600_SD_technical": 9999.0,
                    }
                )
    return pd.DataFrame.from_records(records)


def manual_randomized_block_ss(data: pd.DataFrame, species: str, metric: str) -> tuple:
    subset = data[data["species"].eq(species)]
    grand_mean = subset[metric].mean()
    treatment_means = subset.groupby("DMSO_percent")[metric].mean().reindex(ds.EXPECTED_DMSO_PERCENT)
    block_means = subset.groupby("biological_replicate")[metric].mean().reindex(ds.EXPECTED_BIOLOGICAL_REPLICATES)
    a = len(ds.EXPECTED_DMSO_PERCENT)
    b = len(ds.EXPECTED_BIOLOGICAL_REPLICATES)
    total_ss = ((subset[metric] - grand_mean) ** 2).sum()
    treatment_ss = b * ((treatment_means - grand_mean) ** 2).sum()
    block_ss = a * ((block_means - grand_mean) ** 2).sum()
    residual_ss = total_ss - treatment_ss - block_ss
    return treatment_ss, block_ss, residual_ss


def test_incomplete_block_design_is_rejected() -> None:
    data = make_biological_summary().iloc[:-1].copy()

    with pytest.raises(ds.DMSOStatisticsValidationError, match="Missing randomized-block cells"):
        ds.validate_biological_summary_data(data)


def test_duplicate_rows_are_rejected() -> None:
    data = make_biological_summary()
    data = pd.concat([data, data.iloc[[0]]], ignore_index=True)

    with pytest.raises(ds.DMSOStatisticsValidationError, match="Duplicate biological replicate"):
        ds.validate_biological_summary_data(data)


def test_missing_or_nonnumeric_metric_values_are_rejected() -> None:
    data = make_biological_summary()
    data["AUC_0_17h_OD_h_mean"] = data["AUC_0_17h_OD_h_mean"].astype(object)
    data.loc[0, "AUC_0_17h_OD_h_mean"] = "placeholder"

    with pytest.raises(ds.DMSOStatisticsValidationError, match="nonnumeric"):
        ds.validate_biological_summary_data(data)


def test_technical_sd_columns_do_not_enter_inferential_analysis() -> None:
    data = ds.validate_biological_summary_data(make_biological_summary())
    changed = data.copy()
    changed["AUC_0_17h_OD_h_SD_technical"] = np.linspace(1, 1_000_000, len(changed))

    original = ds.run_all_randomized_block_anovas(data)
    modified = ds.run_all_randomized_block_anovas(changed)

    pd.testing.assert_series_equal(original["treatment_F"], modified["treatment_F"])


def test_biological_replicate_is_included_as_block(monkeypatch) -> None:
    formulas = []
    original_ols = ds.smf.ols

    def capture_ols(formula, data):
        formulas.append(formula)
        return original_ols(formula, data=data)

    monkeypatch.setattr(ds.smf, "ols", capture_ols)
    data = ds.validate_biological_summary_data(make_biological_summary())
    ds.randomized_block_anova(data, "E_coli", "AUC_0_17h_OD_h_mean")

    assert formulas == ["value ~ C(DMSO_concentration) + C(biological_replicate)"]


def test_randomized_block_anova_sums_of_squares_match_manual_calculation() -> None:
    data = ds.validate_biological_summary_data(make_biological_summary())
    result = ds.randomized_block_anova(data, "E_coli", "AUC_0_17h_OD_h_mean")
    treatment_ss, block_ss, residual_ss = manual_randomized_block_ss(
        data,
        "E_coli",
        "AUC_0_17h_OD_h_mean",
    )

    assert result.treatment_SS == pytest.approx(treatment_ss)
    assert result.block_SS == pytest.approx(block_ss)
    assert result.residual_SS == pytest.approx(residual_ss)


def test_residual_degrees_of_freedom_equal_10() -> None:
    data = ds.validate_biological_summary_data(make_biological_summary())
    result = ds.randomized_block_anova(data, "E_coli", "AUC_0_17h_OD_h_mean")

    assert result.residual_df == 10


def test_treatment_results_unchanged_by_constant_block_offset() -> None:
    data = ds.validate_biological_summary_data(make_biological_summary())
    shifted = data.copy()
    offsets = {"BR1": 100.0, "BR2": -50.0, "BR3": 25.0}
    shifted["AUC_0_17h_OD_h_mean"] = shifted["AUC_0_17h_OD_h_mean"] + shifted["biological_replicate"].map(offsets)

    original = ds.randomized_block_anova(data, "E_coli", "AUC_0_17h_OD_h_mean")
    shifted_result = ds.randomized_block_anova(shifted, "E_coli", "AUC_0_17h_OD_h_mean")

    assert shifted_result.treatment_SS == pytest.approx(original.treatment_SS)
    assert shifted_result.residual_SS == pytest.approx(original.residual_SS)
    assert shifted_result.treatment_F == pytest.approx(original.treatment_F)


def test_tukey_uses_randomized_block_residual_mse() -> None:
    data = ds.validate_biological_summary_data(make_biological_summary())
    anova = ds.randomized_block_anova(data, "E_coli", "AUC_0_17h_OD_h_mean")
    tukey = ds.randomized_block_tukey_hsd(data, anova)
    row = tukey[(tukey["DMSO_group_1"].eq(0.0)) & (tukey["DMSO_group_2"].eq(0.25))].iloc[0]
    expected_se = math.sqrt(anova.residual_MS / anova.number_of_biological_replicates)
    expected_q = abs(row["mean_difference_group_2_minus_group_1"]) / expected_se

    assert row["Tukey_q"] == pytest.approx(expected_q)


def test_independent_groups_pairwise_tukeyhsd_is_not_used() -> None:
    source = inspect.getsource(ds)

    assert "pairwise_tukeyhsd" not in source


def test_all_15_pairwise_comparisons_are_produced_per_species_and_metric() -> None:
    data = ds.validate_biological_summary_data(make_biological_summary())
    anova = ds.randomized_block_anova(data, "E_coli", "AUC_0_17h_OD_h_mean")
    tukey = ds.randomized_block_tukey_hsd(data, anova)

    assert len(tukey) == 15


def test_adjusted_p_values_use_studentized_range_distribution() -> None:
    data = ds.validate_biological_summary_data(make_biological_summary())
    anova = ds.randomized_block_anova(data, "E_coli", "AUC_0_17h_OD_h_mean")
    tukey = ds.randomized_block_tukey_hsd(data, anova)
    row = tukey.iloc[0]
    expected_p = stats.studentized_range.sf(
        row["Tukey_q"],
        anova.number_of_DMSO_conditions,
        anova.residual_df,
    )

    assert row["adjusted_p_value"] == pytest.approx(expected_p)


def test_simultaneous_confidence_intervals_are_calculated_correctly() -> None:
    data = ds.validate_biological_summary_data(make_biological_summary())
    anova = ds.randomized_block_anova(data, "E_coli", "AUC_0_17h_OD_h_mean")
    tukey = ds.randomized_block_tukey_hsd(data, anova)
    row = tukey.iloc[0]
    q_critical = stats.studentized_range.ppf(
        0.95,
        anova.number_of_DMSO_conditions,
        anova.residual_df,
    )
    half_width = q_critical * math.sqrt(anova.residual_MS / anova.number_of_biological_replicates)

    assert row["simultaneous_95CI_lower"] == pytest.approx(
        row["mean_difference_group_2_minus_group_1"] - half_width
    )
    assert row["simultaneous_95CI_upper"] == pytest.approx(
        row["mean_difference_group_2_minus_group_1"] + half_width
    )


def test_star_labels_use_adjusted_p_values() -> None:
    assert ds.significance_label(0.049) == "*"
    assert ds.significance_label(0.009) == "**"
    assert ds.significance_label(0.0009) == "***"
    assert ds.significance_label(0.00009) == "****"
    assert ds.significance_label(0.051) == ""


def test_validated_input_contains_36_biological_summary_rows() -> None:
    data = ds.validate_biological_summary_data(make_biological_summary())

    assert len(data) == 36


def test_complete_tukey_output_contains_120_rows() -> None:
    data = ds.validate_biological_summary_data(make_biological_summary())
    anova = ds.run_all_randomized_block_anovas(data)
    tukey = ds.run_all_randomized_block_tukey(data, anova)

    assert len(tukey) == 120


def test_three_biological_replicate_values_exist_per_condition() -> None:
    data = ds.validate_biological_summary_data(make_biological_summary())
    counts = data.groupby(["species", "DMSO_percent"])["biological_replicate"].nunique()

    assert counts.eq(3).all()


def test_blank_or_technical_well_rows_are_rejected() -> None:
    data = make_biological_summary()
    data["well"] = ""
    data.loc[0, "well"] = "B2"

    with pytest.raises(ds.DMSOStatisticsValidationError, match="technical-well rows"):
        ds.validate_biological_summary_data(data)


def test_original_workbook_is_not_modified_by_reader(tmp_path: Path) -> None:
    workbook = tmp_path / "summary.xlsx"
    with pd.ExcelWriter(workbook) as writer:
        make_biological_summary().to_excel(writer, sheet_name="summary", index=False)
    before_mtime = os.path.getmtime(workbook)

    data, sheet_name, warnings = ds.read_biological_summary_workbook(workbook)

    assert len(data) == 36
    assert sheet_name == "summary"
    assert warnings == []
    assert os.path.getmtime(workbook) == before_mtime


def test_figure_labels_and_displayed_comparisons_are_control_only() -> None:
    analysis = load_analysis_script()
    tukey = pd.DataFrame(
        {
            "species": ["E_coli"] * 6,
            "metric": ["Kz_OD600_per_h_mean"] * 6,
            "DMSO_group_1": [0.0, 0.0, 0.0, 0.0, 0.0, 0.25],
            "DMSO_group_2": [0.25, 0.5, 1.0, 2.0, 4.0, 4.0],
            "adjusted_p_value": [0.06, 0.04, 0.009, 0.0009, 0.00009, 0.00001],
            "statistically_significant": [False, True, True, True, True, True],
            "significance_label": ["", "*", "**", "***", "****", "****"],
        }
    )

    panel = analysis._panel_control_pairs(tukey, "E_coli", "Kz_OD600_per_h_mean")

    assert analysis._metric_title("Kz_OD600_per_h_mean") == "Growth rate"
    assert analysis._metric_title("TLag_h_mean") == "Lag time"
    metric_labels = {metric: ylabel for metric, ylabel, _ in analysis.MAIN_FIGURE_METRICS}
    panel_titles = {metric: title for metric, _, title in analysis.MAIN_FIGURE_METRICS}
    assert metric_labels["Kz_OD600_per_h_mean"] == "Growth rate, $K_z$ (OD$_{600}$·h$^{-1}$)"
    assert metric_labels["TLag_h_mean"] == "Lag time, $T_{\\mathrm{Lag}}$ (h)"
    assert panel_titles["Kz_OD600_per_h_mean"] == "Growth rate"
    assert panel_titles["TLag_h_mean"] == "Lag time"
    assert "Tukey’s multiple-comparisons test" in analysis.CAPTION_MARKDOWN
    assert "Only comparisons with the 0% DMSO control are displayed" in analysis.CAPTION_MARKDOWN
    assert len(panel) == 5
    assert panel["DMSO_group_1"].eq(0.0).all()
    assert not panel["DMSO_group_2"].eq(0.0).any()
    assert panel["figure_annotation_label"].tolist() == ["n.s.", "*", "**", "***", "****"]


def test_two_column_dmso_figure_uses_species_columns_and_metric_rows(tmp_path: Path, monkeypatch) -> None:
    analysis = load_analysis_script()
    data = ds.validate_biological_summary_data(make_biological_summary())
    anova = ds.run_all_randomized_block_anovas(data)
    tukey = ds.run_all_randomized_block_tukey(data, anova)
    calls = []
    original_plot_metric_panel = analysis.plot_metric_panel

    def record_plot_metric_panel(ax, data, tukey_results, species, metric, ylabel, color_map, **kwargs):
        calls.append(
            {
                "species": species,
                "metric": metric,
                "panel_label": kwargs["panel_label"],
                "panel_title": kwargs["panel_title"],
            }
        )
        return original_plot_metric_panel(ax, data, tukey_results, species, metric, ylabel, color_map, **kwargs)

    monkeypatch.setattr(analysis, "plot_metric_panel", record_plot_metric_panel)

    paths = analysis.plot_two_column_biological_metrics(data, tukey, tmp_path)

    assert [path.name for path in paths] == [
        "dmso_biological_metrics_two_column.png",
        "dmso_biological_metrics_two_column.pdf",
    ]
    assert all(path.exists() for path in paths)
    assert [(call["panel_label"], call["species"], call["metric"], call["panel_title"]) for call in calls] == [
        ("(A)", "B_subtilis", "AUC_0_17h_OD_h_mean", "Growth over 0–17 h (AUC)"),
        ("(B)", "E_coli", "AUC_0_17h_OD_h_mean", "Growth over 0–17 h (AUC)"),
        ("(C)", "B_subtilis", "Kz_OD600_per_h_mean", "Growth rate"),
        ("(D)", "E_coli", "Kz_OD600_per_h_mean", "Growth rate"),
        ("(E)", "B_subtilis", "TLag_h_mean", "Lag time"),
        ("(F)", "E_coli", "TLag_h_mean", "Lag time"),
    ]
