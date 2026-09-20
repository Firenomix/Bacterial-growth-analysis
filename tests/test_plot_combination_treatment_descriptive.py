from pathlib import Path
import importlib.util
import math
import sys

import pandas as pd
import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def load_script():
    script_path = PROJECT_ROOT / "analyses" / "plot_combination_treatment_descriptive.py"
    spec = importlib.util.spec_from_file_location("plot_combination_treatment_descriptive", script_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def make_descriptive(relative_auc_scale: str = "percent") -> pd.DataFrame:
    script = load_script()
    rows = []
    for species_index, species in enumerate(script.EXPECTED_SPECIES):
        for replicate_index, replicate in enumerate(script.EXPECTED_REPLICATES):
            for compound_index, compound in enumerate(script.EXPECTED_COMPOUNDS):
                for condition in ["vehicle", "compound_alone", "ampicillin_alone", "combination"]:
                    for metric in script.EXPECTED_METRICS:
                        if condition == "combination":
                            mean = 50.0 + species_index * 10 + replicate_index + compound_index
                            condition_id = f"{compound}_25uM_ampicillin_4ug_mL"
                        elif condition == "ampicillin_alone":
                            mean = 40.0 + species_index * 10 + replicate_index
                            condition_id = "ampicillin_4ug_mL"
                        elif condition == "compound_alone":
                            mean = 90.0
                            condition_id = f"{compound}_25uM"
                        else:
                            mean = 100.0
                            condition_id = "vehicle_1pct_DMSO"

                        if metric == "Kz":
                            mean = mean / 500.0
                        elif metric == "TLag":
                            mean = mean / 10.0
                        elif metric == "relative_AUC" and relative_auc_scale == "ratio":
                            mean = mean / 100.0

                        rows.append(
                            {
                                "species": species,
                                "biological_replicate": replicate,
                                "compound": compound,
                                "compound_concentration_uM": 25.0,
                                "ampicillin_concentration_ug_mL": 4.0,
                                "comparison_condition": condition,
                                "metric": metric,
                                "n_technical_wells": 3,
                                "technical_mean": mean,
                                "technical_SD": 999.0,
                                "condition_id": condition_id,
                            }
                        )
    return pd.DataFrame.from_records(rows)


def test_build_plotting_tables_filters_to_matched_regimen_and_expected_shape() -> None:
    script = load_script()
    data = make_descriptive()
    data.loc[len(data)] = {
        "species": "E_coli",
        "biological_replicate": "BR1",
        "compound": "C1",
        "compound_concentration_uM": 12.5,
        "ampicillin_concentration_ug_mL": 2.0,
        "comparison_condition": "combination",
        "metric": "relative_AUC",
        "n_technical_wells": 3,
        "technical_mean": 999.0,
        "technical_SD": 999.0,
        "condition_id": "old_regimen",
    }

    plot_data, wide_qc, warnings_table, unit_notes = script.build_plotting_tables(data)

    assert len(plot_data) == 60
    assert len(wide_qc) == 20
    assert warnings_table.empty
    assert set(plot_data["biological_replicate"]) == {"BR2", "BR3"}
    assert set(plot_data["treatment"]) == set(script.TREATMENT_ORDER)
    assert set(plot_data["metric"]) == {"relative_AUC", "TLag", "Kz"}
    assert not plot_data["technical_mean"].eq(999.0).any()
    assert unit_notes.set_index("metric").loc["relative_AUC", "conversion_applied"].startswith("none")


def test_ampicillin_contextual_duplicates_are_deduplicated_without_averaging() -> None:
    script = load_script()
    plot_data, _, _, _ = script.build_plotting_tables(make_descriptive())

    amp = plot_data[
        plot_data["species"].eq("E_coli")
        & plot_data["biological_replicate"].eq("BR2")
        & plot_data["metric"].eq("relative_AUC")
        & plot_data["treatment"].eq("Ampicillin 4 ug/mL")
    ]

    assert len(amp) == 1
    assert amp.iloc[0]["technical_mean"] == pytest.approx(40.0)
    assert amp.iloc[0]["source_rows_deduplicated"] == 4


def test_inconsistent_ampicillin_duplicates_raise_informative_error() -> None:
    script = load_script()
    data = make_descriptive()
    mask = (
        data["species"].eq("E_coli")
        & data["biological_replicate"].eq("BR2")
        & data["metric"].eq("relative_AUC")
        & data["comparison_condition"].eq("ampicillin_alone")
        & data["compound"].eq("C4")
    )
    data.loc[mask, "technical_mean"] += 0.001

    with pytest.raises(script.CombinationDescriptivePlotError, match="Repeated ampicillin-only values disagree"):
        script.build_plotting_tables(data)


def test_plotted_value_uses_technical_mean_not_technical_sd() -> None:
    script = load_script()
    plot_data, _, _, _ = script.build_plotting_tables(make_descriptive())
    row = plot_data[
        plot_data["species"].eq("B_subtilis")
        & plot_data["biological_replicate"].eq("BR3")
        & plot_data["metric"].eq("Kz")
        & plot_data["treatment"].eq("C2 + Amp")
    ].iloc[0]

    assert row["technical_SD"] == pytest.approx(999.0)
    assert row["plotted_value"] == pytest.approx(row["technical_mean"])


def test_relative_auc_ratio_values_are_converted_once_to_percent() -> None:
    script = load_script()
    plot_data, _, _, unit_notes = script.build_plotting_tables(make_descriptive(relative_auc_scale="ratio"))
    row = plot_data[
        plot_data["species"].eq("E_coli")
        & plot_data["biological_replicate"].eq("BR2")
        & plot_data["metric"].eq("relative_AUC")
        & plot_data["treatment"].eq("C1 + Amp")
    ].iloc[0]

    assert row["technical_mean"] == pytest.approx(0.5)
    assert row["plotted_value"] == pytest.approx(50.0)
    assert "multiplied by 100" in unit_notes.set_index("metric").loc["relative_AUC", "conversion_applied"]


def test_tlag_minutes_are_converted_to_hours_only_when_needed() -> None:
    script = load_script()
    data = make_descriptive()
    data.loc[data["metric"].eq("TLag"), "technical_mean"] *= 60.0

    plot_data, _, _, unit_notes = script.build_plotting_tables(data)
    row = plot_data[
        plot_data["species"].eq("E_coli")
        & plot_data["biological_replicate"].eq("BR2")
        & plot_data["metric"].eq("TLag")
        & plot_data["treatment"].eq("C1 + Amp")
    ].iloc[0]

    assert row["technical_mean"] == pytest.approx(300.0)
    assert row["plotted_value"] == pytest.approx(5.0)
    assert "divided by 60" in unit_notes.set_index("metric").loc["TLag", "conversion_applied"]


def test_missing_values_are_reported_and_mean_line_is_omitted_for_incomplete_pair() -> None:
    script = load_script()
    data = make_descriptive()
    missing_mask = (
        data["species"].eq("E_coli")
        & data["biological_replicate"].eq("BR3")
        & data["comparison_condition"].eq("combination")
        & data["compound"].eq("C1")
        & data["metric"].eq("Kz")
    )
    data = data.loc[~missing_mask].copy()

    plot_data, wide_qc, _, _ = script.build_plotting_tables(data)
    warnings_table = script.missing_value_warnings(plot_data)
    means = script.mean_line_segments(plot_data, "E_coli", "Kz")

    missing_row = plot_data[
        plot_data["species"].eq("E_coli")
        & plot_data["biological_replicate"].eq("BR3")
        & plot_data["treatment"].eq("C1 + Amp")
        & plot_data["metric"].eq("Kz")
    ].iloc[0]

    assert missing_row["missing_value"]
    assert "Missing expected" in warnings_table.iloc[0]["message"]
    assert "Kz" in wide_qc[wide_qc["missing_metrics"].ne("")].iloc[0]["missing_metrics"]
    assert "C1 + Amp" not in set(means["treatment"])


def test_n_technical_wells_warning_is_reported() -> None:
    script = load_script()
    data = make_descriptive()
    mask = data["comparison_condition"].eq("combination") & data["compound"].eq("C1") & data["metric"].eq("Kz")
    data.loc[mask, "n_technical_wells"] = 2

    _, _, warnings_table, _ = script.build_plotting_tables(data)

    assert not warnings_table.empty
    assert warnings_table["message"].str.contains("expected 3").any()


def test_run_writes_outputs_and_does_not_modify_input(tmp_path: Path) -> None:
    script = load_script()
    data = make_descriptive()
    input_path = tmp_path / "combination_treatment_descriptive_metrics.csv"
    output_dir = tmp_path / "plots"
    data.to_csv(input_path, index=False)
    before = input_path.read_bytes()

    result = script.run(input_path, output_dir)

    assert input_path.read_bytes() == before
    assert len(result["plot_data"]) == 60
    assert len(result["wide_qc"]) == 20
    assert (output_dir / "combination_treatment_descriptive_plot_data_long.csv").exists()
    assert (output_dir / "combination_treatment_descriptive_qc_wide.csv").exists()
    assert len(result["figures"]) == 16
    assert all(path.exists() for path in result["figures"])


def test_two_column_combined_figure_uses_species_columns_and_metric_rows(tmp_path: Path, monkeypatch) -> None:
    script = load_script()
    plot_data, _, _, _ = script.build_plotting_tables(make_descriptive())
    calls = []
    original_draw_panel = script._draw_panel

    def record_draw_panel(ax, panel, metric, species, **kwargs):
        calls.append(
            {
                "species": species,
                "metric": metric,
                "panel_label": kwargs["panel_label"],
                "panel_title": kwargs["panel_title"],
            }
        )
        return original_draw_panel(ax, panel, metric, species, **kwargs)

    monkeypatch.setattr(script, "_draw_panel", record_draw_panel)

    paths = script.plot_two_column_combined_figure(plot_data, tmp_path)

    assert [path.name for path in paths] == [
        "combination_treatment_descriptive_combined_two_column.png",
        "combination_treatment_descriptive_combined_two_column.pdf",
    ]
    assert all(path.exists() for path in paths)
    assert [(call["panel_label"], call["species"], call["metric"], call["panel_title"]) for call in calls] == [
        ("(A)", "B_subtilis", "relative_AUC", "Relative AUC"),
        ("(B)", "E_coli", "relative_AUC", "Relative AUC"),
        ("(C)", "B_subtilis", "Kz", "Growth rate"),
        ("(D)", "E_coli", "Kz", "Growth rate"),
        ("(E)", "B_subtilis", "TLag", "Lag time"),
        ("(F)", "E_coli", "TLag", "Lag time"),
    ]
    assert "\\cdot" not in script.METRIC_SPECS["Kz"].y_label
    assert "OD$_{600}$·h$^{-1}$" in script.METRIC_SPECS["Kz"].y_label
