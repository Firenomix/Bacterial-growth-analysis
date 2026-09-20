from pathlib import Path
import math
import sys

import numpy as np
import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from growth_analysis import (  # noqa: E402
    DEFAULT_A2_EXCLUSION_REASON,
    DMSO_ONLY_DMSO_PERCENT,
    DMSO_ONLY_REPLICATES_BY_DMSO,
    DMSO_ONLY_TIME_MIN,
    EXPECTED_DMSO_PERCENT,
    EXPECTED_TIME_MIN,
    GrowthDataValidationError,
    add_relative_auc,
    apply_time_matched_blank_correction,
    calculate_trapezoidal_auc,
    calculate_w_response,
    construct_well,
    fit_all_wells,
    fit_one_growth_curve,
    modified_gompertz,
    parse_dmso_percent,
    parse_time_label,
    summarise_technical_wells,
    validate_growth_data,
    validate_corrected_growth_data,
)


def make_valid_growth_data() -> pd.DataFrame:
    records = []
    species_to_group = {
        "E_coli": "E. Coli",
        "B_subtilis": "B. Subtilis",
    }

    for row_index, dmso_percent in enumerate(EXPECTED_DMSO_PERCENT):
        well_row = chr(ord("B") + row_index)
        condition_label = "Control" if dmso_percent == 0 else f"DMSO {dmso_percent:g}%"
        for species, well_cols in [("E_coli", range(2, 7)), ("B_subtilis", range(8, 13))]:
            for technical_replicate, well_col in enumerate(well_cols, start=1):
                well = f"{well_row}{well_col}"
                for time_min in EXPECTED_TIME_MIN:
                    time_h = time_min / 60.0
                    records.append(
                        {
                            "experiment_id": "test_experiment",
                            "well": well,
                            "species": species,
                            "condition_label": condition_label,
                            "condition": condition_label,
                            "dmso_percent": dmso_percent,
                            "technical_replicate": technical_replicate,
                            "time_min": time_min,
                            "time_h": time_h,
                            "od600": 0.08 + 0.0001 * time_min,
                            "raw_OD600": 0.08 + 0.0001 * time_min,
                            "blank_OD600": 0.1,
                            "corrected_OD600": 0.08 + 0.0001 * time_min - 0.1,
                            "original_content": condition_label,
                            "original_group": species_to_group[species],
                        }
                    )

    return pd.DataFrame.from_records(records)


def make_growth_data_with_blanks() -> pd.DataFrame:
    sample_data = make_valid_growth_data()[
        [
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
    ].copy()
    blank_records = []
    blank_offsets = {
        "A1": 0.09,
        "A2": 0.09,
        "A3": 0.11,
    }

    for well, starting_od in blank_offsets.items():
        for time_min in EXPECTED_TIME_MIN:
            drift = 0.00002 * time_min if well == "A2" else 0
            blank_records.append(
                {
                    "experiment_id": "test_experiment",
                    "well": well,
                    "species": "Blank",
                    "condition_label": "Blank",
                    "dmso_percent": math.nan,
                    "technical_replicate": math.nan,
                    "time_min": time_min,
                    "od600": starting_od + drift,
                    "original_content": "Blank",
                    "original_group": "Blank",
                }
            )

    return pd.concat([pd.DataFrame.from_records(blank_records), sample_data], ignore_index=True)


def make_metric_ready_corrected_data() -> pd.DataFrame:
    records = []
    species_to_base = {"E_coli": 0.0, "B_subtilis": 0.05}
    for row_index, dmso_percent in enumerate(EXPECTED_DMSO_PERCENT):
        well_row = chr(ord("B") + row_index)
        condition = "Control" if dmso_percent == 0 else f"DMSO {dmso_percent:g}%"
        for species, well_cols in [("E_coli", range(2, 7)), ("B_subtilis", range(8, 13))]:
            for technical_replicate, well_col in enumerate(well_cols, start=1):
                well = f"{well_row}{well_col}"
                A = 1.4 + species_to_base[species] - 0.08 * dmso_percent
                Kz = 0.45 - 0.015 * dmso_percent
                TLag = 4.8 + 0.2 * dmso_percent
                baseline = 0.002 * technical_replicate
                for time_min in EXPECTED_TIME_MIN:
                    time_h = time_min / 60.0
                    change = modified_gompertz(np.array([time_h]), A, Kz, TLag)[0]
                    corrected_od = baseline + change
                    records.append(
                        {
                            "well": well,
                            "species": species,
                            "condition": condition,
                            "dmso_percent": dmso_percent,
                            "time_min": time_min,
                            "time_h": time_h,
                            "raw_OD600": corrected_od + 0.08,
                            "blank_OD600": 0.08,
                            "corrected_OD600": corrected_od,
                            "experiment_id": "test_experiment",
                            "technical_replicate": technical_replicate,
                            "original_content": condition,
                            "original_group": "E. Coli" if species == "E_coli" else "B. Subtilis",
                        }
                    )
    return pd.DataFrame.from_records(records)


def make_dmso_only_corrected_data() -> pd.DataFrame:
    data = make_metric_ready_corrected_data()
    data = data[data["dmso_percent"].isin(DMSO_ONLY_DMSO_PERCENT)].copy()
    data = data[data["time_min"].isin(DMSO_ONLY_TIME_MIN)].copy()
    treatment_mask = data["dmso_percent"].eq(0.0) | data["technical_replicate"].isin([1, 2, 3])
    return data[treatment_mask].reset_index(drop=True)


def run_blank_correction(data: pd.DataFrame):
    return apply_time_matched_blank_correction(
        measurement_data=data,
        blank_well_ids=["A1", "A3"],
        excluded_blank_well_ids=["A2"],
        time_column="time_min",
        raw_od_column="od600",
        excluded_blank_reasons={"A2": DEFAULT_A2_EXCLUSION_REASON},
    )


def test_parse_time_label_hours_and_minutes() -> None:
    assert parse_time_label("0 h ") == 0
    assert parse_time_label("1 h 5 min") == 65
    assert parse_time_label("17 h ") == 1020


def test_construct_well_from_row_and_column() -> None:
    assert construct_well("B", "2") == "B2"
    assert construct_well(" c ", "10") == "C10"


def test_parse_dmso_percent() -> None:
    assert parse_dmso_percent("Control") == 0.0
    assert parse_dmso_percent("DMSO 0.25%") == 0.25
    assert parse_dmso_percent("DMSO 4%") == 4.0


def test_validate_growth_data_accepts_expected_design() -> None:
    data = make_valid_growth_data()
    summary = validate_growth_data(data)

    assert summary.unique_wells == 60
    assert summary.measurements_per_well == 205
    assert summary.time_min_start == 0
    assert summary.time_min_end == 1020
    assert summary.time_step_min == 5
    assert summary.species_count == 2
    assert summary.od600_missing_count == 0


def test_validate_growth_data_rejects_missing_measurement() -> None:
    data = make_valid_growth_data().iloc[:-1].copy()

    with pytest.raises(GrowthDataValidationError, match="205 measurements per well"):
        validate_growth_data(data)


def test_blank_correction_uses_time_matched_mean_blank() -> None:
    data = make_growth_data_with_blanks()
    result = run_blank_correction(data)
    row = result.corrected_data[
        result.corrected_data["well"].eq("B2") & result.corrected_data["time_min"].eq(10)
    ].iloc[0]

    assert row["blank_OD600"] == pytest.approx((0.09 + 0.11) / 2)
    assert row["corrected_OD600"] == pytest.approx(row["raw_OD600"] - row["blank_OD600"])


def test_a2_is_excluded_but_retained_in_blank_qc() -> None:
    data = make_growth_data_with_blanks()
    result = run_blank_correction(data)
    a2_qc = result.blank_qc_summary[result.blank_qc_summary["well"].eq("A2")].iloc[0]

    assert not a2_qc["included_in_correction"]
    assert a2_qc["exclusion_reason"] == DEFAULT_A2_EXCLUSION_REASON
    assert "A2_raw_OD" not in result.blank_trace_used.columns
    assert set(result.blank_qc_summary["well"]) == {"A1", "A2", "A3"}


def test_negative_corrected_values_are_preserved() -> None:
    data = make_growth_data_with_blanks()
    result = run_blank_correction(data)

    assert result.corrected_data["corrected_OD600"].min() < 0


def test_missing_blank_well_produces_informative_error() -> None:
    data = make_growth_data_with_blanks()
    data = data[~data["well"].eq("A3")].copy()

    with pytest.raises(GrowthDataValidationError, match="Missing required blank well A3"):
        run_blank_correction(data)


def test_mismatched_blank_and_sample_time_points_produce_informative_error() -> None:
    data = make_growth_data_with_blanks()
    data = data[~(data["well"].eq("A3") & data["time_min"].eq(1020))].copy()

    with pytest.raises(GrowthDataValidationError, match="Blank and sample time points must match"):
        run_blank_correction(data)


def test_original_raw_od_values_are_not_modified_by_blank_correction() -> None:
    data = make_growth_data_with_blanks()
    original_od = data["od600"].copy(deep=True)

    run_blank_correction(data)

    pd.testing.assert_series_equal(data["od600"], original_od)


def test_all_60_bacterial_wells_remain_after_blank_correction() -> None:
    data = make_growth_data_with_blanks()
    result = run_blank_correction(data)

    assert result.corrected_data["well"].nunique() == 60
    assert len(result.corrected_data) == 60 * 205


def test_blank_wells_are_not_in_processed_sample_data() -> None:
    data = make_growth_data_with_blanks()
    result = run_blank_correction(data)

    assert not result.corrected_data["well"].isin(["A1", "A2", "A3"]).any()
    assert set(result.corrected_data["species"]) == {"E_coli", "B_subtilis"}


def test_trapezoidal_auc_is_correct_for_simple_curve() -> None:
    curve = pd.DataFrame({"time_h": [0.0, 1.0, 2.0], "corrected_OD600": [0.0, 1.0, 1.0]})

    assert calculate_trapezoidal_auc(curve, 0.0, 2.0) == pytest.approx(1.5)


def test_auc_windows_use_correct_boundaries() -> None:
    curve = pd.DataFrame(
        {
            "time_h": [0.0, 12.0, 17.0],
            "corrected_OD600": [1.0, 1.0, 1.0],
        }
    )

    assert calculate_trapezoidal_auc(curve, 0.0, 17.0) == pytest.approx(17.0)
    assert calculate_trapezoidal_auc(curve, 0.0, 12.0) == pytest.approx(12.0)


def test_auc_requires_present_boundaries() -> None:
    curve = pd.DataFrame({"time_h": [0.0, 1.0, 2.0], "corrected_OD600": [0.0, 1.0, 1.0]})

    with pytest.raises(GrowthDataValidationError, match="Required AUC boundaries"):
        calculate_trapezoidal_auc(curve, 0.0, 1.5)


def test_auc_uses_corrected_od_not_w_response() -> None:
    curve = pd.DataFrame({"time_h": [0.0, 1.0], "corrected_OD600": [5.0, 5.0]})

    assert calculate_trapezoidal_auc(curve, 0.0, 1.0) == pytest.approx(5.0)


def test_negative_corrected_od_values_are_preserved_in_auc() -> None:
    curve = pd.DataFrame({"time_h": [0.0, 1.0], "corrected_OD600": [-1.0, -1.0]})

    assert calculate_trapezoidal_auc(curve, 0.0, 1.0) == pytest.approx(-1.0)


def test_relative_auc_matches_species_replicate_and_window() -> None:
    metrics = pd.DataFrame(
        {
            "biological_replicate": ["BR1", "BR1", "BR1", "BR1"],
            "species": ["E_coli", "E_coli", "B_subtilis", "B_subtilis"],
            "DMSO_percent": [0.0, 1.0, 0.0, 1.0],
            "AUC_0_17h_OD_h": [10.0, 5.0, 20.0, 10.0],
            "AUC_0_12h_OD_h": [6.0, 3.0, 12.0, 3.0],
        }
    )

    result = add_relative_auc(metrics)

    assert result.loc[1, "relative_AUC_0_17h_percent"] == pytest.approx(50.0)
    assert result.loc[3, "relative_AUC_0_17h_percent"] == pytest.approx(50.0)
    assert result.loc[3, "relative_AUC_0_12h_percent"] == pytest.approx(25.0)


def test_w_response_is_relative_to_first_measurement() -> None:
    curve = pd.DataFrame({"time_h": [0.0, 1.0, 2.0], "corrected_OD600": [0.2, 0.3, 0.1]})

    result = calculate_w_response(curve)

    assert result["W"].tolist() == pytest.approx([0.0, 0.1, -0.1])


def test_modified_gompertz_matches_specified_equation() -> None:
    t = np.array([2.0])
    A = 1.2
    Kz = 0.3
    TLag = 4.0
    expected = A * np.exp(-np.exp(((math.e * Kz / A) * (TLag - t)) + 1.0))

    assert modified_gompertz(t, A, Kz, TLag)[0] == pytest.approx(expected[0])


def test_synthetic_gompertz_fit_recovers_known_parameters() -> None:
    time_h = np.linspace(0, 17, 80)
    expected = {"A": 1.3, "Kz": 0.42, "TLag": 4.7}
    w = modified_gompertz(time_h, expected["A"], expected["Kz"], expected["TLag"])
    curve = pd.DataFrame({"time_h": time_h, "corrected_OD600": 0.05 + w})

    result = fit_one_growth_curve(curve)

    assert result["gompertz_fit_status"].startswith("success")
    assert result["A_OD600"] == pytest.approx(expected["A"], rel=0.03)
    assert result["Kz_OD600_per_h"] == pytest.approx(expected["Kz"], rel=0.05)
    assert result["TLag_h"] == pytest.approx(expected["TLag"], rel=0.03)


def test_one_failed_fit_does_not_stop_other_wells() -> None:
    data = make_metric_ready_corrected_data()
    data.loc[data["well"].eq("B2"), "corrected_OD600"] = 0.1

    metrics = fit_all_wells(data, "BR1")

    assert len(metrics) == 60
    assert metrics[metrics["well"].eq("B2")]["gompertz_fit_status"].iloc[0].startswith("fit_failed")
    assert metrics[metrics["well"].ne("B2")]["gompertz_fit_status"].str.startswith("success").any()


def test_failed_fits_are_reported_as_nan_with_status() -> None:
    curve = pd.DataFrame({"time_h": [0.0, 1.0, 2.0], "corrected_OD600": [0.0, 0.1, 0.2]})

    result = fit_one_growth_curve(curve)

    assert math.isnan(result["Kz_OD600_per_h"])
    assert math.isnan(result["TLag_h"])
    assert math.isnan(result["A_OD600"])
    assert math.isnan(result["gompertz_R2"])
    assert result["gompertz_fit_status"].startswith("fit_failed")


def test_per_well_output_contains_60_bacterial_wells_and_no_blanks() -> None:
    data = make_metric_ready_corrected_data()
    metrics = fit_all_wells(data, "BR1")

    assert len(metrics) == 60
    assert metrics["well"].nunique() == 60
    assert not metrics["well"].isin(["A1", "A2", "A3"]).any()
    assert set(metrics["biological_replicate"]) == {"BR1"}


def test_biological_replicate_summary_contains_12_rows_and_no_blanks() -> None:
    metrics = fit_all_wells(make_metric_ready_corrected_data(), "BR1")
    summary = summarise_technical_wells(metrics)

    assert len(summary) == 12
    assert not summary["species"].eq("Blank").any()
    assert set(summary["DMSO_percent"]) == set(EXPECTED_DMSO_PERCENT)
    assert set(summary["species"]) == {"E_coli", "B_subtilis"}


def test_summary_mean_uses_corresponding_technical_wells() -> None:
    metrics = fit_all_wells(make_metric_ready_corrected_data(), "BR1")
    summary = summarise_technical_wells(metrics)
    group = metrics[(metrics["species"].eq("E_coli")) & (metrics["DMSO_percent"].eq(1.0))]
    summary_row = summary[(summary["species"].eq("E_coli")) & (summary["DMSO_percent"].eq(1.0))].iloc[0]

    assert summary_row["AUC_0_17h_OD_h_mean"] == pytest.approx(group["AUC_0_17h_OD_h"].mean())


def test_technical_sd_uses_ddof_1() -> None:
    metrics = fit_all_wells(make_metric_ready_corrected_data(), "BR1")
    summary = summarise_technical_wells(metrics)
    group = metrics[(metrics["species"].eq("B_subtilis")) & (metrics["DMSO_percent"].eq(2.0))]
    summary_row = summary[(summary["species"].eq("B_subtilis")) & (summary["DMSO_percent"].eq(2.0))].iloc[0]

    assert summary_row["AUC_0_12h_OD_h_SD_technical"] == pytest.approx(
        group["AUC_0_12h_OD_h"].std(ddof=1)
    )


def test_raw_and_corrected_measurements_are_not_modified_by_metric_calculation() -> None:
    data = make_metric_ready_corrected_data()
    original_raw = data["raw_OD600"].copy(deep=True)
    original_corrected = data["corrected_OD600"].copy(deep=True)

    fit_all_wells(data, "BR1")

    pd.testing.assert_series_equal(data["raw_OD600"], original_raw)
    pd.testing.assert_series_equal(data["corrected_OD600"], original_corrected)


def test_dmso_only_validation_accepts_adjusted_dataset_design() -> None:
    data = make_dmso_only_corrected_data()

    validate_corrected_growth_data(
        data,
        expected_unique_wells=34,
        expected_time_min=DMSO_ONLY_TIME_MIN,
        expected_dmso_percent=DMSO_ONLY_DMSO_PERCENT,
        expected_replicates=DMSO_ONLY_REPLICATES_BY_DMSO,
    )


def test_dmso_only_metrics_and_summary_use_adjusted_design() -> None:
    data = make_dmso_only_corrected_data()
    metrics = fit_all_wells(
        data,
        "DMSO_only_run2",
        auc_windows=[("0_16h35min", 0.0, 995.0 / 60.0), ("0_12h", 0.0, 12.0)],
        expected_unique_wells=34,
        expected_time_min=DMSO_ONLY_TIME_MIN,
        expected_dmso_percent=DMSO_ONLY_DMSO_PERCENT,
        expected_replicates=DMSO_ONLY_REPLICATES_BY_DMSO,
    )
    summary = summarise_technical_wells(metrics)

    assert len(metrics) == 34
    assert metrics["well"].nunique() == 34
    assert len(summary) == 10
    assert set(summary["DMSO_percent"]) == set(DMSO_ONLY_DMSO_PERCENT)
    assert "AUC_0_16h35min_OD_h" in metrics.columns
    assert "relative_AUC_0_16h35min_percent" in metrics.columns
    assert not metrics["well"].isin(["A1", "A2", "A3"]).any()


def test_full_validation_rejects_dmso_only_dataset_without_adjusted_mode() -> None:
    data = make_dmso_only_corrected_data()

    with pytest.raises(GrowthDataValidationError, match="Expected 60 bacterial sample wells"):
        validate_corrected_growth_data(data)
