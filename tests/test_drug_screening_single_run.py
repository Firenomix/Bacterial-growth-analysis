from pathlib import Path
import importlib.util
import inspect
import math
import sys

import numpy as np
import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from growth_analysis import GrowthDataValidationError  # noqa: E402


def load_script():
    script_path = PROJECT_ROOT / "analyses" / "drug_screening_single_run.py"
    spec = importlib.util.spec_from_file_location("drug_screening_single_run", script_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def time_label(time_min: int) -> str:
    hours = time_min // 60
    minutes = time_min % 60
    if minutes == 0:
        return f"{hours} h"
    return f"{hours} h {minutes} min"


def treatment_triplets():
    treatments = [("Vehicle Control", "1% DMSO")]
    treatments.extend([("Positive Control (Ampicillin)", group) for group in ["2ug/mL", "8ug/mL"]])
    for compound in ["C1", "C2", "C3", "C4"]:
        for concentration in [6.25, 12.5, 25, 50, 100]:
            treatments.append((compound, f"{concentration:g}uM"))
    for concentration in [12.5, 25, 50, 100]:
        treatments.append(("PenAg", f"{concentration:g}uM"))
    for compound in ["C1 + Ampicillin", "C2 + Ampicillin", "C3+Ampicillin", "C4+Ampicillin"]:
        treatments.append((compound, "12.5uM+ 2ug/mL"))
    assert len(treatments) == 31
    return treatments


def make_synthetic_workbook(path: Path, extra_timepoints: bool = True, missing_blank: bool = False) -> Path:
    times = list(range(0, 1046, 5))
    if extra_timepoints:
        times.extend(range(1050, 1071, 5))
    rows = [["User: USER"], ["Path: ignored"], ["Test ID: 1"], ["Test Name: growth"], ["Date: 01/01/2026"], ["Time: 00:00:00"], ["ID1: incorrect_metadata"], ["Absorbance"], [], []]
    header = ["Well\nRow", "Well\nCol", "Content", "Group"] + ["Raw Data (600)"] * len(times)
    time_row = [np.nan, np.nan, "Time", np.nan] + [time_label(t) for t in times]
    rows.extend([header, time_row])

    wells = [f"{row}{col}" for row in "ABCDEFGH" for col in range(1, 13)]
    blank_wells = {"G4", "G5", "G6"}
    if missing_blank:
        blank_wells.remove("G6")
    sample_wells = [well for well in wells if well not in {"G4", "G5", "G6"}]
    assignments = []
    for content, group in treatment_triplets():
        assignments.extend([(content, group)] * 3)
    assert len(assignments) == 93
    assignment_lookup = dict(zip(sample_wells, assignments))

    for well in wells:
        row = well[0]
        col = int(well[1:])
        if well in blank_wells:
            content, group = "Blank", "Blank"
            values = [0.10 + 0.00001 * t for t in times]
        elif well in {"G4", "G5", "G6"}:
            content, group = assignment_lookup[sample_wells[0]]
            values = [0.08 + 0.001 * (t / 60.0) for t in times]
        else:
            content, group = assignment_lookup[well]
            condition_index = sample_wells.index(well) // 3
            values = [0.08 + 0.002 * condition_index + 0.001 * (t / 60.0) for t in times]
        rows.append([row, col, content, group] + values)

    pd.DataFrame(rows).to_excel(path, sheet_name="All Cycles", header=False, index=False)
    return path


def test_run_id_uses_user_supplied_labels() -> None:
    script = load_script()

    assert script.build_run_id("E_coli", "BR1") == "E_coli_BR1"
    assert script.build_run_id("B_subtilis", "BR3") == "B_subtilis_BR3"


def test_treatment_parser_preserves_different_combination_regimens() -> None:
    script = load_script()

    br1 = script.standardize_drug_treatment("C1 + Ampicillin", "12.5uM+ 2ug/mL")
    br2 = script.standardize_drug_treatment("C1 + Ampicillin", "25uM+4ug/mL")

    assert br1["condition_id"] == "C1_12.5uM_ampicillin_2ug_mL"
    assert br2["condition_id"] == "C1_25uM_ampicillin_4ug_mL"
    assert br1["condition_id"] != br2["condition_id"]


def test_four_and_eight_ug_ampicillin_controls_are_accepted(tmp_path: Path) -> None:
    script = load_script()
    workbook = make_synthetic_workbook(tmp_path / "drug.xlsx")
    data, _ = script.import_drug_screening_workbook(workbook, "E_coli", "BR2")
    mask = (
        data["original_content"].eq("Positive Control (Ampicillin)")
        & data["original_group"].astype(str).str.contains("2")
    )
    data.loc[mask, "original_group"] = "4ug/mL"
    data.loc[mask, "ampicillin_concentration_ug_mL"] = 4.0
    data.loc[mask, "condition_id"] = "ampicillin_4ug_mL"
    data.loc[mask, "condition"] = "Ampicillin 4 ug/mL"
    data.loc[mask, "condition_label"] = "Ampicillin 4 ug/mL"

    script.validate_drug_screening_measurements(data)

    ampicillin_ids = set(data.loc[data["treatment_type"].eq("ampicillin_control"), "condition_id"])
    assert ampicillin_ids == {"ampicillin_4ug_mL", "ampicillin_8ug_mL"}


def test_excel_import_trims_to_17h25_and_uses_authoritative_cli_labels(tmp_path: Path) -> None:
    script = load_script()
    workbook = make_synthetic_workbook(tmp_path / "drug.xlsx", extra_timepoints=True)

    data, metadata = script.import_drug_screening_workbook(workbook, "E_coli", "BR1", max_time_min=1045)

    assert metadata["ID1"] == "incorrect_metadata"
    assert set(data.loc[data["treatment_type"].ne("blank"), "species"]) == {"E_coli"}
    assert set(data["biological_replicate"]) == {"BR1"}
    assert data["run_id"].unique().tolist() == ["E_coli_BR1"]
    assert data["time_min"].nunique() == 210
    assert data["time_min"].max() == 1045
    assert data["well"].nunique() == 96


def test_missing_blank_well_is_informative(tmp_path: Path) -> None:
    script = load_script()
    workbook = make_synthetic_workbook(tmp_path / "drug.xlsx", missing_blank=True)
    data, _ = script.import_drug_screening_workbook(workbook, "E_coli", "BR1")

    with pytest.raises(GrowthDataValidationError, match="Expected blank wells"):
        script.validate_drug_screening_measurements(data)


def test_blank_correction_subtracts_time_matched_g4_g5_g6_and_preserves_raw(tmp_path: Path) -> None:
    script = load_script()
    workbook = make_synthetic_workbook(tmp_path / "drug.xlsx")
    data, _ = script.import_drug_screening_workbook(workbook, "B_subtilis", "BR2")
    original_raw = data["od600"].copy(deep=True)

    result = script.apply_time_matched_blank_correction(data, script.BLANK_WELLS, time_column="time_min", raw_od_column="od600")
    corrected = result.corrected_data
    row = corrected[(corrected["well"].eq("A1")) & (corrected["time_min"].eq(10))].iloc[0]

    assert row["blank_OD600"] == pytest.approx(0.10 + 0.00001 * 10)
    assert row["corrected_OD600"] == pytest.approx(row["raw_OD600"] - row["blank_OD600"])
    pd.testing.assert_series_equal(data["od600"], original_raw)
    assert corrected["corrected_OD600"].min() < 0


def test_corrected_data_retains_93_nonblank_wells_and_19530_measurements(tmp_path: Path) -> None:
    script = load_script()
    workbook = make_synthetic_workbook(tmp_path / "drug.xlsx")
    data, _ = script.import_drug_screening_workbook(workbook, "E_coli", "BR1")
    result = script.apply_time_matched_blank_correction(data, script.BLANK_WELLS, time_column="time_min", raw_od_column="od600")

    script.validate_corrected_drug_screening_data(result.corrected_data)

    assert result.corrected_data["well"].nunique() == 93
    assert len(result.corrected_data) == 93 * 210
    assert not result.corrected_data["well"].isin(script.BLANK_WELLS).any()


def test_qc_flags_high_od_without_removing_measurements(tmp_path: Path) -> None:
    script = load_script()
    workbook = make_synthetic_workbook(tmp_path / "drug.xlsx")
    data, _ = script.import_drug_screening_workbook(workbook, "E_coli", "BR1")
    data.loc[data["well"].eq("A1") & data["time_min"].eq(5), "od600"] = 1.2
    result = script.apply_time_matched_blank_correction(data, script.BLANK_WELLS, time_column="time_min", raw_od_column="od600")
    metrics = script.calculate_technical_well_metrics(result.corrected_data, "E_coli_BR1", "BR1")

    flags = script.build_quality_control_flags(data, result.corrected_data, metrics)

    assert "raw_OD600_greater_or_equal_1" in set(flags["flag_type"])
    assert len(result.corrected_data) == 93 * 210


def test_metrics_are_per_well_before_summary_and_vehicle_relative_auc_is_100(
    tmp_path: Path,
    monkeypatch,
) -> None:
    script = load_script()
    workbook = make_synthetic_workbook(tmp_path / "drug.xlsx")
    data, _ = script.import_drug_screening_workbook(workbook, "E_coli", "BR1")
    result = script.apply_time_matched_blank_correction(data, script.BLANK_WELLS, time_column="time_min", raw_od_column="od600")

    def fake_fit(curve, time_column="time_h", response_column="corrected_OD600"):
        return {
            "A_OD600": 1.0,
            "Kz_OD600_per_h": 0.2,
            "TLag_h": 3.0,
            "gompertz_R2": 0.99,
            "gompertz_fit_status": "success",
        }

    monkeypatch.setattr(script, "fit_one_growth_curve", fake_fit)
    metrics = script.calculate_technical_well_metrics(result.corrected_data, "E_coli_BR1", "BR1")
    summary = script.summarise_drug_screening_biological_replicate(metrics)
    vehicle = metrics[metrics["condition_id"].eq("vehicle_1pct_DMSO")]

    assert len(metrics) == 93
    assert vehicle["relative_AUC_percent"].mean() == pytest.approx(100.0)
    assert len(summary) == 31
    assert summary["n_technical_wells"].eq(3).all()
    row = summary[summary["condition_id"].eq("C1_6.25uM")].iloc[0]
    group = metrics[metrics["condition_id"].eq("C1_6.25uM")]
    assert row["AUC_0_17h25min_OD_h_mean"] == pytest.approx(group["AUC_0_17h25min_OD_h"].mean())
    assert row["AUC_0_17h25min_OD_h_SD_technical"] == pytest.approx(group["AUC_0_17h25min_OD_h"].std(ddof=1))


def test_failed_fits_are_retained_and_documented(tmp_path: Path, monkeypatch) -> None:
    script = load_script()
    workbook = make_synthetic_workbook(tmp_path / "drug.xlsx")
    data, _ = script.import_drug_screening_workbook(workbook, "E_coli", "BR1")
    result = script.apply_time_matched_blank_correction(data, script.BLANK_WELLS, time_column="time_min", raw_od_column="od600")

    def fake_fit(curve, time_column="time_h", response_column="corrected_OD600"):
        if curve["well"].iloc[0] == "A1":
            return {
                "A_OD600": math.nan,
                "Kz_OD600_per_h": math.nan,
                "TLag_h": math.nan,
                "gompertz_R2": math.nan,
                "gompertz_fit_status": "fit_failed: synthetic failure",
            }
        return {
            "A_OD600": 1.0,
            "Kz_OD600_per_h": 0.2,
            "TLag_h": 3.0,
            "gompertz_R2": 0.99,
            "gompertz_fit_status": "success",
        }

    monkeypatch.setattr(script, "fit_one_growth_curve", fake_fit)
    metrics = script.calculate_technical_well_metrics(result.corrected_data, "E_coli_BR1", "BR1")
    summary = script.summarise_drug_screening_biological_replicate(metrics)

    failed = metrics[metrics["well"].eq("A1")].iloc[0]
    assert np.isnan(failed["A_OD600"])
    assert failed["gompertz_fit_warning"] == "synthetic failure"
    assert len(metrics) == 93
    assert summary["n_failed_gompertz_fits"].sum() == 1


def test_single_run_script_does_not_perform_inferential_statistics() -> None:
    script = load_script()
    source = inspect.getsource(script)

    assert "anova" not in source.lower()
    assert "tukey" not in source.lower()
