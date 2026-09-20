from pathlib import Path
import hashlib
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
    script_path = PROJECT_ROOT / "analyses" / "combine_drug_screening_growth_curves.py"
    spec = importlib.util.spec_from_file_location("combine_drug_screening_growth_curves", script_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def matched_conditions():
    conditions = [
        ("vehicle_control", "DMSO", math.nan, math.nan, "vehicle_1pct_DMSO", "Vehicle Control (1% DMSO)"),
        ("ampicillin_control", "Ampicillin", math.nan, 2.0, "ampicillin_2ug_mL", "Ampicillin 2 ug/mL"),
        ("ampicillin_control", "Ampicillin", math.nan, 8.0, "ampicillin_8ug_mL", "Ampicillin 8 ug/mL"),
    ]
    for compound in ["C1", "C2", "C3", "C4"]:
        for concentration in [6.25, 12.5, 25.0, 50.0, 100.0]:
            conditions.append(
                (
                    "compound_monotherapy",
                    compound,
                    concentration,
                    math.nan,
                    f"{compound}_{concentration:g}uM",
                    f"{compound} {concentration:g} uM",
                )
            )
    for concentration in [12.5, 25.0, 50.0, 100.0]:
        conditions.append(
            (
                "penag_monotherapy",
                "PenAg",
                concentration,
                math.nan,
                f"PenAg_{concentration:g}uM",
                f"PenAg {concentration:g} uM",
            )
        )
    assert len(conditions) == 27
    return conditions


def combination_conditions(replicate: str):
    if replicate == "BR1":
        compound_concentration = 12.5
        amp = 2.0
    else:
        compound_concentration = 25.0
        amp = 4.0
    return [
        (
            "combination",
            compound,
            compound_concentration,
            amp,
            f"{compound}_{compound_concentration:g}uM_ampicillin_{amp:g}ug_mL",
            f"{compound} {compound_concentration:g} uM + ampicillin {amp:g} ug/mL",
        )
        for compound in ["C1", "C2", "C3", "C4"]
    ]


def make_combined_growth_data(species: str = "E_coli") -> pd.DataFrame:
    records = []
    times = list(range(0, 1046, 5))
    replicate_offsets = {"BR1": 0.0, "BR2": 0.3, "BR3": 0.9}
    well_counter = 1
    for replicate in ["BR1", "BR2", "BR3"]:
        conditions = matched_conditions() + combination_conditions(replicate)
        for condition_index, condition in enumerate(conditions):
            treatment_type, compound, compound_conc, amp, condition_id, label = condition
            for technical_replicate in [1, 2, 3]:
                well = f"W{well_counter}"
                well_counter += 1
                for time_min in times:
                    value = (
                        replicate_offsets[replicate]
                        + condition_index * 0.01
                        + technical_replicate * 0.001
                        + time_min * 0.0001
                    )
                    if replicate == "BR1" and condition_id == "C1_6.25uM" and technical_replicate == 1 and time_min == 0:
                        value = -0.2
                    if replicate == "BR3" and condition_id == "C1_6.25uM" and technical_replicate == 3 and time_min == 1045:
                        value = 1.2
                    records.append(
                        {
                            "species": species,
                            "biological_replicate": replicate,
                            "well": well,
                            "technical_replicate": technical_replicate,
                            "treatment_type": treatment_type,
                            "compound": compound,
                            "compound_concentration_uM": compound_conc,
                            "ampicillin_concentration_ug_mL": amp,
                            "condition_id": condition_id,
                            "condition": label,
                            "time_min": time_min,
                            "time_h": time_min / 60.0,
                            "corrected_OD600": value,
                        }
                    )
    return pd.DataFrame.from_records(records)


def test_br1_br2_br3_are_required_exactly_once() -> None:
    script = load_script()
    data = make_combined_growth_data()
    data = data[data["biological_replicate"].ne("BR3")].copy()

    with pytest.raises(GrowthDataValidationError, match="BR1, BR2 and BR3"):
        script.validate_combined_inputs(data, "E_coli")


def test_incorrect_species_assignment_is_rejected() -> None:
    script = load_script()
    data = make_combined_growth_data("B_subtilis")

    with pytest.raises(GrowthDataValidationError, match="requested species E_coli"):
        script.validate_combined_inputs(data, "E_coli")


def test_inputs_must_have_identical_210_time_points() -> None:
    script = load_script()
    data = make_combined_growth_data()
    data = data[~((data["biological_replicate"].eq("BR2")) & (data["well"].eq("W94")) & (data["time_min"].eq(1045)))]

    with pytest.raises(GrowthDataValidationError, match="does not contain one measurement at every time point"):
        script.validate_combined_inputs(data, "E_coli")


def test_hierarchical_averaging_uses_br_means_and_ddof_1() -> None:
    script = load_script()
    data = make_combined_growth_data()
    matching = script.build_condition_matching_summary(data)
    matched_ids = script.matched_condition_ids(matching)
    br_means = script.calculate_biological_replicate_mean_curves(data, matched_ids)
    combined = script.calculate_combined_mean_sd_curves(br_means)
    row = combined[(combined["condition_id"].eq("C1_6.25uM")) & (combined["time_min"].eq(10))].iloc[0]
    source = br_means[(br_means["condition_id"].eq("C1_6.25uM")) & (br_means["time_min"].eq(10))]

    expected_br_values = source.set_index("biological_replicate")["biological_replicate_mean_corrected_OD600"]
    assert row["combined_mean_corrected_OD600"] == pytest.approx(expected_br_values.mean())
    assert row["combined_SD_corrected_OD600"] == pytest.approx(expected_br_values.std(ddof=1))


def test_nine_technical_wells_are_not_pooled_and_within_run_sds_are_not_averaged() -> None:
    script = load_script()
    data = make_combined_growth_data()
    matching = script.build_condition_matching_summary(data)
    br_means = script.calculate_biological_replicate_mean_curves(data, script.matched_condition_ids(matching))
    combined = script.calculate_combined_mean_sd_curves(br_means)
    row = combined[(combined["condition_id"].eq("C1_6.25uM")) & (combined["time_min"].eq(10))].iloc[0]
    source_tech = data[(data["condition_id"].eq("C1_6.25uM")) & (data["time_min"].eq(10))]
    source_br = br_means[(br_means["condition_id"].eq("C1_6.25uM")) & (br_means["time_min"].eq(10))]

    assert row["combined_SD_corrected_OD600"] != pytest.approx(source_tech["corrected_OD600"].std(ddof=1))
    assert row["combined_SD_corrected_OD600"] != pytest.approx(source_br["technical_well_SD_corrected_OD600"].mean())


def test_negative_and_high_od_values_are_preserved() -> None:
    script = load_script()
    data = make_combined_growth_data()
    matching = script.build_condition_matching_summary(data)
    br_means = script.calculate_biological_replicate_mean_curves(data, script.matched_condition_ids(matching))

    assert data["corrected_OD600"].min() < 0
    assert data["corrected_OD600"].max() >= 1.0
    assert br_means["biological_replicate_mean_corrected_OD600"].min() < 0


def test_matched_and_combination_table_sizes_and_exclusions() -> None:
    script = load_script()
    data = make_combined_growth_data()
    matching = script.build_condition_matching_summary(data)
    matched_ids = script.matched_condition_ids(matching)
    br_means = script.calculate_biological_replicate_mean_curves(data, matched_ids)
    combined = script.calculate_combined_mean_sd_curves(br_means)
    combination = script.calculate_combination_run_level_curves(data)

    assert len(matched_ids) == 27
    assert len(combined) == 27 * 210
    assert not combined["treatment_type"].eq("combination").any()
    assert len(combination) == (4 + 2) * 3 * 210
    assert set(combination["compound"]) == {"Ampicillin", "C1", "C2", "C3", "C4"}
    assert set(combination["treatment_type"]) == {"ampicillin_control", "combination"}
    assert set(combination["biological_replicate"]) == {"BR1", "BR2", "BR3"}


def test_combination_regimens_remain_distinguished() -> None:
    script = load_script()
    data = make_combined_growth_data()
    combination = script.calculate_combination_run_level_curves(data)
    c1 = combination[combination["compound"].eq("C1")].drop_duplicates(
        ["biological_replicate", "compound_concentration_uM", "ampicillin_concentration_ug_mL", "condition_id"]
    )

    assert c1.loc[c1["biological_replicate"].eq("BR1"), "condition_id"].iloc[0] == "C1_12.5uM_ampicillin_2ug_mL"
    assert set(c1.loc[c1["biological_replicate"].isin(["BR2", "BR3"]), "condition_id"]) == {
        "C1_25uM_ampicillin_4ug_mL"
    }


def test_changed_ampicillin_control_concentrations_become_unmatched_not_fatal() -> None:
    script = load_script()
    data = make_combined_growth_data()
    mask = data["biological_replicate"].isin(["BR2", "BR3"]) & data["condition_id"].eq("ampicillin_2ug_mL")
    data.loc[mask, "ampicillin_concentration_ug_mL"] = 4.0
    data.loc[mask, "condition_id"] = "ampicillin_4ug_mL"
    data.loc[mask, "condition"] = "Ampicillin 4 ug/mL"

    matching = script.build_condition_matching_summary(data)
    matched_ids = script.matched_condition_ids(matching)
    br_means = script.calculate_biological_replicate_mean_curves(data, matched_ids)
    combined = script.calculate_combined_mean_sd_curves(br_means)

    assert len(matched_ids) == 26
    assert len(combined) == 26 * 210
    assert "ampicillin_2ug_mL" not in matched_ids
    assert "ampicillin_4ug_mL" not in matched_ids
    unmatched = matching[~matching["eligible_for_combined_mean_SD"]]
    assert {"ampicillin_2ug_mL", "ampicillin_4ug_mL"}.issubset(set(unmatched["condition_id"]))


def test_positive_controls_are_in_combination_run_level_curves() -> None:
    script = load_script()
    data = make_combined_growth_data()
    combination = script.calculate_combination_run_level_curves(data)

    controls = combination[combination["treatment_type"].eq("ampicillin_control")]

    assert not controls.empty
    assert set(controls["condition_id"]) == {"ampicillin_2ug_mL", "ampicillin_8ug_mL"}
    assert len(controls) == 2 * 3 * 210


def test_input_csv_files_remain_unchanged(tmp_path: Path) -> None:
    script = load_script()
    data = make_combined_growth_data()
    paths = {}
    for replicate in ["BR1", "BR2", "BR3"]:
        path = tmp_path / f"{replicate}.csv"
        data[data["biological_replicate"].eq(replicate)].to_csv(path, index=False)
        paths[replicate] = path
    before = {replicate: hashlib.sha256(path.read_bytes()).hexdigest() for replicate, path in paths.items()}

    loaded = script.read_blank_corrected_inputs(paths, requested_species="E_coli")

    after = {replicate: hashlib.sha256(path.read_bytes()).hexdigest() for replicate, path in paths.items()}
    assert len(loaded) == len(data)
    assert after == before


def test_no_inferential_tests_are_implemented() -> None:
    script = load_script()
    source = inspect.getsource(script).lower()

    assert "anova" not in source
    assert "tukey" not in source
    assert "ols(" not in source
    assert "pairwise" not in source
