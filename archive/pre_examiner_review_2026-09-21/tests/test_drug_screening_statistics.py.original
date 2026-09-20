from pathlib import Path
import importlib.util
import inspect
import math
import sys

import numpy as np
import pandas as pd
import pytest
from scipy import stats

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))


def load_script():
    script_path = PROJECT_ROOT / "analyses" / "drug_screening_statistics.py"
    spec = importlib.util.spec_from_file_location("drug_screening_statistics", script_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def make_br_metrics():
    script = load_script()
    records = []
    block_offsets = {"BR1": 0.0, "BR2": 30.0, "BR3": 60.0}
    effects = {
        "vehicle_1pct_DMSO": 0.0,
        "C1_6.25uM": -3.0,
        "C1_12.5uM": -5.0,
        "C1_25uM": -8.0,
        "C1_50uM": -11.0,
        "C1_100uM": -15.0,
    }
    for replicate in script.EXPECTED_REPLICATES:
        for index, condition_id in enumerate(script.FAMILY_ORDER["schiff_C1"]):
            condition_effect = effects[condition_id]
            interaction_noise = {"BR1": -0.06, "BR2": 0.02, "BR3": 0.04}[replicate] * (index % 3)
            records.append(
                {
                    "run_id": f"E_coli_{replicate}",
                    "biological_replicate": replicate,
                    "species": "E_coli",
                    "treatment_type": "vehicle_control" if condition_id == "vehicle_1pct_DMSO" else "compound_monotherapy",
                    "compound": "DMSO" if condition_id == "vehicle_1pct_DMSO" else "C1",
                    "compound_concentration_uM": math.nan if condition_id == "vehicle_1pct_DMSO" else [6.25, 12.5, 25.0, 50.0, 100.0][index - 1],
                    "ampicillin_concentration_ug_mL": math.nan,
                    "condition_id": condition_id,
                    "condition": condition_id,
                    "n_technical_wells": 3,
                    "raw_AUC_mean": 100.0 + block_offsets[replicate] + condition_effect + index * 0.3 + interaction_noise,
                    "relative_AUC_mean": 100.0 + condition_effect,
                    "Kz_mean": 0.2 + index * 0.01 + block_offsets[replicate] * 0.0005,
                    "TLag_mean": 4.0 + index * 0.2 + block_offsets[replicate] * 0.005,
                    "n_valid_gompertz_fits": 3,
                    "n_failed_gompertz_fits": 0,
                    "eligible_for_AUC_inference": False,
                    "eligible_for_Kz_inference": False,
                    "eligible_for_TLag_inference": False,
                }
            )
    return pd.DataFrame.from_records(records)


def make_complete_br_metrics():
    script = load_script()
    condition_ids = []
    for family in script.build_family_specs():
        condition_ids.extend(family.condition_ids)
    condition_ids = list(dict.fromkeys(condition_ids))
    records = []
    for species_index, species in enumerate(script.EXPECTED_SPECIES):
        for replicate_index, replicate in enumerate(script.EXPECTED_REPLICATES):
            for condition_index, condition_id in enumerate(condition_ids):
                treatment_type = "vehicle_control" if condition_id == "vehicle_1pct_DMSO" else "compound_monotherapy"
                compound = "DMSO"
                concentration = math.nan
                if condition_id.startswith("PenAg"):
                    treatment_type = "penag_monotherapy"
                    compound = "PenAg"
                    concentration = float(condition_id.split("_")[1].replace("uM", ""))
                elif condition_id.startswith("C"):
                    compound, concentration_text = condition_id.split("_", 1)
                    concentration = float(concentration_text.replace("uM", ""))
                records.append(
                    {
                        "run_id": f"{species}_{replicate}",
                        "biological_replicate": replicate,
                        "species": species,
                        "treatment_type": treatment_type,
                        "compound": compound,
                        "compound_concentration_uM": concentration,
                        "ampicillin_concentration_ug_mL": math.nan,
                        "condition_id": condition_id,
                        "condition": condition_id,
                        "n_technical_wells": 3,
                        "raw_AUC_mean": 200 + species_index * 40 + replicate_index * 30 + condition_index * 5 + replicate_index * condition_index * 0.25,
                        "relative_AUC_mean": 100 - condition_index,
                        "Kz_mean": 0.2 + species_index * 0.02 + condition_index * 0.01 + replicate_index * 0.003,
                        "TLag_mean": 3.0 + species_index * 0.2 + condition_index * 0.1 + replicate_index * 0.02,
                        "n_valid_gompertz_fits": 3,
                        "n_failed_gompertz_fits": 0,
                        "eligible_for_AUC_inference": False,
                        "eligible_for_Kz_inference": False,
                        "eligible_for_TLag_inference": False,
                    }
                )
    return pd.DataFrame.from_records(records)


def empty_frame(columns):
    return pd.DataFrame(columns=list(columns))


def test_schiff_block_anova_dimensions_and_no_interaction() -> None:
    script = load_script()
    family = script.FamilySpec("E_coli", "C1", "schiff", script.FAMILY_ORDER["schiff_C1"])
    data = script.family_dataset(make_br_metrics(), family, "AUC_0_1045_min")
    anova = script.randomized_block_anova(data, family, "AUC_0_1045_min")

    assert anova.total_model_observations == 18
    assert anova.condition_df == 5
    assert anova.block_df == 2
    assert anova.residual_df == 10
    assert anova.model_status == "fitted"


def test_auc_family_dataset_uses_raw_auc_not_relative_auc() -> None:
    script = load_script()
    family = script.FamilySpec("E_coli", "C1", "schiff", script.FAMILY_ORDER["schiff_C1"])
    br_metrics = make_br_metrics()
    data = script.family_dataset(br_metrics, family, "AUC_0_1045_min")
    first = data[data["condition_id"].eq("C1_6.25uM") & data["biological_replicate"].eq("BR2")].iloc[0]

    assert first["value"] == pytest.approx(first["raw_AUC_mean"])
    assert first["value"] != pytest.approx(first["relative_AUC_mean"])


def test_complete_models_produce_expected_total_tukey_comparisons() -> None:
    script = load_script()
    anova, tukey, letters, diagnostics, exclusions = script.run_inferential_models(make_complete_br_metrics())

    assert exclusions == []
    assert len(anova) == 30
    assert len(tukey[tukey["metric"].eq("AUC_0_1045_min")]) == 140
    assert len(tukey[tukey["metric"].eq("Kz")]) == 140
    assert len(tukey[tukey["metric"].eq("TLag")]) == 140
    assert len(tukey) == 420
    assert sorted(anova["residual_df"].unique().tolist()) == [8, 10]
    assert not letters.empty
    assert not diagnostics.empty


def test_penag_block_anova_dimensions() -> None:
    script = load_script()
    records = []
    for replicate in script.EXPECTED_REPLICATES:
        for index, condition_id in enumerate(script.FAMILY_ORDER["penag_PenAg"]):
            records.append(
                {
                    "biological_replicate": replicate,
                    "species": "E_coli",
                    "condition_id": condition_id,
                    "treatment_type": "vehicle_control" if index == 0 else "penag_monotherapy",
                    "compound": "DMSO" if index == 0 else "PenAg",
                    "compound_concentration_uM": math.nan if index == 0 else script.PENAG_CONCENTRATIONS[index - 1],
                    "raw_AUC_mean": 50 + index + {"BR1": 0, "BR2": 5, "BR3": 9}[replicate],
                    "relative_AUC_mean": 100 - index,
                    "Kz_mean": 0.1 + index,
                    "TLag_mean": 3 + index,
                    "n_valid_gompertz_fits": 3,
                }
            )
    br_metrics = pd.DataFrame.from_records(records)
    family = script.FamilySpec("E_coli", "PenAg", "penag", script.FAMILY_ORDER["penag_PenAg"])
    data = script.family_dataset(br_metrics, family, "AUC_0_1045_min")
    anova = script.randomized_block_anova(data, family, "AUC_0_1045_min")

    assert anova.total_model_observations == 15
    assert anova.condition_df == 4
    assert anova.block_df == 2
    assert anova.residual_df == 8


def test_validation_requires_each_biological_replicate_once_per_species() -> None:
    script = load_script()
    growth = empty_frame(script.GROWTH_REQUIRED)
    technical = empty_frame(script.TECHNICAL_REQUIRED)
    summary = empty_frame(script.SUMMARY_REQUIRED)
    runs = empty_frame(script.RUN_REQUIRED)
    runs.loc[0, ["species", "biological_replicate", "analysis_endpoint_min", "number_of_retained_time_points"]] = [
        "E_coli",
        "BR1",
        script.MAX_TIME_MIN,
        len(script.TIMEPOINTS),
    ]

    with pytest.raises(script.DrugScreeningStatisticsError, match="must contain BR1, BR2 and BR3 exactly once"):
        script.validate_loaded_data(growth, technical, summary, runs)


def test_tukey_uses_block_residual_mse_and_studentized_range() -> None:
    script = load_script()
    family = script.FamilySpec("E_coli", "C1", "schiff", script.FAMILY_ORDER["schiff_C1"])
    data = script.family_dataset(make_br_metrics(), family, "AUC_0_1045_min")
    anova = script.randomized_block_anova(data, family, "AUC_0_1045_min")
    tukey = script.tukey_from_block(data, anova, family)
    row = tukey.iloc[0]

    expected_se = math.sqrt(anova.residual_MS / 3)
    expected_q = abs(row["mean_difference_condition_1_minus_condition_2"]) / expected_se
    expected_p = float(stats.studentized_range.sf(expected_q, 6, 10))

    assert len(tukey) == 15
    assert row["Tukey_standard_error"] == pytest.approx(expected_se)
    assert row["q_statistic"] == pytest.approx(expected_q)
    assert row["Tukey_adjusted_p_value"] == pytest.approx(expected_p)


def test_od600_linearity_qc_flags_raw_threshold_and_preserves_rows() -> None:
    script = load_script()
    growth = pd.DataFrame(
        {
            "run_id": ["run1", "run1", "run1"],
            "species": ["E_coli", "E_coli", "E_coli"],
            "biological_replicate": ["BR1", "BR1", "BR1"],
            "condition_id": ["vehicle_1pct_DMSO", "vehicle_1pct_DMSO", "vehicle_1pct_DMSO"],
            "condition": ["vehicle", "vehicle", "vehicle"],
            "well": ["A1", "A1", "A1"],
            "time_min": [0, 5, 10],
            "raw_OD600": [0.1, 1.0, 1.2],
            "corrected_OD600": [-0.02, 0.9, 1.1],
        }
    )

    qc = script.od600_linearity_qc(growth)

    assert len(growth) == 3
    assert len(qc) == 1
    assert bool(qc["raw_OD600_threshold_reached"].iloc[0])
    assert qc["n_raw_measurements_ge_1"].iloc[0] == 2
    assert qc["first_time_min_raw_OD600_ge_1"].iloc[0] == 5
    assert qc["action"].iloc[0] == "flagged_preserved_not_excluded"


def test_compact_letters_reproduce_pairwise_matrix() -> None:
    script = load_script()
    family = script.FamilySpec("E_coli", "C1", "schiff", script.FAMILY_ORDER["schiff_C1"])
    data = script.family_dataset(make_br_metrics(), family, "AUC_0_1045_min")
    anova = script.randomized_block_anova(data, family, "AUC_0_1045_min")
    tukey = script.tukey_from_block(data, anova, family)
    letters = script.compact_letter_display(tukey, family.condition_ids)

    script.validate_compact_letters(tukey, letters)
    assert set(letters["condition_id"]) == set(family.condition_ids)


def make_combination_technical():
    script = load_script()
    records = []
    for species in script.EXPECTED_SPECIES:
        for replicate in script.EXPECTED_REPLICATES:
            compound_conc = 12.5 if replicate == "BR1" else 25.0
            amp_conc = 2.0 if replicate == "BR1" else 4.0
            shared_conditions = [
                ("vehicle_control", "DMSO", math.nan, math.nan, "vehicle_1pct_DMSO", 100),
                ("ampicillin_control", "Ampicillin", math.nan, amp_conc, f"ampicillin_{amp_conc:g}ug_mL", 35),
            ]
            for treatment_type, comp, comp_conc, amp, condition_id, base_auc in shared_conditions:
                for technical_replicate in [1, 2, 3]:
                    records.append(
                        {
                            "run_id": f"{species}_{replicate}",
                            "biological_replicate": replicate,
                            "species": species,
                            "well": f"{condition_id}_{technical_replicate}",
                            "technical_replicate": technical_replicate,
                            "treatment_type": treatment_type,
                            "compound": comp,
                            "compound_concentration_uM": comp_conc,
                            "ampicillin_concentration_ug_mL": amp,
                            "condition_id": condition_id,
                            "condition": condition_id,
                            script.RAW_AUC: base_auc + technical_replicate,
                            script.REL_AUC: base_auc + technical_replicate,
                            script.KZ: 0.1 + technical_replicate * 0.01,
                            script.TLAG: 4.0 + technical_replicate * 0.1,
                            "gompertz_R2": 0.99,
                            "gompertz_fit_status": "success",
                            "gompertz_fit_warning": "",
                        }
                    )
            for compound in script.SCHIFF_COMPOUNDS:
                conditions = [
                    ("compound", "compound_monotherapy", compound, compound_conc, math.nan, f"{compound}_{compound_conc:g}uM", 92),
                    ("combo", "combination", compound, compound_conc, amp_conc, f"{compound}_{compound_conc:g}uM_ampicillin_{amp_conc:g}ug_mL", 55),
                ]
                for _, treatment_type, comp, comp_conc, amp, condition_id, base_auc in conditions:
                    for technical_replicate in [1, 2, 3]:
                        records.append(
                            {
                                "run_id": f"{species}_{replicate}",
                                "biological_replicate": replicate,
                                "species": species,
                                "well": f"{compound}_{condition_id}_{technical_replicate}",
                                "technical_replicate": technical_replicate,
                                "treatment_type": treatment_type,
                                "compound": comp,
                                "compound_concentration_uM": comp_conc,
                                "ampicillin_concentration_ug_mL": amp,
                                "condition_id": condition_id,
                                "condition": condition_id,
                                script.RAW_AUC: base_auc + technical_replicate,
                                script.REL_AUC: base_auc + technical_replicate,
                                script.KZ: 0.1 + technical_replicate * 0.01,
                                script.TLAG: 4.0 + technical_replicate * 0.1,
                                "gompertz_R2": 0.99,
                                "gompertz_fit_status": "success",
                                "gompertz_fit_warning": "",
                            }
                        )
    return pd.DataFrame.from_records(records).drop_duplicates(["species", "biological_replicate", "compound", "condition_id", "well"])


def test_combination_regimens_remain_separate_and_delta_uses_condition_means() -> None:
    script = load_script()
    descriptive, delta, exclusions = script.combination_condition_sets(make_combination_technical())

    assert exclusions.empty
    assert len(delta) == 24
    br1 = delta[(delta["species"].eq("E_coli")) & (delta["biological_replicate"].eq("BR1")) & (delta["compound"].eq("C1"))].iloc[0]
    br2 = delta[(delta["species"].eq("E_coli")) & (delta["biological_replicate"].eq("BR2")) & (delta["compound"].eq("C1"))].iloc[0]
    assert br1["compound_concentration_uM"] == 12.5
    assert br1["ampicillin_concentration_ug_mL"] == 2.0
    assert br2["compound_concentration_uM"] == 25.0
    assert br2["ampicillin_concentration_ug_mL"] == 4.0
    assert br1["delta_relative_AUC_vs_ampicillin"] == pytest.approx(20.0)
    assert br1["interpretation"] == "possible attenuation of ampicillin activity"
    assert set(descriptive["comparison_condition"]) == {"vehicle", "compound_alone", "ampicillin_alone", "combination"}


def test_combination_growth_fit_metrics_use_successful_fits_only() -> None:
    script = load_script()
    technical = make_combination_technical()
    mask = (
        technical["species"].eq("E_coli")
        & technical["biological_replicate"].eq("BR1")
        & technical["compound"].eq("C1")
        & technical["treatment_type"].eq("combination")
        & technical["technical_replicate"].eq(3)
    )
    technical.loc[mask, "gompertz_fit_status"] = "failed: synthetic test failure"
    technical.loc[mask, script.KZ] = 99.0
    technical.loc[mask, script.TLAG] = 99.0

    descriptive, delta, exclusions = script.combination_condition_sets(technical)

    assert exclusions.empty
    combo_kz = descriptive[
        descriptive["species"].eq("E_coli")
        & descriptive["biological_replicate"].eq("BR1")
        & descriptive["compound"].eq("C1")
        & descriptive["comparison_condition"].eq("combination")
        & descriptive["metric"].eq("Kz")
    ].iloc[0]
    combo_tlag = descriptive[
        descriptive["species"].eq("E_coli")
        & descriptive["biological_replicate"].eq("BR1")
        & descriptive["compound"].eq("C1")
        & descriptive["comparison_condition"].eq("combination")
        & descriptive["metric"].eq("TLag")
    ].iloc[0]
    delta_row = delta[
        delta["species"].eq("E_coli")
        & delta["biological_replicate"].eq("BR1")
        & delta["compound"].eq("C1")
    ].iloc[0]

    assert combo_kz["n_technical_wells"] == 2
    assert combo_kz["technical_mean"] == pytest.approx((0.11 + 0.12) / 2)
    assert combo_tlag["n_technical_wells"] == 2
    assert combo_tlag["technical_mean"] == pytest.approx((4.1 + 4.2) / 2)
    assert delta_row["delta_Kz_vs_ampicillin"] == pytest.approx(((0.11 + 0.12) / 2) - 0.12)


def test_independent_groups_tukey_hsd_is_not_used() -> None:
    script = load_script()
    source = inspect.getsource(script)

    assert "pairwise_tukeyhsd" not in source
