from pathlib import Path
import importlib.util
import inspect
import math
import sys

import numpy as np
import pandas as pd
from scipy import stats

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
ANALYSES_DIR = PROJECT_ROOT / "analyses"
for path in [SRC_DIR, ANALYSES_DIR]:
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))


def load_script():
    script_path = PROJECT_ROOT / "analyses" / "drug_screening_thesis_figures.py"
    spec = importlib.util.spec_from_file_location("drug_screening_thesis_figures", script_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def make_complete_br_metrics():
    script = load_script()
    conditions = list(dict.fromkeys(script.SCHIFF_AUC_CONDITIONS + script.PENAG_AUC_CONDITIONS))
    rows = []
    for species_index, species in enumerate(script.EXPECTED_SPECIES):
        for replicate_index, replicate in enumerate(script.EXPECTED_REPLICATES):
            for condition_index, condition_id in enumerate(conditions):
                compound = script.condition_compound(condition_id)
                treatment_type = "vehicle_control"
                if compound in script.SCHIFF_COMPOUNDS:
                    treatment_type = "compound_monotherapy"
                if compound == "PenAg":
                    treatment_type = "penag_monotherapy"
                rows.append(
                    {
                        "run_id": f"{species}_{replicate}",
                        "biological_replicate": replicate,
                        "species": species,
                        "treatment_type": treatment_type,
                        "compound": compound,
                        "compound_concentration_uM": script.condition_concentration(condition_id),
                        "ampicillin_concentration_ug_mL": math.nan,
                        "condition_id": condition_id,
                        "condition": condition_id,
                        "n_technical_wells": 3,
                        "raw_AUC_mean": 50 + species_index * 40 + replicate_index * 15 + condition_index * 4 + replicate_index * condition_index * 0.2,
                        "relative_AUC_mean": 100 - condition_index,
                        "Kz_mean": 0.2 + condition_index * 0.01,
                        "TLag_mean": 3.0 + condition_index * 0.2 + replicate_index * 0.1,
                        "n_valid_gompertz_fits": 3,
                        "n_failed_gompertz_fits": 0,
                        "eligible_for_AUC_inference": False,
                        "eligible_for_Kz_inference": False,
                        "eligible_for_TLag_inference": False,
                    }
                )
    return pd.DataFrame.from_records(rows)


def make_technical_rows():
    script = load_script()
    rows = []
    conditions = script.SCHIFF_AUC_CONDITIONS + script.PENAG_AUC_CONDITIONS + ["C1_25uM_ampicillin_4ug_mL"]
    for condition_id in conditions:
        compound = "C1" if "ampicillin" in condition_id else script.condition_compound(condition_id)
        treatment_type = "combination" if "ampicillin" in condition_id else "vehicle_control"
        if compound in script.SCHIFF_COMPOUNDS and treatment_type != "combination":
            treatment_type = "compound_monotherapy"
        if compound == "PenAg":
            treatment_type = "penag_monotherapy"
        rows.append(
            {
                "run_id": "E_coli_BR1",
                "biological_replicate": "BR1",
                "species": "E_coli",
                "well": condition_id,
                "technical_replicate": 1,
                "treatment_type": treatment_type,
                "compound": compound,
                "compound_concentration_uM": 25.0 if "ampicillin" in condition_id else script.condition_concentration(condition_id),
                "ampicillin_concentration_ug_mL": 4.0 if "ampicillin" in condition_id else math.nan,
                "condition_id": condition_id,
                "condition": condition_id,
                script.RAW_AUC: 10.0,
                script.REL_AUC: 90.0,
                script.TLAG: 4.0,
                "gompertz_R2": 0.99,
                "gompertz_fit_status": "success",
                "gompertz_fit_warning": "",
            }
        )
    return pd.DataFrame.from_records(rows)


def test_thesis_models_have_expected_dimensions_and_tukey_counts() -> None:
    script = load_script()
    anova, complete_tukey, vehicle = script.run_thesis_models(make_complete_br_metrics())

    assert len(anova) == 12
    assert sorted(anova["residual_df"].unique().tolist()) == [4, 8, 10]
    assert len(complete_tukey) == 146
    assert len(vehicle[vehicle["model_family"].eq("schiff_relative_auc")]) == 40
    assert len(vehicle[vehicle["model_family"].eq("penag_relative_auc")]) == 8
    assert len(vehicle[vehicle["model_family"].eq("penag_lag_time")]) == 4
    assert set(vehicle["vehicle_condition"]) == {script.VEHICLE}
    assert not vehicle["treatment_condition"].eq(script.VEHICLE).any()


def test_penag_lag_time_uses_only_vehicle_12_5_and_25_um() -> None:
    script = load_script()
    lag_specs = [spec for spec in script.thesis_families() if spec.figure_family == "penag_lag_time"]

    assert len(lag_specs) == 2
    for spec in lag_specs:
        assert list(spec.family.condition_ids) == script.PENAG_TLAG_CONDITIONS
        assert "PenAg_50uM" not in spec.family.condition_ids
        assert "PenAg_100uM" not in spec.family.condition_ids


def test_raw_auc_is_used_for_inference_and_relative_auc_for_plotting() -> None:
    script = load_script()
    spec = next(family for family in script.thesis_families() if family.figure_family == "schiff_relative_auc")
    data = make_complete_br_metrics()
    family_data = script.family_dataset(data, spec.family, spec.metric)
    row = family_data[family_data["condition_id"].eq("C1_6.25uM") & family_data["biological_replicate"].eq("BR2")].iloc[0]

    assert row["value"] == row["raw_AUC_mean"]
    assert row["value"] != row["relative_AUC_mean"]
    assert spec.plotted_metric == "relative_AUC_percent"


def test_filtered_p_values_match_complete_block_tukey() -> None:
    script = load_script()
    _, complete_tukey, vehicle = script.run_thesis_models(make_complete_br_metrics())
    selected = vehicle[vehicle["estimability_status"].eq("estimable")].iloc[0]
    pair = complete_tukey[
        complete_tukey["species"].eq(selected["species"])
        & complete_tukey["compound"].eq(selected["compound"])
        & complete_tukey["metric"].eq(selected["metric"])
        & (
            (
                complete_tukey["condition_1"].eq(selected["vehicle_condition"])
                & complete_tukey["condition_2"].eq(selected["treatment_condition"])
            )
            | (
                complete_tukey["condition_2"].eq(selected["vehicle_condition"])
                & complete_tukey["condition_1"].eq(selected["treatment_condition"])
            )
        )
    ].iloc[0]

    assert selected["Tukey_adjusted_p_value"] == pair["Tukey_adjusted_p_value"]
    expected_p = float(stats.studentized_range.sf(pair["q_statistic"], pair["number_of_groups"], pair["residual_df"]))
    assert pair["Tukey_adjusted_p_value"] == expected_p


def test_schiff_nonsignificant_comparisons_are_not_drawn_but_penag_are() -> None:
    script = load_script()
    _, _, vehicle = script.run_thesis_models(make_complete_br_metrics())

    schiff_ns = vehicle[vehicle["model_family"].eq("schiff_relative_auc") & vehicle["significance_label"].eq("n.s.")]
    assert not schiff_ns["drawn_on_figure"].any()
    penag_auc = vehicle[vehicle["model_family"].eq("penag_relative_auc")]
    penag_lag = vehicle[vehicle["model_family"].eq("penag_lag_time")]
    assert penag_auc["drawn_on_figure"].all()
    assert penag_lag["drawn_on_figure"].all()


def test_thesis_metric_points_exclude_combination_treatments() -> None:
    script = load_script()
    points = script.thesis_metric_technical_data(make_technical_rows())

    assert "combination" not in set(points["treatment_type"])
    assert "C1_25uM_ampicillin_4ug_mL" not in set(points["condition_id"])
    assert not points["figure_family"].str.contains("combination").any()


def test_growth_curve_summary_uses_biological_replicate_means_and_ddof_one() -> None:
    script = load_script()
    rows = []
    br_values = {"BR1": 1.0, "BR2": 2.0, "BR3": 4.0}
    for replicate, value in br_values.items():
        for tech in [1, 2, 3]:
            rows.append(
                {
                    "species": "E_coli",
                    "biological_replicate": replicate,
                    "condition_id": script.VEHICLE,
                    "well": f"{replicate}_{tech}",
                    "time_min": 0,
                    "time_h": 0.0,
                    "corrected_OD600": value,
                }
            )
    summary = script.summarize_growth_curves(pd.DataFrame.from_records(rows), [script.VEHICLE])
    row = summary.iloc[0]

    assert row["combined_mean_OD600"] == np.mean([1.0, 2.0, 4.0])
    assert row["combined_SD_OD600"] == np.std([1.0, 2.0, 4.0], ddof=1)
    assert row["n_biological_replicates"] == 3


def test_no_compact_letters_pairwise_tukeyhsd_or_dunnett_in_thesis_script() -> None:
    script = load_script()
    source = inspect.getsource(script)

    assert "compact_letter" not in source
    assert "pairwise_tukeyhsd" not in source
    assert "Dunnett" not in source
    assert "dunnett" not in source
