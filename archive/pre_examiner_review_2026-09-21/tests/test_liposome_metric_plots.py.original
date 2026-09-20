import math
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import pytest
from scipy import stats


PROJECT_ROOT = Path(__file__).resolve().parents[1]
ANALYSES_DIR = PROJECT_ROOT / "analyses"
if str(ANALYSES_DIR) not in sys.path:
    sys.path.insert(0, str(ANALYSES_DIR))

import liposome_metric_plots as lmp  # noqa: E402


def make_block_means() -> pd.DataFrame:
    family = next(f for f in lmp.family_specs() if f.key == "empty_liposomes")
    treatment_effects = {"PBS": 0.0, "Empty 25 uM lipid": 1.0, "Empty 50 uM lipid": 2.0}
    block_effects = {"BR1": -0.5, "BR2": 0.0, "BR3": 0.25, "BR4": 0.25}
    residuals = {
        "BR1": [0.02, -0.03, 0.01],
        "BR2": [-0.01, 0.04, -0.02],
        "BR3": [0.03, -0.02, 0.00],
        "BR4": [-0.04, 0.01, 0.03],
    }
    rows = []
    for rep, block in block_effects.items():
        for order, group in enumerate(family.groups):
            treatment = group["label"]
            rows.append(
                {
                    "species": "B. Subtilis",
                    "biological_replicate": rep,
                    "treatment": treatment,
                    "treatment_order": order,
                    "technical_mean": 10.0 + treatment_effects[treatment] + block + 0.05 * order + residuals[rep][order],
                    "technical_SD": 0.2,
                    "n_technical_wells": 3,
                }
            )
    return pd.DataFrame(rows)


def test_replicate_structure_validation_detects_complete_block() -> None:
    family = next(f for f in lmp.family_specs() if f.key == "empty_liposomes")
    ok, reason = lmp.validate_replicate_structure(make_block_means(), family, "auc_0_19h")
    assert ok is True
    assert reason == "complete"


def test_raw_plate_rows_are_decoded_to_treatment_and_species() -> None:
    content, species = lmp.decode_raw_plate_row("O", 18, "Sample X2", "K")
    assert content == "Loaded Liposomes (50uM) + free/associated PenAg (6.25uM)"
    assert species == "B. Subtilis"
    content, species = lmp.decode_raw_plate_row("K", 18, "Sample X2", "H")
    assert content == "Co-delivery PenAg (3.125uM) + Empty Liposomes (50uM)"
    assert species == "B. Subtilis"
    content, species = lmp.decode_raw_plate_row("K", 3, "Blank B", "H")
    assert content == "Co-delivery PenAg (3.125uM) + Empty Liposomes (50uM)"
    assert species == "Blank"


def test_stacked_cycle_workbook_import_reconstructs_plate_wells(tmp_path: Path) -> None:
    excel = pd.DataFrame(index=range(23), columns=range(24), dtype=object)
    excel.iloc[0, 0] = "Cycle 1 (0 h)"
    excel.iloc[1, 1] = "Raw Data (600)"
    excel.iloc[2, 1:24] = list(range(1, 24))
    excel.iloc[3, 0] = "A"
    excel.iloc[3, 3] = 0.10
    excel.iloc[3, 9] = 0.20
    excel.iloc[4, 0] = "D"
    excel.iloc[4, 3] = 0.30
    excel.iloc[4, 9] = 0.40
    excel.iloc[5, 0] = "P"
    excel.iloc[5, 18] = 0.50
    excel.iloc[20, 0] = "Cycle 2 (0 h 6 min)"
    excel.iloc[21, 1] = "Raw Data (600)"
    excel.iloc[22, 1:24] = list(range(1, 24))
    workbook = tmp_path / "stacked.xlsx"
    excel.to_excel(workbook, sheet_name="Cycle export", index=False, header=False)

    imported = lmp.import_workbook(workbook, "BRX")

    assert set(imported["time_h"]) == {0.0}
    assert {"A3", "A9", "D3", "D9", "P18"}.issubset(set(imported["well"]))
    assert imported.loc[imported["well"] == "A3", "species"].iloc[0] == "Blank"
    assert imported.loc[imported["well"] == "A9", "species"].iloc[0] == "E. Coli"
    assert imported.loc[imported["well"] == "D9", "condition_type"].iloc[0] == "Free PenAg"
    assert imported.loc[imported["well"] == "D9", "drug_uM"].iloc[0] == pytest.approx(6.25)
    assert imported.loc[imported["well"] == "P18", "condition_type"].iloc[0] == "Loaded liposomes"
    assert imported.loc[imported["well"] == "P18", "drug_uM"].iloc[0] == pytest.approx(3.125)
    assert imported.loc[imported["well"] == "P18", "lipid_uM"].iloc[0] == pytest.approx(25.0)


def test_replicate_structure_validation_requires_three_blocks() -> None:
    family = next(f for f in lmp.family_specs() if f.key == "empty_liposomes")
    incomplete = make_block_means().loc[lambda data: data["biological_replicate"].isin(["BR1", "BR2"])]
    ok, reason = lmp.validate_replicate_structure(incomplete, family, "auc_0_19h")
    assert ok is False
    assert "Fewer than 3 biological replicates" in reason


def test_replicate_structure_validation_allows_two_blocks_when_requested() -> None:
    family = next(f for f in lmp.family_specs() if f.key == "empty_liposomes")
    two_blocks = make_block_means().loc[lambda data: data["biological_replicate"].isin(["BR1", "BR2"])]
    ok, reason = lmp.validate_replicate_structure(two_blocks, family, "auc_0_19h", minimum_blocks=2)
    result = lmp.randomized_block_anova(two_blocks, family, "auc_0_19h", "B. Subtilis", minimum_blocks=2)
    assert ok is True
    assert reason == "complete"
    assert result.number_of_biological_replicates == 2
    assert result.residual_df == 2


def test_empty_inference_tables_keep_headers_when_no_models_are_estimable() -> None:
    metrics = pd.DataFrame(
        [
            {
                "species": "B. Subtilis",
                "biological_replicate": "BR1",
                "condition_type": "PBS control",
                "drug_uM": 0.0,
                "lipid_uM": 0.0,
                "well": "A1",
                "auc_0_19h": 1.0,
            }
        ]
    )
    gompertz = pd.DataFrame(
        columns=[
            "fit_status",
            "species",
            "biological_replicate",
            "condition_type",
            "drug_uM",
            "lipid_uM",
            "well",
            "lag_h",
            "mu_max",
        ]
    )
    _, _, anova, tukey, warnings = lmp.build_analysis_tables(metrics, gompertz)
    assert list(anova.columns) == lmp.ANOVA_COLUMNS
    assert list(tukey.columns) == lmp.TUKEY_COLUMNS
    assert anova.empty
    assert tukey.empty
    assert warnings


def test_randomized_block_anova_residual_degrees_of_freedom() -> None:
    family = next(f for f in lmp.family_specs() if f.key == "empty_liposomes")
    result = lmp.randomized_block_anova(make_block_means(), family, "auc_0_19h", "B. Subtilis")
    assert result.treatment_df == 2
    assert result.block_df == 3
    assert result.residual_df == 6
    assert result.total_model_observations == 12


def test_tukey_uses_block_residual_mse_and_studentized_range() -> None:
    family = next(f for f in lmp.family_specs() if f.key == "empty_liposomes")
    data = make_block_means()
    anova = lmp.randomized_block_anova(data, family, "auc_0_19h", "B. Subtilis")
    tukey = lmp.tukey_from_block(data, anova, family)
    row = tukey.loc[
        (tukey["group_1"] == "PBS") & (tukey["group_2"] == "Empty 25 uM lipid")
    ].iloc[0]
    means = data.groupby("treatment")["technical_mean"].mean()
    expected_se = math.sqrt(anova.residual_MS / 4)
    expected_q = abs(means["PBS"] - means["Empty 25 uM lipid"]) / expected_se
    expected_p = float(stats.studentized_range.sf(expected_q, 3, 6))
    assert row["Tukey_standard_error"] == pytest.approx(expected_se)
    assert row["Tukey_q"] == pytest.approx(expected_q)
    assert row["Tukey_adjusted_p_value"] == pytest.approx(expected_p)


def test_hierarchical_mean_and_sd_are_from_correct_levels() -> None:
    per_well = pd.DataFrame(
        [
            {"species": "B. Subtilis", "biological_replicate": "BR1", "treatment": "PBS", "treatment_order": 0, "auc_0_19h": 9.0},
            {"species": "B. Subtilis", "biological_replicate": "BR1", "treatment": "PBS", "treatment_order": 0, "auc_0_19h": 11.0},
            {"species": "B. Subtilis", "biological_replicate": "BR2", "treatment": "PBS", "treatment_order": 0, "auc_0_19h": 13.0},
            {"species": "B. Subtilis", "biological_replicate": "BR2", "treatment": "PBS", "treatment_order": 0, "auc_0_19h": 15.0},
        ]
    )
    br_means = lmp.biological_replicate_means(per_well, "auc_0_19h")
    assert br_means.loc[br_means["biological_replicate"] == "BR1", "technical_mean"].iloc[0] == pytest.approx(10.0)
    assert br_means.loc[br_means["biological_replicate"] == "BR1", "technical_SD"].iloc[0] == pytest.approx(np.std([9.0, 11.0], ddof=1))
    summary = lmp.make_summary_table(br_means.assign(family_key="empty_liposomes", family_label="Empty liposome control", metric="auc_0_19h"))
    assert summary["biological_mean"].iloc[0] == pytest.approx(12.0)
    assert summary["biological_SD"].iloc[0] == pytest.approx(np.std([10.0, 14.0], ddof=1))


def make_blank_correction_input() -> pd.DataFrame:
    rows = []

    def add(content: str, species: str, condition_type: str, well: str, values: list[float]) -> None:
        for time_h, od600_raw in enumerate(values):
            rows.append(
                {
                    "species": species,
                    "content": content,
                    "condition_label": content,
                    "condition_type": condition_type,
                    "drug_uM": 0.0,
                    "lipid_uM": 0.0,
                    "well": well,
                    "time_h": float(time_h),
                    "od600_raw": od600_raw,
                }
            )

    add("PBS only", "Blank", "PBS control", "A3", [0.10, 0.20])
    add("PBS only", "Blank", "PBS control", "A4", [0.30, 0.40])
    add("Empty Liposomes (25uM)", "Blank", "Empty liposomes", "E3", [0.90, 1.00])
    add("Free PenAg (3.125uM)", "Blank", "Free PenAg", "D3", [0.05, 0.06])
    add("Loaded Liposomes (25uM) + free/associated PenAg (3.125uM)", "Blank", "Loaded liposomes", "L3", [0.80, 0.90])
    add("Empty Liposomes (25uM)", "B. Subtilis", "Empty liposomes", "F18", [0.70, 0.80])
    add("Loaded Liposomes (25uM) + free/associated PenAg (3.125uM)", "E. Coli", "Loaded liposomes", "L9", [0.90, 1.00])
    add("Free PenAg (3.125uM)", "E. Coli", "Free PenAg", "D9", [0.50, 0.60])
    return pd.DataFrame(rows)


def test_pbs_blank_source_corrects_liposome_wells_with_time_matched_pbs_blank() -> None:
    corrected = lmp.apply_liposome_blank_correction(make_blank_correction_input(), liposome_blank_source="pbs")
    empty_t1 = corrected.loc[(corrected["well"] == "F18") & (corrected["time_h"] == 1.0)].iloc[0]
    loaded_t1 = corrected.loc[(corrected["well"] == "L9") & (corrected["time_h"] == 1.0)].iloc[0]
    free_t1 = corrected.loc[(corrected["well"] == "D9") & (corrected["time_h"] == 1.0)].iloc[0]
    assert empty_t1["blank_mean_od600"] == pytest.approx(0.30)
    assert empty_t1["od600_corrected"] == pytest.approx(0.50)
    assert loaded_t1["blank_mean_od600"] == pytest.approx(0.30)
    assert loaded_t1["od600_corrected"] == pytest.approx(0.70)
    assert free_t1["blank_mean_od600"] == pytest.approx(0.06)
    assert free_t1["od600_corrected"] == pytest.approx(0.54)


def test_matching_blank_source_keeps_previous_liposome_blank_rule() -> None:
    corrected = lmp.apply_liposome_blank_correction(make_blank_correction_input(), liposome_blank_source="matching")
    empty_t1 = corrected.loc[(corrected["well"] == "F18") & (corrected["time_h"] == 1.0)].iloc[0]
    loaded_t1 = corrected.loc[(corrected["well"] == "L9") & (corrected["time_h"] == 1.0)].iloc[0]
    assert empty_t1["blank_mean_od600"] == pytest.approx(0.90)
    assert empty_t1["od600_corrected"] == pytest.approx(-0.10)
    assert loaded_t1["blank_mean_od600"] == pytest.approx(0.90)
    assert loaded_t1["od600_corrected"] == pytest.approx(0.10)


def test_selected_pairs_match_reference_plot_families() -> None:
    families = {family.key: family for family in lmp.family_specs()}
    assert families["empty_liposomes"].selected_pairs == (
        ("PBS", "Empty 25 uM lipid"),
        ("PBS", "Empty 50 uM lipid"),
        ("Empty 25 uM lipid", "Empty 50 uM lipid"),
    )
    assert ("Free 3.125 uM", "Co-delivery + 25 uM lipid") in families["formulation_3_125uM"].selected_pairs
    assert ("Co-delivery + 50 uM lipid", "Loaded + 50 uM lipid") in families["formulation_6_25uM"].selected_pairs


def test_shared_axis_config_uses_all_comparable_panels_in_one_figure() -> None:
    family = next(f for f in lmp.family_specs() if f.key == "empty_liposomes")
    per_well = pd.DataFrame(
        [
            {
                "species": "B. Subtilis",
                "family_key": family.key,
                "metric": "auc_0_19h",
                "treatment": "PBS",
                "biological_replicate": "BR1",
                "metric_value": 1.0,
            },
            {
                "species": "E. Coli",
                "family_key": family.key,
                "metric": "auc_0_19h",
                "treatment": "PBS",
                "biological_replicate": "BR1",
                "metric_value": 12.0,
            },
        ]
    )
    br_means = pd.DataFrame(
        [
            {
                "species": "B. Subtilis",
                "family_key": family.key,
                "metric": "auc_0_19h",
                "treatment": "PBS",
                "technical_mean": 1.0,
            },
            {
                "species": "E. Coli",
                "family_key": family.key,
                "metric": "auc_0_19h",
                "treatment": "PBS",
                "technical_mean": 12.0,
            },
        ]
    )
    shared = lmp.shared_axis_config(
        per_well,
        br_means,
        pd.DataFrame(),
        [(family, "B. Subtilis"), (family, "E. Coli")],
        "auc_0_19h",
    )
    b_subtilis_only = lmp.shared_axis_config(
        per_well,
        br_means,
        pd.DataFrame(),
        [(family, "B. Subtilis")],
        "auc_0_19h",
    )
    assert shared["upper"] > b_subtilis_only["upper"]
    assert shared["upper"] > 12.0


def test_shared_y_axes_keep_tick_labels_visible_on_each_panel() -> None:
    family = next(f for f in lmp.family_specs() if f.key == "empty_liposomes")
    rows = []
    for species, value in [("B. Subtilis", 7.0), ("E. Coli", 10.0)]:
        rows.append(
            {
                "species": species,
                "family_key": family.key,
                "metric": "auc_0_19h",
                "treatment": "PBS",
                "treatment_order": 0,
                "biological_replicate": "BR1",
                "metric_value": value,
                "technical_mean": value,
            }
        )
    per_well = pd.DataFrame(rows)
    br_means = pd.DataFrame(rows)
    axis_config = lmp.shared_axis_config(
        per_well,
        br_means,
        pd.DataFrame(),
        [(family, "B. Subtilis"), (family, "E. Coli")],
        "auc_0_19h",
    )
    fig, axes = lmp.plt.subplots(1, 2, sharey=True)
    for ax, species in zip(axes, ["B. Subtilis", "E. Coli"]):
        lmp.plot_panel(ax, per_well, br_means, pd.DataFrame(), family, "auc_0_19h", species, axis_config=axis_config)
    fig.canvas.draw()
    assert all(label.get_visible() for label in axes[1].get_yticklabels())
    lmp.plt.close(fig)


def test_independent_group_test_calls_are_absent() -> None:
    source = (ANALYSES_DIR / "liposome_metric_plots.py").read_text(encoding="utf-8").lower()
    banned_calls = ["ttest_ind", "anova_oneway", "pairwise_tukeyhsd", "gameshowell"]
    assert not any(call in source for call in banned_calls)
