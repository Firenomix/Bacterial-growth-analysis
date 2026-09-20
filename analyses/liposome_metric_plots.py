"""Calculate and plot liposome metrics with biological replicate blocking.

This script is intentionally separate from the legacy liposome growth-curve
analyses. It can read annotated workbooks or previously processed per-well
metric/Gompertz CSV pairs. Technical-well metrics are averaged within each
biological replicate before inference. Eligible treatment families are fitted
as ``metric ~ treatment + biological_replicate`` and Tukey comparisons use the
blocked-model residual mean square and degrees of freedom.

The ``--liposome-blank-source`` option records whether liposome-containing
wells use their matching plate blanks or the PBS blank mean. Missing and
failed model fits remain missing. By default, inferential output requires at
least three biological replicates; a lower explicit threshold is intended for
clearly labelled exploratory reanalysis only.
"""

from __future__ import annotations

import argparse
import itertools
import math
import re
import sys
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from scipy.optimize import curve_fit


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "results" / "liposome_metric_plots"

SPECIES_ORDER = ["B. Subtilis", "E. Coli"]
SPECIES_DISPLAY = {
    "B. Subtilis": r"$\it{B.\ subtilis}$",
    "E. Coli": r"$\it{E.\ coli}$",
}

METRIC_SPECS = {
    "auc_0_19h": {
        "display": "AUC",
        "ylabel": "Selected-blank-corrected AUC (0-19 h)",
        "source": "metrics",
        "column": "auc_0_19h",
    },
    "lag_h": {
        "display": "Lag time",
        "ylabel": "Modified Gompertz lag time (h)",
        "source": "gompertz",
        "column": "lag_h",
    },
    "mu_max": {
        "display": "Growth rate",
        "ylabel": r"Modified Gompertz maximum OD slope (OD$_{600}$ h$^{-1}$)",
        "source": "gompertz",
        "column": "mu_max",
    },
}

GROUP_STYLE = {
    "Vehicle": {"colour": "#6B7280", "marker": "o"},
    "PBS": {"colour": "#6B7280", "marker": "o"},
    "Free 3.125 uM": {"colour": "#3D6F9F", "marker": "o"},
    "Free 6.25 uM": {"colour": "#2F5D8A", "marker": "o"},
    "Empty 25 uM lipid": {"colour": "#B279A2", "marker": "s"},
    "Empty 50 uM lipid": {"colour": "#8E5F82", "marker": "^"},
    "Co-delivery + 25 uM lipid": {"colour": "#E28F2D", "marker": "s"},
    "Co-delivery + 50 uM lipid": {"colour": "#C96F16", "marker": "^"},
    "Loaded + 25 uM lipid": {"colour": "#3D8E63", "marker": "D"},
    "Loaded + 50 uM lipid": {"colour": "#2F7D55", "marker": "D"},
}

SINGLE_WELL_BLANK_RULES = {
    "Co-delivery PenAg (6.25uM) + Empty Liposomes (50uM)": "J3",
    "Co-delivery PenAg (3.125uM) + Empty Liposomes (50uM)": "K3",
    "Co-delivery PenAg (6.25uM) + Empty Liposomes (25uM)": "L3",
    "Co-delivery PenAg (3.125uM) + Empty Liposomes (25uM)": "M3",
}

LIPOSOME_CONDITION_TYPES = {"Empty liposomes", "Co-delivery", "Loaded liposomes"}

RAW_PLATE_GROUP_CONTENT = {
    "A": "PBS only",
    "B": "0.0625% DMSO",
    "C": "Free PenAg (6.25uM)",
    "D": "Free PenAg (3.125uM)",
    "E": "Empty Liposomes (50uM)",
    "F": "Empty Liposomes (25uM)",
    "G": "Co-delivery PenAg (6.25uM) + Empty Liposomes (50uM)",
    "H": "Co-delivery PenAg (3.125uM) + Empty Liposomes (50uM)",
    "I": "Co-delivery PenAg (6.25uM) + Empty Liposomes (25uM)",
    "J": "Co-delivery PenAg (3.125uM) + Empty Liposomes (25uM)",
    "K": "Loaded Liposomes (50uM) + free/associated PenAg (6.25uM)",
    "L": "Loaded Liposomes (25uM) + free/associated PenAg (3.125uM)",
}

STACKED_PLATE_ROW_CONTENT = {
    "A": "PBS only",
    "B": "0.0625% DMSO",
    "D": "Free PenAg (6.25uM)",
    "E": "Free PenAg (3.125uM)",
    "G": "Empty Liposomes (50uM)",
    "H": "Empty Liposomes (25uM)",
    "J": "Co-delivery PenAg (6.25uM) + Empty Liposomes (50uM)",
    "K": "Co-delivery PenAg (3.125uM) + Empty Liposomes (50uM)",
    "L": "Co-delivery PenAg (6.25uM) + Empty Liposomes (25uM)",
    "M": "Co-delivery PenAg (3.125uM) + Empty Liposomes (25uM)",
    "O": "Loaded Liposomes (50uM) + free/associated PenAg (6.25uM)",
    "P": "Loaded Liposomes (25uM) + free/associated PenAg (3.125uM)",
}


@dataclass(frozen=True)
class FamilySpec:
    """Treatment groups and displayed comparisons for one analysis family."""
    key: str
    label: str
    groups: tuple[dict, ...]
    selected_pairs: tuple[tuple[str, str], ...]


@dataclass(frozen=True)
class BlockAnovaResult:
    """Blocked-ANOVA components used by the Tukey calculation."""
    species: str
    family_key: str
    family_label: str
    metric: str
    number_of_conditions: int
    number_of_biological_replicates: int
    total_model_observations: int
    treatment_df: int
    block_df: int
    residual_df: int
    treatment_SS: float
    block_SS: float
    residual_SS: float
    treatment_MS: float
    block_MS: float
    residual_MS: float
    treatment_F: float
    treatment_p_value: float
    block_F: float
    block_p_value: float
    model_status: str = "fitted"

    def to_dict(self) -> dict:
        return self.__dict__.copy()


class LiposomeMetricError(ValueError):
    """Raised when liposome data or replicate structure cannot be analysed."""


ANOVA_COLUMNS = [
    "species",
    "family_key",
    "family_label",
    "metric",
    "number_of_conditions",
    "number_of_biological_replicates",
    "total_model_observations",
    "treatment_df",
    "block_df",
    "residual_df",
    "treatment_SS",
    "block_SS",
    "residual_SS",
    "treatment_MS",
    "block_MS",
    "residual_MS",
    "treatment_F",
    "treatment_p_value",
    "block_F",
    "block_p_value",
    "model_status",
]

TUKEY_COLUMNS = [
    "species",
    "family_key",
    "family_label",
    "metric",
    "group_1",
    "group_2",
    "mean_difference_group_1_minus_group_2",
    "Tukey_standard_error",
    "Tukey_q",
    "Tukey_q_critical_0_05",
    "Tukey_adjusted_p_value",
    "simultaneous_ci_low",
    "simultaneous_ci_high",
    "residual_MSE",
    "residual_df",
    "number_of_groups",
    "statistically_significant",
    "significance",
]


def safe_name(text: object) -> str:
    return re.sub(r"[^A-Za-z0-9]+", "_", str(text)).strip("_")


def parse_key_value_path(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("Expected BR_ID=path")
    key, path = value.split("=", 1)
    key = key.strip()
    if not key:
        raise argparse.ArgumentTypeError("Biological replicate id is empty")
    return key, Path(path.strip().strip('"'))


def time_to_hours(time_text: object) -> float:
    text = str(time_text).strip()
    hours_match = re.search(r"(\d+(?:\.\d+)?)\s*h", text, flags=re.IGNORECASE)
    minute_match = re.search(r"(\d+(?:\.\d+)?)\s*min", text, flags=re.IGNORECASE)
    if not hours_match and not minute_match:
        raise LiposomeMetricError(f"Could not parse time label: {time_text!r}")
    hours = float(hours_match.group(1)) if hours_match else 0.0
    minutes = float(minute_match.group(1)) if minute_match else 0.0
    return hours + minutes / 60.0


def first_number_before(text: object, marker: str) -> float:
    match = re.search(
        rf"{re.escape(marker)}\s*\((\d+(?:\.\d+)?)\s*uM\)",
        str(text),
        flags=re.IGNORECASE,
    )
    return float(match.group(1)) if match else np.nan


def classify_condition(content: object) -> str:
    text = str(content).strip()
    if text == "PBS only":
        return "PBS control"
    if "DMSO" in text:
        return "Vehicle control"
    if text.startswith("Free PenAg"):
        return "Free PenAg"
    if text.startswith("Empty Liposomes"):
        return "Empty liposomes"
    if text.startswith("Co-delivery"):
        return "Co-delivery"
    if text.startswith("Loaded Liposomes"):
        return "Loaded liposomes"
    return "Other"


def extract_drug_concentration(content: object) -> float:
    text = str(content)
    if text.startswith("Free PenAg"):
        return first_number_before(text, "Free PenAg")
    if text.startswith("Co-delivery"):
        return first_number_before(text, "Co-delivery PenAg")
    match = re.search(
        r"free/associated PenAg\s*\((\d+(?:\.\d+)?)\s*uM\)",
        text,
        flags=re.IGNORECASE,
    )
    return float(match.group(1)) if match else 0.0


def extract_lipid_concentration(content: object) -> float:
    text = str(content)
    if text.startswith("Empty Liposomes"):
        return first_number_before(text, "Empty Liposomes")
    if text.startswith("Co-delivery"):
        match = re.search(
            r"Empty Liposomes\s*\((\d+(?:\.\d+)?)\s*uM\)",
            text,
            flags=re.IGNORECASE,
        )
        return float(match.group(1)) if match else np.nan
    if text.startswith("Loaded Liposomes"):
        return first_number_before(text, "Loaded Liposomes")
    return 0.0


def normalize_species(group: object) -> str:
    text = str(group).strip()
    if text == "B. Subtilits":
        return "B. Subtilis"
    return text


def decode_raw_plate_row(well_row: object, well_col: object, content: object, group: object) -> tuple[str, str]:
    """Decode raw BMG Sample X1/X2 rows into treatment and species labels."""

    group_key = str(group).strip().upper()
    col = int(well_col)
    text = str(content).strip()
    if group_key not in RAW_PLATE_GROUP_CONTENT or not (text.startswith("Sample X") or text.startswith("Blank")):
        return text, normalize_species(group)
    decoded_content = RAW_PLATE_GROUP_CONTENT[group_key]
    if 3 <= col <= 5:
        return decoded_content, "Blank"
    if 9 <= col <= 14:
        return decoded_content, "E. Coli"
    if 18 <= col <= 23:
        return decoded_content, "B. Subtilis"
    return decoded_content, normalize_species(group)


def decode_stacked_plate_row(well_row: object, well_col: object) -> tuple[str, str]:
    """Decode stacked-cycle exports where condition identity is the physical plate row."""

    row_key = str(well_row).strip().upper()
    col = int(well_col)
    if row_key not in STACKED_PLATE_ROW_CONTENT:
        return str(well_row).strip(), "Other"
    content = STACKED_PLATE_ROW_CONTENT[row_key]
    if 3 <= col <= 5:
        return content, "Blank"
    if 9 <= col <= 14:
        return content, "E. Coli"
    if 18 <= col <= 23:
        return content, "B. Subtilis"
    return content, "Other"


def short_condition_label(row: pd.Series) -> str:
    condition_type = row["condition_type"]
    drug = float(row["drug_uM"]) if pd.notna(row["drug_uM"]) else np.nan
    lipid = float(row["lipid_uM"]) if pd.notna(row["lipid_uM"]) else np.nan
    if condition_type == "PBS control":
        return "PBS"
    if condition_type == "Vehicle control":
        return "0.0625% DMSO"
    if condition_type == "Free PenAg":
        return f"Free {drug:g} uM"
    if condition_type == "Empty liposomes":
        return f"Empty {lipid:g} uM lipid"
    if condition_type == "Co-delivery":
        return f"Co-delivery {drug:g} uM + {lipid:g} uM lipid"
    if condition_type == "Loaded liposomes":
        return f"Loaded {lipid:g} uM lipid + {drug:g} uM PenAg"
    return str(row["content"])


def selected_label(row: pd.Series) -> str | None:
    condition_type = row["condition_type"]
    drug = float(row["drug_uM"]) if pd.notna(row["drug_uM"]) else np.nan
    lipid = float(row["lipid_uM"]) if pd.notna(row["lipid_uM"]) else np.nan
    if condition_type == "Vehicle control":
        return "Vehicle"
    if condition_type == "PBS control":
        return "PBS"
    if condition_type == "Free PenAg" and np.isclose(drug, 3.125):
        return "Free 3.125 uM"
    if condition_type == "Free PenAg" and np.isclose(drug, 6.25):
        return "Free 6.25 uM"
    if condition_type == "Empty liposomes" and np.isclose(lipid, 25.0):
        return "Empty 25 uM lipid"
    if condition_type == "Empty liposomes" and np.isclose(lipid, 50.0):
        return "Empty 50 uM lipid"
    if condition_type == "Co-delivery" and np.isclose(lipid, 25.0):
        return "Co-delivery + 25 uM lipid"
    if condition_type == "Co-delivery" and np.isclose(lipid, 50.0):
        return "Co-delivery + 50 uM lipid"
    if condition_type == "Loaded liposomes" and np.isclose(lipid, 25.0):
        return "Loaded + 25 uM lipid"
    if condition_type == "Loaded liposomes" and np.isclose(lipid, 50.0):
        return "Loaded + 50 uM lipid"
    return None


def modified_gompertz(time_h: np.ndarray, y0: float, amplitude: float, mu_max: float, lag_h: float) -> np.ndarray:
    amplitude = np.maximum(amplitude, 1e-9)
    exponent = (mu_max * np.e / amplitude) * (lag_h - time_h) + 1.0
    exponent = np.clip(exponent, -50, 50)
    return y0 + amplitude * np.exp(-np.exp(exponent))


def time_to_threshold(time_h: np.ndarray, values: np.ndarray, threshold: float) -> float:
    crossing = np.where(values >= threshold)[0]
    if len(crossing) == 0:
        return np.nan
    index = int(crossing[0])
    if index == 0:
        return float(time_h[0])
    t1, t2 = time_h[index - 1], time_h[index]
    y1, y2 = values[index - 1], values[index]
    if y2 == y1:
        return float(t2)
    return float(t1 + (threshold - y1) / (y2 - y1) * (t2 - t1))


def fit_gompertz(time_h: np.ndarray, od_values: np.ndarray) -> dict:
    time_h = np.asarray(time_h, dtype=float)
    od_values = np.asarray(od_values, dtype=float)
    baseline = float(np.median(od_values[: min(5, len(od_values))]))
    signal_range = float(np.max(od_values) - np.min(od_values))
    if signal_range < 0.05:
        return _failed_fit("skipped_low_signal", signal_range)

    amplitude_guess = max(float(np.max(od_values) - baseline), 0.01)
    gradient_guess = max(float(np.max(np.gradient(od_values, time_h))), 0.01)
    lag_guess = time_to_threshold(time_h, od_values, baseline + 0.1 * amplitude_guess)
    if np.isnan(lag_guess):
        lag_guess = 2.0

    lower = [-0.30, 0.001, 0.001, 0.0]
    upper = [0.50, 3.00, 3.00, float(np.max(time_h))]
    initial = [
        np.clip(baseline, lower[0] + 1e-6, upper[0] - 1e-6),
        np.clip(amplitude_guess, lower[1] + 1e-6, upper[1] - 1e-6),
        np.clip(gradient_guess, lower[2] + 1e-6, upper[2] - 1e-6),
        np.clip(lag_guess, lower[3] + 1e-6, upper[3] - 1e-6),
    ]
    try:
        parameters, _ = curve_fit(
            modified_gompertz,
            time_h,
            od_values,
            p0=initial,
            bounds=(lower, upper),
            maxfev=30000,
        )
    except (RuntimeError, ValueError, FloatingPointError) as exc:
        result = _failed_fit(f"fit_failed: {exc}", signal_range)
        return result

    fitted = modified_gompertz(time_h, *parameters)
    residuals = od_values - fitted
    rss = float(np.sum(residuals**2))
    tss = float(np.sum((od_values - np.mean(od_values)) ** 2))
    r_squared = 1.0 - rss / tss if tss > 0 else np.nan
    return {
        "fit_status": "ok" if r_squared >= 0.80 else "poor_fit",
        "signal_range": signal_range,
        "y0": float(parameters[0]),
        "A": float(parameters[1]),
        "mu_max": float(parameters[2]),
        "lag_h": float(parameters[3]),
        "rss": rss,
        "r_squared": float(r_squared),
    }


def _failed_fit(status: str, signal_range: float) -> dict:
    return {
        "fit_status": status,
        "signal_range": signal_range,
        "y0": np.nan,
        "A": np.nan,
        "mu_max": np.nan,
        "lag_h": np.nan,
        "rss": np.nan,
        "r_squared": np.nan,
    }


def estimate_specific_growth_rate(time_h: np.ndarray, od_values: np.ndarray, window_h: float = 1.5, min_od: float = 0.03) -> dict:
    time_h = np.asarray(time_h, dtype=float)
    od_values = np.asarray(od_values, dtype=float)
    valid = np.isfinite(time_h) & np.isfinite(od_values) & (od_values > min_od)
    time_h = time_h[valid]
    od_values = od_values[valid]
    if len(time_h) < 5:
        return {
            "specific_growth_rate_h_inv": np.nan,
            "specific_growth_rate_r_squared": np.nan,
            "growth_rate_window_start_h": np.nan,
            "growth_rate_window_end_h": np.nan,
        }
    order = np.argsort(time_h)
    time_h = time_h[order]
    od_values = od_values[order]
    best_slope = best_r_squared = best_start = best_end = np.nan
    for start_time in time_h:
        in_window = (time_h >= start_time) & (time_h <= start_time + window_h)
        if in_window.sum() < 5:
            continue
        window_time = time_h[in_window]
        window_od = od_values[in_window]
        if window_time[-1] - window_time[0] < 0.8 * window_h:
            continue
        slope, intercept = np.polyfit(window_time, np.log(window_od), 1)
        fitted = slope * window_time + intercept
        rss = float(np.sum((np.log(window_od) - fitted) ** 2))
        tss = float(np.sum((np.log(window_od) - np.mean(np.log(window_od))) ** 2))
        r_squared = 1.0 - rss / tss if tss > 0 else np.nan
        if slope > 0 and (np.isnan(best_slope) or slope > best_slope):
            best_slope = float(slope)
            best_r_squared = float(r_squared)
            best_start = float(window_time[0])
            best_end = float(window_time[-1])
    return {
        "specific_growth_rate_h_inv": best_slope,
        "specific_growth_rate_r_squared": best_r_squared,
        "growth_rate_window_start_h": best_start,
        "growth_rate_window_end_h": best_end,
    }


def read_measurement_sheet(workbook_path: Path, sheet_name: str = "All Cycles") -> tuple[pd.DataFrame, str]:
    workbook = pd.ExcelFile(workbook_path, engine="openpyxl")
    candidates = []
    if sheet_name in workbook.sheet_names:
        candidates.append(sheet_name)
    candidates.extend(sheet for sheet in workbook.sheet_names if sheet not in candidates)
    for candidate in candidates:
        excel = pd.read_excel(workbook_path, sheet_name=candidate, header=None, engine="openpyxl")
        if excel.eq("Raw Data (600)").any(axis=1).any():
            return excel, candidate
    raise LiposomeMetricError(
        f"No raw OD600 columns found in {workbook_path}. Checked sheets: {workbook.sheet_names}"
    )


def build_import_record(
    replicate_id: str,
    workbook_path: Path,
    sheet_name: str,
    source_row: int,
    well_row: object,
    well_col: object,
    content: object,
    group: object,
    time_h: float,
    od600: object,
) -> dict:
    well = f"{str(well_row).strip()}{int(well_col)}"
    content_text, species = decode_raw_plate_row(well_row, well_col, content, group)
    return {
        "biological_replicate": replicate_id,
        "source_row": int(source_row),
        "source_file": str(workbook_path),
        "source_sheet": sheet_name,
        "well": well,
        "content": content_text,
        "species": species,
        "condition_type": classify_condition(content_text),
        "drug_uM": extract_drug_concentration(content_text),
        "lipid_uM": extract_lipid_concentration(content_text),
        "time_h": float(time_h),
        "od600_raw": float(od600),
    }


def import_stacked_cycle_sheet(
    excel: pd.DataFrame,
    workbook_path: Path,
    replicate_id: str,
    sheet_name: str,
    header_rows: list[int],
) -> pd.DataFrame:
    records = []
    for block_index, header_row in enumerate(header_rows):
        time_label = excel.iloc[header_row - 1, 0] if header_row > 0 else np.nan
        time_h = time_to_hours(time_label)
        next_header = header_rows[block_index + 1] if block_index + 1 < len(header_rows) else len(excel)
        column_row = header_row + 1
        data_start_row = header_row + 2
        well_columns = [
            column
            for column in excel.columns
            if pd.notna(excel.iloc[column_row, column]) and str(excel.iloc[column_row, column]).strip().replace(".0", "").isdigit()
        ]
        for source_row in range(data_start_row, next_header):
            well_row = excel.iloc[source_row, 0]
            if pd.isna(well_row) or str(well_row).strip().upper() not in STACKED_PLATE_ROW_CONTENT:
                continue
            for column in well_columns:
                well_col = excel.iloc[column_row, column]
                od600 = excel.iloc[source_row, column]
                if pd.isna(od600):
                    continue
                content_text, species = decode_stacked_plate_row(well_row, well_col)
                if species == "Other":
                    continue
                records.append(
                    build_import_record(
                        replicate_id,
                        workbook_path,
                        sheet_name,
                        source_row,
                        well_row,
                        well_col,
                        content_text,
                        species,
                        time_h,
                        od600,
                    )
                )
    return pd.DataFrame.from_records(records)


def import_workbook(workbook_path: Path, replicate_id: str, sheet_name: str = "All Cycles") -> pd.DataFrame:
    if not workbook_path.exists():
        raise FileNotFoundError(f"Workbook not found for {replicate_id}: {workbook_path}")
    excel, actual_sheet = read_measurement_sheet(workbook_path, sheet_name=sheet_name)
    header_matches = excel.eq("Raw Data (600)")
    matching_rows = header_matches.any(axis=1)
    if not matching_rows.any():
        raise LiposomeMetricError(f"No raw OD600 columns found in {workbook_path}")
    header_rows = [int(row) for row in matching_rows[matching_rows].index]
    header_row = header_rows[0]
    time_row = header_row + 1
    data_start_row = header_row + 2
    od_columns = [column for column in excel.columns if excel.iloc[header_row, column] == "Raw Data (600)"]
    if not od_columns:
        raise LiposomeMetricError(f"No raw OD600 columns found in {workbook_path}")
    if len(header_rows) > 1 and len(od_columns) == 1:
        tidy = import_stacked_cycle_sheet(excel, workbook_path, replicate_id, actual_sheet, header_rows)
        if tidy.empty:
            raise LiposomeMetricError(f"No records imported from stacked-cycle workbook {workbook_path}")
        labels = tidy.drop_duplicates(["content", "condition_type", "drug_uM", "lipid_uM"]).copy()
        label_map = {row["content"]: short_condition_label(row) for _, row in labels.iterrows()}
        tidy["condition_label"] = tidy["content"].map(label_map)
        return tidy
    time_hours = [time_to_hours(excel.iloc[time_row, column]) for column in od_columns]
    records = []
    for source_row, row in excel.iloc[data_start_row:].iterrows():
        well_row, well_col, content, group = row.iloc[0], row.iloc[1], row.iloc[2], row.iloc[3]
        if any(pd.isna(value) for value in [well_row, well_col, content, group]):
            continue
        for column, time_h in zip(od_columns, time_hours):
            od600 = row[column]
            if pd.isna(od600):
                continue
            records.append(
                build_import_record(
                    replicate_id,
                    workbook_path,
                    actual_sheet,
                    source_row,
                    well_row,
                    well_col,
                    content,
                    group,
                    time_h,
                    od600,
                )
            )
    tidy = pd.DataFrame.from_records(records)
    if tidy.empty:
        raise LiposomeMetricError(f"No records imported from {workbook_path}")
    labels = tidy.drop_duplicates(["content", "condition_type", "drug_uM", "lipid_uM"]).copy()
    label_map = {row["content"]: short_condition_label(row) for _, row in labels.iterrows()}
    tidy["condition_label"] = tidy["content"].map(label_map)
    return tidy


def apply_liposome_blank_correction(tidy: pd.DataFrame, liposome_blank_source: str = "matching") -> pd.DataFrame:
    if liposome_blank_source not in {"matching", "pbs"}:
        raise LiposomeMetricError("liposome_blank_source must be 'matching' or 'pbs'.")
    blank_data = tidy.loc[tidy["species"] == "Blank"].copy()
    growth_raw = tidy.loc[tidy["species"].isin(["E. Coli", "B. Subtilis"])].copy()
    if blank_data.empty or growth_raw.empty:
        raise LiposomeMetricError("Expected both Blank rows and bacterial sample rows.")

    blank_curve = (
        blank_data.groupby(["content", "condition_label", "time_h"], as_index=False)
        .agg(blank_mean_od600=("od600_raw", "mean"), blank_sd_od600=("od600_raw", "std"), blank_n=("od600_raw", "count"))
    )
    blank_curve["blank_source_well"] = "all matching blank wells"
    special_curves = []
    for content, well in SINGLE_WELL_BLANK_RULES.items():
        special = blank_data.loc[(blank_data["content"] == content) & (blank_data["well"] == well)]
        if special.empty:
            continue
        special_curve = (
            special.groupby(["content", "condition_label", "time_h"], as_index=False)
            .agg(blank_mean_od600=("od600_raw", "mean"), blank_sd_od600=("od600_raw", "std"), blank_n=("od600_raw", "count"))
        )
        special_curve["blank_source_well"] = well
        special_curves.append(special_curve)
    if special_curves:
        blank_curve = blank_curve.loc[~blank_curve["content"].isin(SINGLE_WELL_BLANK_RULES)].copy()
        blank_curve = pd.concat([blank_curve, *special_curves], ignore_index=True)

    growth_raw["blank_source_content"] = growth_raw["content"]
    blank_for_correction = blank_curve[["content", "time_h", "blank_mean_od600", "blank_source_well"]].rename(
        columns={
            "content": "blank_source_content",
            "blank_mean_od600": "blank_timepoint_mean_od600",
            "blank_source_well": "blank_timepoint_source_well",
        }
    )
    t0_time = float(blank_curve["time_h"].min())
    t0_blank = blank_curve.loc[
        np.isclose(blank_curve["time_h"], t0_time),
        ["content", "blank_mean_od600", "blank_source_well"],
    ].rename(
        columns={
            "content": "blank_source_content",
            "blank_mean_od600": "blank_t0_mean_od600",
            "blank_source_well": "blank_t0_source_well",
        }
    )
    corrected = growth_raw.merge(blank_for_correction, on=["blank_source_content", "time_h"], how="left")
    corrected = corrected.merge(t0_blank, on="blank_source_content", how="left")
    corrected["_use_t0_blank"] = corrected["condition_type"].isin(["Empty liposomes", "Co-delivery"])
    corrected["_use_pbs_blank"] = corrected["condition_type"].isin(LIPOSOME_CONDITION_TYPES) & (liposome_blank_source == "pbs")

    if liposome_blank_source == "pbs":
        pbs_blanks = blank_data.loc[blank_data["condition_type"] == "PBS control"].copy()
        if pbs_blanks.empty:
            raise LiposomeMetricError("PBS blank correction requested, but no PBS blank rows were found.")
        pbs_blank_curve = (
            pbs_blanks.groupby("time_h", as_index=False)
            .agg(pbs_blank_mean_od600=("od600_raw", "mean"), pbs_blank_n=("od600_raw", "count"))
        )
        corrected = corrected.merge(pbs_blank_curve, on="time_h", how="left")
        if corrected.loc[corrected["_use_pbs_blank"], "pbs_blank_mean_od600"].isna().any():
            missing_times = (
                corrected.loc[corrected["_use_pbs_blank"] & corrected["pbs_blank_mean_od600"].isna(), "time_h"]
                .drop_duplicates()
                .sort_values()
                .tolist()
            )
            raise LiposomeMetricError(f"PBS blank correction requested, but PBS blanks are missing time points: {missing_times}")
    else:
        corrected["pbs_blank_mean_od600"] = np.nan
        corrected["pbs_blank_n"] = np.nan

    corrected["blank_mean_od600"] = np.where(
        corrected["_use_pbs_blank"],
        corrected["pbs_blank_mean_od600"],
        np.where(
            corrected["_use_t0_blank"],
            corrected["blank_t0_mean_od600"],
            corrected["blank_timepoint_mean_od600"],
        ),
    )
    corrected["blank_source_well"] = np.where(
        corrected["_use_pbs_blank"],
        "PBS blank wells mean",
        np.where(
            corrected["_use_t0_blank"],
            corrected["blank_t0_source_well"],
            corrected["blank_timepoint_source_well"],
        ),
    )
    if corrected["blank_mean_od600"].isna().any():
        missing = corrected.loc[corrected["blank_mean_od600"].isna(), "content"].drop_duplicates().tolist()
        raise LiposomeMetricError(f"Missing blank correction for conditions: {missing}")
    corrected["od600_corrected"] = corrected["od600_raw"] - corrected["blank_mean_od600"]
    return corrected


def calculate_workbook_metrics(workbook_path: Path, replicate_id: str, liposome_blank_source: str = "matching") -> tuple[pd.DataFrame, pd.DataFrame]:
    corrected = apply_liposome_blank_correction(import_workbook(workbook_path, replicate_id), liposome_blank_source=liposome_blank_source)
    metric_rows = []
    gompertz_rows = []
    group_cols = ["biological_replicate", "species", "content", "condition_label", "condition_type", "drug_uM", "lipid_uM", "well"]
    for keys, curve in corrected.groupby(group_cols, dropna=False):
        curve = curve.sort_values("time_h")
        time_h = curve["time_h"].to_numpy(dtype=float)
        od = curve["od600_corrected"].to_numpy(dtype=float)
        before_14h = time_h <= 14.0
        baseline = float(np.median(od[time_h <= 1.0]))
        maximum = float(np.max(od))
        threshold = baseline + 0.1 * (maximum - baseline)
        common = dict(zip(group_cols, keys))
        metric_rows.append(
            {
                **common,
                "baseline_od600": baseline,
                "final_od600": float(od[-1]),
                "max_od600": maximum,
                "auc_0_14h": float(np.trapezoid(od[before_14h], time_h[before_14h])),
                "auc_0_19h": float(np.trapezoid(od, time_h)),
                "time_to_10pct_growth_h": time_to_threshold(time_h, od, threshold),
                **estimate_specific_growth_rate(time_h, od),
            }
        )
        gompertz_rows.append({**common, **fit_gompertz(time_h, od)})
    return pd.DataFrame.from_records(metric_rows), pd.DataFrame.from_records(gompertz_rows)


def load_processed_metric_pair(metrics_path: Path, gompertz_path: Path, replicate_id: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    metrics = pd.read_csv(metrics_path)
    gompertz = pd.read_csv(gompertz_path)
    for table, path in [(metrics, metrics_path), (gompertz, gompertz_path)]:
        missing = {"species", "condition_type", "drug_uM", "lipid_uM", "well"} - set(table.columns)
        if missing:
            raise LiposomeMetricError(f"{path} is missing columns: {sorted(missing)}")
        table.insert(0, "biological_replicate", replicate_id)
        table["source_file"] = str(path)
    return metrics, gompertz


def family_specs() -> list[FamilySpec]:
    families = [
        FamilySpec(
            key="free_penag",
            label="Free PenAg dose response",
            groups=(
                {"label": "Vehicle", "condition_type": "Vehicle control"},
                {"label": "Free 3.125 uM", "condition_type": "Free PenAg", "drug_uM": 3.125},
                {"label": "Free 6.25 uM", "condition_type": "Free PenAg", "drug_uM": 6.25},
            ),
            selected_pairs=(
                ("Vehicle", "Free 3.125 uM"),
                ("Vehicle", "Free 6.25 uM"),
                ("Free 3.125 uM", "Free 6.25 uM"),
            ),
        ),
        FamilySpec(
            key="empty_liposomes",
            label="Empty liposome control",
            groups=(
                {"label": "PBS", "condition_type": "PBS control"},
                {"label": "Empty 25 uM lipid", "condition_type": "Empty liposomes", "lipid_uM": 25.0},
                {"label": "Empty 50 uM lipid", "condition_type": "Empty liposomes", "lipid_uM": 50.0},
            ),
            selected_pairs=(
                ("PBS", "Empty 25 uM lipid"),
                ("PBS", "Empty 50 uM lipid"),
                ("Empty 25 uM lipid", "Empty 50 uM lipid"),
            ),
        ),
    ]
    for drug, loaded_lipid in [(3.125, 25.0), (6.25, 50.0)]:
        free = f"Free {drug:g} uM"
        co25 = "Co-delivery + 25 uM lipid"
        co50 = "Co-delivery + 50 uM lipid"
        loaded = f"Loaded + {loaded_lipid:g} uM lipid"
        families.append(
            FamilySpec(
                key=f"formulation_{safe_name(f'{drug:g}uM')}",
                label=f"{drug:g}uM PenAg",
                groups=(
                    {"label": free, "condition_type": "Free PenAg", "drug_uM": drug},
                    {"label": co25, "condition_type": "Co-delivery", "drug_uM": drug, "lipid_uM": 25.0},
                    {"label": co50, "condition_type": "Co-delivery", "drug_uM": drug, "lipid_uM": 50.0},
                    {"label": loaded, "condition_type": "Loaded liposomes", "drug_uM": drug, "lipid_uM": loaded_lipid},
                ),
                selected_pairs=(
                    (free, co25),
                    (free, co50),
                    (free, loaded),
                    (co25 if np.isclose(loaded_lipid, 25.0) else co50, loaded),
                ),
            )
        )
    return families


def select_family_rows(table: pd.DataFrame, family: FamilySpec, species: str, metric: str) -> pd.DataFrame:
    rows = []
    value_col = METRIC_SPECS[metric]["column"]
    for order, group in enumerate(family.groups):
        mask = (table["species"] == species) & (table["condition_type"] == group["condition_type"])
        if "drug_uM" in group:
            mask &= np.isclose(pd.to_numeric(table["drug_uM"], errors="coerce"), group["drug_uM"])
        if "lipid_uM" in group:
            mask &= np.isclose(pd.to_numeric(table["lipid_uM"], errors="coerce"), group["lipid_uM"])
        selected = table.loc[mask].copy()
        selected["treatment"] = group["label"]
        selected["treatment_order"] = order
        rows.append(selected)
    if not rows:
        return pd.DataFrame()
    data = pd.concat(rows, ignore_index=True)
    data[value_col] = pd.to_numeric(data[value_col], errors="coerce")
    return data.dropna(subset=[value_col]).copy()


def biological_replicate_means(per_well: pd.DataFrame, metric: str) -> pd.DataFrame:
    value_col = METRIC_SPECS[metric]["column"]
    return (
        per_well.groupby(
            ["species", "biological_replicate", "treatment", "treatment_order"],
            as_index=False,
            observed=False,
        )
        .agg(
            technical_mean=(value_col, "mean"),
            technical_SD=(value_col, lambda values: float(np.std(values, ddof=1)) if len(values) > 1 else np.nan),
            n_technical_wells=(value_col, "count"),
        )
    )


def validate_replicate_structure(br_means: pd.DataFrame, family: FamilySpec, metric: str, minimum_blocks: int = 3) -> tuple[bool, str]:
    if br_means.empty:
        return False, "No valid biological-replicate means."
    expected_treatments = [group["label"] for group in family.groups]
    replicates = sorted(br_means["biological_replicate"].dropna().unique().tolist())
    if len(replicates) < minimum_blocks:
        return False, f"Fewer than {minimum_blocks} biological replicates are available."
    observed = set(zip(br_means["biological_replicate"], br_means["treatment"]))
    missing = [(rep, treatment) for rep in replicates for treatment in expected_treatments if (rep, treatment) not in observed]
    if missing:
        return False, f"Incomplete randomized-block cells for {metric}: {missing}"
    return True, "complete"


def randomized_block_anova(
    br_means: pd.DataFrame,
    family: FamilySpec,
    metric: str,
    species: str,
    minimum_blocks: int = 3,
) -> BlockAnovaResult:
    ok, reason = validate_replicate_structure(br_means, family, metric, minimum_blocks=minimum_blocks)
    if not ok:
        raise LiposomeMetricError(reason)

    data = br_means.copy()
    treatment_order = [group["label"] for group in family.groups]
    replicate_order = sorted(data["biological_replicate"].unique().tolist())
    data["treatment"] = pd.Categorical(data["treatment"], categories=treatment_order, ordered=True)
    data["biological_replicate"] = pd.Categorical(data["biological_replicate"], categories=replicate_order, ordered=True)
    values = data["technical_mean"].to_numpy(dtype=float)
    grand_mean = float(np.mean(values))
    treatment_means = data.groupby("treatment", observed=False)["technical_mean"].mean().reindex(treatment_order)
    block_means = data.groupby("biological_replicate", observed=False)["technical_mean"].mean().reindex(replicate_order)
    a = len(treatment_order)
    b = len(replicate_order)
    total_ss = float(np.sum((values - grand_mean) ** 2))
    treatment_ss = float(b * np.sum((treatment_means - grand_mean) ** 2))
    block_ss = float(a * np.sum((block_means - grand_mean) ** 2))
    residual_ss = max(float(total_ss - treatment_ss - block_ss), 0.0)
    treatment_df = a - 1
    block_df = b - 1
    residual_df = treatment_df * block_df
    treatment_ms = treatment_ss / treatment_df
    block_ms = block_ss / block_df
    residual_ms = residual_ss / residual_df if residual_df > 0 else np.nan
    treatment_f = treatment_ms / residual_ms if residual_ms > 0 else math.inf
    block_f = block_ms / residual_ms if residual_ms > 0 else math.inf
    return BlockAnovaResult(
        species=species,
        family_key=family.key,
        family_label=family.label,
        metric=metric,
        number_of_conditions=a,
        number_of_biological_replicates=b,
        total_model_observations=len(data),
        treatment_df=treatment_df,
        block_df=block_df,
        residual_df=residual_df,
        treatment_SS=treatment_ss,
        block_SS=block_ss,
        residual_SS=residual_ss,
        treatment_MS=treatment_ms,
        block_MS=block_ms,
        residual_MS=residual_ms,
        treatment_F=treatment_f,
        treatment_p_value=float(stats.f.sf(treatment_f, treatment_df, residual_df)) if np.isfinite(treatment_f) else 0.0,
        block_F=block_f,
        block_p_value=float(stats.f.sf(block_f, block_df, residual_df)) if np.isfinite(block_f) else 0.0,
    )


def tukey_from_block(br_means: pd.DataFrame, anova: BlockAnovaResult, family: FamilySpec) -> pd.DataFrame:
    """Calculate all-pair Tukey tests from blocked-model residual variance."""
    treatment_order = [group["label"] for group in family.groups]
    means = br_means.groupby("treatment", observed=False)["technical_mean"].mean().reindex(treatment_order)
    tukey_se = math.sqrt(anova.residual_MS / anova.number_of_biological_replicates) if anova.residual_MS >= 0 else np.nan
    q_crit = float(stats.studentized_range.ppf(0.95, anova.number_of_conditions, anova.residual_df))
    rows = []
    for first, second in itertools.combinations(treatment_order, 2):
        difference = float(means.loc[first] - means.loc[second])
        q_stat = abs(difference) / tukey_se if tukey_se > 0 else math.inf
        p_value = float(stats.studentized_range.sf(q_stat, anova.number_of_conditions, anova.residual_df)) if np.isfinite(q_stat) else 0.0
        half_width = q_crit * tukey_se if np.isfinite(tukey_se) else np.nan
        rows.append(
            {
                "species": anova.species,
                "family_key": family.key,
                "family_label": family.label,
                "metric": anova.metric,
                "group_1": first,
                "group_2": second,
                "mean_difference_group_1_minus_group_2": difference,
                "Tukey_standard_error": tukey_se,
                "Tukey_q": q_stat,
                "Tukey_q_critical_0_05": q_crit,
                "Tukey_adjusted_p_value": p_value,
                "simultaneous_ci_low": difference - half_width,
                "simultaneous_ci_high": difference + half_width,
                "residual_MSE": anova.residual_MS,
                "residual_df": anova.residual_df,
                "number_of_groups": anova.number_of_conditions,
                "statistically_significant": bool(p_value < 0.05),
                "significance": p_value_to_label(p_value),
            }
        )
    return pd.DataFrame.from_records(rows)


def p_value_to_label(p_value: float) -> str:
    if pd.isna(p_value):
        return ""
    if p_value < 0.0001:
        return "****"
    if p_value < 0.001:
        return "***"
    if p_value < 0.01:
        return "**"
    if p_value < 0.05:
        return "*"
    return "n.s."


def build_analysis_tables(
    metrics: pd.DataFrame,
    gompertz: pd.DataFrame,
    minimum_statistical_brs: int = 3,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, list[str]]:
    all_per_well = []
    all_br_means = []
    anova_rows = []
    tukey_rows = []
    warnings_list = []
    for family in family_specs():
        for metric in METRIC_SPECS:
            source = metrics if METRIC_SPECS[metric]["source"] == "metrics" else gompertz.loc[gompertz["fit_status"] == "ok"].copy()
            for species in SPECIES_ORDER:
                family_rows = select_family_rows(source, family, species, metric)
                if family_rows.empty:
                    warnings_list.append(f"No rows for {species}, {family.label}, {metric}.")
                    continue
                family_rows["family_key"] = family.key
                family_rows["family_label"] = family.label
                family_rows["metric"] = metric
                family_rows["metric_value"] = family_rows[METRIC_SPECS[metric]["column"]]
                all_per_well.append(family_rows)
                br_means = biological_replicate_means(family_rows, metric)
                br_means["family_key"] = family.key
                br_means["family_label"] = family.label
                br_means["metric"] = metric
                br_means["metric_display"] = METRIC_SPECS[metric]["display"]
                all_br_means.append(br_means)
                ok, reason = validate_replicate_structure(br_means, family, metric, minimum_blocks=minimum_statistical_brs)
                if not ok:
                    warnings_list.append(f"Descriptive only for {species}, {family.label}, {metric}: {reason}")
                    continue
                anova = randomized_block_anova(br_means, family, metric, species, minimum_blocks=minimum_statistical_brs)
                anova_rows.append(anova.to_dict())
                tukey_rows.append(tukey_from_block(br_means, anova, family))
    return (
        pd.concat(all_per_well, ignore_index=True) if all_per_well else pd.DataFrame(),
        pd.concat(all_br_means, ignore_index=True) if all_br_means else pd.DataFrame(),
        pd.DataFrame.from_records(anova_rows, columns=ANOVA_COLUMNS),
        pd.concat(tukey_rows, ignore_index=True) if tukey_rows else pd.DataFrame(columns=TUKEY_COLUMNS),
        warnings_list,
    )


def make_summary_table(br_means: pd.DataFrame) -> pd.DataFrame:
    if br_means.empty:
        return pd.DataFrame()
    return (
        br_means.groupby(["species", "family_key", "family_label", "metric", "treatment", "treatment_order"], as_index=False)
        .agg(
            biological_mean=("technical_mean", "mean"),
            biological_SD=("technical_mean", lambda values: float(np.std(values, ddof=1)) if len(values) > 1 else np.nan),
            n_biological_replicates=("technical_mean", "count"),
            min_n_technical_wells=("n_technical_wells", "min"),
            max_n_technical_wells=("n_technical_wells", "max"),
        )
        .sort_values(["species", "family_key", "metric", "treatment_order"])
    )


def add_significance_bracket(ax, x1: int, x2: int, y: float, height: float, label: str) -> None:
    ax.plot([x1, x1, x2, x2], [y, y + height, y + height, y], color="black", linewidth=1.0, clip_on=False)
    ax.text(
        (x1 + x2) / 2,
        y + height,
        label,
        ha="center",
        va="bottom",
        fontsize=9,
        fontweight="bold",
        bbox={"facecolor": "white", "edgecolor": "none", "pad": 0.3},
        clip_on=False,
    )


def panel_metric_values(per_well: pd.DataFrame, br_means: pd.DataFrame, family: FamilySpec, metric: str, species: str) -> list[float]:
    labels = [group["label"] for group in family.groups]
    panel_per_well = per_well.loc[
        (per_well["family_key"] == family.key)
        & (per_well["metric"] == metric)
        & (per_well["species"] == species)
    ].copy()
    panel_br_means = br_means.loc[
        (br_means["family_key"] == family.key)
        & (br_means["metric"] == metric)
        & (br_means["species"] == species)
    ].copy()
    values = pd.to_numeric(panel_per_well["metric_value"], errors="coerce").dropna().to_list()
    for label in labels:
        replicate_means = panel_br_means.loc[panel_br_means["treatment"] == label, "technical_mean"].dropna()
        if replicate_means.empty:
            continue
        mean = float(replicate_means.mean())
        sd = float(np.std(replicate_means, ddof=1)) if len(replicate_means) > 1 else 0.0
        values.extend([mean - sd, mean + sd])
    return [float(value) for value in values]


def selected_pair_labels(tukey: pd.DataFrame, family: FamilySpec, metric: str, species: str) -> list[tuple[str, str, str]]:
    labels = [group["label"] for group in family.groups]
    selected = [pair for pair in family.selected_pairs if pair[0] in labels and pair[1] in labels]
    if tukey.empty:
        return []
    pair_lookup = {
        frozenset([row.group_1, row.group_2]): row.significance
        for row in tukey.loc[
            (tukey["species"] == species)
            & (tukey["family_key"] == family.key)
            & (tukey["metric"] == metric)
        ].itertuples(index=False)
    }
    return [
        (first, second, pair_lookup[frozenset([first, second])])
        for first, second in selected
        if frozenset([first, second]) in pair_lookup
    ]


def make_axis_config(values: list[float], metric: str, bracket_count: int) -> dict[str, float]:
    if values:
        y_min = min(values)
        y_max = max(values)
        minimum_span = {"auc_0_19h": 1.0, "lag_h": 0.5, "mu_max": 0.05}.get(metric, 1.0)
        span = max(y_max - y_min, abs(y_max) * 0.12, minimum_span)
    else:
        y_min, y_max, span = 0.0, 1.0, 1.0
    lower = min(0.0, y_min - 0.10 * span)
    bracket_step = 0.11 * span
    bracket_height = 0.025 * span
    bracket_base = y_max + 0.09 * span
    if bracket_count:
        upper = bracket_base + max(bracket_count - 1, 0) * bracket_step + bracket_height + 0.10 * span
    else:
        upper = y_max + 0.12 * span
    return {
        "lower": float(lower),
        "upper": float(upper),
        "bracket_base": float(bracket_base),
        "bracket_step": float(bracket_step),
        "bracket_height": float(bracket_height),
    }


def shared_axis_config(
    per_well: pd.DataFrame,
    br_means: pd.DataFrame,
    tukey: pd.DataFrame,
    panels: list[tuple[FamilySpec, str]],
    metric: str,
) -> dict[str, float]:
    values = []
    max_brackets = 0
    for family, species in panels:
        values.extend(panel_metric_values(per_well, br_means, family, metric, species))
        max_brackets = max(max_brackets, len(selected_pair_labels(tukey, family, metric, species)))
    return make_axis_config(values, metric, max_brackets)


def plot_panel(
    ax,
    per_well: pd.DataFrame,
    br_means: pd.DataFrame,
    tukey: pd.DataFrame,
    family: FamilySpec,
    metric: str,
    species: str,
    show_ylabel: bool = True,
    axis_config: dict[str, float] | None = None,
) -> None:
    labels = [group["label"] for group in family.groups]
    x_positions = np.arange(len(labels))
    all_y = []
    rng = np.random.default_rng(12345)
    panel_per_well = per_well.loc[
        (per_well["family_key"] == family.key)
        & (per_well["metric"] == metric)
        & (per_well["species"] == species)
    ].copy()
    panel_br_means = br_means.loc[
        (br_means["family_key"] == family.key)
        & (br_means["metric"] == metric)
        & (br_means["species"] == species)
    ].copy()
    for index, label in enumerate(labels):
        tech = panel_per_well.loc[panel_per_well["treatment"] == label]
        style = GROUP_STYLE[label]
        for replicate_id, replicate_rows in tech.groupby("biological_replicate"):
            values = pd.to_numeric(replicate_rows["metric_value"], errors="coerce").dropna().to_numpy(dtype=float)
            if len(values) == 0:
                continue
            offsets = np.linspace(-0.13, 0.13, len(values)) if len(values) > 1 else np.array([0.0])
            offsets = offsets + rng.normal(0, 0.006, size=len(offsets))
            ax.scatter(
                np.full(len(values), index) + offsets,
                values,
                s=28,
                marker=style["marker"],
                color=style["colour"],
                edgecolor="#1F2937",
                linewidth=0.55,
                alpha=0.62,
                zorder=2,
            )
            all_y.extend(values.tolist())
        replicate_means = panel_br_means.loc[panel_br_means["treatment"] == label, "technical_mean"].dropna()
        if not replicate_means.empty:
            mean = float(replicate_means.mean())
            sd = float(np.std(replicate_means, ddof=1)) if len(replicate_means) > 1 else 0.0
            ax.errorbar(index, mean, yerr=sd, fmt="none", color="black", linewidth=1.35, capsize=4.5, zorder=4)
            ax.hlines(mean, index - 0.24, index + 0.24, color="black", linewidth=2.0, zorder=5)
            all_y.extend([mean - sd, mean + sd])

    bracket_pairs = selected_pair_labels(tukey, family, metric, species)
    if axis_config is None:
        axis_config = make_axis_config(all_y, metric, len(bracket_pairs))
    for level, (first, second, label) in enumerate(bracket_pairs):
        add_significance_bracket(
            ax,
            labels.index(first),
            labels.index(second),
            axis_config["bracket_base"] + level * axis_config["bracket_step"],
            axis_config["bracket_height"],
            label,
        )
    ax.set_ylim(axis_config["lower"], axis_config["upper"])
    ax.set_xticks(x_positions)
    ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=8)
    ax.set_title(f"{SPECIES_DISPLAY[species]} - {family.label}", fontsize=10)
    if show_ylabel:
        ax.set_ylabel(METRIC_SPECS[metric]["ylabel"], fontsize=9)
    ax.tick_params(axis="y", labelleft=True)
    ax.grid(axis="y", color="#E5E7EB", linewidth=0.7)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def save_figure(fig: plt.Figure, output_base: Path) -> list[Path]:
    output_base.parent.mkdir(parents=True, exist_ok=True)
    png = output_base.with_suffix(".png")
    pdf = output_base.with_suffix(".pdf")
    fig.savefig(png, dpi=300, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    plt.close(fig)
    return [png, pdf]


def metric_phrase(metric: str) -> str:
    display = METRIC_SPECS[metric]["display"]
    return display if display == "AUC" else display.lower()


def plot_empty_liposome_figures(per_well: pd.DataFrame, br_means: pd.DataFrame, tukey: pd.DataFrame, output_dir: Path) -> list[Path]:
    paths = []
    family = next(f for f in family_specs() if f.key == "empty_liposomes")
    for metric in METRIC_SPECS:
        panels = [(family, species) for species in SPECIES_ORDER]
        axis_config = shared_axis_config(per_well, br_means, tukey, panels, metric)
        fig, axes = plt.subplots(1, 2, figsize=(9.4, 4.4), sharey=True)
        for ax, species in zip(axes, SPECIES_ORDER):
            plot_panel(ax, per_well, br_means, tukey, family, metric, species, show_ylabel=True, axis_config=axis_config)
        fig.suptitle(f"Do empty liposomes alter {metric_phrase(metric)}?", fontsize=12)
        fig.text(0.5, 0.02, "Treatment", ha="center", fontsize=9)
        fig.tight_layout(rect=(0, 0.05, 1, 0.94))
        paths.extend(save_figure(fig, output_dir / f"empty_liposomes_{metric}"))
    return paths


def plot_formulation_figures(per_well: pd.DataFrame, br_means: pd.DataFrame, tukey: pd.DataFrame, output_dir: Path) -> list[Path]:
    paths = []
    formulation_families = [f for f in family_specs() if f.key.startswith("formulation_")]
    for metric in METRIC_SPECS:
        panels = [(family, species) for species in SPECIES_ORDER for family in formulation_families]
        axis_config = shared_axis_config(per_well, br_means, tukey, panels, metric)
        fig, axes = plt.subplots(2, 2, figsize=(10.0, 8.4), sharey=True)
        for row_index, species in enumerate(SPECIES_ORDER):
            for col_index, family in enumerate(formulation_families):
                plot_panel(
                    axes[row_index, col_index],
                    per_well,
                    br_means,
                    tukey,
                    family,
                    metric,
                    species,
                    show_ylabel=col_index == 0,
                    axis_config=axis_config,
                )
        fig.suptitle(f"Loaded liposome formulation effects on {metric_phrase(metric)}", fontsize=12)
        fig.text(0.5, 0.02, "Treatment", ha="center", fontsize=9)
        fig.tight_layout(rect=(0, 0.05, 1, 0.95))
        paths.extend(save_figure(fig, output_dir / f"formulation_comparisons_{metric}"))
    return paths


def write_warning_file(output_dir: Path, warnings_list: Iterable[str]) -> Path:
    path = output_dir / "liposome_metric_plot_warnings.txt"
    text = "\n".join(warnings_list) if warnings_list else "No warnings."
    path.write_text(text + "\n", encoding="utf-8")
    return path


def run_analysis(
    workbooks: dict[str, Path],
    processed_pairs: dict[str, tuple[Path, Path]],
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    liposome_blank_source: str = "matching",
    minimum_statistical_brs: int = 3,
) -> dict:
    metric_tables = []
    gompertz_tables = []
    for replicate_id, workbook_path in workbooks.items():
        metrics, gompertz = calculate_workbook_metrics(workbook_path, replicate_id, liposome_blank_source=liposome_blank_source)
        metric_tables.append(metrics)
        gompertz_tables.append(gompertz)
    for replicate_id, (metrics_path, gompertz_path) in processed_pairs.items():
        metrics, gompertz = load_processed_metric_pair(metrics_path, gompertz_path, replicate_id)
        metric_tables.append(metrics)
        gompertz_tables.append(gompertz)
    if not metric_tables:
        raise LiposomeMetricError("Provide at least one --workbook or --processed-pair input.")
    metrics = pd.concat(metric_tables, ignore_index=True)
    gompertz = pd.concat(gompertz_tables, ignore_index=True)

    output_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = output_dir / "liposome_metrics_by_well.csv"
    gompertz_path = output_dir / "liposome_gompertz_by_well.csv"
    metrics.to_csv(metrics_path, index=False)
    gompertz.to_csv(gompertz_path, index=False)

    per_well, br_means, anova, tukey, warning_messages = build_analysis_tables(
        metrics,
        gompertz,
        minimum_statistical_brs=minimum_statistical_brs,
    )
    summary = make_summary_table(br_means)
    per_well_path = output_dir / "liposome_selected_metric_values_by_well.csv"
    br_path = output_dir / "liposome_biological_replicate_metric_means.csv"
    summary_path = output_dir / "liposome_hierarchical_summary.csv"
    anova_path = output_dir / "liposome_randomized_block_anova.csv"
    tukey_path = output_dir / "liposome_randomized_block_tukey_all_pairs.csv"
    per_well.to_csv(per_well_path, index=False)
    br_means.to_csv(br_path, index=False)
    summary.to_csv(summary_path, index=False)
    anova.to_csv(anova_path, index=False)
    tukey.to_csv(tukey_path, index=False)
    warning_path = write_warning_file(output_dir, warning_messages)

    figure_dir = output_dir / "figures"
    figure_paths = []
    figure_paths.extend(plot_empty_liposome_figures(per_well, br_means, tukey, figure_dir))
    figure_paths.extend(plot_formulation_figures(per_well, br_means, tukey, figure_dir))
    return {
        "metrics": metrics,
        "gompertz": gompertz,
        "per_well": per_well,
        "br_means": br_means,
        "summary": summary,
        "anova": anova,
        "tukey": tukey,
        "warnings": warning_messages,
        "paths": {
            "metrics": metrics_path,
            "gompertz": gompertz_path,
            "per_well": per_well_path,
            "br_means": br_path,
            "summary": summary_path,
            "anova": anova_path,
            "tukey": tukey_path,
            "warnings": warning_path,
            "figures": figure_paths,
        },
    }


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line interface for workbook and processed inputs."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workbook", action="append", default=[], type=parse_key_value_path, help="Biological replicate workbook as BR1=path. Repeat for BR2-BR4.")
    parser.add_argument("--processed-pair", action="append", default=[], help="Processed metrics and Gompertz CSVs as BR1=metrics.csv,gompertz.csv.")
    parser.add_argument(
        "--liposome-blank-source",
        choices=["matching", "pbs"],
        default="matching",
        help="Use matching blanks, or use the PBS blank trace for liposome-containing sample wells.",
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--minimum-statistical-brs",
        type=int,
        default=3,
        help="Minimum number of biological replicates required before randomized-block ANOVA/Tukey tests are run.",
    )
    return parser


def parse_processed_pairs(values: list[str]) -> dict[str, tuple[Path, Path]]:
    pairs = {}
    for value in values:
        if "=" not in value or "," not in value:
            raise argparse.ArgumentTypeError("Expected BR_ID=metrics.csv,gompertz.csv")
        replicate_id, paths = value.split("=", 1)
        metrics_path, gompertz_path = paths.split(",", 1)
        pairs[replicate_id.strip()] = (Path(metrics_path.strip().strip('"')), Path(gompertz_path.strip().strip('"')))
    return pairs


def main(argv: list[str] | None = None) -> int:
    """Command-line entry point for the liposome metric workflow."""
    args = build_parser().parse_args(argv)
    workbooks = dict(args.workbook)
    processed_pairs = parse_processed_pairs(args.processed_pair)
    result = run_analysis(
        workbooks=workbooks,
        processed_pairs=processed_pairs,
        output_dir=args.output_dir,
        liposome_blank_source=args.liposome_blank_source,
        minimum_statistical_brs=args.minimum_statistical_brs,
    )
    print("Liposome metric plotting complete.")
    print(f"- output: {args.output_dir.resolve()}")
    print(f"- liposome blank source: {args.liposome_blank_source}")
    print(f"- minimum statistical BRs: {args.minimum_statistical_brs}")
    print(f"- per-well selected rows: {len(result['per_well'])}")
    print(f"- biological-replicate means: {len(result['br_means'])}")
    print(f"- randomized-block ANOVA models fitted: {len(result['anova'])}")
    print(f"- Tukey all-pair comparisons: {len(result['tukey'])}")
    print(f"- warnings: {len(result['warnings'])}")
    print("- figures:")
    for path in result["paths"]["figures"]:
        print(f"  {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
