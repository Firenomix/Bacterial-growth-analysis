"""Randomized-block statistics for biological DMSO summaries."""

from __future__ import annotations

import itertools
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from scipy import stats


EXPECTED_BIOLOGICAL_REPLICATES = ["BR1", "BR2", "BR3"]
EXPECTED_SPECIES = ["B_subtilis", "E_coli"]
EXPECTED_DMSO_PERCENT = [0.0, 0.25, 0.5, 1.0, 2.0, 4.0]
INFERENTIAL_METRICS = [
    "AUC_0_17h_OD_h_mean",
    "AUC_0_12h_OD_h_mean",
    "Kz_OD600_per_h_mean",
    "TLag_h_mean",
]
DESCRIPTIVE_METRICS = INFERENTIAL_METRICS + ["A_OD600_mean"]
TECHNICAL_SD_PATTERN = re.compile(r"_SD_technical$")


class DMSOStatisticsValidationError(ValueError):
    """Raised when biological-summary data fails randomized-block validation."""


@dataclass(frozen=True)
class RandomizedBlockAnovaResult:
    """ANOVA quantities for a single species and metric."""

    species: str
    metric: str
    number_of_biological_replicates: int
    number_of_DMSO_conditions: int
    treatment_SS: float
    treatment_df: int
    treatment_MS: float
    treatment_F: float
    treatment_p_value: float
    block_SS: float
    block_df: int
    block_MS: float
    block_F: float
    block_p_value: float
    residual_SS: float
    residual_df: int
    residual_MS: float
    partial_eta_squared_treatment: float

    def to_dict(self) -> dict:
        return self.__dict__.copy()


def significance_label(p_value: float) -> str:
    """Return star labels from multiplicity-adjusted p-values."""

    if pd.isna(p_value) or p_value >= 0.05:
        return ""
    if p_value < 0.0001:
        return "****"
    if p_value < 0.001:
        return "***"
    if p_value < 0.01:
        return "**"
    return "*"


def _format_errors(errors: Sequence[str]) -> str:
    return "DMSO statistics validation failed:\n- " + "\n- ".join(errors)


def _normalise_species(value: object) -> str:
    text = str(value).strip()
    mapping = {
        "E. coli": "E_coli",
        "E. Coli": "E_coli",
        "E_coli": "E_coli",
        "B. subtilis": "B_subtilis",
        "B. Subtilis": "B_subtilis",
        "B_subtilis": "B_subtilis",
    }
    return mapping.get(text, text)


def _normalise_condition(value: object, dmso_percent: float) -> str:
    text = str(value).strip()
    if text and text.lower() != "nan":
        return text
    if dmso_percent == 0:
        return "Control"
    return f"DMSO {dmso_percent:g}%"


def read_biological_summary_workbook(
    input_path: Union[str, Path],
    sheet_name: Optional[str] = None,
) -> Tuple[pd.DataFrame, str, List[str]]:
    """Read the workbook sheet containing biological-replicate summaries.

    Repeated header rows and empty separator rows are removed explicitly because
    the supplied workbook is laid out as three pasted BR tables on one sheet.
    """

    input_path = Path(input_path)
    if not input_path.exists():
        raise FileNotFoundError(f"Workbook does not exist: {input_path}")

    sheets = pd.read_excel(input_path, sheet_name=None, dtype=object)
    required_columns = {
        "biological_replicate",
        "species",
        "condition",
        "DMSO_percent",
        *INFERENTIAL_METRICS,
    }
    candidates = []
    for name, sheet in sheets.items():
        columns = {str(column).strip() for column in sheet.columns}
        if required_columns.issubset(columns):
            candidates.append(name)

    if sheet_name:
        if sheet_name not in sheets:
            raise DMSOStatisticsValidationError(
                f"Requested sheet {sheet_name!r} was not found. Available sheets: {list(sheets)}"
            )
        if sheet_name not in candidates:
            raise DMSOStatisticsValidationError(
                f"Requested sheet {sheet_name!r} does not contain required columns: {sorted(required_columns)}"
            )
        selected_sheet = sheet_name
    elif len(candidates) == 1:
        selected_sheet = candidates[0]
    elif not candidates:
        raise DMSOStatisticsValidationError(
            f"No workbook sheet contains required columns: {sorted(required_columns)}"
        )
    else:
        raise DMSOStatisticsValidationError(
            f"Multiple workbook sheets contain required columns; specify one with --sheet-name: {candidates}"
        )

    raw = sheets[selected_sheet].copy()
    raw.columns = [str(column).strip() for column in raw.columns]
    raw = raw.dropna(how="all").copy()
    repeated_header_mask = raw["biological_replicate"].astype(str).str.strip().eq("biological_replicate")
    cleaned = raw[~repeated_header_mask].copy()
    warnings = []
    removed_rows = int(len(raw) - len(cleaned))
    if removed_rows:
        warnings.append(f"Removed {removed_rows} repeated header row(s) from workbook sheet {selected_sheet}.")

    return cleaned.reset_index(drop=True), selected_sheet, warnings


def validate_biological_summary_data(data: pd.DataFrame) -> pd.DataFrame:
    """Validate and standardize complete BR x species x DMSO summary data."""

    errors: List[str] = []
    required_columns = {
        "biological_replicate",
        "species",
        "condition",
        "DMSO_percent",
        *INFERENTIAL_METRICS,
    }
    missing_columns = required_columns.difference(data.columns)
    if missing_columns:
        errors.append(f"Missing required columns: {sorted(missing_columns)}")
        raise DMSOStatisticsValidationError(_format_errors(errors))

    if "well" in data.columns:
        nonempty_wells = data["well"].notna() & data["well"].astype(str).str.strip().ne("")
        if nonempty_wells.any():
            errors.append("Input appears to contain technical-well rows because a non-empty well column is present.")

    cleaned = data.copy()
    cleaned["biological_replicate"] = cleaned["biological_replicate"].astype(str).str.strip()
    cleaned["species"] = cleaned["species"].map(_normalise_species)
    cleaned["DMSO_percent"] = pd.to_numeric(cleaned["DMSO_percent"], errors="coerce")
    cleaned["condition"] = [
        _normalise_condition(condition, dmso)
        for condition, dmso in zip(cleaned["condition"], cleaned["DMSO_percent"])
    ]

    for metric in INFERENTIAL_METRICS:
        non_numeric_mask = pd.to_numeric(cleaned[metric], errors="coerce").isna()
        if non_numeric_mask.any():
            bad_values = sorted(
                cleaned.loc[non_numeric_mask, metric].astype(str).str.strip().drop_duplicates().tolist()
            )
            errors.append(f"Metric {metric} contains missing, placeholder, or nonnumeric values: {bad_values}")
        cleaned[metric] = pd.to_numeric(cleaned[metric], errors="coerce")

    for metric in DESCRIPTIVE_METRICS:
        if metric in cleaned.columns:
            cleaned[metric] = pd.to_numeric(cleaned[metric], errors="coerce")

    if cleaned["species"].eq("Blank").any() or cleaned["condition"].eq("Blank").any():
        errors.append("Blank rows must not be present in the biological-summary workbook.")

    observed_replicates = sorted(cleaned["biological_replicate"].dropna().unique().tolist())
    if observed_replicates != EXPECTED_BIOLOGICAL_REPLICATES:
        errors.append(
            f"Expected biological replicates {EXPECTED_BIOLOGICAL_REPLICATES}, found {observed_replicates}"
        )

    observed_species = sorted(cleaned["species"].dropna().unique().tolist())
    if observed_species != EXPECTED_SPECIES:
        errors.append(f"Expected species {EXPECTED_SPECIES}, found {observed_species}")

    observed_dmso = sorted(cleaned["DMSO_percent"].dropna().astype(float).unique().tolist())
    if observed_dmso != EXPECTED_DMSO_PERCENT:
        errors.append(f"Expected DMSO concentrations {EXPECTED_DMSO_PERCENT}, found {observed_dmso}")

    duplicates = cleaned.duplicated(["biological_replicate", "species", "DMSO_percent"], keep=False)
    if duplicates.any():
        duplicate_keys = (
            cleaned.loc[duplicates, ["biological_replicate", "species", "DMSO_percent"]]
            .drop_duplicates()
            .to_dict(orient="records")
        )
        errors.append(f"Duplicate biological replicate x species x DMSO rows found: {duplicate_keys}")

    expected_rows = (
        len(EXPECTED_BIOLOGICAL_REPLICATES)
        * len(EXPECTED_SPECIES)
        * len(EXPECTED_DMSO_PERCENT)
    )
    if len(cleaned) != expected_rows:
        errors.append(f"Expected {expected_rows} biological-summary rows, found {len(cleaned)}")

    expected_index = pd.MultiIndex.from_product(
        [EXPECTED_BIOLOGICAL_REPLICATES, EXPECTED_SPECIES, EXPECTED_DMSO_PERCENT],
        names=["biological_replicate", "species", "DMSO_percent"],
    )
    observed_index = pd.MultiIndex.from_frame(
        cleaned[["biological_replicate", "species", "DMSO_percent"]]
    )
    missing_design = expected_index.difference(observed_index)
    if len(missing_design):
        errors.append(
            "Missing randomized-block cells: "
            + str([tuple(item) for item in missing_design.tolist()])
        )

    if errors:
        raise DMSOStatisticsValidationError(_format_errors(errors))

    cleaned = cleaned.sort_values(
        ["biological_replicate", "species", "DMSO_percent"],
        key=lambda series: series.map(
            {value: index for index, value in enumerate(EXPECTED_BIOLOGICAL_REPLICATES)}
            if series.name == "biological_replicate"
            else {value: index for index, value in enumerate(EXPECTED_SPECIES)}
            if series.name == "species"
            else {value: index for index, value in enumerate(EXPECTED_DMSO_PERCENT)}
        ),
    ).reset_index(drop=True)

    return cleaned


def calculate_descriptive_statistics(
    data: pd.DataFrame,
    metrics: Optional[Sequence[str]] = None,
) -> pd.DataFrame:
    """Calculate biological-replicate descriptive statistics by species and DMSO."""

    metrics = [metric for metric in (metrics or DESCRIPTIVE_METRICS) if metric in data.columns]
    records = []
    for (species, dmso_percent), group in data.groupby(["species", "DMSO_percent"], sort=True):
        for metric in metrics:
            values = pd.to_numeric(group[metric], errors="coerce")
            records.append(
                {
                    "species": species,
                    "DMSO_percent": float(dmso_percent),
                    "metric": metric,
                    "n_biological_replicates": int(values.count()),
                    "mean": float(values.mean()),
                    "SD": float(values.std(ddof=1)),
                    "SEM": float(values.std(ddof=1) / math.sqrt(values.count())),
                    "minimum": float(values.min()),
                    "maximum": float(values.max()),
                }
            )
    return pd.DataFrame.from_records(records)


def randomized_block_anova(
    data: pd.DataFrame,
    species: str,
    metric: str,
) -> RandomizedBlockAnovaResult:
    """Fit and report additive randomized-block ANOVA for one species and metric."""

    subset = data[data["species"].eq(species)].copy()
    if subset.empty:
        raise DMSOStatisticsValidationError(f"No rows available for species {species!r}")
    if metric not in subset.columns:
        raise DMSOStatisticsValidationError(f"Metric column {metric!r} is missing.")

    analysis = subset[["biological_replicate", "DMSO_percent", metric]].copy()
    analysis = analysis.rename(columns={metric: "value"})
    analysis["value"] = pd.to_numeric(analysis["value"], errors="raise")
    analysis["DMSO_concentration"] = pd.Categorical(
        analysis["DMSO_percent"].astype(float),
        categories=EXPECTED_DMSO_PERCENT,
        ordered=True,
    )
    analysis["biological_replicate"] = pd.Categorical(
        analysis["biological_replicate"],
        categories=EXPECTED_BIOLOGICAL_REPLICATES,
        ordered=True,
    )

    # The fitted model is retained to make the design explicit; the balanced
    # randomized-block sums of squares below are equivalent and easier to audit.
    smf.ols("value ~ C(DMSO_concentration) + C(biological_replicate)", data=analysis).fit()

    a = len(EXPECTED_DMSO_PERCENT)
    b = len(EXPECTED_BIOLOGICAL_REPLICATES)
    grand_mean = float(analysis["value"].mean())
    treatment_means = analysis.groupby("DMSO_concentration", observed=False)["value"].mean()
    block_means = analysis.groupby("biological_replicate", observed=False)["value"].mean()

    total_ss = float(((analysis["value"] - grand_mean) ** 2).sum())
    treatment_ss = float(b * ((treatment_means - grand_mean) ** 2).sum())
    block_ss = float(a * ((block_means - grand_mean) ** 2).sum())
    residual_ss = float(total_ss - treatment_ss - block_ss)
    if abs(residual_ss) < 1e-12:
        residual_ss = 0.0

    treatment_df = a - 1
    block_df = b - 1
    residual_df = (a - 1) * (b - 1)
    treatment_ms = treatment_ss / treatment_df
    block_ms = block_ss / block_df
    residual_ms = residual_ss / residual_df
    treatment_f = treatment_ms / residual_ms if residual_ms > 0 else math.inf
    block_f = block_ms / residual_ms if residual_ms > 0 else math.inf
    treatment_p = float(stats.f.sf(treatment_f, treatment_df, residual_df))
    block_p = float(stats.f.sf(block_f, block_df, residual_df))
    eta = treatment_ss / (treatment_ss + residual_ss) if (treatment_ss + residual_ss) else math.nan

    return RandomizedBlockAnovaResult(
        species=species,
        metric=metric,
        number_of_biological_replicates=b,
        number_of_DMSO_conditions=a,
        treatment_SS=treatment_ss,
        treatment_df=treatment_df,
        treatment_MS=treatment_ms,
        treatment_F=treatment_f,
        treatment_p_value=treatment_p,
        block_SS=block_ss,
        block_df=block_df,
        block_MS=block_ms,
        block_F=block_f,
        block_p_value=block_p,
        residual_SS=residual_ss,
        residual_df=residual_df,
        residual_MS=residual_ms,
        partial_eta_squared_treatment=eta,
    )


def run_all_randomized_block_anovas(
    data: pd.DataFrame,
    metrics: Optional[Sequence[str]] = None,
) -> pd.DataFrame:
    """Run randomized-block ANOVA for every species and metric."""

    metrics = list(metrics or INFERENTIAL_METRICS)
    rows = [
        randomized_block_anova(data, species, metric).to_dict()
        for species in EXPECTED_SPECIES
        for metric in metrics
    ]
    return pd.DataFrame.from_records(rows)


def randomized_block_tukey_hsd(
    data: pd.DataFrame,
    anova_result: RandomizedBlockAnovaResult,
) -> pd.DataFrame:
    """Run randomized-block Tukey HSD using the ANOVA residual MSE."""

    species = anova_result.species
    metric = anova_result.metric
    subset = data[data["species"].eq(species)].copy()
    treatment_means = (
        subset.groupby("DMSO_percent", observed=False)[metric]
        .mean()
        .reindex(EXPECTED_DMSO_PERCENT)
    )
    a = anova_result.number_of_DMSO_conditions
    b = anova_result.number_of_biological_replicates
    mse = anova_result.residual_MS
    residual_df = anova_result.residual_df
    tukey_se = math.sqrt(mse / b)
    q_critical = float(stats.studentized_range.ppf(0.95, a, residual_df))

    rows = []
    for group_1, group_2 in itertools.combinations(EXPECTED_DMSO_PERCENT, 2):
        mean_1 = float(treatment_means.loc[group_1])
        mean_2 = float(treatment_means.loc[group_2])
        difference = mean_2 - mean_1
        q_value = abs(difference) / tukey_se if tukey_se > 0 else math.inf
        adjusted_p = float(stats.studentized_range.sf(q_value, a, residual_df))
        ci_half_width = q_critical * tukey_se
        rows.append(
            {
                "species": species,
                "metric": metric,
                "DMSO_group_1": float(group_1),
                "DMSO_group_2": float(group_2),
                "mean_group_1": mean_1,
                "mean_group_2": mean_2,
                "mean_difference_group_2_minus_group_1": difference,
                "Tukey_q": float(q_value),
                "residual_df": int(residual_df),
                "adjusted_p_value": adjusted_p,
                "simultaneous_95CI_lower": float(difference - ci_half_width),
                "simultaneous_95CI_upper": float(difference + ci_half_width),
                "statistically_significant": bool(adjusted_p < 0.05),
                "significance_label": significance_label(adjusted_p),
            }
        )

    return pd.DataFrame.from_records(rows)


def run_all_randomized_block_tukey(
    data: pd.DataFrame,
    anova_results: pd.DataFrame,
) -> pd.DataFrame:
    """Run Tukey all-pairwise tests for all ANOVA rows."""

    rows = []
    for result in anova_results.to_dict(orient="records"):
        anova_result = RandomizedBlockAnovaResult(**result)
        rows.append(randomized_block_tukey_hsd(data, anova_result))
    return pd.concat(rows, ignore_index=True)


def detect_value_warnings(data: pd.DataFrame) -> List[str]:
    """Return non-fatal warnings for values that look inconsistent in scale."""

    warnings: List[str] = []
    for metric in INFERENTIAL_METRICS:
        grouped = data.groupby(["species", "biological_replicate"])[metric]
        for (species, replicate), values in grouped:
            median = float(values.median())
            if median == 0:
                continue
            large = values[values.abs() > abs(median) * 5]
            for row_index, value in large.items():
                row = data.loc[row_index]
                warnings.append(
                    f"{replicate} {species} {row['DMSO_percent']:g}% {metric}={value:g} "
                    f"is more than 5x that replicate/species median; retained unchanged."
                )
    return warnings
