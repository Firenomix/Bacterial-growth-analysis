# Bacterial Growth Analysis

Reproducible Python workflows for importing, validating, blank-correcting,
quantifying, statistically analysing, and plotting bacterial OD600 growth
experiments. Raw instrument exports remain outside the repository and are
supplied explicitly on the command line.

## Repository layout and provenance

The repository preserves the analysis hierarchy used in the thesis:

| Path | Role |
| --- | --- |
| `src/growth_analysis.py` | Shared parsing, validation, blank correction, AUC, modified-Gompertz fitting, and QC plotting functions. |
| `analyses/dmso_tolerance.py` | Processes and quantifies one DMSO-tolerance biological replicate; it does not perform cross-replicate inference. |
| `analyses/drug_screening_single_run.py` | Processes one species-specific drug-screening workbook and averages technical wells within that run. |
| `analyses/combine_drug_screening_growth_curves.py` | Combines already corrected BR1-BR3 curves descriptively, preserving hierarchical averaging. |
| `analyses/drug_screening_statistics.py` | Runs randomized-block ANOVA and blocked-residual Tukey comparisons on biological-replicate means. |
| `analyses/drug_screening_thesis_figures.py` | Generates thesis-ready monotherapy figures and their figure-specific audit tables. |
| `analyses/plot_combination_treatment_descriptive.py` | Plots the matched BR2/BR3 combination regimen descriptively; it performs no inferential test. |
| `analyses/liposome_metric_plots.py` | Imports or loads liposome metrics, applies the selected blank rule, and runs blocked inference when enough BRs are available. |
| `tests/` | Unit and integration tests for validation, calculations, blocked inference, and plotting contracts. |
| `results/` | Generated tables and figures. These are derived outputs, not raw data. |

### Original-to-repository filename map

The supplied scripts did not need upload-suffix removal; their descriptive
filenames were retained and organized by responsibility:

| Original supplied filename | Final repository path |
| --- | --- |
| `growth_analysis.py` | `src/growth_analysis.py` |
| `dmso_tolerance.py` | `analyses/dmso_tolerance.py` |
| `drug_screening_single_run.py` | `analyses/drug_screening_single_run.py` |
| `combine_drug_screening_growth_curves.py` | `analyses/combine_drug_screening_growth_curves.py` |
| `drug_screening_statistics.py` | `analyses/drug_screening_statistics.py` |
| `drug_screening_thesis_figures.py` | `analyses/drug_screening_thesis_figures.py` |
| `plot_combination_treatment_descriptive.py` | `analyses/plot_combination_treatment_descriptive.py` |
| `liposome_metric_plots.py` | `analyses/liposome_metric_plots.py` |

### Installation

Python 3.9 or later is recommended. Install the declared dependencies from the
repository root:

```powershell
python -m pip install -r requirements.txt
```

### Execution order

The workflows are related but independent; DMSO, drug-screening, and liposome
datasets are not combined with one another.

1. DMSO: run `dmso_tolerance.py` once for each input biological replicate. Run
   the separate DMSO cross-replicate statistics stage only after the relevant
   replicate summaries exist.
2. Drug screening: run `drug_screening_single_run.py` for each species and BR.
3. Drug-screening curves: run `combine_drug_screening_growth_curves.py`
   separately for each species after BR1-BR3 corrected CSVs exist.
4. Drug-screening inference: run `drug_screening_statistics.py` after all six
   species-by-BR run directories exist.
5. Drug-screening presentation: run `drug_screening_thesis_figures.py` from the
   same six run directories. Optionally run
   `plot_combination_treatment_descriptive.py` on the descriptive metrics table
   produced by the statistics stage.
6. Liposomes: run `liposome_metric_plots.py` directly from labelled workbooks
   or from its paired processed metric files. This workflow is independent of
   the drug-screening stages.

The first analysis examines the effect of different DMSO concentrations on the
growth of *Escherichia coli* K-12 MG1655 and *Bacillus subtilis* ATCC 6051.

## DMSO-tolerance raw import and QC

This repository keeps analysis code separate from raw instrument exports. The
annotated BMG CLARIOstar CSV is read from a command-line path and is not
modified.

Run the import, validation, QC-flagging, blank-correction, and growth-curve
plotting stages:

```powershell
python analyses/dmso_tolerance.py --input "path\to\dmso_tolerance_annotated.csv.csv" --replicate-id BR1
```

By default, outputs are written under the biological-replicate folder, for
example `results/dmso_tolerance/BR1/`.

- `processed_growth_data.csv`: tidy long-form OD600 measurements.
- `quality_control_summary.csv`: validation summary and QC flag counts.
- `quality_control_flags.csv`: OD600 values greater than or equal to 1 and
  unusually large consecutive-reading changes.
- `individual_raw_growth_curves_E_coli.png`
- `individual_raw_growth_curves_B_subtilis.png`
- `mean_sd_growth_curves_E_coli.png`
- `mean_sd_growth_curves_B_subtilis.png`

The tidy processed data contains `experiment_id`, `well`, `species`,
`condition_label`, `dmso_percent`, `technical_replicate`, `time_min`, `od600`,
`original_content`, and `original_group`. Original `Content` and `Group` labels
are retained for traceability, while `Group` is standardized internally as
`E_coli` and `B_subtilis`.

## Blank correction

Blank wells are identified from rows where `Content` is `Blank` and `Group` is
`Blank`. The expected blank wells are A1, A2, and A3. A1 and A3 are averaged at
each individual time point to create the blank trace used for correction. A2 is
retained in blank QC outputs but excluded from the blank trace because of:
`Abnormal upward blank drift compared with A1 and A3.`

For each bacterial sample well and time point:

```text
blank-corrected OD600 = raw OD600 - time-matched mean blank OD600
```

The correction uses the corresponding blank value at each time point rather
than a single constant. Raw OD600 values are retained, negative corrected values
are preserved, and the original raw input CSV remains unchanged.

Additional blank-corrected outputs are written to the same run folder:

- `blank_qc_summary.csv`
- `blank_trace_used.csv`
- `blank_corrected_growth_data.csv`
- `blank_well_qc.png`
- `individual_blank_corrected_growth_curves_E_coli.png`
- `individual_blank_corrected_growth_curves_B_subtilis.png`
- `mean_sd_blank_corrected_growth_curves_E_coli.png`
- `mean_sd_blank_corrected_growth_curves_B_subtilis.png`

The blank-correction step does not time-zero-correct, smooth, clip, delete, or
exclude bacterial measurements.

## AUC and Gompertz metrics

After blank correction, the BR-level quantification uses `corrected_OD600` and
`time_h`. Blank correction is not applied a second time.

Two trapezoidal AUC windows are calculated from observed blank-corrected OD600:

- `AUC_0_17h_OD_h`: primary complete-experiment AUC, reported in OD600.h.
- `AUC_0_12h_OD_h`: secondary early-growth AUC, reported in OD600.h.

Relative AUC values are presentation aids only. They are calculated separately
within each biological replicate, species, and AUC window as 100 times the
individual well AUC divided by the mean AUC of the matching five 0% DMSO wells.
The unnormalised AUC values remain the primary values for later analysis.

Each bacterial well is fitted independently with the exact modified Gompertz
equation:

```text
W(t) = A exp[-exp((e Kz / A) (TLag - t) + 1)]
```

where:

- `W(t) = corrected_OD600(t) - corrected_OD600(0)` for the same well.
- `Kz_OD600_per_h` is the absolute maximum OD600 growth rate at the inflection
  point, in OD600.h^-1. It is not a specific growth rate.
- `TLag_h` is the model-derived lag time, in hours.
- `A_OD600` is the fitted upper asymptote of the change in OD600.

The five technical wells for each species and DMSO condition are averaged into
one biological-replicate result. No inferential statistics are performed at this
stage.

The BR1 command is:

```powershell
python analyses/dmso_tolerance.py --input "path\to\dmso_tolerance_annotated.csv.csv" --replicate-id BR1
```

Future BR2 and BR3 datasets can be analysed independently with the same command
by changing only `--input` and `--replicate-id`:

```powershell
python analyses/dmso_tolerance.py --input "path\to\BR2_dmso_tolerance_annotated.csv.csv" --replicate-id BR2
python analyses/dmso_tolerance.py --input "path\to\BR3_dmso_tolerance_annotated.csv.csv" --replicate-id BR3
```

## DMSO-only dataset mode

The smaller `DMSO only.csv` export can be analysed with the same pipeline using
the DMSO-only validation profile. This profile expects:

- 34 bacterial sample wells.
- 200 time points from 0 to 16 h 35 min.
- Control, 0.25%, 0.5%, 1%, and 2% DMSO.
- Five control wells per species and three technical wells per species for each
  DMSO-treated condition.

Because this export does not contain a 17 h measurement, the complete-experiment
AUC is labelled `AUC_0_16h35min_OD_h`. The secondary early-growth AUC remains
`AUC_0_12h_OD_h`.

Run it with:

```powershell
python analyses/dmso_tolerance.py --input "path\to\DMSO only.csv" --replicate-id DMSO_only_run2 --dataset-mode dmso-only
```

If `--dataset-mode` is omitted, `auto` mode will detect this layout and use the
DMSO-only profile. You can still force the full 60-well profile with
`--dataset-mode full`.

## Drug-screening single-run workflow

The drug-screening workbooks remain as raw Excel files outside the repository.
Each command processes exactly one biological-replicate workbook and uses the
user-supplied species and biological-replicate labels to construct the run ID.
Instrument metadata is retained for record-keeping only and is not used to
decide the replicate identity.

Run one workbook with:

```powershell
python analyses/drug_screening_single_run.py --input "path\to\workbook.xlsx" --species E_coli --biological-replicate BR1 --max-time-min 1045
```

Accepted species are `E_coli` and `B_subtilis`. Accepted biological-replicate
identifiers are `BR1`, `BR2`, and `BR3`. The run ID is built automatically, for
example `E_coli_BR1` or `B_subtilis_BR3`. Outputs are written to:

```text
results/drug_screening/<species>/<run_id>/
```

The analysis interval is 0 to 17 h 25 min inclusive (`time_min <= 1045`),
which gives 210 retained 5-minute time points per well. Workbooks with later
measurements are trimmed before blank correction, plotting, AUC calculation, or
model fitting. No interpolation or extrapolation is performed.

Blank wells are identified only where `Content = Blank` and `Group = Blank`.
The expected blank wells are G4, G5, and G6. At each retained time point, the
blank trace is the mean OD600 of G4, G5, and G6. For every nonblank well:

```text
blank-corrected OD600 = raw OD600 - time-matched mean blank OD600
```

Raw OD600 values are preserved, negative corrected values are retained, and
the 1% DMSO vehicle-control wells are not used as blanks.

Treatment annotations are standardised into treatment type, compound,
compound concentration in uM, ampicillin concentration in ug/mL, and
`condition_id`. The single-run validator requires the 1% DMSO vehicle control
and two positive-control ampicillin concentrations, but it does not require the
same ampicillin concentrations in every biological replicate. BR-specific
ampicillin controls, such as 2, 4, or 8 ug/mL, are preserved as separate
conditions. Combination regimens are also kept distinct. For example, BR1
conditions using 12.5 uM compound plus 2 ug/mL ampicillin are not pooled with
BR2/BR3 conditions using 25 uM compound plus 4 ug/mL ampicillin.

Per-run outputs are:

- `run_summary.csv`
- `blank_qc_summary.csv`
- `blank_trace_used.csv`
- `quality_control_flags.csv`
- `blank_corrected_growth_data.csv`
- `technical_well_growth_metrics.csv`
- `biological_replicate_summary.csv`
- `blank_well_qc.png/.pdf`
- `individual_blank_corrected_growth_curves.png/.pdf`
- `mean_sd_blank_corrected_growth_curves.png/.pdf`
- `gompertz_fits.png/.pdf`

Metrics are calculated separately for every nonblank technical well before
averaging. AUC is calculated from observed blank-corrected OD600 over 0 to
17 h 25 min. Relative AUC is normalised to the mean AUC of the three 1% DMSO
vehicle-control wells in the same workbook, so the vehicle-control mean
relative AUC is 100%. The existing modified-Gompertz implementation is reused
with `W(t) = corrected_OD600(t) - corrected_OD600(t = 0)`.

The biological-replicate summary averages the three technical wells within
each treatment condition. These per-run summary files are intended for the
later combined BR1-BR3 statistical script. Later statistical graphs can display
all technical-well points, but inferential analysis must use the biological
replicate means (`n = 3`) and must only combine conditions with the same
treatment definition across biological replicates.

No inferential statistics, p-values, or significance annotations are generated
by `drug_screening_single_run.py`.

## Drug-screening combined growth curves

After the three single-run outputs have been generated for a species, combine
the blank-corrected growth curves descriptively with:

```powershell
python analyses/combine_drug_screening_growth_curves.py --input-br1 "results\drug_screening\E_coli\E_coli_BR1\blank_corrected_growth_data.csv" --input-br2 "results\drug_screening\E_coli\E_coli_BR2\blank_corrected_growth_data.csv" --input-br3 "results\drug_screening\E_coli\E_coli_BR3\blank_corrected_growth_data.csv" --species E_coli
```

Run the same command separately for `B_subtilis`, changing the three input
paths and `--species`. By default the outputs are written to:

```text
results/drug_screening/<species>/combined_biological_replicates/
```

The script reads the three `blank_corrected_growth_data.csv` files; it does not
combine existing PNG images. It validates that each input belongs to the
requested species, that BR1, BR2, and BR3 are present exactly once, that all
files contain the same 210 time points from 0 to 1045 minutes, and that each
condition has three technical wells within each biological replicate.

For matched conditions, averaging is hierarchical:

```text
biological-replicate mean = mean of the three technical wells within one BR
combined mean = mean of the three biological-replicate means
combined SD = sample SD across the three biological-replicate means
```

The combined SD is therefore between-biological-replicate variation with
`n = 3`. The nine technical wells are not pooled, within-run technical SDs are
not averaged, and SEM is not used.

The matched-condition figure includes only conditions that are identical in
BR1, BR2, and BR3: the 1% DMSO vehicle control, any ampicillin controls present
with the same concentration in all three biological replicates, C1-C4
monotherapy concentrations, and PenAg monotherapy concentrations. Ampicillin
controls whose concentrations differ between biological replicates are retained
in `condition_matching_summary.csv` but are not averaged across unmatched
conditions. The compound-ampicillin combination treatments are also not
averaged because BR1 used 12.5 uM compound plus 2 ug/mL ampicillin, while BR2
and BR3 used 25 uM compound plus 4 ug/mL ampicillin. Those unmatched
combination regimens are shown as separate run-level curves. The same
run-level figure also includes an ampicillin positive-control panel for
reference, preserving BR-specific positive-control concentrations such as 2,
4, and 8 ug/mL.

Combined-growth outputs are:

- `condition_matching_summary.csv`
- `biological_replicate_mean_growth_curves.csv`
- `combined_mean_sd_growth_curves.csv`
- `combination_treatment_run_level_growth_curves.csv`
- `combined_mean_sd_blank_corrected_growth_curves_<species>.png/.pdf`
- `combination_treatment_run_level_growth_curves_<species>.png/.pdf`

This combined-growth script is descriptive only. It does not perform
inferential statistics, p-values, or significance annotations.

## Drug-screening statistical analysis

After all six single-run folders have been generated, run the drug-screening
statistical analysis with:

```powershell
python analyses/drug_screening_statistics.py `
  --e-coli-br1 "results\drug_screening\E_coli\E_coli_BR1" `
  --e-coli-br2 "results\drug_screening\E_coli\E_coli_BR2" `
  --e-coli-br3 "results\drug_screening\E_coli\E_coli_BR3" `
  --b-subtilis-br1 "results\drug_screening\B_subtilis\B_subtilis_BR1" `
  --b-subtilis-br2 "results\drug_screening\B_subtilis\B_subtilis_BR2" `
  --b-subtilis-br3 "results\drug_screening\B_subtilis\B_subtilis_BR3"
```

By default, outputs are saved to:

```text
results/drug_screening/statistics/
```

The script reads each run's `blank_corrected_growth_data.csv`,
`technical_well_growth_metrics.csv`, `biological_replicate_summary.csv`, and
`run_summary.csv`. It validates species and biological-replicate labels from
the tables themselves, not only from paths.

The primary endpoint is raw AUC over 0 to 1045 min, equivalent to 0 to 17 h
25 min. Raw AUC is used for inference because relative AUC normalises the
vehicle-control mean within each run to 100%. Relative AUC is used for figure
presentation. The normalised vehicle biological-replicate means therefore have
zero between-run SD, which is expected and not a validation failure.

Technical wells are analysed at the technical-well stage, then averaged within
each biological run. Inferential statistics use one biological-replicate mean
per condition per run, so the statistical sample size is `n = 3`, not `n = 9`.
The plotted overall mean is the mean of BR1, BR2, and BR3 means. The plotted SD
is the sample SD across those three biological-replicate means with `ddof=1`;
technical wells are not pooled across runs to inflate residual degrees of
freedom.

For each eligible species-compound-metric family, the model is:

```text
metric ~ condition + biological_replicate
```

This is a randomised-complete-block one-way ANOVA with biological replicate as
the fixed block. No condition-by-block interaction is fitted because each
condition-by-block cell contains one biological-replicate mean. Schiff-base
models contain 18 observations and 10 residual degrees of freedom. PenAg
models contain 15 observations and 8 residual degrees of freedom.

Tukey comparisons use the residual mean square and residual degrees of freedom
from the blocked model:

```text
Tukey SE = sqrt(residual MS / 3 biological replicates)
```

Ordinary independent-groups Tukey HSD is not used because it would ignore the
matched BR1-BR3 block structure. Compact letters are calculated from the full
Tukey-adjusted pairwise comparison matrix. Conditions sharing a letter are not
significantly different; conditions with no letter in common are significantly
different at adjusted `p < 0.05`. On relative-AUC figures, the compact letters
come from the raw-AUC statistical model.

Kz and TLag values come from the existing modified-Gompertz fits. Failed or
non-interpretable fits are not replaced with zero. A Kz or TLag family is only
tested when all condition-by-block means are complete and based on valid fits;
otherwise the available values are shown descriptively and the reason is
reported in `statistical_analysis_exclusions.csv`.

Ampicillin controls are retained in the combined tables but are not included in
the C1-C4 or PenAg monotherapy concentration-response models. Combination
treatments are analysed descriptively by run because the dose changed between
BR1 and BR2-BR3. The four within-run comparison conditions are present where
available: vehicle, matched compound alone, matched ampicillin alone, and
compound plus ampicillin. The limitation is not absence of the four conditions;
it is that BR1 used 12.5 uM compound plus 2 ug/mL ampicillin, whereas BR2 and
BR3 used 25 uM compound plus 4 ug/mL ampicillin. The current combination
analysis can describe possible attenuation of ampicillin activity, but it
cannot confirm antagonism statistically.

The combination-minus-ampicillin table reports the difference between condition
means:

```text
combination mean relative AUC - matched ampicillin-alone mean relative AUC
```

A positive value means the combination allowed more growth than ampicillin
alone, consistent with possible attenuation of ampicillin activity. A negative
value means the combination inhibited growth more than ampicillin alone.

OD600 linearity QC is based primarily on raw instrument OD600 values. Raw
measurements at or above 1.0 are flagged and preserved; they are not capped,
removed, or used to shorten/recalculate the AUC interval. Corrected OD600
values at or above 1.0 are counted descriptively only.

Main outputs include:

- `combined_technical_well_metrics.csv`
- `combined_biological_replicate_metrics.csv`
- `randomised_block_anova_results.csv`
- `tukey_block_pairwise_results.csv`
- `compact_letter_display.csv`
- `compound_selection_relative_auc_summary.csv`
- `fit_availability_summary.csv`
- `od600_linearity_qc_summary.csv`
- `statistical_analysis_exclusions.csv`
- `combination_treatment_descriptive_metrics.csv`
- `combination_treatment_delta_vs_ampicillin.csv`
- `statistical_model_diagnostics_summary.csv`
- grouped-dot statistical figures, combination descriptive figures, a
  compound-selection heatmap, and residual diagnostic plots.

## Drug-screening thesis figures

After the six single-run folders have been generated, create the final
monotherapy thesis figures with:

```powershell
python analyses/drug_screening_thesis_figures.py `
  --e-coli-br1 "results\drug_screening\E_coli\E_coli_BR1" `
  --e-coli-br2 "results\drug_screening\E_coli\E_coli_BR2" `
  --e-coli-br3 "results\drug_screening\E_coli\E_coli_BR3" `
  --b-subtilis-br1 "results\drug_screening\B_subtilis\B_subtilis_BR1" `
  --b-subtilis-br2 "results\drug_screening\B_subtilis\B_subtilis_BR2" `
  --b-subtilis-br3 "results\drug_screening\B_subtilis\B_subtilis_BR3"
```

By default, outputs are saved to:

```text
results/drug_screening/statistics/thesis_figures/
```

The script creates ten PNG/PDF figure pairs:

- `schiff_base_100uM_growth_curves_E_coli`
- `schiff_base_100uM_growth_curves_B_subtilis`
- `schiff_base_relative_auc_all_concentrations_E_coli`
- `schiff_base_relative_auc_all_concentrations_B_subtilis`
- `penag_growth_curves_all_concentrations_E_coli`
- `penag_growth_curves_all_concentrations_B_subtilis`
- `penag_relative_auc_E_coli`
- `penag_relative_auc_B_subtilis`
- `penag_lag_time_E_coli`
- `penag_lag_time_B_subtilis`

Growth curves are averaged hierarchically: the three technical wells are first
averaged within each biological replicate at every time point, then the plotted
line is the mean of the BR1-BR3 means and the ribbon is the sample SD across
those three biological-replicate means (`ddof=1`). No significance brackets are
placed on growth curves.

Relative AUC figures show `relative_AUC_percent` for presentation, but
statistical inference uses raw AUC over the complete 0-1045 min interval. This
keeps the randomised-block ANOVA from being run on the normalised vehicle
response, which is fixed at 100% within each run. The inferential unit is the
biological-replicate mean (`n = 3`), not the nine technical wells.

The thesis figures use the existing randomised-block model:

```text
metric ~ condition + biological_replicate
```

Tukey-adjusted all-pair comparisons are calculated from the block-model
residual mean square and residual degrees of freedom. The figures then display
only vehicle-versus-treatment brackets. This keeps the complete Tukey family
adjustment and is more conservative than a control-specific Dunnett test. The
long Schiff-base relative-AUC figures show only significant vehicle brackets;
unmarked eligible comparisons were not significant after Tukey adjustment.
PenAg relative-AUC and lag-time figures show all planned vehicle brackets,
including `n.s.` where applicable. Compact-letter displays are not used in
these thesis figures.

PenAg lag-time figures and models include only 1% DMSO, 12.5 uM PenAg and
25 uM PenAg. PenAg 50 and 100 uM are excluded from TLag analysis because
strongly suppressed or non-growing curves do not provide a biologically
interpretable lag time. Failed fits are not replaced with zero. Combination
treatments are outside the scope of this thesis-figure script.

The thesis-figure tables are:

- `thesis_figure_technical_well_data.csv`
- `thesis_figure_biological_replicate_data.csv`
- `randomised_block_anova_thesis_models.csv`
- `thesis_figure_vehicle_tukey_comparisons.csv`
- `penag_lag_fit_availability.csv`
- `thesis_figure_manifest.csv`

## Combination-treatment descriptive figures

This optional presentation stage reads the long-format
`combination_treatment_descriptive_metrics.csv` created by
`drug_screening_statistics.py`. It is restricted to the matched BR2/BR3 regimen
of 25 uM compound plus 4 ug/mL ampicillin and the corresponding 4 ug/mL
ampicillin-only control.

```powershell
python analyses/plot_combination_treatment_descriptive.py `
  --input "results\drug_screening\statistics\combination_treatment_descriptive_metrics.csv" `
  --output-dir "results\drug_screening\statistics\combination_treatment_descriptive_plots"
```

The plotted value is `technical_mean`, which is already the mean of the three
technical wells within one biological replicate. `technical_SD` is not used to
reconstruct observations. Repeated contextual rows for the ampicillin-only
control must agree within strict numerical tolerance before one copy is
retained. BR2 and BR3 are displayed as separate points; a short mean line is
drawn only when both values are available. This script performs no statistical
test and adds no significance annotations.

Outputs include plot-ready long and wide QC tables, unit-conversion notes,
warnings, and individual/combined PNG and PDF figures.

## Liposome metric workflow

`liposome_metric_plots.py` is a separate workflow for the annotated liposome
experiments. It can calculate per-well AUC and modified-Gompertz metrics from
raw workbooks or consume paired processed metric files. Technical wells are
averaged within each biological replicate before any inference.

Example using raw workbooks:

```powershell
python analyses/liposome_metric_plots.py `
  --workbook "BR1=path\to\BR1.xlsx" `
  --workbook "BR2=path\to\BR2.xlsx" `
  --workbook "BR3=path\to\BR3.xlsx" `
  --liposome-blank-source pbs `
  --output-dir "results\liposome_metric_plots"
```

`--liposome-blank-source matching` uses the condition-matched blank rules;
`pbs` uses the time-matched PBS blank mean for liposome-containing wells. The
selected rule is recorded in the outputs. Raw OD values are retained, and
missing or failed Gompertz fits are not replaced with zero.

For each eligible species, treatment family, and metric, inference uses the
biological-replicate means in the model:

```text
metric ~ treatment + biological_replicate
```

Tukey all-pair comparisons use the residual mean square and degrees of freedom
from that blocked model. The default minimum is three biological replicates.
`--minimum-statistical-brs 2` is available only as an explicit exploratory
override and should be reported with its low residual degrees of freedom. If
fewer than the selected minimum are available, descriptive outputs are still
created and the reason for omitting inference is written to the warning file.

Main outputs are per-well metrics, biological-replicate means, hierarchical
summaries, randomized-block ANOVA and Tukey tables, a warnings text file, and
publication-quality PNG/PDF metric figures under the selected output directory.

## BR1-BR3 randomized-block statistics

The workbook-level statistics stage uses the completed biological-replicate
summary workbook, where technical wells have already been averaged within each
biological experiment. The statistical units are BR1, BR2, and BR3, so the
sample size is `n = 3` independent biological experiments.

The design is analysed separately for each species and metric as a complete
randomized-block experiment:

```text
response = DMSO concentration effect + biological experiment block effect + residual
```

This preserves the matching of DMSO concentrations within each biological
experiment. An ordinary independent-groups one-way ANOVA is not used because it
would ignore the BR1-BR3 block structure. The standard independent-groups Tukey
HSD implementation is also not used directly. Instead, Tukey all-pairwise
comparisons are calculated from the randomized-block residual mean square:

```text
Tukey SE = sqrt(residual MS / number of biological blocks)
q = absolute treatment-mean difference / Tukey SE
```

Adjusted p-values and simultaneous 95% confidence intervals use the
studentized-range distribution with six DMSO concentrations and the
randomized-block residual degrees of freedom.

Inferential testing is performed on raw biological-replicate means, not relative
AUC percentages. Relative AUC remains descriptive/presentation-only. The primary
endpoint is `AUC_0_17h_OD_h_mean`; `AUC_0_12h_OD_h_mean` is the supporting
secondary early-growth endpoint. `Kz_OD600_per_h_mean` and `TLag_h_mean` are
model-derived endpoints. `A_OD600_mean` is retained descriptively when present
but is not tested inferentially in this stage.

Significance stars in the figures are based only on Tukey-adjusted p-values:

```text
p >= 0.05   no star
p < 0.05    *
p < 0.01    **
p < 0.001   ***
p < 0.0001  ****
```

Run the BR1-BR3 statistical analysis with:

```powershell
python analyses/dmso_statistics.py --input "path\to\DMSO biological repeaets summarised 3x.xlsx" --output-dir results/dmso_statistics
```

The source workbook remains outside the repository and is not modified. Outputs
are written to `results/dmso_statistics/`, including validated biological data,
descriptive statistics, randomized-block ANOVA results, randomized-block Tukey
comparisons, significant Tukey comparisons, a Markdown summary, and PNG/PDF
figures.

Run tests:

```powershell
python -m pytest
```
