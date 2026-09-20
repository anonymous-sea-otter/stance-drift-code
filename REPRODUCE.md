# Reproducing Figures

This document provides verified commands for reproducing all figures in the paper.

---

## Prerequisites

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Set API keys (only needed for running new experiments):
   ```bash
   export OPENAI_API_KEY="your_openai_api_key"
   export TOGETHER_API_KEY="your_together_api_key"
   export GEMINI_API_KEY="your_gemini_api_key"
   ```

---

## Quick Reference

| Table/Figure | Script | Command |
|--------------|--------|---------|
| Figure 2A, 2B | `visualization.py` | `python visualization.py --topic_id <ID> --model gpt_5_4 --title "..." --panel ...` |
| Figure 2C | `visualization.py` | `python visualization.py --ideal-only --topic_id dummy` |
| Table 1A, 1B | `faithfulness_metric.py` | `python faithfulness_metric.py --latex-tables` |
| Figure 4 | `faithfulness_metric.py` | `python faithfulness_metric.py` |
| Figure S3A, S3B | `visualization.py` | `python visualization.py --topic_id <ID> --model gpt_5_4 --title "..." --panel ...` |
| Table S1-S4 | `faithfulness_metric.py` | `python faithfulness_metric.py --latex-tables --comprehensive` |
| Figure S4 | `faithfulness_metric.py` | `python faithfulness_metric.py --comprehensive` |
| Figure S6 | `success_rate_metric.py` | `python success_rate_metric.py --all-letters-only` |
| Data S1 (all sheets) | `export_data_s1.py` | `python export_data_s1.py` |
| SDR tables (subset) | `faithfulness_metric.py` | `python faithfulness_metric.py --sdr-tables` |
| SDR tables (14 / 18 models) | `faithfulness_metric.py` | `python faithfulness_metric.py --sdr-tables --with-ablations` / `--comprehensive` |
| SDR/SPR variation tables | `faithfulness_metric.py` | `python faithfulness_metric.py --sdr-tables --models <list> --output-suffix _6model_variations` |
| Figure S5 (GPT-5.4 configurations) and Tables S5–S10 (SDR) | `compare_prompt_variations.py`, `faithfulness_metric.py --sdr-tables --comprehensive` | see below |

---

## Detailed Commands

### Transition Matrix Heatmaps (Figure 2, S3)

**Script:** `visualization.py`

**Output:** `figures/`

```bash
# Figure 2C - Identity matrix (perfect stance preservation)
python visualization.py --ideal-only --topic_id dummy

# Figure 2A - Example: Polarization pattern (replace TOPIC_ID_1 with actual proposition ID)
python visualization.py --topic_id 2401 --model gpt_5_4 --title '\"We should increase fuel tax\"\n-Empirical Transition Matrix' --panel left

# Figure 2B - Example: Mixed pattern (replace TOPIC_ID_2 with actual proposition ID)
python visualization.py --topic_id 3234 --model gpt_5_4 --title '\"The use of AI should be abandoned\"\n-Empirical Transition Matrix' --panel center

# Figure S3A, S3B - SE-annotated versions are generated alongside each mean-only figure
# (outputs: debate_speech_{TOPIC_ID}_{model}_prompt_1.pdf and debate_speech_{TOPIC_ID}_{model}_prompt_1_se.pdf)
```

**Options:**
- `--topic_id`: Proposition topic_id from propositions.json (required unless `--ideal-only`)
- `--model`: Internal model name (default: `gpt_4o_mini`)
- `--prompt_id`: Prompt ID (default: `1`)
- `--title`: Plot title (default: `"Empirical Transition Matrix"`)
- `--panel`: Panel layout — `left`, `center`, `right`, or `single` (default: `single`)
- `--ideal-only`: Only generate the ideal identity matrix figure

---

### Model Bar Plots (prerequisite for Table 1, S1–S4, Figure S4-5)

**Script:** `postprocess.py`

**Output:** `postprocess_results/{model}/` and `postprocess_results/model_comparison.csv`

Run these commands to generate per-model bar plots and populate `model_comparison.csv`, which is used by `faithfulness_metric.py --latex-tables` to produce Table 1A,B and Table S1–S4.

```bash
# Base model
python postprocess.py --model gpt_4o_mini

# Variants (reversed option order, temperature=0)
python postprocess.py --model gpt_4o_mini --suffix _reversed
python postprocess.py --model gpt_4o_mini --temp 0

# Other models
python postprocess.py --model gpt_4_1
python postprocess.py --model gpt_5_4
python postprocess.py --model gpt_3_5_turbo
python postprocess.py --model gemma_3n_e4b
python postprocess.py --model llama3_3_70b
python postprocess.py --model llama3_1_8b
python postprocess.py --model llama4_maverick
python postprocess.py --model qwen3_a3b

# Prompt-variation / reasoning-effort runs (reflection ladder)
python postprocess.py --model gpt_4o_mini --suffix _reflection --prompt_id reflection
python postprocess.py --model gpt_5_4 --suffix _reflection --prompt_id reflection
python postprocess.py --model gpt_5_4 --suffix _reflection_third_person --prompt_id reflection_third_person
python postprocess.py --model gpt_5_4 --suffix _reflection_third_person_medium --prompt_id reflection_third_person_effort_medium
```

**Output files per model:**
- `barplot_mean_probabilities.pdf` — Mean diagonal probabilities
- `barplot_includes_1.pdf` — Confidence intervals including 1
- `barplot_includes_1_bonferroni.pdf` — Bonferroni-corrected CIs
- `barplot_success_rates.pdf` — Success rates
- `summary.txt` — Statistical summary
- `postprocess_detailed_results.csv` — Per-proposition detailed results
- `postprocess_success_rates_results.csv` — Success rates at multiple thresholds

**Cross-model output:**
- `postprocess_results/model_comparison.csv` — Accumulates mean and SE of the four SPR-type metrics and the five drift-class masses across models (updated on each run)

---

### Model Comparison Tables (Table 1, S1-S4)

**Script:** `faithfulness_metric.py`

**Output:** `figures/`

**Prerequisites:** Run `postprocess.py` for all models first (generates `model_comparison.csv`).

```bash
# Table 1A, 1B - Subset (9 models): Mean probability & Bonferroni CI inclusion
python faithfulness_metric.py --latex-tables

# Table S1-S4 - Comprehensive (18 models): All 4 metrics
python faithfulness_metric.py --latex-tables --comprehensive

# Subset + 5 GPT-4o-mini ablations (14 models): All 4 metrics
python faithfulness_metric.py --latex-tables --with-ablations
```

**Output files:**
- `figures/latex_tables_subset.tex` — 4 LaTeX table snippets (subset models)
- `figures/latex_tables_comprehensive.tex` — 4 LaTeX table snippets (all models)
- `figures/latex_tables_subset_ablations.tex` — 4 LaTeX table snippets (subset + ablations)
- `figures/latex_tables_{subset,comprehensive,subset_ablations}_{spr,bonferroni_ci_inclusion,ci_inclusion,success_rate}.tex` — the same tables, one file per metric

---

### SPR Comparison Bar Charts (Figure 4, S4)

**Script:** `faithfulness_metric.py`

**Output:** `figures/`

```bash
# Figure 4 - Stance Preservation Rate comparison (9 models)
python faithfulness_metric.py

# Figure S4 - Comprehensive comparison (18 models)
python faithfulness_metric.py --comprehensive
```

**Options:**
- `--comprehensive`: Include all 18 models (adds the gpt_4o_mini ablations and the reflection / reasoning-effort variant runs)
- `--latex-tables`: Generate LaTeX table snippets instead of bar plots
- `--errorbar`: Show error bars
- `--neutral-only`: Use only neutral (letter C) data
- `--models`: Comma-separated ad-hoc model list (overrides subset/comprehensive)
- `--variant-models`: Extra models to color as variants (lighter blue), for use with `--models`
- `--output-suffix`: Appended to output filenames to avoid clobbering published figures

---

### Success Rate Analysis (Figure S6)

**Script:** `success_rate_metric.py`

**Output:** `figures/`

**Prerequisites:** Run `postprocess.py` for all models first.

```bash
# Figure S6(A) - Pass rate vs threshold curves (a proposition-stance combination passes if its mean diagonal probability exceeds the threshold)
# Figure S6(B) - AUC of the pass-rate curve, normalized by 0.5
python success_rate_metric.py --all-letters-only
```

**Output files:**
- `success_rate_vs_threshold_all_letters.pdf`
- `auc_barplot_all_letters.pdf`
- `auc_results_all_letters.csv`

---

### SDR Tables and Prompt-Variation Comparison (reflection / reasoning-effort runs)

**Scripts:** `faithfulness_metric.py`, `success_rate_metric.py`, `compare_prompt_variations.py`

**Prerequisites:** Run `postprocess.py` for the target models first (including the four reflection-ladder runs above). SDR (Stance Drift Rate) partitions the off-diagonal transition mass into five mutually exclusive drift classes — Polarization (Pol: B→A, D→E), Moderation (Mod: A→B, E→D), Moderation to neutral (ModNeut: A/B/D/E→C), Flipping (Flip: opposite side, 8 cells), and Deviation from neutrality (Dev: C→A/B/D/E) — which together with the diagonal SPR sum to 1. Same-direction drift = Pol + Mod; direction-changing drift = ModNeut + Flip + Dev.

```bash
# SDR LaTeX tables for the 9-model subset
# (outputs: figures/latex_tables_sdr_subset.tex (combined) plus one file per table,
#           figures/latex_tables_sdr_subset_{summary,polarization,moderation,
#           moderation_to_neutral,flipping,deviation_from_neutral}.tex)
python faithfulness_metric.py --sdr-tables
# Same for the 14-model subset_ablations set and the 18-model comprehensive set
python faithfulness_metric.py --sdr-tables --with-ablations
python faithfulness_metric.py --sdr-tables --comprehensive

# SDR + SPR tables for the 6 GPT prompt-variation runs
# (outputs: figures/latex_tables_sdr_subset_6model_variations.tex,
#           figures/latex_tables_subset_6model_variations.tex, each plus one file per table / metric
#           named as above with the _6model_variations suffix)
python faithfulness_metric.py --sdr-tables --models gpt_4o_mini,gpt_4o_mini_reflection,gpt_5_4,gpt_5_4_reflection,gpt_5_4_reflection_third_person,gpt_5_4_reflection_third_person_medium --output-suffix _6model_variations
python faithfulness_metric.py --latex-tables --models gpt_4o_mini,gpt_4o_mini_reflection,gpt_5_4,gpt_5_4_reflection,gpt_5_4_reflection_third_person,gpt_5_4_reflection_third_person_medium --output-suffix _6model_variations

# GPT-5.4 prompt/reasoning ladder (default -> reflection -> third-person reflection -> third-person reflection, medium reasoning)
# (outputs: postprocess_results/variation_comparison/{spr_by_stance,drift_modes_long,ladder_overall}.csv,
#           {spr_by_stance,polarization_by_stance}.pdf; drift_modes_long.csv carries mean and SE per
#           rung, stance (incl. Overall) and class; ladder_overall.csv the per-rung overall values and
#           the change from the previous rung and from vanilla, which make_macros.py cross-checks)
python compare_prompt_variations.py

# Reflection-vs-baseline comparison figures
# (outputs: figures/average_diagonal_probability_by_model_reflection_compare.pdf,
#           figures/{success_rate_vs_threshold,auc_barplot}_all_letters_reflection_compare.pdf,
#           figures/auc_results_all_letters_reflection_compare.csv)
python faithfulness_metric.py --models gpt_4o_mini,gpt_4o_mini_reflection,gpt_5_4,gpt_5_4_reflection --variant-models gpt_4o_mini_reflection,gpt_5_4_reflection --output-suffix _reflection_compare
python success_rate_metric.py --all-letters-only --models gpt_4o_mini,gpt_4o_mini_reflection,gpt_5_4,gpt_5_4_reflection --output-suffix _reflection_compare
```

---

### Supplementary Data S1 (Multi-Sheet Excel)

**Script:** `export_data_s1.py`

**Output:** `figures/data_S1.xlsx`

**Prerequisites:** Run `postprocess.py` for all models and `success_rate_metric.py` first.

```bash
python export_data_s1.py
```

**Sheets (12 total):**

| Sheet | Corresponds To |
|-------|---------------|
| `Fig2A_S3A` | Transition matrix for Fig 2A & S3A (topic 2401, gpt-5.4) |
| `Fig2B_S3B` | Transition matrix for Fig 2B & S3B (topic 3234, gpt-5.4) |
| `Table1A` | Success rate (mean diagonal probability) by initial stance (9 models) |
| `Table1B` | Bonferroni-Corrected CI Inclusion Rate (9 models) |
| `Fig5` | SPR bar chart (9 models) |
| `TableS1` | Success rate by initial stance (18 runs) |
| `TableS2` | Bonferroni-Corrected CI Inclusion Rate (18 runs) |
| `TableS3` | CI Inclusion Rate (18 runs) |
| `TableS4` | Pass rate at threshold 0.95 (18 runs) |
| `FigS4` | SPR bar chart (18 runs) |
| `FigS6A` | Pass rate vs threshold curves (18 runs) |
| `FigS6B` | AUC of the pass-rate curve, normalized (18 runs) |

---

### Ablation Studies

Generate data for ablation analysis (used by `success_rate_metric.py` and `faithfulness_metric.py`):

```bash
# Assertion prompt ablation (prompt_id 9)
python postprocess.py --model gpt_4o_mini --suffix _assert --prompt_id 9

# Multiple summarization ablation
python postprocess.py --model gpt_4o_mini --suffix _multiple_summarization

# In-context learning ablation (prompt_id 8)
python postprocess.py --model gpt_4o_mini --suffix _in_context --prompt_id 8
```

---

### Running New Experiments: Prompt Variants and Reasoning Effort

**Script:** `main.py` (requires API keys)

- `--prompt_choice` accepts the numeric variants `1`–`9` or a named variant: `reflection`, `reflection_third_person`. The choice is embedded in output filenames as `_prompt_<choice>`.
- `--reasoning_effort {none,minimal,low,medium,high}` is only valid for reasoning models (`gpt-5*`). Omitting it (or passing `none`) sends no parameter and leaves filenames unchanged; the other levels are sent to the API and label output filenames with `_effort_<level>`.

```bash
# Example: the gpt_5_4_reflection_third_person_medium run
python main.py gpt-5.4 results --start_idx 0 --end_idx 112 --batch --prompt_choice reflection_third_person --reasoning_effort medium
```

Result files land in `results/{model_slug}/debate_speech_{topic_id}_{model_slug}_prompt_{choice}[_effort_<level>]{.json,.log,.pdf,_raw.json}`. Note the directory is named by the model slug only — the four published reflection runs were moved to suffix-named directories (e.g. `results/gpt_5_4_reflection_third_person_medium/`) so `postprocess.py --suffix` can find them.

---

## Notes

- **Temperature=0** outputs save to `postprocess_results/{model}_temp0/`
- Each `postprocess.py` run appends to `postprocess_results/model_comparison.csv`; re-running for the same model updates its rows
- A failed encode (an all-NaN rep row in `_raw.json`) is excluded from every mean, CI and success rate; the denominator is the number of valid reps, recorded as `N_valid` in `postprocess_detailed_results.csv` (99 for six cells in the two Llama runs, 100 elsewhere)
- `postprocess.py` also writes the five SDR drift-class columns (`Mean_Pol`, `Mean_Mod`, `Mean_ModNeut`, `Mean_Flip`, `Mean_Dev`) into `postprocess_detailed_results.csv` and `model_comparison.csv`; these feed `faithfulness_metric.py --sdr-tables`, `compare_prompt_variations.py`, and the manuscript's number macros (the manuscript sources are not part of this copy)

---

## Dependencies

See `requirements.txt` for full list:
- numpy, pandas, matplotlib, seaborn, scipy
- openai, together, google-genai (for running new experiments)
- sentence-transformers, scikit-learn (for clustering)

---

*Last updated: September 2026*

