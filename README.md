# LLM stance drift on debate speeches

Code for studying how large language models **preserve or shift self-reported stance** when answering multiple-choice items tied to debate-speech propositions. The pipeline runs model calls, builds **empirical transition matrices** (stance before vs. after summarization / encoding), and aggregates **faithfulness** and **success-rate** metrics for paper figures and tables. It also covers **prompt-variation / reasoning-effort runs** (reflection prompts, `--reasoning_effort` for gpt-5* models) and **SDR (Stance Drift Rate)** analysis decomposing off-diagonal drift into five classes (polarization, moderation, moderation to neutral, flipping, deviation from neutrality).

## Repository layout

| Area | Role |
|------|------|
| `main.py` | Run LLM experiments over the debate-speeches dataset (`propositions.json` / `example.json`); writes per-topic logs and JSON under `results/`. |
| `postprocess.py` | Aggregate runs into bar plots, CSV summaries, and `postprocess_results/model_comparison.csv`. |
| `faithfulness_metric.py` | Model-comparison bar plots and LaTeX table snippets (subset / comprehensive). |
| `success_rate_metric.py` | Pass-rate vs. threshold curves and AUC summaries. |
| `compare_prompt_variations.py` | GPT-5.4 prompt/reasoning ladder comparison (SPR + SDR drift modes by stance). |
| `visualization.py` | Transition-matrix heatmaps for selected topics. |
| `export_data_s1.py` | Multi-sheet supplementary Excel (`figures/data_S1.xlsx`). |
| `human_extraction/` | Human survey postprocessing, majority-gold scoring (`compute_majority_gold_scores.py`), and related utilities. The human labels are in `human_labels_3234.csv` (see below). |
| `figures/` | Generated plots and LaTeX fragments. |

Supporting modules include `utils.py`, `chat_client.py`, and `name_maps.py`.

## What this copy contains

This is an anonymized copy prepared for double-blind review.

- `results/` holds, for every run, the per-proposition transition matrix (`*.json`), the raw per-trial extraction distributions (`*_raw.json`), the run log with the prompts and the generated arguments (`*.log`), and the transition-matrix heatmap (`*.pdf`).
- `postprocess_results/` and `figures/` hold the aggregated statistics, tables and figures of the paper.
- The human study is distributed as the de-identified label file `human_extraction/human_labels_3234.csv`: one row per annotator (numbered 1 to 9), one column per argument (`1_Q2` … `100_Q2`), each cell a stance letter A–E. The survey export is not distributed. `postprocess_human_extraction.py` and `compute_majority_gold_scores.py` read this file by default and reproduce the committed workbooks and `majority_gold_scores.csv`.

## Setup

Install Python packages used across the scripts (scientific stack plus API clients where you rerun experiments). **Figure reproduction** and a fuller dependency list are documented in [`REPRODUCE.md`](REPRODUCE.md).

To run **new** API-backed experiments, set the keys your stack needs, for example:

- `OPENAI_API_KEY`
- `TOGETHER_API_KEY`
- `GEMINI_API_KEY`

(Exact usage depends on the model and client paths in `main.py` / `utils.py`.)

## Reproducing paper outputs

Verified commands for every figure, table, and supplementary export are in **[`REPRODUCE.md`](REPRODUCE.md)**.

Typical flow: ensure `postprocess_results/` is populated (from `postprocess.py` on your models), then run `faithfulness_metric.py`, `success_rate_metric.py`, `visualization.py`, and `export_data_s1.py` as described there.

## Citation

If you use this repository, cite the associated paper.
