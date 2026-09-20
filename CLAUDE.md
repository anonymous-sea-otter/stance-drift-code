# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this repo is

Research code for a paper on **LLM stance drift**: give a model a stance letter (A–E, "Agree strongly" → "Disagree strongly") on a debate proposition, have it *decode* the letter into a short argument, then *encode* the argument back into a letter using top-logprobs. Repeating this 100× per initial stance yields a 5×5 **empirical transition matrix** per proposition. Downstream scripts aggregate these into **SPR** (Stance Preservation Rate = diagonal mass) and **SDR** (Stance Drift Rate, off-diagonal mass partitioned into five classes: Polarization / Moderation / Moderation to neutral / Flipping / Deviation from neutrality) for paper figures and LaTeX tables.

There is no test suite, linter config, or build step. All scripts are run directly with `python` from the repo root (paths like `results/`, `propositions.json`, `postprocess_results/` are relative to CWD).

## Commands

`REPRODUCE.md` is the authoritative list of verified commands for every figure/table. Key ones:

```bash
pip install -r requirements.txt          # together / google-genai are NOT pinned here; install separately if running non-OpenAI models

# Run a new experiment (needs OPENAI_API_KEY / TOGETHER_API_KEY / GEMINI_API_KEY)
python main.py gpt-5.4 results --start_idx 0 --end_idx 112 --batch --prompt_choice reflection_third_person --reasoning_effort medium

# Aggregate one model's raw results -> postprocess_results/{model}{suffix}/ + model_comparison.csv
python postprocess.py --model gpt_5_4
python postprocess.py --model gpt_5_4 --suffix _reflection --prompt_id reflection   # variant run

# Paper outputs (all read postprocess_results/, write figures/)
python faithfulness_metric.py                     # Fig 4 (9-model SPR bars);  --comprehensive = Fig S4, 18 models
python faithfulness_metric.py --latex-tables      # Table 1A/1B;              --comprehensive = Tables S1–S4
python faithfulness_metric.py --sdr-tables        # SDR LaTeX tables
python success_rate_metric.py --all-letters-only  # Fig S6
python visualization.py --topic_id 2401 --model gpt_5_4 --panel left --title "..."   # transition-matrix heatmap
python compare_prompt_variations.py               # Fig S5: GPT-5.4 configurations (default, reflection, third-person reflection, medium reasoning)
python export_data_s1.py                          # figures/data_S1.xlsx (run last; depends on all of the above)

python cancel_batch_jobs.py                       # cancel every in-flight OpenAI batch job
```

`run_expt.sh` is a stale SLURM wrapper from an earlier (BBQ-topic) version of the project; its CLI shape does not match the current `main.py`.

## Pipeline and data flow

```
propositions.json (112 topics; example.json holds 2 held-out ICL examples)
   │  main.py  →  utils.estimate_tran_mat  →  utils.decode_and_encode  →  chat_client.UnifiedChatClient
   ▼
results/{model_slug}/debate_speech_{topic_id}_{model_slug}_prompt_{choice}[_effort_<lvl>]{.json,_raw.json,.log,.pdf}
   │  postprocess.py --model M --suffix S --prompt_id P
   ▼
postprocess_results/{M}{S}/postprocess_detailed_results.csv   (per-proposition × stance rows, incl. Mean_Pol/Mod/ModNeut/Flip/Dev and N_valid)
postprocess_results/model_comparison.csv                       (one block of 5 rows per model; upserted on each run)
   │  faithfulness_metric.py / success_rate_metric.py / compare_prompt_variations.py / export_data_s1.py
   ▼
figures/*.pdf, figures/*.tex, figures/data_S1.xlsx
```

**Result file formats.** `*_prompt_1.json` is the 5×5 mean transition matrix (row = initial letter, col = encoded letter). `*_raw.json` is a dict keyed by a permutation string (currently always `"[['A','B','C','D','E'], ['A','B','C','D','E']]"`) → array of shape `(100 reps, 5 initial, 5 encoded)`. All statistics downstream are computed from `_raw.json`.

**Model naming has two layers.** API model strings (`gpt-5.4`, `meta-llama/Llama-3.3-70B-Instruct-Turbo`) map to filesystem slugs (`gpt_5_4`, `llama3_3_70b`) via `MODEL_NAME_MAP` in `main.py`. Variant runs (ablations, reflection prompts, reasoning effort) are distinguished by a **directory suffix**, not by the slug inside the filename: `results/gpt_5_4_reflection/debate_speech_..._gpt_5_4_prompt_reflection_raw.json`. `main.py` writes to `results/{slug}/` only, so variant runs must be moved into a suffix-named directory by hand before `postprocess.py --suffix` can find them. `--temp 0` in `postprocess.py` appends `_temp0` to the output dir.

**Adding a new model or variant** requires touching several hardcoded lists: `MODEL_NAME_MAP` in `main.py`; `MODEL_NAME_LATEX` / `MODEL_NAME_PLOT` in `name_maps.py`; the subset/comprehensive `model_dirs` in `faithfulness_metric.py` and `success_rate_metric.py`; `SUBSET_MODELS` / `COMPREHENSIVE_MODELS` in `export_data_s1.py`; and `VARIANT_MODELS` in `faithfulness_metric.py` if it should be colored as a variant. For ad-hoc comparisons prefer `--models ... --output-suffix ...` so published figures are not clobbered.

## Key modules

- `utils.py` — the experiment core. `generate_prompt()` holds every prompt template (numeric variants 1–9 plus named `reflection` / `reflection_third_person`; note `main.py` casts digit strings to `int` because templates compare `== 1` etc.). `decode_and_encode()` is the sequential path; `create_*_batch_requests` / `submit_batch_and_wait` / `process_*_batch_results` are the OpenAI Batch API path (`--batch`, OpenAI models only). `get_normalized_prob()` deliberately raises on all-zero encode columns rather than producing NaN.
- `chat_client.py` — `UnifiedChatClient` routes by model-name prefix to OpenAI / Together / Gemini and normalizes responses so callers can always read `.choices[0].logprobs.content[0].top_logprobs`. Together Llama models cap `top_logprobs` at 5; gpt-5* models get `reasoning_effort` passed through.
- `postprocess.py` — defines the SDR mode masks (`_build_sdr_masks`, asserted to partition the 5×5 grid) and the one-sided CLT/Hoeffding CI logic. It asserts per-rep `SPR + Pol + Mod + ModNeut + Flip + Dev ≈ 1` on valid reps. A failed encode (all-NaN rep row in `_raw.json`) is excluded from every mean, CI and success rate; `N_valid` in the detailed CSV records the reps used (99 in six cells of the two Llama runs, 100 elsewhere). Per-topic errors propagate; nothing is skipped silently.
- `name_maps.py` — single source of display names for LaTeX and matplotlib labels.
- `human_extraction/` — a separate mini-pipeline comparing AI vs human stance extraction for topic 3234; see `human_extraction/postprocess_summary.md` for its artifact flow. The human labels are distributed as the de-identified file `human_extraction/human_labels_3234.csv` (annotators numbered 1 to 9, one column per argument); the survey export is not distributed.

## Conventions and gotchas

- Committed outputs under `results/`, `postprocess_results/`, and `figures/` are the paper's published numbers. Regenerating them is expected, but don't hand-edit them.
- `postprocess.py` re-runs are idempotent per model label (it drops old rows for that label in `model_comparison.csv` before appending). Older rows (e.g. `gpt_5_2`) may lack the SDR columns.
- Encode uses temperature 0.7 and `max_tokens=10`; decode uses temperature 0.7 and `max_tokens=200` with a 100-word argument limit. `REPITITION_EST_MAT = 100` in `main.py` sets the rep count.
- `utils.py` and `main.py` contain `pdb.set_trace()` fallbacks in except blocks; a failure there will drop into the debugger rather than exit.