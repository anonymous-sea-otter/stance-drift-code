import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import argparse
from name_maps import get_latex_name, get_plot_name, get_plot_name_single_line

# Directory containing all model/suffix result folders
RESULTS_DIR = "postprocess_results"
CSV_NAME = "postprocess_detailed_results.csv"
MODEL_COMPARISON_CSV = "model_comparison.csv"

# Metric definitions: (mean_column, se_column, display_title, file_slug)
# file_slug names the per-metric output file: figures/latex_tables_{mode}_{slug}.tex
LATEX_METRICS = [
    ("Mean_mean", "Mean_standard_error",
     "Success rate (mean diagonal probability) by initial stance", "spr"),
    ("Includes_1_bonferroni_mean", "Includes_1_bonferroni_standard_error",
     "Bonferroni-Corrected CI Inclusion Rate", "bonferroni_ci_inclusion"),
    ("Includes_1_mean", "Includes_1_standard_error",
     "CI Inclusion Rate", "ci_inclusion"),
    ("Success_rate_mean", "Success_rate_standard_error",
     "Pass rate at threshold 0.95 (proportion of propositions whose success rate exceeds it)", "success_rate"),
]

# Subset (9 models) for Figure 5 / Table 1
SUBSET_MODEL_DIRS = [
    "gpt_4o_mini",
    "gpt_4_1",
    "gpt_5_4",
    "gpt_3_5_turbo",
    "gemma_3n_e4b",
    "llama3_3_70b",
    "llama3_1_8b",
    "llama4_maverick",
    "qwen3_a3b",
]

# GPT-4o-mini ablation runs, appended to the subset by --with-ablations
ABLATION_MODEL_DIRS = [
    "gpt_4o_mini_reversed",
    "gpt_4o_mini_temp0",
    "gpt_4o_mini_multiple_summarization",
    "gpt_4o_mini_in_context",
    "gpt_4o_mini_assert",
]

STANCE_ORDER = ["A", "B", "C", "D", "E"]
STANCE_LABELS = {
    "A": "Agree strongly",
    "B": "Agree",
    "C": "Neutral",
    "D": "Disagree",
    "E": "Disagree strongly",
}

# Per-mode per-stance SDR tables (read from model_comparison.csv columns added by postprocess.py)
# (mean_col, se_col, title, slug) -- the 5 SDR modes, in the order used in every SDR table.
SDR_LATEX_METRICS = [
    ("Mean_Pol_mean", "Mean_Pol_standard_error",
     "SDR component: Polarization", "polarization"),
    ("Mean_Mod_mean", "Mean_Mod_standard_error",
     "SDR component: Moderation", "moderation"),
    ("Mean_ModNeut_mean", "Mean_ModNeut_standard_error",
     "SDR component: Moderation to neutral", "moderation_to_neutral"),
    ("Mean_Flip_mean", "Mean_Flip_standard_error",
     "SDR component: Flipping", "flipping"),
    ("Mean_Dev_mean", "Mean_Dev_standard_error",
     "SDR component: Deviation from neutrality", "deviation_from_neutral"),
]

# Cells that are structurally zero by mode definition — render as "---" to distinguish
# from measured zeros.
#   Pol     = {B->A, D->E}            zero for A, C, E
#   Mod     = {A->B, E->D}            zero for B, C, D
#   ModNeut = {A,B,D,E -> C}          zero for C
#   Flip    = 8 opposite-side cells   zero for C
#   Dev     = {C -> A,B,D,E}          zero for A, B, D, E
SDR_STRUCTURAL_ZEROS = {
    "Mean_Pol_mean": {"A", "C", "E"},
    "Mean_Mod_mean": {"B", "C", "D"},
    "Mean_ModNeut_mean": {"C"},
    "Mean_Flip_mean": {"C"},
    "Mean_Dev_mean": {"A", "B", "D", "E"},
}


def table_frac(x):
    """Fraction with 3 decimals and no leading zero (".593"): the manuscript's
    convention for every table cell; prose and plots keep the leading zero."""
    s = f"{x:.3f}"
    if s == "-0.000":
        s = "0.000"
    return s[1:] if s.startswith("0.") else s


def latex_escape(s):
    """Escape underscores for LaTeX (fallback for unknown names)."""
    return s.replace("_", "\\_")


def bar_color_for_model(model_key, variant_models):
    """Opaque bar color by model family (GPT main / GPT variant / Llama / Gemma / Qwen)."""
    gpt_main = "#3d74ae"
    gpt_variant = "#6a9fd4"  # lighter blue, same hue family as gpt_main
    if model_key in variant_models:
        return gpt_variant
    if model_key.startswith("gpt_"):
        return gpt_main
    if "llama" in model_key:
        return "#ee8539"
    if "gemma" in model_key:
        return "#529d40"
    if "qwen" in model_key:
        return "#8c69b7"
    return "#7F7F7F"


def generate_latex_tables(model_dirs, comprehensive=False, output_suffix=None,
                          mode_label=None):
    """
    Generate LaTeX table snippets from model_comparison.csv.

    Produces 4 tables (one per metric). Each table has:
    - Rows: initial stances (Agree strongly ... Disagree strongly)
    - Columns: model/variant names
    - Cells: mean (SE) in percentage with 1 decimal place

    Writes one combined file, figures/latex_tables_{mode_label}{output_suffix}.tex,
    containing all 4 tables, plus one file per metric,
    figures/latex_tables_{mode_label}{output_suffix}_{slug}.tex, each holding exactly
    that metric's "% title" comment line and tabular block.

    Args:
        model_dirs: list of model directory names to include
        comprehensive: if True, label output as comprehensive
        output_suffix: if given, appended before ".tex" so an ad-hoc --models run
            does not clobber the published figures/latex_tables_{subset,comprehensive}.tex
        mode_label: explicit label for the output filename; defaults to
            "comprehensive" or "subset" based on `comprehensive`

    Returns:
        path to the saved combined .tex file, or None on error
    """
    comparison_file = os.path.join(RESULTS_DIR, MODEL_COMPARISON_CSV)
    if not os.path.exists(comparison_file):
        print(f"Error: {comparison_file} not found. "
              "Run postprocess.py for all models first.")
        return None

    df = pd.read_csv(comparison_file)

    # Filter to requested models, preserving order
    df = df[df["Model"].isin(model_dirs)]
    available_models = [m for m in model_dirs if m in df["Model"].values]

    if not available_models:
        print("Error: No matching models found in model_comparison.csv.")
        return None

    n_stances = len(STANCE_ORDER)
    col_spec = "l" + "c" * n_stances

    output_lines = []
    metric_blocks = []  # (slug, [lines of that metric's block])

    for mean_col, se_col, title, slug in LATEX_METRICS:
        block_start = len(output_lines)
        # Comment header
        output_lines.append(f"% {title}")
        output_lines.append(f"\\begin{{tabular}}{{{col_spec}}}")
        output_lines.append("\t\t\\\\")
        output_lines.append("\t\t\\hline")

        # Header row: stance names as columns (the stub column holds the model names)
        header = "Model"
        for letter in STANCE_ORDER:
            header += f" & {STANCE_LABELS[letter]}"
        header += " \\\\"
        output_lines.append(f"\t\t{header}")
        output_lines.append("\t\t\\hline")

        # Data rows: one row per model; cells are fractions, 3 decimals, no leading
        # zero (the manuscript's table convention, DECISIONS.md 2026-09-13).
        for model in available_models:
            row = get_latex_name(model)
            for letter in STANCE_ORDER:
                cell = df[(df["Model"] == model) & (df["Letter"] == letter)]
                if len(cell) == 0:
                    row += " & N/A"
                else:
                    mean_val = cell[mean_col].values[0]
                    se_val = cell[se_col].values[0]
                    row += f" & {table_frac(mean_val)} ({table_frac(se_val)})"
            row += " \\\\"
            output_lines.append(f"\t\t{row}")

        output_lines.append("\t\t\\hline")
        output_lines.append("\\end{tabular}")
        metric_blocks.append((slug, output_lines[block_start:]))
        output_lines.append("")
        output_lines.append("")

    # Save combined file (all 4 tables; format unchanged)
    if mode_label is None:
        mode_label = "comprehensive" if comprehensive else "subset"
    output_stem = f"figures/latex_tables_{mode_label}"
    if output_suffix:
        output_stem += output_suffix
    output_file = f"{output_stem}.tex"
    os.makedirs("figures", exist_ok=True)
    with open(output_file, "w") as f:
        f.write("\n".join(output_lines))

    print(f"LaTeX tables saved to: {output_file}")
    print(f"  Models included: {len(available_models)}")
    print(f"  Tables generated: {len(LATEX_METRICS)}")

    # Save one file per metric: "% title" + tabular block + single trailing newline
    for slug, block_lines in metric_blocks:
        metric_file = f"{output_stem}_{slug}.tex"
        with open(metric_file, "w") as f:
            f.write("\n".join(block_lines) + "\n")
        print(f"  Per-metric table saved to: {metric_file}")

    return output_file


def generate_sdr_latex_tables(model_dirs, output_suffix=None, mode_label=None):
    """
    Generate SDR (Stance Drift Rate) LaTeX tables:
      1. Summary table: rows=models, cols=[SPR, Pol, Mod, ModNeut, Flip, Dev];
         the five pattern columns are the components of SDR = 1 - SPR;
         fractions with 3 decimals (no leading zero), each row sums to 1.
         Aggregation: stage-1 mean over the 5 initial stances within each (topic, perm);
         stage-2 mean ± SE across those (topic, perm) samples.
      2. Five per-stance per-mode tables (same shape as the SPR table); structurally
         zero cells rendered as "---".

    Writes one combined file, figures/latex_tables_sdr_{mode_label}{output_suffix}.tex,
    containing all 6 tables, plus one file per table,
    figures/latex_tables_sdr_{mode_label}{output_suffix}_{slug}.tex (slugs: summary,
    polarization, moderation, moderation_to_neutral, flipping, deviation_from_neutral),
    each holding exactly that table's "% title" comment line and tabular block.

    Args:
        model_dirs: list of model directory names to include
        output_suffix: if given, appended before ".tex" so an ad-hoc --models run
            does not clobber the published files
        mode_label: label for the output filename (subset / subset_ablations /
            comprehensive); defaults to "subset"

    Returns:
        path to the saved combined .tex file, or None on error
    """
    # ----- Stage 1+2: summary table from postprocess_detailed_results.csv -----
    summary_rows = {}
    available_models = []
    n_samples = {}
    mode_cols = ["SPR", "Pol", "Mod", "ModNeut", "Flip", "Dev"]
    raw_col_of = {
        "SPR": "Mean",
        "Pol": "Mean_Pol",
        "Mod": "Mean_Mod",
        "ModNeut": "Mean_ModNeut",
        "Flip": "Mean_Flip",
        "Dev": "Mean_Dev",
    }
    mode_cols_raw = [raw_col_of[c] for c in mode_cols]
    for model in model_dirs:
        csv_path = os.path.join(RESULTS_DIR, model, CSV_NAME)
        if not os.path.exists(csv_path):
            print(f"  Skipping {model}: {csv_path} missing")
            continue
        df_m = pd.read_csv(csv_path)
        missing = [c for c in mode_cols_raw if c not in df_m.columns]
        if missing:
            print(f"  Skipping {model}: missing columns {missing}. "
                  "Re-run postprocess.py to regenerate.")
            continue
        topic_level = df_m.groupby(["Proposition_ID", "Perm"])[mode_cols_raw].mean().reset_index()
        n = len(topic_level)
        means = topic_level[mode_cols_raw].mean()
        ses = topic_level[mode_cols_raw].std(ddof=1) / np.sqrt(n)
        summary_rows[model] = {
            c: (means[raw_col_of[c]] * 100, ses[raw_col_of[c]] * 100) for c in mode_cols
        }
        n_samples[model] = n
        available_models.append(model)

    if not available_models:
        print("Error: no SDR-ready CSVs found. Run postprocess.py for the target models first.")
        return None

    mode_labels = {
        "SPR":     "SPR",
        # Pattern columns are the components of the SDR (= 1 - SPR); the column
        # heads carry the pattern name only, the caption states the relation.
        "Pol":     "Polarization",
        "Mod":     "Moderation",
        "ModNeut": "Moderation to neutral",
        "Flip":    "Flipping",
        "Dev":     "Deviation from neutrality",
    }
    output_lines = []
    metric_blocks = []  # (slug, [lines of that table's block])

    frac3 = table_frac

    block_start = len(output_lines)
    output_lines.append(
        "% SDR Summary: rows=models, columns=[SPR, 5 drift patterns = components of SDR (1 - SPR)]; fractions of the "
        "transition mass, each row sums to 1. Stage-1: mean over 5 initial stances per "
        "(topic, perm). Stage-2: mean (SE) across (topic, perm) samples."
    )
    output_lines.append(f"\\begin{{tabular}}{{l{'c' * len(mode_cols)}}}")
    output_lines.append("\t\t\\\\")
    output_lines.append("\t\t\\hline")
    header = "Model"
    for c in mode_cols:
        header += f" & {mode_labels[c]}"
    header += " \\\\"
    output_lines.append(f"\t\t{header}")
    output_lines.append("\t\t\\hline")
    for model in available_models:
        row = get_latex_name(model)
        for c in mode_cols:
            mean_pct, se_pct = summary_rows[model][c]
            row += f" & {frac3(mean_pct / 100)} ({frac3(se_pct / 100)})"
        row += " \\\\"
        output_lines.append(f"\t\t{row}")
    output_lines.append("\t\t\\hline")
    output_lines.append("\\end{tabular}")
    metric_blocks.append(("summary", output_lines[block_start:]))
    output_lines.append("")
    output_lines.append("")

    # ----- Per-stance per-mode tables (from already-aggregated model_comparison.csv) -----
    comparison_file = os.path.join(RESULTS_DIR, MODEL_COMPARISON_CSV)
    if not os.path.exists(comparison_file):
        print(f"  Warning: {comparison_file} missing — skipping per-stance SDR tables.")
    else:
        df_cmp = pd.read_csv(comparison_file)
        df_cmp = df_cmp[df_cmp["Model"].isin(available_models)]
        n_stances = len(STANCE_ORDER)
        col_spec = "l" + "c" * n_stances
        for mean_col, se_col, title, slug in SDR_LATEX_METRICS:
            if mean_col not in df_cmp.columns:
                print(f"  Skipping '{title}': {mean_col} missing in model_comparison.csv. "
                      "Re-run postprocess.py.")
                continue
            zeros = SDR_STRUCTURAL_ZEROS.get(mean_col, set())
            block_start = len(output_lines)
            output_lines.append(f"% {title}")
            output_lines.append(f"\\begin{{tabular}}{{{col_spec}}}")
            output_lines.append("\t\t\\\\")
            output_lines.append("\t\t\\hline")
            header = "Model"
            for letter in STANCE_ORDER:
                header += f" & {STANCE_LABELS[letter]}"
            header += " \\\\"
            output_lines.append(f"\t\t{header}")
            output_lines.append("\t\t\\hline")
            for model in available_models:
                row = get_latex_name(model)
                for letter in STANCE_ORDER:
                    if letter in zeros:
                        row += " & ---"
                        continue
                    cell = df_cmp[(df_cmp["Model"] == model) & (df_cmp["Letter"] == letter)]
                    if len(cell) == 0:
                        row += " & N/A"
                    else:
                        mean_val = cell[mean_col].values[0]
                        se_val = cell[se_col].values[0]
                        row += f" & {table_frac(mean_val)} ({table_frac(se_val)})"
                row += " \\\\"
                output_lines.append(f"\t\t{row}")
            output_lines.append("\t\t\\hline")
            output_lines.append("\\end{tabular}")
            metric_blocks.append((slug, output_lines[block_start:]))
            output_lines.append("")
            output_lines.append("")

    # Save combined file (all tables; format unchanged)
    if mode_label is None:
        mode_label = "subset"
    output_stem = f"figures/latex_tables_sdr_{mode_label}"
    if output_suffix:
        output_stem += output_suffix
    output_file = f"{output_stem}.tex"
    os.makedirs("figures", exist_ok=True)
    with open(output_file, "w") as f:
        f.write("\n".join(output_lines))
    print(f"SDR LaTeX tables saved to: {output_file}")
    print(f"  Models included: {len(available_models)} (per-model n_samples={n_samples})")
    print(f"  Tables generated: {len(metric_blocks)}")

    # Save one file per table: "% title" + tabular block + single trailing newline
    for slug, block_lines in metric_blocks:
        metric_file = f"{output_stem}_{slug}.tex"
        with open(metric_file, "w") as f:
            f.write("\n".join(block_lines) + "\n")
        print(f"  Per-table file saved to: {metric_file}")

    return output_file


def main():
    parser = argparse.ArgumentParser(
        description="Plot average diagonal probability for each model."
    )
    parser.add_argument(
        "--errorbar", action="store_true", help="Plot error bars (std) for each bar"
    )
    parser.add_argument(
        "--neutral-only",
        action="store_true",
        help="Use only neutral letter (C) data instead of all letters",
    )
    parser.add_argument(
        "--metric",
        choices=["raw", "negative_log", "baseline"],
        default="raw",
        help="Metric type: raw (default), negative_log, or baseline (difference from gpt_3_5_turbo)",
    )
    parser.add_argument(
        "--comprehensive",
        action="store_true",
        help="Use comprehensive model list (18 models) instead of subset (9 models)",
    )
    parser.add_argument(
        "--with-ablations",
        action="store_true",
        help="Use the 9-model subset followed by the five GPT-4o-mini ablation runs "
             "(14 models; output label 'subset_ablations'). Not combinable with "
             "--comprehensive.",
    )
    parser.add_argument(
        "--latex-tables",
        action="store_true",
        help="Generate LaTeX table snippets from model_comparison.csv "
             "(subset: Table 1A,B; comprehensive: Table S1-S4; "
             "--with-ablations: 14-model subset_ablations set). Writes the combined "
             "figures/latex_tables_{mode}.tex plus one file per metric, "
             "figures/latex_tables_{mode}_{spr,bonferroni_ci_inclusion,ci_inclusion,"
             "success_rate}.tex",
    )
    parser.add_argument(
        "--sdr-tables",
        action="store_true",
        help="Generate SDR (Stance Drift Rate) LaTeX tables: a summary table "
             "(rows=models, cols=[SPR, Pol, Mod, ModNeut, Flip, Dev]; rows sum to "
             "~100%%) plus 5 per-stance per-mode tables. --with-ablations / "
             "--comprehensive select the model set exactly as for --latex-tables. "
             "Writes the combined figures/latex_tables_sdr_{mode}.tex plus one file "
             "per table, figures/latex_tables_sdr_{mode}_{summary,polarization,"
             "moderation,moderation_to_neutral,flipping,deviation_from_neutral}.tex",
    )
    parser.add_argument(
        "--models",
        type=str,
        default=None,
        help="Comma-separated list of model dirs to plot. Overrides the hardcoded "
             "subset/comprehensive list when provided (e.g., for ad-hoc comparisons).",
    )
    parser.add_argument(
        "--variant-models",
        type=str,
        default=None,
        help="Comma-separated list of model dirs to color as variants (lighter blue). "
             "Use with --models to distinguish intervention runs from baselines.",
    )
    parser.add_argument(
        "--output-suffix",
        type=str,
        default=None,
        help="Suffix appended to the output PDF filename (before .pdf). "
             "Useful to avoid clobbering the published figure when running ad-hoc lists.",
    )
    args = parser.parse_args()

    if args.comprehensive and args.with_ablations:
        parser.error("--comprehensive and --with-ablations are mutually exclusive")

    # Find all model/suffix subdirectories
    # Variant models (ablations / prompt-variation runs) — lighter GPT-adjacent blue
    VARIANT_MODELS = {
        "gpt_4o_mini_reversed",
        "gpt_4o_mini_temp0",
        "gpt_4o_mini_multiple_summarization",
        "gpt_4o_mini_in_context",
        "gpt_4o_mini_assert",
        "gpt_4o_mini_reflection",
        "gpt_5_4_reflection",
        "gpt_5_4_reflection_third_person",
        "gpt_5_4_reflection_third_person_medium",
    }
    if args.variant_models:
        VARIANT_MODELS = VARIANT_MODELS | {
            m.strip() for m in args.variant_models.split(",") if m.strip()
        }

    if args.comprehensive:
        # Comprehensive list (18 models) for Figure S8
        model_dirs = SUBSET_MODEL_DIRS + ABLATION_MODEL_DIRS + [
            "gpt_4o_mini_reflection",
            "gpt_5_4_reflection",
            "gpt_5_4_reflection_third_person",
            "gpt_5_4_reflection_third_person_medium",
        ]
        mode_label = "comprehensive"
    elif args.with_ablations:
        # Subset + 5 GPT-4o-mini ablations (14 models)
        model_dirs = SUBSET_MODEL_DIRS + ABLATION_MODEL_DIRS
        mode_label = "subset_ablations"
    else:
        # Subset list (9 models) for Figure 5
        model_dirs = list(SUBSET_MODEL_DIRS)
        mode_label = "subset"

    # Ad-hoc override. Without --output-suffix the published subset/comprehensive
    # files would be overwritten with the ad-hoc list (and copied into the
    # manuscript by check.sh), so the suffix is required here.
    if args.models:
        if (args.latex_tables or args.sdr_tables) and not args.output_suffix:
            parser.error("--models with --latex-tables/--sdr-tables requires --output-suffix")
        model_dirs = [m.strip() for m in args.models.split(",") if m.strip()]

    # If --latex-tables, generate tables and exit
    if args.latex_tables:
        generate_latex_tables(model_dirs, comprehensive=args.comprehensive,
                              output_suffix=args.output_suffix,
                              mode_label=mode_label)
        return

    # If --sdr-tables, generate SDR tables and exit
    if args.sdr_tables:
        generate_sdr_latex_tables(model_dirs, output_suffix=args.output_suffix,
                                  mode_label=mode_label)
        return

    # Store average diagonal probabilities for each model
    model_averages = {}
    model_stds = {}

    # For baseline comparison, we need to load all data first
    all_model_data = {}

    for model_dir in model_dirs:
        csv_path = os.path.join(RESULTS_DIR, model_dir, CSV_NAME)
        if not os.path.exists(csv_path):
            continue

        df = pd.read_csv(csv_path)

        # Filter data based on letter selection
        if args.neutral_only:
            df = df[df["Letter"] == "C"]

        # Store the data for baseline comparison
        if args.metric == "baseline":
            all_model_data[model_dir] = df["Mean"].values
        else:
            # Compute average diagonal probability across selected rows
            if args.metric == "negative_log":
                # Use negative log of mean values
                avg_diagonal_prob = -np.log(df["Mean"] + 1e-6).mean()
                std_diagonal_prob = np.log(df["Mean"] + 1e-6).std()
            else:
                # Use raw mean values
                avg_diagonal_prob = (df["Mean"]).mean()
                std_diagonal_prob = (df["Mean"]).std()

            model_averages[model_dir] = avg_diagonal_prob
            model_stds[model_dir] = std_diagonal_prob

    # Handle baseline comparison
    if args.metric == "baseline":
        baseline_model = "gpt_3_5_turbo"  #'gpt_3_5_turbo'
        if baseline_model not in all_model_data:
            print(f"Error: Baseline model '{baseline_model}' not found in results.")
            return

        baseline_data = all_model_data[baseline_model]

        # Compute differences from baseline for each model
        for model_dir, model_data in all_model_data.items():
            if model_dir == baseline_model:
                continue

            # Ensure same length by truncating to minimum length
            min_length = min(len(baseline_data), len(model_data))
            baseline_subset = baseline_data[:min_length]
            model_subset = model_data[:min_length]

            # Compute differences from baseline
            differences = model_subset - baseline_subset

            # Calculate mean and std of differences
            avg_difference = differences.mean()
            std_difference = differences.std()

            model_averages[model_dir] = avg_difference
            model_stds[model_dir] = std_difference

    # Create horizontal bar plot (models top→bottom: highest SPR first for raw/baseline;
    # for negative_log, smallest -log(SPR) = highest SPR first)
    triples = list(zip(
        model_averages.keys(),
        model_averages.values(),
        model_stds.values(),
    ))
    if args.metric == "negative_log":
        triples.sort(key=lambda t: t[1])
    else:
        triples.sort(key=lambda t: t[1], reverse=True)
    if triples:
        models, averages, stds = map(list, zip(*triples))
    else:
        models, averages, stds = [], [], []
    n = len(models)
    y_pos = np.arange(n)

    # Dense charts (comprehensive / ad-hoc --models lists) carry long variant
    # names: render them on multiple lines (as defined in name_maps) so the
    # label gutter stays narrow. The subset chart keeps its published layout.
    dense_labels = args.comprehensive or bool(args.models)
    fig_w = 18 if args.comprehensive else 16
    row_h = 0.9 if dense_labels else 0.75
    fig_h = max(8.0, row_h * n + 3)
    plt.figure(figsize=(fig_w, fig_h))
    ax = plt.gca()

    colors = [bar_color_for_model(m, VARIANT_MODELS) for m in models]
    bar_kwargs = {"height": 0.7, "alpha": 1.0, "color": colors}
    if args.errorbar:
        bars = ax.barh(
            y_pos,
            averages,
            xerr=stds,
            capsize=5,
            ecolor="black",
            **bar_kwargs,
        )
    else:
        bars = ax.barh(y_pos, averages, **bar_kwargs)

    ax.invert_yaxis()

    xlabel_fs = 24
    ytick_fs = 17 if dense_labels else 24
    spr_title = "Stance Preservation Rate (SPR)"
    if args.neutral_only:
        spr_title += " (neutral only)"

    if args.metric == "negative_log":
        ax.set_xlabel(f"{spr_title}\n-log(SPR)", fontsize=xlabel_fs)
        metric_type = "Negative Log"
    elif args.metric == "baseline":
        ax.set_xlabel(
            f"{spr_title}\nDifference from {baseline_model} "
            f"(baseline mean SPR: {baseline_subset.mean():.3f})",
            fontsize=xlabel_fs,
        )
        metric_type = "Baseline"
    else:
        ax.set_xlabel(spr_title, fontsize=xlabel_fs, fontweight="semibold")
        metric_type = "Raw"

    if args.metric == "raw":
        xmax = max(1.05, max(averages) * 1.05) if averages else 1.05
        ax.set_xlim(0, xmax)
        ax.axvline(
            1.0,
            color="grey",
            linestyle="-",
            linewidth=2,
            label="Perfect fidelity (SPR=1.0)",
        )
        ax.axvline(
            0.2,
            color="black",
            linestyle="--",
            linewidth=2,
            label="Uniform random baseline (SPR=0.2)",
        )
        ax.legend(
            fontsize=16 if dense_labels else 18,
            loc="lower right",
            bbox_to_anchor=(0.98, 0.02) if dense_labels else (0.90, 0.02),
        )
    elif args.metric == "baseline":
        lo, hi = min(averages), max(averages)
        pad = 0.05 * (hi - lo) if hi != lo else 0.05 * (abs(hi) + 0.1)
        ax.set_xlim(lo - pad, hi + pad)
    else:
        xmax = max(averages) * 1.05 if averages else 1.0
        ax.set_xlim(0, xmax if xmax > 0 else 1.0)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    ax.tick_params(axis="x", labelsize=24, width=2, length=6)
    plt.setp(ax.get_xticklabels(), fontweight="semibold")
    label_fn = get_plot_name if dense_labels else get_plot_name_single_line
    models_labels = [label_fn(model) for model in models]
    ax.set_yticks(y_pos)
    ax.set_yticklabels(models_labels, fontsize=ytick_fs)
    plt.setp(ax.get_yticklabels(), ha="right", rotation=0, fontweight="semibold")

    for bar, avg in zip(bars, averages):
        cx = bar.get_x() + bar.get_width() / 2
        cy = bar.get_y() + bar.get_height() / 2
        # Fractions in plots carry no leading zero (".593"), as in the tables.
        ax.text(
            cx,
            cy,
            f"{avg:.3f}",
            ha="center",
            va="center",
            fontsize=xlabel_fs,
            color="white",
            fontweight="semibold",
        )

    ax.grid(axis="x", alpha=0.3)
    plt.tight_layout()
    
    # Set output filename based on mode
    if args.comprehensive:
        output_filename = "figures/average_diagonal_probability_by_model_comprehensive.pdf"
    else:
        output_filename = "figures/average_diagonal_probability_by_model.pdf"
    if args.output_suffix:
        output_filename = output_filename.replace(".pdf", f"{args.output_suffix}.pdf")
    
    # Ensure figures directory exists
    os.makedirs("figures", exist_ok=True)
    
    plt.savefig(output_filename, dpi=500, bbox_inches="tight")
    plt.close()

    # Print summary statistics
    data_type = "Neutral Only" if args.neutral_only else "All Letters"
    if args.metric == "negative_log":
        metric_type = "Negative Log"
    elif args.metric == "baseline":
        metric_type = f"Baseline (vs {baseline_model})"
    else:
        metric_type = "Raw"

    mode_type = "Comprehensive (18 models, Figure S8)" if args.comprehensive else "Subset (9 models, Figure 5)"
    print(f"Average Diagonal Probability by Model ({metric_type}, {data_type}, {mode_type}):")
    print("=" * 70)
    for model, avg in model_averages.items():
        std = model_stds[model]
        print(f"{model}: {avg:.4f} ± {std:.4f}")

    print(f"\nPlot saved: {output_filename}")
    print(f"Total models analyzed: {len(model_averages)}")
    if args.comprehensive:
        print("Using comprehensive model list (18 models).")
    else:
        print("Using subset model list (9 models).")
    if args.errorbar:
        print("Error bars (std) are included.")
    if args.neutral_only:
        print("Using only neutral letter (C) data.")
    else:
        print("Using all letters data.")
    if args.metric == "negative_log":
        print("Using negative log scale.")
    elif args.metric == "baseline":
        print(f"Using baseline comparison ({baseline_model}).")
    else:
        print("Using raw scale.")


if __name__ == "__main__":
    main()
