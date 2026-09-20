"""
Postprocess: Statistical Analysis of LLM Probability Outputs

This script performs postprocessing and statistical analysis on LLM output probability matrices
from debate speech experiments. It generates four bar plot visualizations for:
1. Percentage of confidence intervals including 1
2. Percentage of Bonferroni-corrected confidence intervals including 1
3. Average probability of stance preservation
4. Pass rates (mean diagonal probability > threshold, per proposition and initial stance)

Results are aggregated across propositions (from propositions.json) for each initial stance.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import json
import os
import argparse
from scipy import stats

# Configuration
plt.style.use("ggplot")

LETTERS = ["A", "B", "C", "D", "E"]

# Statistical analysis parameters
confidence = 0.99
epsilon = 1e-10
method = "clt"  # 'hoeffding', 'clt'
THRE_SUCCESS = 0.95  # Pass-rate threshold; overridden in main() by --threshold (default 0.95)

THRE_SUCCESS_SQ = np.round(np.arange(0, 1.05, 0.05), 2)


def _build_sdr_masks(letters=LETTERS):
    """5x5 mode masks over (initial_idx, encoded_idx). Mutually exclusive, jointly
    exhaustive with the diagonal (each off-diagonal cell assigned to exactly one mode)."""
    likert = {"A": +2, "B": +1, "C": 0, "D": -1, "E": -2}
    r = np.array([likert[l] for l in letters])
    s = np.sign(r)
    m = np.abs(r)
    n = len(letters)
    pol = np.zeros((n, n))
    mod = np.zeros((n, n))
    modneut = np.zeros((n, n))
    flip = np.zeros((n, n))
    dev = np.zeros((n, n))
    diag = np.eye(n)
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            if s[i] != 0 and s[i] == s[j] and m[j] > m[i]:
                pol[i, j] = 1  # Polarization: B->A, D->E
            elif s[i] != 0 and s[j] == s[i] and m[j] < m[i]:
                mod[i, j] = 1  # Moderation (same side): A->B, E->D
            elif s[i] != 0 and s[j] == 0:
                modneut[i, j] = 1  # Moderation to neutral: A,B,D,E -> C
            elif s[i] != 0 and s[j] != 0 and s[i] != s[j]:
                flip[i, j] = 1  # Flipping: opposite side (8 cells)
            elif s[i] == 0 and s[j] != 0:
                dev[i, j] = 1  # Deviation from neutrality: C -> A,B,D,E
    assert np.array_equal(pol + mod + modneut + flip + dev + diag, np.ones((n, n))), (
        "SDR masks do not partition the 5x5 grid"
    )
    return {"Pol": pol, "Mod": mod, "ModNeut": modneut, "Flip": flip, "Dev": dev}


SDR_MASKS = _build_sdr_masks()
# Five drift classes (writting/DECISIONS.md 2026-09-10, TASK 1.1a). Same-direction
# drift = Pol + Mod; direction-changing drift = Flip + Dev + ModNeut.
SDR_MODES = ["Pol", "Mod", "ModNeut", "Flip", "Dev"]


def load_json(file_path):
    """Load JSON data from file."""
    with open(file_path, "r") as f:
        return json.load(f)


def _normalized_reps(data, perm_key, temp=1):
    """(reps, 5 initial, 5 encoded) array after temperature scaling. A failed encode is
    stored as an all-NaN row; it stays NaN here and is excluded from every mean and
    count downstream (the denominator is the number of valid reps, not 100)."""
    arr = np.asarray(data[perm_key], dtype=float)
    invalid = np.isnan(arr).any(axis=2)  # (reps, 5)
    if temp == 0:
        concentrated = np.zeros_like(arr)
        concentrated[
            np.arange(arr.shape[0])[:, None, None],
            np.arange(arr.shape[1])[None, :, None],
            np.nan_to_num(arr, nan=-np.inf).argmax(axis=2)[:, :, None],
        ] = 1
        arr = concentrated
    else:
        arr = arr ** temp
        sums = arr.sum(axis=2, keepdims=True)
        if not (sums[~invalid] > 0).all():
            raise ValueError("all-zero encode row in a valid rep; cannot normalize")
        arr = arr / sums  # Normalize to sum to 1
    arr[invalid] = np.nan
    return arr


def extract_diagonal_probs(data, perm_key, letters=LETTERS, temp=1):
    """
    Extract the diagonal (i,i) probabilities for each initial letter from all experiments.

    Args:
        data: Experimental data containing probability matrices
        perm_key: Key for accessing specific permutation data
        letters: List of response letters ['A', 'B', 'C', 'D', 'E']

    Returns:
        dict: {letter: [list of 100 diagonal probabilities]}
    """
    diag_probs = {}
    # 100 experiments, 5 initials, dim-5 probability vectors; failed encodes are NaN
    arr = _normalized_reps(data, perm_key, temp=temp)
    for i, l in enumerate(letters):
        diag_probs[l] = arr[:, i, i]  # take the i-th column for i-th initial

    return diag_probs


def extract_sdr_mode_probs(data, perm_key, letters=LETTERS, temp=1):
    """
    For each initial letter i, return per-rep probability mass falling into each SDR mode.

    Modes (Polarization, Moderation, Moderation to neutral, Flipping,
    DeviationFromNeutral) partition the off-diagonal cells; together with the
    diagonal (SPR) they sum to 1.0 per rep.

    Returns:
        dict: {letter: {mode_name: np.ndarray of shape (n_reps,)}}
    """
    arr = _normalized_reps(data, perm_key, temp=temp)  # (reps, 5 initial, 5 encoded)

    mode_probs = {l: {} for l in letters}
    for mode in SDR_MODES:
        mask = SDR_MASKS[mode]  # (5,5)
        # (reps,5,5) * (5,5) → sum over encoded axis → (reps,5); NaN for failed encodes
        per_rep = (arr * mask[None, :, :]).sum(axis=2)
        for i, l in enumerate(letters):
            mode_probs[l][mode] = per_rep[:, i]
    return mode_probs


def compute_confidence_interval(samples, confidence=0.95, method="clt"):
    """
    Compute confidence interval for the mean of samples.
    One sided test:
        Null hypothesis: the mean is equal to 1
        Alternative hypothesis: the mean is less than 1

    Parameters:
        samples (array-like): Data samples
        confidence (float): Confidence level (default 0.95)
        method (str): 'clt' for central limit theorem (default), 'hoeffding' for Hoeffding's inequality

    Returns:
        tuple: (mean, lower_bound, upper_bound)
    """

    # Failed encodes (NaN) are excluded: the mean and the CI use the valid reps only.
    samples = np.asarray(samples, dtype=float)
    samples = samples[np.isfinite(samples)]

    n = len(samples)
    mean = np.mean(samples)
    upper = mean
    lower = mean

    if method == "clt":
        std = np.std(samples, ddof=1)
        se = std / np.sqrt(n)
        h = stats.t.ppf(confidence, n - 1) * se
        upper = mean + h
        lower = 0

    elif method == "hoeffding":
        # Hoeffding's inequality for bounded variables in [a, b]
        # For probabilities, a=0, b=1
        eps = np.sqrt((1 / (2 * n)) * np.log(1 / (1 - confidence)))
        upper = mean + eps
        lower = 0

    return mean, lower, upper


def letter_to_option(letter):
    """Convert letter to response option."""
    letter = letter.strip().upper()
    if letter == "A":
        return "Agree strongly"
    elif letter == "B":
        return "Agree"
    elif letter == "C":
        return "Neutral"
    elif letter == "D":
        return "Disagree"
    elif letter == "E":
        return "Disagree strongly"


def create_barplot(data, value_column, title, filename, model_name, color="steelblue"):
    """
    Create and save a bar plot visualization.

    Args:
        data: DataFrame with results
        value_column: Column to use for bar values
        title: Plot title
        filename: Output filename (without path)
        model_name: Model name for directory structure
        color: Bar color
    """
    # Create output directory
    output_dir = f"postprocess_results/{model_name}"
    os.makedirs(output_dir, exist_ok=True)

    # Create full file path
    full_filename = os.path.join(output_dir, filename)

    # Compute mean and SE for each initial stance (letter)
    grouped = data.groupby("Letter")[value_column]
    means = grouped.mean()
    ses = grouped.apply(lambda x: np.std(x, ddof=1) / np.sqrt(len(x)))

    # Ensure ordering follows LETTERS
    labels = [letter_to_option(l) for l in LETTERS]
    mean_values = [means[l] for l in LETTERS]
    se_values = [ses[l] for l in LETTERS]

    # Convert to percentages for display
    mean_pct = [m * 100 for m in mean_values]
    se_pct = [s * 100 for s in se_values]

    fig, ax = plt.subplots(figsize=(17, 10))
    x_pos = np.arange(len(LETTERS))
    bars = ax.bar(
                x_pos,
                mean_pct,
                yerr=se_pct,
                capsize=30,
                width=0.85,
                color=color,
                edgecolor="#333333",
                alpha=0.5,
                error_kw={"linewidth": 2.2, "ecolor": "#2B2B2B"},
            )

    # Add value labels in the middle of each bar: "mean (SE)"
    for i, (bar, m, s) in enumerate(zip(bars, mean_pct, se_pct)):
        ax.text(
            bar.get_x() + bar.get_width() / 2.0, 
            max(bar.get_height()-10.0, 0.0),
            f"{m:.1f} ({s:.1f})",
            ha="center", 
            va="bottom", 
            fontsize=25,            # slightly larger
            fontweight="semibold",  # lighter than bold
            color="black",       
        )

    ax.set_xticks(x_pos)
    ax.set_xticklabels(labels, fontsize=24, fontweight="bold")
    ax.set_xlabel("Initial Stance", fontsize=30, fontweight="bold")
    ax.set_ylabel("Percentage (%)", fontsize=30, fontweight="bold", labelpad=12)
    title_with_annot = title + "\n(Values in %; Annot: Mean (SE))"
    ax.set_title(title_with_annot, fontsize=34, fontweight="bold", pad=18)
    ax.set_ylim(0, min(115, max(mean_pct) + max(se_pct) + 15))
    ax.tick_params(axis="y", labelsize=24)
    ax.yaxis.grid(True, linestyle="-", linewidth=0.6, alpha=0.25)
    ax.set_axisbelow(True)

    plt.tight_layout()
    plt.savefig(full_filename, bbox_inches="tight", dpi=500)
    plt.close()
    print(f"Saved bar plot: {full_filename}")


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Postprocess LLM probability outputs and generate statistical analysis"
    )

    parser.add_argument(
        "--model",
        type=str,
        default="gpt_4o_mini",
        help="Model name (default: gpt_4o_mini)",
    )

    parser.add_argument(
        "--threshold",
        type=float,
        default=0.95,
        help="Threshold for success rate calculation (default: 0.95)",
    )

    parser.add_argument(
        "--suffix",
        type=str,
        default="",
        help="Additional suffix for model directory (e.g., '_reversed')",
    )

    parser.add_argument(
        "--confidence",
        type=float,
        default=0.99,
        help="Confidence level for statistical analysis (default: 0.99)",
    )

    parser.add_argument(
        "--epsilon",
        type=float,
        default=1e-10,
        help="Epsilon parameter for statistical calculations (default: 1e-10)",
    )

    parser.add_argument(
        "--temp",
        type=float,
        default=1,
        help="Temperature parameter for temperature scaling (default: 1)",
    )
    parser.add_argument(
        "--method",
        type=str,
        default="clt",
        choices=["clt", "hoeffding"],
        help="Statistical method for confidence intervals (default: clt)",
    )
    parser.add_argument(
        "--prompt_id",
        type=str,
        default="1",
        help="Prompt ID to use for file naming (default: '1'). Accepts strings like 'reflection'.",
    )

    return parser.parse_args()


def main():
    """Main processing function."""
    args = parse_arguments()

    # Set parameters from command line arguments
    MODEL = args.model
    SUFFIX = args.suffix
    THRE_SUCCESS = args.threshold
    CONFIDENCE = args.confidence
    EPSILON = args.epsilon
    METHOD = args.method
    TEMP = args.temp
    PROMPT_ID = args.prompt_id

    print("Starting postprocessing analysis...")
    print(f"Model: {MODEL}{SUFFIX}")
    print(f"Success threshold: {THRE_SUCCESS}")
    print(f"Confidence level: {CONFIDENCE}")
    print(f"Method: {METHOD}")
    print(f"Epsilon: {EPSILON}")
    print("-" * 50)

    # Load propositions
    propositions = load_json("propositions.json")
    num_propositions = len(propositions)
    print(f"Loaded {num_propositions} propositions from propositions.json")

    results = []
    success_rates_results = []
    files_processed = 0

    for prop in propositions:
        topic_id = prop["topic_id"]
        topic = prop["topic"]

        file_path = (
            f"results/{MODEL}{SUFFIX}/debate_speech_{topic_id}"
            f"_{MODEL}_prompt_{PROMPT_ID}_raw.json"
        )

        if not os.path.exists(file_path):
            continue  # skip missing files

        try:
            data = load_json(file_path)
            files_processed += 1

            for perm_key in data.keys():
                diag_probs = extract_diagonal_probs(
                    data, perm_key=perm_key, letters=LETTERS, temp=TEMP
                )
                sdr_probs = extract_sdr_mode_probs(
                    data, perm_key=perm_key, letters=LETTERS, temp=TEMP
                )

                for i, l in enumerate(LETTERS):
                    samples = diag_probs[l]
                    mean, ci_low, ci_high = compute_confidence_interval(
                        samples, confidence=CONFIDENCE, method=METHOD
                    )
                    mean, ci_low_bonferroni, ci_high_bonferroni = (
                        compute_confidence_interval(
                            samples,
                            confidence=1 - (1 - CONFIDENCE) / num_propositions,
                            method=METHOD,
                        )
                    )

                    includes_1 = ci_low <= 1 <= ci_high
                    includes_1_bonferroni = (
                        ci_low_bonferroni <= 1 <= ci_high_bonferroni
                    )
                    valid = np.isfinite(samples)  # failed encodes are excluded everywhere
                    n_valid = int(valid.sum())
                    if n_valid < 2:
                        raise AssertionError(
                            f"topic {topic_id} letter {l}: only {n_valid} valid reps; "
                            "a mean and CI need at least 2"
                        )
                    # Pass rate at the threshold (SI Table S4): the combination passes if its
                    # success rate, the diagonal probability averaged over the valid trials,
                    # exceeds the threshold; the per-model value is the proportion of
                    # propositions that pass (DECISIONS.md 2026-09-13; was per trial).
                    success_rate = float(mean > THRE_SUCCESS)

                    # Per-rep partition check: diagonal + 5 modes ≈ 1 on the valid
                    # reps. Failed encodes (all-NaN rows) are NaN here and excluded
                    # from every mean above, so the denominators are N_valid.
                    total_per_rep = samples + sum(sdr_probs[l][mode] for mode in SDR_MODES)
                    finite = np.isfinite(total_per_rep)
                    if not np.allclose(total_per_rep[finite], 1.0, atol=1e-6):
                        raise AssertionError(
                            f"SDR partition violated at topic {topic_id} letter {l} "
                            f"perm {perm_key}: max |sum-1| = "
                            f"{np.max(np.abs(total_per_rep[finite] - 1)):.3e}"
                        )

                    results.append(
                        {
                            "Proposition_ID": topic_id,
                            "Proposition": topic,
                            "Perm": perm_key,
                            "Letter": l,
                            "Mean": mean,
                            "CI_low": ci_low,
                            "CI_high": ci_high,
                            "Includes_1": includes_1,
                            "Includes_1_bonferroni": includes_1_bonferroni,
                            "Success_rate": success_rate,
                            **{
                                f"Mean_{mode}": float(np.nanmean(sdr_probs[l][mode]))
                                for mode in SDR_MODES
                            },
                            "N_valid": n_valid,
                        }
                    )

                    success_rates_result = {
                        "Proposition_ID": topic_id,
                        "Proposition": topic,
                        "Perm": perm_key,
                        "Letter": l,
                    }
                    for thre_success in THRE_SUCCESS_SQ:
                        success_rates_result[f"Success_rate_{thre_success}"] = (
                            float(mean > thre_success)  # same per-combination rule as above
                        )
                    success_rates_results.append(success_rates_result)

        except (OSError, ValueError) as e:
            # Unreadable or malformed file only. A computation error (partition
            # assertion, too few valid reps) propagates: a silently dropped topic
            # would change every downstream mean without any sign of it.
            raise RuntimeError(f"cannot read {file_path}: {e}") from e

    print(f"Processed {files_processed} / {num_propositions} proposition files")
    if len(results) != len(LETTERS) * files_processed:
        raise AssertionError(
            f"{len(results)} result rows for {files_processed} files; expected {len(LETTERS)} per file"
        )

    if not results:
        print("No results found! Check your file paths and configuration.")
        return

    # Create results DataFrame
    results_df = pd.DataFrame(results)
    print(f"\nTotal results: {len(results_df)} rows")

    # Summary by initial stance (Letter)
    def standard_error(x):
        return np.std(x, ddof=1) / np.sqrt(len(x))

    summary_by_stance = (
        results_df.groupby("Letter")[
            ["Includes_1", "Includes_1_bonferroni", "Mean", "Success_rate"]
            + [f"Mean_{mode}" for mode in SDR_MODES]
        ]
        .agg(["mean", standard_error])
    )
    # Flatten multi-level columns
    summary_by_stance.columns = [
        f"{col}_{stat}" for col, stat in summary_by_stance.columns
    ]
    summary_by_stance = summary_by_stance.reset_index()
    # Add readable stance labels
    summary_by_stance.insert(
        1, "Stance", summary_by_stance["Letter"].apply(letter_to_option)
    )

    print("\nSummary by initial stance:")
    print(summary_by_stance.to_string(index=False))

    # Save summary to text file
    if TEMP == 0:
        SUFFIX = SUFFIX + "_temp0"
    output_dir = f"postprocess_results/{MODEL}{SUFFIX}"
    os.makedirs(output_dir, exist_ok=True)
    summary_file = os.path.join(output_dir, "summary.txt")

    with open(summary_file, "w") as f:
        f.write("Summary by Initial Stance - Statistical Analysis Results\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"Model: {MODEL}{SUFFIX}\n")
        f.write(f"Propositions processed: {files_processed} / {num_propositions}\n")
        f.write(f"Success threshold: {THRE_SUCCESS}\n")
        f.write(f"Confidence level: {CONFIDENCE}\n")
        f.write(f"Method: {METHOD}\n")
        f.write(f"Epsilon: {EPSILON}\n")
        f.write(f"Temperature: {TEMP}\n")
        f.write("=" * 60 + "\n\n")
        f.write(summary_by_stance.to_string(index=False))
        f.write("\n\n")
        f.write("Column Descriptions:\n")
        f.write(
            "- Includes_1_mean: Percentage of confidence intervals that include 1.0\n"
        )
        f.write(
            "- Includes_1_bonferroni_mean: Same with Bonferroni correction "
            f"(÷{num_propositions} propositions)\n"
        )
        f.write("- Mean_mean: Average diagonal probability (stance preservation)\n")
        f.write(
            "- Success_rate_mean: Pass rate, the fraction of propositions whose mean diagonal "
            "probability exceeds the threshold\n"
        )
        f.write(
            "- Mean_{Pol,Mod,ModNeut,Flip,Dev}_mean: Average off-diagonal mass per drift "
            "class (polarization, moderation, moderation to neutral, flipping, "
            "deviation from neutrality); with Mean_mean they sum to 1\n"
        )
        f.write("- *_standard_error: Standard error of the corresponding metric\n")
        f.write(
            "- Failed encodes (all-NaN reps in _raw.json) are excluded from every mean; "
            "N_valid in postprocess_detailed_results.csv counts the reps used\n"
        )

    print(f"Saved summary to: {summary_file}")

    # Create and save bar plots
    print("\nGenerating bar plots...")

    # Bar plot 1: Percentage of CIs including 1
    create_barplot(
        results_df,
        "Includes_1",
        f"Percentage of Confidence Intervals Including 1\n"
        f"({int(CONFIDENCE * 100)}% confidence interval, 1-sided)",
        "barplot_includes_1.pdf",
        f"{MODEL}{SUFFIX}",
        "#7A2E2E",
    )

    # Bar plot 2: Percentage of CIs including 1 (Bonferroni corrected)
    create_barplot(
        results_df,
        "Includes_1_bonferroni",
        f"Percentage of Confidence Intervals Including 1\n"
        f"(Bonferroni-corrected)",
        "barplot_includes_1_bonferroni.pdf",
        f"{MODEL}{SUFFIX}",
        "#8F4E2A",
    )

    # Bar plot 3: Average probability of stance preservation
    create_barplot(
        results_df,
        "Mean",
        "Average Probability of Stance Preservation",
        "barplot_mean_probabilities.pdf",
        f"{MODEL}{SUFFIX}",
        "#2F4B7C",
    )

    # Bar plot 4: Success rates
    create_barplot(
        results_df,
        "Success_rate",
        f"Pass Rates\n(Fraction of propositions with mean probability > {THRE_SUCCESS})",
        "barplot_success_rates.pdf",
        f"{MODEL}{SUFFIX}",
        "#2E6F62",
    )

    # Save success rates results to CSV
    success_rates_results_df = pd.DataFrame(success_rates_results)
    success_rates_results_df.to_csv(
        f"postprocess_results/{MODEL}{SUFFIX}/postprocess_success_rates_results.csv",
        index=False,
    )

    # Save detailed results to CSV
    results_df.to_csv(
        f"postprocess_results/{MODEL}{SUFFIX}/postprocess_detailed_results.csv",
        index=False,
    )

    # Save / update cross-model comparison CSV
    model_label = f"{MODEL}{SUFFIX}"
    model_summary = summary_by_stance.copy()
    model_summary.insert(0, "Model", model_label)

    comparison_file = "postprocess_results/model_comparison.csv"
    if os.path.exists(comparison_file):
        existing = pd.read_csv(comparison_file)
        # Remove old rows for this model (allows re-runs to update)
        existing = existing[existing["Model"] != model_label]
        combined = pd.concat([existing, model_summary], ignore_index=True)
    else:
        combined = model_summary

    combined.to_csv(comparison_file, index=False)
    print(f"Updated cross-model comparison: {comparison_file}")

    print("\nAnalysis complete!")
    print("Generated files:")
    print(f"- postprocess_results/{MODEL}{SUFFIX}/summary.txt")
    print(f"- postprocess_results/{MODEL}{SUFFIX}/barplot_includes_1.pdf")
    print(f"- postprocess_results/{MODEL}{SUFFIX}/barplot_includes_1_bonferroni.pdf")
    print(f"- postprocess_results/{MODEL}{SUFFIX}/barplot_mean_probabilities.pdf")
    print(f"- postprocess_results/{MODEL}{SUFFIX}/barplot_success_rates.pdf")
    print(f"- postprocess_results/{MODEL}{SUFFIX}/postprocess_detailed_results.csv")
    print(f"- postprocess_results/{MODEL}{SUFFIX}/postprocess_success_rates_results.csv")
    print(f"- {comparison_file}")


if __name__ == "__main__":
    main()
