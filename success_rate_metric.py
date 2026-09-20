import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import argparse
from name_maps import get_plot_name

# Directory containing all model/suffix result folders
RESULTS_DIR = "postprocess_results"
CSV_NAME = "postprocess_detailed_results.csv"  #'postprocess_success_rates_results.csv'

THRE_SUCCESS_SQ = np.round(np.arange(0, 1.05, 0.05), 2)


def main():
    parser = argparse.ArgumentParser(
        description="Plot average success rate curves for each model/suffix setting."
    )
    parser.add_argument(
        "--errorbar", action="store_true", help="Plot error bars (std) for each curve"
    )
    parser.add_argument(
        "--all-letters-only", action="store_true", help="Only output all letters results, skip neutral outputs"
    )
    parser.add_argument(
        "--models",
        type=str,
        default=None,
        help="Comma-separated list of model dirs to plot. Overrides the hardcoded 14-model list.",
    )
    parser.add_argument(
        "--output-suffix",
        type=str,
        default=None,
        help="Suffix appended to all output filenames (before extension), to avoid clobbering the published figures.",
    )
    args = parser.parse_args()

    # Create metrics directory if it doesn't exist
    metrics_dir = "figures"
    os.makedirs(metrics_dir, exist_ok=True)

    def with_sfx(filename):
        if not args.output_suffix:
            return filename
        base, ext = os.path.splitext(filename)
        return f"{base}{args.output_suffix}{ext}"

    # Fixed comprehensive model list (18 models)
    model_dirs = [
        "gpt_4o_mini",
        "gpt_4_1",
        "gpt_5_4",
        "gpt_3_5_turbo",
        "gemma_3n_e4b",
        "llama3_3_70b",
        "llama3_1_8b",
        "llama4_maverick",
        "qwen3_a3b",
        "gpt_4o_mini_reversed",
        "gpt_4o_mini_temp0",
        "gpt_4o_mini_multiple_summarization",
        "gpt_4o_mini_in_context",
        "gpt_4o_mini_assert",
        "gpt_4o_mini_reflection",
        "gpt_5_4_reflection",
        "gpt_5_4_reflection_third_person",
        "gpt_5_4_reflection_third_person_medium",
    ]

    if args.models:
        model_dirs = [m.strip() for m in args.models.split(",") if m.strip()]

    # Store results for plotting
    curves = {}
    curves_std = {}
    curves_C = {}
    curves_C_std = {}

    # Store AUC results
    auc_results_all = []
    auc_results_neutral = []

    for model_dir in model_dirs:
        csv_path = os.path.join(RESULTS_DIR, model_dir, CSV_NAME)
        if not os.path.exists(csv_path):
            continue
        df = pd.read_csv(csv_path)
        for thre_success in THRE_SUCCESS_SQ:
            if thre_success > 1.0:
                thre_success = 1 - 1e-6  # Avoid exact 1.0 for numerical stability
            df[f"Success_rate_{thre_success}"] = df["Mean"] > thre_success

        # Find all threshold columns
        threshold_cols = [col for col in df.columns if col.startswith("Success_rate_")]
        # Extract threshold values from column names
        thresholds = [float(col.split("_")[-1]) for col in threshold_cols]
        # Compute mean and std across all rows for each threshold
        avg_success = df[threshold_cols].mean(axis=0).values
        std_success = df[threshold_cols].std(axis=0).values
        curves[model_dir] = (thresholds, avg_success)
        curves_std[model_dir] = std_success

        # Compute AUC for all letters using trapezoidal rule (starting from threshold 0.5)
        threshold_mask = np.array(thresholds) >= 0.5
        thresholds_filtered = np.array(thresholds)[threshold_mask]
        avg_success_filtered = avg_success[threshold_mask]
        auc_all = np.trapz(avg_success_filtered, thresholds_filtered)
        # Normalize by maximum possible AUC (0.5) to get 0-1 scale
        auc_all_normalized = auc_all / 0.5
        auc_results_all.append({"Model": model_dir, "AUC": auc_all_normalized})

        # For letter 'C' only (skip if --all-letters-only is set)
        if not args.all_letters_only:
            df_C = df[df["Letter"] == "C"]
            avg_success_C = df_C[threshold_cols].mean(axis=0).values
            std_success_C = df_C[threshold_cols].std(axis=0).values
            curves_C[model_dir] = (thresholds, avg_success_C)
            curves_C_std[model_dir] = std_success_C

            # Compute AUC for neutral (letter C) only (starting from threshold 0.5)
            avg_success_C_filtered = avg_success_C[threshold_mask]
            auc_neutral = np.trapz(avg_success_C_filtered, thresholds_filtered)
            # Normalize by maximum possible AUC (0.5) to get 0-1 scale
            auc_neutral_normalized = auc_neutral / 0.5
            auc_results_neutral.append({"Model": model_dir, "AUC": auc_neutral_normalized})

    # Plot 1: All letters
    plt.figure(figsize=(25, 15))
    for model_dir, (thresholds, avg_success) in curves.items():
        if args.errorbar:
            std_success = curves_std[model_dir]
            plt.errorbar(
                thresholds,
                avg_success,
                yerr=std_success,
                marker="o",
                label=get_plot_name(model_dir).replace("-\n", "-").replace("\n", " "),
                capsize=3,
            )
        else:
            plt.plot(thresholds, avg_success, marker="o", label=get_plot_name(model_dir).replace("-\n", "-").replace("\n", " "))
    plt.xlabel("Threshold", fontsize=30, fontweight="bold")
    plt.ylabel("Pass rate", fontsize=30, fontweight="bold")
    plt.yticks(fontsize=30, fontweight="bold")
    plt.xticks(fontsize=30, fontweight="bold")
    plt.xlim(0.5, 1.0)  # Align with AUC computation range (0.5 to 1.0)
    plt.title("Pass rate vs. threshold", fontsize=40, fontweight="bold")
    # Legend below the axes so that 18 labels neither clip nor cover the curves
    plt.legend(prop={'size': 26}, ncol=3, loc="upper center", bbox_to_anchor=(0.5, -0.1), frameon=False)
    plt.tight_layout()
    plt.savefig(
        os.path.join(metrics_dir, with_sfx("success_rate_vs_threshold_all_letters.pdf")),  dpi=500, bbox_inches="tight"
    )
    plt.close()

    # Plot 2: Only letter 'C' (Neutral) - skip if --all-letters-only is set
    if not args.all_letters_only:
        plt.figure(figsize=(10, 6))
        for model_dir, (thresholds, avg_success_C) in curves_C.items():
            if args.errorbar:
                std_success_C = curves_C_std[model_dir]
                plt.errorbar(
                    thresholds,
                    avg_success_C,
                    yerr=std_success_C,
                    marker="o",
                    label=get_plot_name(model_dir).replace("-\n", "-").replace("\n", " "),
                    capsize=3,
                )
            else:
                plt.plot(thresholds, avg_success_C, marker="o", label=get_plot_name(model_dir).replace("-\n", "-").replace("\n", " "))
        plt.xlabel("Threshold")
        plt.ylabel("Average Success Rate (Neutral)")
        plt.title("Threshold vs. Average Success Rate (Neutral)")
        plt.legend(prop={'size': 30}, ncol=2)
        plt.tight_layout()
        plt.savefig(
            os.path.join(metrics_dir, with_sfx("success_rate_vs_threshold_Neutral.pdf")),  dpi=500
        )
        plt.close()

    # Save AUC results to CSV files
    # Convert to DataFrames and sort by AUC (high to low)
    auc_df_all = pd.DataFrame(auc_results_all).sort_values("AUC", ascending=False)

    # Save to CSV files
    auc_df_all.to_csv(
        os.path.join(metrics_dir, with_sfx("auc_results_all_letters.csv")), index=False
    )
    
    if not args.all_letters_only:
        auc_df_neutral = pd.DataFrame(auc_results_neutral).sort_values(
            "AUC", ascending=False
        )
        auc_df_neutral.to_csv(
            os.path.join(metrics_dir, with_sfx("auc_results_neutral.csv")), index=False
        )

    # Create bar plots for AUC results
    def format_model_name(model_name):
        return get_plot_name(model_name)

    # Function to assign colors based on model type
    def get_model_color(model_name):
        model_lower = model_name.lower()
        # Reflection variants — checked first so they don't fall through to baselines
        reflection_variants = {
            "gpt_4o_mini_reflection": "#7FE5CC",  # lighter teal (vs gpt_4o_mini #2AADA6)
            "gpt_5_4_reflection": "#A5A8F4",  # lighter indigo (vs gpt_5_4 #6366F1)
            "gpt_5_4_reflection_third_person": "#C3C5F8",  # next indigo tint
            "gpt_5_4_reflection_third_person_medium": "#DDDEFB",  # lightest indigo tint
        }
        if model_lower in reflection_variants:
            return reflection_variants[model_lower]
        # Check for GPT-4o-mini variants first (most specific)
        gpt4o_mini_variants = {
            "gpt_4o_mini_reversed": "#4ECDC4",  # Teal
            "gpt_4o_mini_temp0": "#3DBDB6",  # Darker teal
            "gpt_4o_mini_multiple_summarization": "#6ED4CC",  # Lighter teal
            "gpt_4o_mini": "#2AADA6",  # Darkest teal
            "gpt_4o_mini_in_context": "#5BCCC4",  # Medium teal
            "gpt_4o_mini_assert": "#7FE5CC",  # Light teal
        }

        for variant, color in gpt4o_mini_variants.items():
            if variant in model_lower:
                return color

        # Check for other GPT-4 variants
        if "gpt_4_1" in model_lower:
            return "#1E90FF"  # Dodger blue
        elif "gpt_4" in model_lower or "gpt-4" in model_lower:
            return "#4ECDC4"  # Default GPT-4 teal
        elif "gpt_5_4" in model_lower or "gpt-5.4" in model_lower:
            return "#6366F1"  # Indigo

        # Check for LLaMA variants
        if "llama3_1_8b" in model_lower:
            return "#A8E6CF"  # Light mint green
        elif "llama3_3_70b" in model_lower:
            return "#7FB069"  # Darker green
        elif "llama4" in model_lower:
            return "#50C878"  # Emerald green
        elif "llama" in model_lower:
            return "#90EE90"  # Default light green for other LLaMA variants

        # Check for other model families
        if (
            "gpt_3_5" in model_lower
            or "gpt-3.5" in model_lower
            or "gpt3.5" in model_lower
        ):
            return "#FF6B6B"  # Light red
        elif "qwen" in model_lower:
            return "#FF8C00"  # Dark orange
        elif "claude" in model_lower:
            return "#95E1D3"  # Light green
        elif "gemini" in model_lower:
            return "#F38BA8"  # Pink
        elif "mistral" in model_lower:
            return "#FFD93D"  # Yellow
        elif "gemma" in model_lower:
            return "#DDA0DD"  # Plum color for Gemma
        else:
            return "#B8B8B8"  # Gray for unknown models

    # Format model names for plotting
    auc_df_all_formatted = auc_df_all.copy()
    auc_df_all_formatted["Model_Formatted"] = auc_df_all_formatted["Model"].apply(
        format_model_name
    )

    if not args.all_letters_only:
        auc_df_neutral_formatted = auc_df_neutral.copy()
        auc_df_neutral_formatted["Model_Formatted"] = auc_df_neutral_formatted[
            "Model"
        ].apply(format_model_name)

    # Plot 3: AUC Bar Plot - All Letters
    # Width scales with the number of models (~2.16 per bar ≈ the original 30.16 at n=14)
    plt.figure(figsize=(max(12, 2.16 * len(auc_df_all_formatted)), 12))

    # Get colors for each model
    colors = [get_model_color(model) for model in auc_df_all_formatted["Model"]]

    bars = plt.bar(
        range(len(auc_df_all_formatted)), auc_df_all_formatted["AUC"], color=colors
    )
    plt.xlabel("Model", fontsize=30, fontweight="bold")
    plt.ylabel("Normalized AUC", fontsize=30, fontweight="bold")
    plt.title("AUC of the pass-rate curve", fontsize=40, fontweight="bold")
    plt.xticks(
        range(len(auc_df_all_formatted)),
        auc_df_all_formatted["Model_Formatted"],
        rotation=60,
        ha="right",
        rotation_mode="anchor",
        fontweight="bold",
        fontsize=26
    )
    plt.yticks(fontsize=30, fontweight="bold")

    # Add value labels on bars
    for i, bar in enumerate(bars):
        height = bar.get_height()
        plt.text(
            bar.get_x() + bar.get_width() / 2.0,
            bar.get_height() / 2, # height + 0.01,
            f"{height:.3f}",
            ha="center",
            va="center",
            fontsize=30,
            color="black",
            fontweight="bold",
        )

    plt.tight_layout()
    plt.savefig(
        os.path.join(metrics_dir, with_sfx("auc_barplot_all_letters.pdf")),
         dpi=500,
        bbox_inches="tight",
    )
    plt.close()

    # Plot 4: AUC Bar Plot - Neutral (Letter C) - skip if --all-letters-only is set
    if not args.all_letters_only:
        plt.figure(figsize=(max(12, 2.16 * len(auc_df_neutral_formatted)), 12))

        # Get colors for each model
        colors_neutral = [
            get_model_color(model) for model in auc_df_neutral_formatted["Model"]
        ]

        bars = plt.bar(
            range(len(auc_df_neutral_formatted)),
            auc_df_neutral_formatted["AUC"],
            color=colors_neutral,
        )
        plt.xlabel("Model", fontsize=30, fontweight="bold")
        plt.ylabel("Area Under Curve (AUC)", fontsize=30, fontweight="bold")
        plt.title("AUC Comparison - Neutral (Letter C)", fontsize=40, fontweight="bold")
        plt.yticks(fontsize=30, fontweight="bold")
        plt.xticks(
            range(len(auc_df_neutral_formatted)),
            auc_df_neutral_formatted["Model_Formatted"],
            rotation=60,
            ha="right",
            rotation_mode="anchor",
            fontweight="bold",
            fontsize=26
        )

        # Add value labels on bars
        for i, bar in enumerate(bars):
            height = bar.get_height()
            plt.text(
                bar.get_x() + bar.get_width() / 2.0,
                bar.get_height() / 2, # height + 0.01,
                f"{height:.3f}",
                ha="center",
                va="center",
                fontsize=30,
                color="black",
                fontweight="bold",
            )

        plt.tight_layout()
        plt.savefig(
            os.path.join(metrics_dir, with_sfx("auc_barplot_neutral.pdf")),
             dpi=500,
            bbox_inches="tight",
        )
        plt.close()

    print("Plots saved:")
    print(f" - {os.path.join(metrics_dir, 'success_rate_vs_threshold_all_letters.pdf')}")
    print(f" - {os.path.join(metrics_dir, 'auc_barplot_all_letters.pdf')}")
    if not args.all_letters_only:
        print(f" - {os.path.join(metrics_dir, 'success_rate_vs_threshold_Neutral.pdf')}")
        print(f" - {os.path.join(metrics_dir, 'auc_barplot_neutral.pdf')}")
    print("AUC results saved:")
    print(f" - {os.path.join(metrics_dir, 'auc_results_all_letters.csv')}")
    if not args.all_letters_only:
        print(f" - {os.path.join(metrics_dir, 'auc_results_neutral.csv')}")
    if args.errorbar:
        print("Error bars (std) are included.")
if __name__ == "__main__":
    main()
