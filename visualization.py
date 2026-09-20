import os
import json
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from postprocess import letter_to_option

LETTERS = ["A", "B", "C", "D", "E"]


def get_raw_json_path(topic_id, model, prompt_id, results_dir="results"):
    """
    Construct path to a raw results JSON file.

    Args:
        topic_id (str): Proposition topic ID (e.g., "1161")
        model (str): Internal model name (e.g., "gpt_4o_mini")
        prompt_id (int): Prompt ID number
        results_dir (str): Base results directory

    Returns:
        tuple: (json_file_path, base_name_for_plots)
    """
    base_name = f"debate_speech_{topic_id}_{model}_prompt_{prompt_id}"
    json_file_path = os.path.join(
        results_dir, model, base_name + "_raw.json"
    )
    return json_file_path, base_name

def visualize_transition_matrices(results_tranmat, letter_list, title, plot_file_path, present_se = False, panel='left'):
    if panel=='left': 
        cbar = False
        width = 14.724
    elif panel=='right':
        cbar = True
        width = 14.308
    elif panel=='center':
        cbar = False
        width = 11.557 
    else:
        cbar = True
        width = 17.846
    
    fig, ax = plt.subplots(
        1, 1, figsize=(width, 15)
        )

    if present_se:
        mean_pct = pd.DataFrame(results_tranmat["mean"] * 100)
        se_pct = pd.DataFrame(results_tranmat["se"] * 100)
        # Create annotation matrix: "mean (se)" in percentage with 1 decimal place
        annot = mean_pct.apply(lambda col: col.map(lambda x: f"{x:.1f}")) + \
                "\n(" + se_pct.apply(lambda col: col.map(lambda x: f"{x:.1f}")) + ")"
        # Update title to mention percent
        title = title + "\n(Values in %; Annot: Mean (SE))"
        sns.heatmap(
            mean_pct,
            ax=ax,
            cmap="Reds",
            cbar=cbar,
            annot=annot.values,
            annot_kws={"fontsize": 35},
            fmt="",
            vmin=0,
            vmax=100
            )
    else:
        # Display as percentage with 1 decimal place
        data_pct = pd.DataFrame(results_tranmat * 100)
        annot = data_pct.apply(lambda col: col.map(lambda x: f"{x:.1f}"))
        title = title + "\n(Values in %)"
        sns.heatmap(
            data_pct, ax=ax, annot=annot.values, fmt="",
            cmap="Reds", cbar=cbar, vmin=0, vmax=100,
            annot_kws={"fontsize": 35}
        )

    xticks = np.arange(0, len(letter_list), 1)
    letter_xticks = [letter_to_option(letter_list[tick]) for tick in list(xticks)]
    yticks = np.arange(len(letter_list) - 1, -1, -1)
    letter_yticks = [letter_to_option(letter_list[tick]) for tick in list(yticks)]

    ax.set_title(title, fontsize=36, fontweight="bold")
    ax.set_xlabel("To State", fontsize=35, fontweight="bold")
    ax.set_xticks(xticks + 0.5, [tick.replace(" ", "\n") for tick in letter_xticks], fontsize=35, rotation=30, fontweight="bold", ha="center")

    if panel=='right' or panel=='center':
        ax.set_ylabel(None)
        ax.set_yticks([],[])
    if panel=='right' or panel=='single':
        ax.figure.axes[-1].yaxis.set_tick_params(labelsize=35)
    if panel=='left' or panel=='single':
        ax.set_ylabel("From State", fontsize=35, fontweight="bold")
        ax.set_yticks(yticks + 0.5, [tick.replace(" ", "\n") for tick in letter_yticks], fontsize=35, rotation=30, fontweight="bold", ha="right")


    plt.tight_layout()
    plt.savefig(f"{plot_file_path}.pdf", bbox_inches="tight", dpi=500)
    #plt.show()
    
    print(f"Plot reconstructed and saved to: {plot_file_path}.pdf")
    return

def load_and_compute(json_file_path):
    """Load raw JSON and compute mean and SE of the transition matrix."""
    with open(json_file_path, "r") as f:
        data = json.load(f)
    perm_key = list(data.keys())[0]
    arr = np.asarray(data[perm_key])
    mean = np.mean(arr, axis=0)
    se = np.std(arr, axis=0, ddof=1) / np.sqrt(arr.shape[0])
    return mean, se


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Visualize empirical transition matrices from debate speech experiments."
    )
    parser.add_argument(
        "--topic_id",
        type=str,
        required=True,
        help="Proposition topic_id from propositions.json (e.g., 1161)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="gpt_4o_mini",
        help="Internal model name (default: gpt_4o_mini)",
    )
    parser.add_argument(
        "--prompt_id",
        type=int,
        default=1,
        help="Prompt ID (default: 1)",
    )
    parser.add_argument(
        "--title",
        type=str,
        default="Empirical Transition Matrix",
        help="Title for the plot (default: 'Empirical Transition Matrix')",
    )
    parser.add_argument(
        "--panel",
        type=str,
        default="single",
        choices=["left", "center", "right", "single"],
        help="Panel layout for figure sizing/colorbar (default: single)",
    )
    parser.add_argument(
        "--ideal-only",
        action="store_true",
        help="Only generate the ideal identity matrix figure",
    )
    return parser.parse_args()


def main():
    args = parse_arguments()
    # Decode escape sequences in title (e.g., \n -> actual newline)
    if args.title:
        args.title = args.title.encode().decode('unicode_escape')
    os.makedirs("figures", exist_ok=True)

    # Always generate the ideal identity matrix
    if args.ideal_only:
        visualize_transition_matrices(
            np.eye(5), LETTERS,
            "Perfect Stance Preservation\n-Identity Matrix",
            "figures/ideal",
            present_se=False, panel="right",
        )
        return

    # Load data for the given topic_id
    json_file_path, base_name = get_raw_json_path(
        topic_id=args.topic_id,
        model=args.model,
        prompt_id=args.prompt_id,
    )

    if not os.path.exists(json_file_path):
        print(f"Error: File not found: {json_file_path}")
        return

    mean, se = load_and_compute(json_file_path)
    plot_file_path = os.path.join("figures", base_name)

    # Generate mean-only heatmap
    visualize_transition_matrices(
        mean, LETTERS, args.title,
        plot_file_path + f"_{args.panel}", present_se=False, panel=args.panel,
    )

    # Generate SE-annotated heatmap
    visualize_transition_matrices(
        {"mean": mean, "se": se}, LETTERS, args.title,
        plot_file_path + f"_se_{args.panel}", present_se=True, panel=args.panel,
    )


if __name__ == "__main__":
    main()