"""
Compare stance-drift across the GPT-5.4 prompt / reasoning variations.

Reads ``postprocess_results/model_comparison.csv`` (produced by ``postprocess.py``)
and, for the 4-rung GPT-5.4 ladder

    gpt_5_4 (default configuration)  ->  reflection  ->  third-person reflection
    ->  third-person reflection with medium reasoning effort

emits a side-by-side comparison of, per initial stance (and overall):

  * SPR  (``Mean_mean``): diagonal stance-preservation rate
  * the five SDR drift modes (Pol / Mod / ModNeut / Flip / Dev) that partition
    the off-diagonal mass (``Mean_{Pol,Mod,ModNeut,Flip,Dev}_mean``)

Outputs to ``postprocess_results/variation_comparison/``:
  * ``spr_by_stance.csv``        wide SPR table (stance x variation, mean & SE)
  * ``drift_modes_long.csv``     tidy (variation, stance, SPR, Pol, Mod, ModNeut, Flip,
                                 Dev, each with a ``_se`` twin); stance "Overall" rows
                                 use the two-stage aggregation of the SDR summary table
  * ``ladder_overall.csv``       one row per rung: overall mean/SE per quantity and the
                                 change from the previous rung and from vanilla (fractions)
  * ``spr_by_stance.pdf``        grouped bars: SPR by stance, one bar per variation
  * ``polarization_by_stance.pdf`` grouped bars: Pol drift by stance
  * a Markdown summary printed to stdout

Run AFTER ``postprocess.py`` has populated ``model_comparison.csv`` for each
variation in the ladder. Pure pandas/matplotlib -- no API needed.

    python results/compare_prompt_variations.py
"""

import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

LETTERS = ["A", "B", "C", "D", "E"]
STANCE_LABEL = {
    "A": "Agree\nstrongly",
    "B": "Agree",
    "C": "Neutral",
    "D": "Disagree",
    "E": "Disagree\nstrongly",
}

# The ladder, in order, with concise labels for legends/columns. Each step adds
# exactly one factor to the previous, so differences isolate that factor.
LADDER = [
    ("gpt_5_4", "default"),
    ("gpt_5_4_reflection", "reflection"),
    ("gpt_5_4_reflection_third_person", "third-person reflection"),
    ("gpt_5_4_reflection_third_person_medium", "third-person reflection, medium reasoning"),
]

MODES = ["Pol", "Mod", "ModNeut", "Flip", "Dev"]
MODE_FULL = {
    "Pol": "Polarization",
    "Mod": "Moderation",
    "ModNeut": "Moderation to neutral",
    "Flip": "Flipping",
    "Dev": "Deviation from neutrality",
}
# Colour-blind-friendly sequence (one per ladder rung).
BAR_COLORS = ["#4C72B0", "#55A868", "#C44E52", "#8172B3"]


def _grouped_bar(df_wide, se_wide, value_name, title, out_path, labels):
    """Grouped bar: x = stance, one bar per ladder rung. df_wide rows=stance(+Overall)."""
    stances = list(df_wide.index)
    n_grp = len(stances)
    n_bar = len(labels)
    x = np.arange(n_grp)
    width = 0.8 / n_bar

    fig, ax = plt.subplots(figsize=(max(11, 2.0 * n_grp), 7))
    for k, (model, lab) in enumerate(labels):
        vals = df_wide[model].to_numpy()  # fractions (plots keep the leading zero; DECISIONS.md 2026-09-13)
        errs = se_wide[model].to_numpy() if se_wide is not None else None
        ax.bar(
            x + (k - (n_bar - 1) / 2) * width,
            vals,
            width=width,
            yerr=errs,
            capsize=4,
            color=BAR_COLORS[k % len(BAR_COLORS)],
            edgecolor="#333333",
            label=lab,
            error_kw={"linewidth": 1.2, "ecolor": "#444444"},
        )
    xt = [STANCE_LABEL.get(s, s) for s in stances]
    ax.set_xticks(x)
    ax.set_xticklabels(xt, fontsize=13, fontweight="bold")
    ax.set_ylabel(value_name, fontsize=15, fontweight="bold")
    ax.set_xlabel("Initial stance", fontsize=15, fontweight="bold")
    ax.set_title(title, fontsize=16, fontweight="bold", pad=12)
    ax.set_ylim(0, 1.05)
    ax.legend(fontsize=12, ncol=2, framealpha=0.9)
    ax.yaxis.grid(True, linestyle="-", linewidth=0.6, alpha=0.25)
    ax.set_axisbelow(True)
    plt.tight_layout()
    plt.savefig(out_path, bbox_inches="tight", dpi=300)
    plt.close()
    print(f"  saved {out_path}")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--comparison_csv",
        default="postprocess_results/model_comparison.csv",
        help="Path to the model_comparison.csv produced by postprocess.py",
    )
    ap.add_argument(
        "--out_dir",
        default="postprocess_results/variation_comparison",
        help="Directory for the comparison tables/figures",
    )
    args = ap.parse_args()

    df = pd.read_csv(args.comparison_csv)
    present = [(m, lab) for (m, lab) in LADDER if m in set(df.Model.unique())]
    missing = [m for (m, _) in LADDER if m not in set(df.Model.unique())]
    if missing:
        print(f"WARNING: ladder models absent from {args.comparison_csv}: {missing}")
    if not present:
        raise SystemExit("No ladder models found; run postprocess.py first.")

    os.makedirs(args.out_dir, exist_ok=True)
    models = [m for (m, _) in present]

    sub = df[df.Model.isin(models)].copy()
    # Stable stance ordering A..E for pivots.
    sub["Letter"] = pd.Categorical(sub["Letter"], categories=LETTERS, ordered=True)

    # --- Wide SPR table (mean & SE), stance x model, with an Overall row. ----
    spr = sub.pivot(index="Letter", columns="Model", values="Mean_mean")[models]
    spr_se = sub.pivot(index="Letter", columns="Model", values="Mean_standard_error")[models]
    # Overall = unweighted mean over the 5 equal-n stance means (= grand mean).
    spr.loc["Overall"] = spr.mean(axis=0)
    spr_se.loc["Overall"] = spr_se.mean(axis=0)  # rough; per-stance SEs already small

    # --- Tidy long table: mean and SE per rung, stance and class; plus Overall rows.
    # Per-stance rows come from model_comparison.csv (mean and SE over the 112
    # propositions). Overall rows are two-stage, as in
    # faithfulness_metric.generate_sdr_latex_tables: mean over the five stances within
    # each (Proposition_ID, Perm), then mean and SE = std(ddof=1)/sqrt(n) across those.
    quantities = [("SPR", "Mean")] + [(m, f"Mean_{m}") for m in MODES]
    long_rows = []
    for _, r in sub.iterrows():
        row = {"variation": dict(present)[r.Model], "model": r.Model, "stance": r.Letter}
        for q, col in quantities:
            row[q] = r[f"{col}_mean"]
            row[f"{q}_se"] = r[f"{col}_standard_error"]
        long_rows.append(row)
    overall = {}
    for model, lab in present:
        detail_csv = os.path.join(os.path.dirname(args.comparison_csv), model, "postprocess_detailed_results.csv")
        det = pd.read_csv(detail_csv)
        cols = [col for _, col in quantities]
        topic_level = det.groupby(["Proposition_ID", "Perm"])[cols].mean()
        n = len(topic_level)
        means, ses = topic_level.mean(), topic_level.std(ddof=1) / np.sqrt(n)
        row = {"variation": lab, "model": model, "stance": "Overall"}
        for q, col in quantities:
            row[q] = means[col]
            row[f"{q}_se"] = ses[col]
        long_rows.append(row)
        overall[model] = row
    stance_order = LETTERS + ["Overall"]
    long_df = pd.DataFrame(long_rows)
    long_df["stance"] = pd.Categorical(long_df["stance"], categories=stance_order, ordered=True)
    long_df["model"] = pd.Categorical(long_df["model"], categories=models, ordered=True)
    long_df = long_df.sort_values(["model", "stance"]).reset_index(drop=True)
    long_df["stance"] = long_df["stance"].astype(str)
    long_df["model"] = long_df["model"].astype(str)

    # --- Ladder table: one row per rung, overall mean/SE, and the change from the
    # previous rung and from the vanilla rung (fractions; no test, no CI). -------
    ladder_rows = []
    for k, (model, lab) in enumerate(present):
        row = {"rung": k, "variation": lab, "model": model}
        for q, _ in quantities:
            row[q] = overall[model][q]
            row[f"{q}_se"] = overall[model][f"{q}_se"]
            row[f"{q}_change_prev"] = (overall[model][q] - overall[present[k - 1][0]][q]) if k else np.nan
            row[f"{q}_change_vanilla"] = overall[model][q] - overall[present[0][0]][q]
        ladder_rows.append(row)
    ladder_df = pd.DataFrame(ladder_rows)

    spr_csv = os.path.join(args.out_dir, "spr_by_stance.csv")
    long_csv = os.path.join(args.out_dir, "drift_modes_long.csv")
    ladder_csv = os.path.join(args.out_dir, "ladder_overall.csv")
    spr.to_csv(spr_csv)
    long_df.to_csv(long_csv, index=False)
    ladder_df.to_csv(ladder_csv, index=False)
    print(f"  saved {spr_csv}")
    print(f"  saved {long_csv}")
    print(f"  saved {ladder_csv}")

    # --- Figures -----------------------------------------------------------
    pol = sub.pivot(index="Letter", columns="Model", values="Mean_Pol_mean")[models]
    pol_se = sub.pivot(index="Letter", columns="Model", values="Mean_Pol_standard_error")[models]
    _grouped_bar(
        spr.drop(index="Overall"), spr_se.drop(index="Overall"),
        "Success rate",
        "GPT-5.4: success rate by initial stance",
        os.path.join(args.out_dir, "spr_by_stance.pdf"), present,
    )
    _grouped_bar(
        pol, pol_se,
        "Mass of polarization",
        "GPT-5.4: polarization by initial stance",
        os.path.join(args.out_dir, "polarization_by_stance.pdf"), present,
    )

    # --- Markdown summary to stdout ----------------------------------------
    labels = [lab for (_, lab) in present]
    print("\n### SPR (diagonal stance-preservation), % by initial stance\n")
    header = "| Stance | " + " | ".join(labels) + " |"
    print(header)
    print("|" + "---|" * (len(labels) + 1))
    for stance in LETTERS + ["Overall"]:
        cells = [f"{spr.loc[stance, m] * 100:.1f}" for m in models]
        name = "Overall" if stance == "Overall" else f"{stance} ({STANCE_LABEL[stance].replace(chr(10), ' ')})"
        print(f"| {name} | " + " | ".join(cells) + " |")

    for mode in MODES:
        print(f"\n### {MODE_FULL[mode]} ({mode}), % by initial stance, mean (SE)\n")
        print(header)
        print("|" + "---|" * (len(labels) + 1))
        for stance in stance_order:
            cells = []
            for m in models:
                r = long_df[(long_df.model == m) & (long_df.stance == stance)].iloc[0]
                cells.append(f"{r[mode] * 100:.1f} ({r[mode + '_se'] * 100:.1f})")
            name = "Overall" if stance == "Overall" else f"{stance} ({STANCE_LABEL[stance].replace(chr(10), ' ')})"
            print(f"| {name} | " + " | ".join(cells) + " |")

    print("\n### Change in overall mass from the previous rung, percentage points\n")
    print("| Rung | " + " | ".join(q for q, _ in quantities) + " |")
    print("|" + "---|" * (len(quantities) + 1))
    for _, r in ladder_df.iterrows():
        cells = ["" if np.isnan(r[f"{q}_change_prev"]) else f"{r[f'{q}_change_prev'] * 100:+.1f}" for q, _ in quantities]
        print(f"| {r['variation']} | " + " | ".join(cells) + " |")

    print("\nDone.")


if __name__ == "__main__":
    main()
