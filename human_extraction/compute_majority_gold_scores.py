"""
Human-majority gold: descriptive mean and **standard errors (SE)** for survey-level metrics.

Gold G_i per row is ``majority_human_extract`` (human vote-share argmax; A–E tie-break
from ``postprocess_human_extraction.argmax_with_tie_info``).

Human cohort
------------
For each Qualtrics respondent p, score_p = (1/100) * sum_i Y_{p,i}, where
Y_{p,i} = 1 if the parsed letter in column ``i_Q2`` equals G_{i-1}, else 0.
Missing / unparsable cells: counted as 0 (wrong) with denominator 100 unless
``--missing-exclude`` (then normalize by number of valid item responses).

AI survey-level (per-encode agreement; no AI majority label)
------------------------------------------------------------
For row i, let n = ``ai_total_votes_n`` (expected 9). Let G be the gold letter.
The count of encodes equal to G is ``ai_total_count_{G}``. Define
X_i = p_hat_i = (that count) / n (the **sample proportion** of votes matching G).
Survey mean S_AI = (1/100) * sum_i X_i.

Two different “per-question” SD notions (do not mix them up)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Let z_1..z_n be the n binary indicators (1 = vote equals G, else 0), so p_hat = mean(z).

* **Sample SD of the n votes** (spread of 0/1 draws): s = sqrt( (n/(n-1)) * p_hat * (1-p_hat) ).
  This is *not* what we plug into the survey-level propagation for S_AI.

* **Estimated variance of X_i = p_hat** (sampling variance of the *proportion*):
  plugin Var_hat ≈ p_hat*(1-p_hat)/n; unbiased (for Var(p_hat) under Bernoulli) uses
  Var_hat = s**2 / n = p_hat*(1-p_hat)/(n-1). We use the **(n-1)** denominator here.

Human SE (cohort mean)
^^^^^^^^^^^^^^^^^^^^^^
Let bar(H) = mean_p score_p over respondents with finite scores, N = that count.
**SE(bar(H)) = SD(score_p) / sqrt(N)** (sample SD with ddof=1). This matches the
role of the AI propagated **SE(S_AI)** below: uncertainty for a **survey-level mean**,
not between-person spread.

Propagated SE for S_AI (always reported)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
If items are uncorrelated: Var(S_AI) = (1/100**2) * sum_i Var(X_i), hence
**SE_hat(S_AI) = (1/100) * sqrt( sum_i Var_hat(X_i) )**.

**Not** sqrt(sum_i s_i) / 100 and **not** sum_i sqrt(Var_hat(X_i)) / 100: independent
summands add **variances**, not standard deviations; the square root applies **once**
after the sum.

Assumption comment (1): Across items, X_i are treated as uncorrelated for this
propagation; real items often correlate positively, so this SE can be optimistic
(smaller than a dependence-aware uncertainty).

Assumption comment (2): Within each item, Var_hat uses the Bernoulli / exchangeable
sampling model for the n encodes (IID at rate p_i).

Inputs: 9-encode workbook (``*_9encode.xlsx``) with human majority + ``ai_total_count_*``,
and the de-identified label CSV (``human_labels_3234.csv``: annotator number, ``1_Q2``..``100_Q2``).

Outputs: prints summary to stdout and writes a one-row CSV (``--output-csv``).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Reuse Qualtrics parsing from sibling module (run from repo root or human_extraction/).
SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from postprocess_human_extraction import (  # noqa: E402
    LETTERS,
    load_qualtrics_response_rows,
    parse_letter,
)


AI_TOTAL_COUNT_COLS = [f"ai_total_count_{L}" for L in LETTERS]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Compute human cohort mean and SE(mean) vs majority_human_extract, and AI S_AI "
            "mean / propagated SE from 9-encode vote counts (consistent uncertainty scale)."
        )
    )
    p.add_argument(
        "--xlsx-9encode",
        type=Path,
        default=Path("human_extraction/debata_speeches_3234_filtered_with_human_9encode.xlsx"),
        help="Workbook with majority_human_extract and ai_total_count_A..E (+ ai_total_votes_n).",
    )
    p.add_argument(
        "--human-csv",
        type=Path,
        default=Path("human_extraction/human_labels_3234.csv"),
        help="De-identified label CSV (annotator number, 1_Q2..100_Q2) aligned to workbook row order.",
    )
    p.add_argument(
        "--missing-exclude",
        action="store_true",
        help="Per participant, average only over items with a parsed letter (denominator = valid count).",
    )
    p.add_argument(
        "--output-csv",
        type=Path,
        default=Path("human_extraction/majority_gold_scores.csv"),
        help="Write one-row summary of metrics and metadata to this CSV path.",
    )
    return p.parse_args()


def _gold_letter(cell) -> str:
    s = str(cell).strip().upper()
    if len(s) != 1 or s not in LETTERS:
        raise ValueError(f"Invalid majority_human_extract value: {cell!r}")
    return s


def validate_workbook(df: pd.DataFrame) -> None:
    if len(df) != 100:
        raise ValueError(f"Expected 100 rows in workbook, got {len(df)}")
    if "majority_human_extract" not in df.columns:
        raise ValueError("Workbook missing column majority_human_extract")
    missing = [c for c in AI_TOTAL_COUNT_COLS if c not in df.columns]
    if missing:
        raise ValueError(
            f"Workbook missing ai_total_count_* columns (need 9-encode output): {missing}"
        )
    if "ai_total_votes_n" not in df.columns:
        raise ValueError("Workbook missing ai_total_votes_n (need 9-encode output)")


def participant_scores(
    human_df: pd.DataFrame,
    gold: list[str],
    *,
    missing_exclude: bool,
) -> np.ndarray:
    item_cols = [f"{k}_Q2" for k in range(1, 101)]
    missing_cols = [c for c in item_cols if c not in human_df.columns]
    if missing_cols:
        raise ValueError(f"Human CSV missing expected columns (sample): {missing_cols[:5]}")

    scores: list[float] = []
    for _, row in human_df.iterrows():
        correct = 0
        valid = 0
        for j, col in enumerate(item_cols):
            letter = parse_letter(row.get(col, ""))
            g = gold[j]
            if letter is None:
                if not missing_exclude:
                    correct += 0
                continue
            valid += 1
            if letter == g:
                correct += 1
        if missing_exclude:
            if valid == 0:
                scores.append(float("nan"))
            else:
                scores.append(correct / valid)
        else:
            scores.append(correct / 100.0)
    return np.asarray(scores, dtype=float)


def ai_survey_mean_and_se(df: pd.DataFrame) -> tuple[float, float, np.ndarray]:
    """
    Returns S_AI, propagated SE(S_AI), and per-row X_i = p_hat_i (length 100).

    Var_hat(X_i) = p_hat*(1-p_hat)/(n-1) for n>1 (same as s^2/n with sample SD
    s = sqrt(n/(n-1)*p_hat*(1-p_hat)) of the n binary votes). Not the plugin p(1-p)/n.

    SE(S_AI) = (1/n_rows) * sqrt(sum_i Var_hat(X_i)) under uncorrelated items.
    """
    gold_series = df["majority_human_extract"].map(_gold_letter)
    n_votes = df["ai_total_votes_n"].to_numpy(dtype=float)
    if not np.allclose(n_votes, n_votes[0]) or int(round(n_votes[0])) != n_votes[0]:
        raise ValueError(f"Unexpected ai_total_votes_n values: unique={np.unique(n_votes)}")
    n0 = int(round(n_votes[0]))
    if n0 < 2:
        raise ValueError(f"Need ai_total_votes_n >= 2 for variance of proportion, got {n0}")

    counts_mat = df[AI_TOTAL_COUNT_COLS].to_numpy(dtype=float)
    letter_to_idx = {L: i for i, L in enumerate(LETTERS)}

    x_list: list[float] = []
    var_list: list[float] = []
    for i in range(len(df)):
        g = gold_series.iloc[i]
        idx = letter_to_idx[g]
        hit = float(counts_mat[i, idx])
        nv = float(n_votes[i])
        if abs(np.sum(counts_mat[i]) - nv) > 1e-6:
            raise ValueError(f"Row {i}: ai_total_count_* sum {np.sum(counts_mat[i])} != ai_total_votes_n {nv}")
        p_hat = hit / nv
        x_list.append(p_hat)
        # Var(sample proportion) estimator: s^2/n = p_hat*(1-p_hat)/(n-1); NOT sum of per-vote SDs.
        var_list.append(p_hat * (1.0 - p_hat) / (nv - 1.0))

    x = np.asarray(x_list, dtype=float)
    s_ai = float(np.mean(x))
    se_ai = float(np.sqrt(np.sum(var_list)) / len(df))
    return s_ai, se_ai, x


def main() -> None:
    args = parse_args()
    df = pd.read_excel(args.xlsx_9encode, engine="openpyxl")
    df = df.reset_index(drop=True)
    validate_workbook(df)

    gold = [_gold_letter(v) for v in df["majority_human_extract"].tolist()]
    human_df = load_qualtrics_response_rows(args.human_csv)
    scores = participant_scores(human_df, gold, missing_exclude=args.missing_exclude)
    finite = scores[np.isfinite(scores)]
    if finite.size == 0:
        raise ValueError("No finite participant scores (check CSV and --missing-exclude).")

    mean_h = float(np.mean(finite))
    n_fin = int(finite.size)
    sd_between = float(np.std(finite, ddof=1)) if n_fin > 1 else float("nan")
    se_mean_h = float(sd_between / np.sqrt(n_fin)) if n_fin > 1 and np.isfinite(sd_between) else float("nan")

    s_ai, se_ai, _x = ai_survey_mean_and_se(df)
    n_votes_per_row = int(round(float(df["ai_total_votes_n"].iloc[0])))

    missing_policy = (
        "exclude_from_denominator_per_participant"
        if args.missing_exclude
        else "missing_counts_wrong_denominator_100"
    )
    out_row = {
        "human_mean_score_p": mean_h,
        "human_se_mean_score_p": se_mean_h,
        "human_sd_between_participants": sd_between,
        "n_respondents": len(human_df),
        "n_participants_finite_score": n_fin,
        "missing_policy": missing_policy,
        "s_ai_mean_per_row_hit_rate": s_ai,
        "s_ai_propagated_se": se_ai,
        "n_items": len(df),
        "ai_n_votes_per_row": n_votes_per_row,
        "var_xi_formula": (
            "p_hat*(1-p_hat)/(n-1) per item; SE(S_AI)=(1/n_items)*sqrt(sum_i Var_hat(X_i))"
        ),
        "xlsx_9encode": args.xlsx_9encode.as_posix(),
        "human_csv": args.human_csv.as_posix(),
        "assumption_across_items": (
            "X_i treated as uncorrelated for propagated SE; positive cross-item correlation "
            "may make this SE smaller than dependence-aware uncertainty."
        ),
        "assumption_within_item": (
            "Var(X_i) uses Bernoulli/exchangeable encodes; p_hat*(1-p_hat)/(n-1) = s^2/n for binary votes."
        ),
    }
    out_path = args.output_csv
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([out_row]).to_csv(out_path, index=False, encoding="utf-8")

    print("--- Majority-gold descriptive scores ---")
    print(f"Workbook: {args.xlsx_9encode.resolve()}")
    print(f"Human CSV: {args.human_csv.resolve()}")
    print(f"Respondents (rows): {len(human_df)}")
    print(f"Missing policy: {'exclude from denominator per participant' if args.missing_exclude else 'count as wrong; denominator 100'}")
    print()
    print(f"Human mean(score_p):        {mean_h:.6f}")
    print(f"Human SE(mean score_p):     {se_mean_h:.6f}  (SD_between / sqrt(N); N={n_fin})")
    print(f"Human SD between participants (descriptive): {sd_between:.6f}")
    print()
    print(f"S_AI (mean per-row encode hit rate vs G_i): {s_ai:.6f}")
    print(f"Propagated SE(S_AI):                        {se_ai:.6f}")
    print()
    print("Assumption comment (report next to propagated SE(S_AI)):")
    print("  (1) Across items: X_i treated as uncorrelated; positive cross-item correlation")
    print("      can make this SE smaller than a dependence-aware uncertainty.")
    print("  (2) Within item: Var(X_i) estimated as p_hat*(1-p_hat)/(n-1) (= s^2/n for binary votes);")
    print("      Bernoulli / exchangeable encodes. Not sqrt(sum of per-item vote SDs).")
    print()
    print(f"Wrote CSV: {out_path.resolve()}")


if __name__ == "__main__":
    main()
