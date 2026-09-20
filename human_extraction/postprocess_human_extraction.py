"""
Postprocess human extraction labels: merge into filtered debate xlsx with
prob_human columns, majority labels, tie flags, and accuracy / tie diagnostics.
Never overwrites the input filtered workbook by default.

The human labels are read from the de-identified file
``human_extraction/human_labels_3234.csv``: one row per annotator (numbered 1 to 9),
one column per argument (``1_Q2`` .. ``100_Q2``), each cell a stance letter A-E.
The survey export it was derived from is not distributed.
"""

from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path

import numpy as np
import pandas as pd

LETTERS = ["A", "B", "C", "D", "E"]
AI_PROB_COLS = [f"prob_AI_extract_{L}" for L in LETTERS]
HUMAN_PROB_COLS = [f"prob_human_extraction_{L}" for L in LETTERS]
DATA_ROW_START_RE = re.compile(r"^\d+$")  # first column: annotator number
LETTER_FROM_CELL_RE = re.compile(r"^([A-E])(?:\)|$)", re.IGNORECASE)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Merge human extraction ratings into filtered debate xlsx."
    )
    p.add_argument(
        "--filtered-xlsx",
        type=Path,
        default=Path("human_extraction/debata_speeches_3234_filtered.xlsx"),
        help="Input filtered workbook (read-only unless same as output by mistake).",
    )
    p.add_argument(
        "--human-csv",
        type=Path,
        default=Path("human_extraction/human_labels_3234.csv"),
        help="De-identified label CSV: annotator number, then 1_Q2..100_Q2 (letters A-E).",
    )
    p.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output xlsx path. Default: <filtered_stem>_with_human.xlsx next to input.",
    )
    p.add_argument(
        "--summary-sheet",
        action="store_true",
        help="Add a second sheet 'summary' with accuracy and tie tables.",
    )
    return p.parse_args()


def default_output_path(filtered_xlsx: Path) -> Path:
    return filtered_xlsx.with_name(f"{filtered_xlsx.stem}_with_human.xlsx")


def load_qualtrics_response_rows(csv_path: Path) -> pd.DataFrame:
    """Load the label CSV: header row 0, data rows start with the annotator number in column 0."""
    with csv_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.reader(f)
        try:
            header = next(reader)
        except StopIteration as exc:
            raise ValueError(f"Empty CSV: {csv_path}") from exc

        ncols = len(header)
        data_rows: list[list[str]] = []
        for row in reader:
            if not row:
                continue
            first = row[0].strip()
            if DATA_ROW_START_RE.match(first):
                if len(row) < ncols:
                    row = row + [""] * (ncols - len(row))
                else:
                    row = row[:ncols]
                data_rows.append(row)

    if not data_rows:
        raise ValueError(
            f"No data rows found in {csv_path} (expected an annotator number in the first column)."
        )

    return pd.DataFrame(data_rows, columns=header)


def parse_letter(cell: str) -> str | None:
    if cell is None or (isinstance(cell, float) and np.isnan(cell)):
        return None
    s = str(cell).strip()
    if not s:
        return None
    m = LETTER_FROM_CELL_RE.match(s)
    if not m:
        return None
    return m.group(1).upper()


def human_proportions_per_item(
    human_df: pd.DataFrame, item_cols: list[str]
) -> tuple[np.ndarray, np.ndarray]:
    """
    Returns:
        props: shape (100, 5) proportions per item column order
        counts_valid: shape (100,) number of valid responses per item
    """
    n_items = len(item_cols)
    counts = np.zeros((n_items, 5), dtype=float)
    valid_per_item = np.zeros(n_items, dtype=int)

    for _, row in human_df.iterrows():
        for j, col in enumerate(item_cols):
            letter = parse_letter(row.get(col, ""))
            if letter is None:
                continue
            idx = LETTERS.index(letter)
            counts[j, idx] += 1.0
            valid_per_item[j] += 1

    props = np.zeros_like(counts)
    for j in range(n_items):
        n = valid_per_item[j]
        if n > 0:
            props[j, :] = counts[j, :] / n
    return props, valid_per_item


def argmax_with_tie_info(
    probs: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, list[list[str]]]:
    """
    probs: (n_rows, 5)
    Returns resolved letter (deterministic tie-break A..E), is_tie bool, tied_letters lists.
    """
    n = probs.shape[0]
    resolved = []
    is_tie = np.zeros(n, dtype=bool)
    tied_lists: list[list[str]] = []

    for i in range(n):
        p = probs[i]
        m = float(np.max(p))
        at_max = [LETTERS[j] for j in range(5) if abs(p[j] - m) < 1e-12]
        tied_lists.append(at_max)
        is_tie[i] = len(at_max) > 1
        # deterministic: smallest letter in A..E order among tied
        resolved.append(min(at_max, key=LETTERS.index))

    return np.array(resolved), is_tie, tied_lists


def main() -> None:
    args = parse_args()
    filtered_path: Path = args.filtered_xlsx
    human_path: Path = args.human_csv
    out_path = args.output if args.output is not None else default_output_path(filtered_path)

    if out_path.resolve() == filtered_path.resolve():
        raise ValueError(
            "Refusing to write: --output matches --filtered-xlsx. Choose a different output path."
        )

    base = pd.read_excel(filtered_path, engine="openpyxl")
    if len(base) != 100:
        raise ValueError(f"Expected 100 rows in filtered xlsx, got {len(base)}")

    for c in AI_PROB_COLS:
        if c not in base.columns:
            raise ValueError(f"Missing column {c} in {filtered_path}")

    item_cols = [f"{k}_Q2" for k in range(1, 101)]
    human_df = load_qualtrics_response_rows(human_path)
    missing = [c for c in item_cols if c not in human_df.columns]
    if missing:
        raise ValueError(f"Missing expected columns in human CSV: {missing[:5]}...")

    props, _valid_per_item = human_proportions_per_item(human_df, item_cols)
    for j, letter in enumerate(LETTERS):
        base[HUMAN_PROB_COLS[j]] = props[:, j]

    ai_mat = base[AI_PROB_COLS].to_numpy(dtype=float)
    human_mat = base[HUMAN_PROB_COLS].to_numpy(dtype=float)

    maj_ai, tie_ai, tied_ai = argmax_with_tie_info(ai_mat)
    maj_h, tie_h, tied_h = argmax_with_tie_info(human_mat)

    base["majority_AI_extract"] = maj_ai
    base["majority_human_extract"] = maj_h
    base["tie_AI_extract"] = tie_ai
    base["tie_human_extract"] = tie_h
    base["tied_letters_AI"] = [",".join(x) for x in tied_ai]
    base["tied_letters_human"] = [",".join(x) for x in tied_h]

    agree = maj_ai == maj_h
    accuracy = float(np.mean(agree))
    no_tie_mask = ~(tie_ai | tie_h)
    acc_no_tie = (
        float(np.mean(agree[no_tie_mask])) if np.any(no_tie_mask) else float("nan")
    )

    row_idx_ai = np.where(tie_ai)[0].tolist()
    row_idx_h = np.where(tie_h)[0].tolist()
    row_idx_both = np.where(tie_ai & tie_h)[0].tolist()

    def describe_rows(indices: list[int]) -> str:
        parts = []
        for i in indices:
            stance = base.iloc[i].get("initial stance", "")
            rep = base.iloc[i].get("rep", "")
            parts.append(f"row_index={i} (stance={stance}, rep={rep})")
        return "; ".join(parts) if parts else "(none)"

    print("--- Human extraction postprocess ---")
    print(f"Respondents: {len(human_df)}")
    print(f"Accuracy (resolved majority AI == resolved majority human): {accuracy:.4f}")
    print(
        f"Accuracy excluding rows with any tie (AI or human): {acc_no_tie:.4f} "
        f"(n={int(no_tie_mask.sum())})"
    )
    print(f"Tie count AI: {int(tie_ai.sum())}; rows: {row_idx_ai}")
    print(f"  {describe_rows(row_idx_ai)}")
    print(f"Tie count human: {int(tie_h.sum())}; rows: {row_idx_h}")
    print(f"  {describe_rows(row_idx_h)}")
    print(f"Tie count both: {len(row_idx_both)}; rows: {row_idx_both}")
    print(f"  {describe_rows(row_idx_both)}")
    print(f"Output: {out_path.resolve()}")

    col_order = (
        ["proposition", "argument", "initial stance"]
        + ["majority_AI_extract", "majority_human_extract"]
        + ["rep"]
        + AI_PROB_COLS
        + HUMAN_PROB_COLS
        + [
            "tie_AI_extract",
            "tie_human_extract",
            "tied_letters_AI",
            "tied_letters_human",
        ]
    )
    for c in col_order:
        if c not in base.columns:
            raise ValueError(f"Internal error: missing column {c}")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(out_path, engine="openpyxl") as writer:
        base[col_order].to_excel(writer, sheet_name="data", index=False)
        if args.summary_sheet:
            summary_rows = [
                {"metric": "accuracy_resolved", "value": accuracy},
                {"metric": "accuracy_excluding_any_tie_row", "value": acc_no_tie},
                {"metric": "n_rows", "value": 100},
                {"metric": "n_respondents", "value": len(human_df)},
                {"metric": "n_tie_AI", "value": int(tie_ai.sum())},
                {"metric": "n_tie_human", "value": int(tie_h.sum())},
                {"metric": "n_tie_both", "value": len(row_idx_both)},
            ]
            pd.DataFrame(summary_rows).to_excel(writer, sheet_name="summary", index=False)
            tie_mask = tie_ai | tie_h
            tie_export = pd.DataFrame(
                {
                    "row_index": np.arange(100)[tie_mask],
                    "initial stance": base.loc[tie_mask, "initial stance"].values,
                    "rep": base.loc[tie_mask, "rep"].values,
                    "tie_AI_extract": tie_ai[tie_mask],
                    "tie_human_extract": tie_h[tie_mask],
                    "tied_letters_AI": base.loc[tie_mask, "tied_letters_AI"].values,
                    "tied_letters_human": base.loc[tie_mask, "tied_letters_human"].values,
                }
            )
            tie_export.to_excel(writer, sheet_name="tie_rows", index=False)

    print("Done.")


if __name__ == "__main__":
    main()
