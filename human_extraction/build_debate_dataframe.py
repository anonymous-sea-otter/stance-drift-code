import argparse
import json
import re
from pathlib import Path

import pandas as pd


LETTERS = ["A", "B", "C", "D", "E"]
LINE_PATTERN = re.compile(r"Rep\s+(\d+):\s+Generated argument:\s*(.+)$")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build filtered debate dataframe and export to xlsx."
    )
    parser.add_argument(
        "--topic-id",
        required=True,
        help="Topic id to resolve proposition (e.g., 3234).",
    )
    parser.add_argument(
        "--speech-file",
        required=True,
        help="Path to speech text file (one generated argument per line).",
    )
    parser.add_argument(
        "--propositions-file",
        default="propositions.json",
        help="Path to propositions.json.",
    )
    parser.add_argument(
        "--raw-json-file",
        required=True,
        help="Path to raw probability tensor json.",
    )
    parser.add_argument(
        "--output-file",
        default=None,
        help="Output xlsx path. Default: <speech_file_stem>_filtered.xlsx in speech folder.",
    )
    parser.add_argument(
        "--num-stances",
        type=int,
        default=5,
        help="Number of initial stances (default: 5).",
    )
    parser.add_argument(
        "--reps-per-stance",
        type=int,
        default=100,
        help="Expected reps per stance block in speech file (default: 100).",
    )
    parser.add_argument(
        "--filter-max-rep",
        type=int,
        default=19,
        help="Keep rows with rep <= this value for each initial stance (default: 19).",
    )
    return parser.parse_args()


def load_proposition(propositions_file: Path, topic_id: str) -> str:
    with propositions_file.open("r", encoding="utf-8") as f:
        propositions = json.load(f)

    for item in propositions:
        if str(item.get("topic_id")) == str(topic_id):
            return item.get("topic", "")
    raise ValueError(f"topic_id {topic_id} not found in {propositions_file}")


def parse_speech_rows(speech_file: Path, reps_per_stance: int, letters: list[str]) -> list[dict]:
    with speech_file.open("r", encoding="utf-8") as f:
        lines = [line.rstrip("\n") for line in f if line.strip()]

    rows: list[dict] = []
    for idx, line in enumerate(lines):
        match = LINE_PATTERN.search(line)
        if not match:
            raise ValueError(
                f"Failed to parse line {idx + 1} in {speech_file}: expected "
                "'Rep <n>: Generated argument: <text>'"
            )

        rep = int(match.group(1))
        argument = match.group(2).strip()
        stance_idx = idx // reps_per_stance
        if stance_idx >= len(letters):
            raise ValueError(
                f"Row index {idx} implies stance index {stance_idx}, but only "
                f"{len(letters)} stances are configured."
            )

        rows.append(
            {
                "rep": rep,
                "argument": argument,
                "initial stance": letters[stance_idx],
                "initial_stance_index": stance_idx,
                "line_index": idx,
            }
        )

    expected_total = reps_per_stance * len(letters)
    if len(rows) != expected_total:
        raise ValueError(
            f"Expected {expected_total} rows ({len(letters)} stances x {reps_per_stance} reps), "
            f"got {len(rows)} from {speech_file}"
        )

    # Validate rep sequence per stance block.
    for stance_idx, letter in enumerate(letters):
        start = stance_idx * reps_per_stance
        end = start + reps_per_stance
        block_reps = [row["rep"] for row in rows[start:end]]
        expected_reps = list(range(reps_per_stance))
        if block_reps != expected_reps:
            raise ValueError(
                f"Rep sequence mismatch for stance {letter}: expected 0..{reps_per_stance - 1}, "
                f"got first/last reps {block_reps[:3]}...{block_reps[-3:]}"
            )
    return rows


def load_raw_tensor(raw_json_file: Path) -> list:
    with raw_json_file.open("r", encoding="utf-8") as f:
        raw = json.load(f)

    keys = list(raw.keys())
    if len(keys) != 1:
        raise ValueError(
            f"Expected exactly one top-level key in {raw_json_file}, found {len(keys)} keys."
        )
    return raw[keys[0]]


def validate_tensor_shape(tensor: list, reps_per_stance: int, num_stances: int) -> None:
    if len(tensor) != reps_per_stance:
        raise ValueError(
            f"Tensor first dimension must be {reps_per_stance} reps, got {len(tensor)}"
        )

    for rep_idx, by_initial in enumerate(tensor):
        if len(by_initial) != num_stances:
            raise ValueError(
                f"Tensor second dimension must be {num_stances} initials at rep {rep_idx}, "
                f"got {len(by_initial)}"
            )
        for stance_idx, probs in enumerate(by_initial):
            if len(probs) != num_stances:
                raise ValueError(
                    f"Tensor third dimension must be {num_stances} extract probs at rep {rep_idx}, "
                    f"initial {stance_idx}, got {len(probs)}"
                )


def enrich_with_probabilities(rows: list[dict], tensor: list, letters: list[str]) -> None:
    prob_cols = [f"prob_AI_extract_{letter}" for letter in letters]
    for row in rows:
        rep = row["rep"]
        stance_idx = row["initial_stance_index"]
        probs = tensor[rep][stance_idx]
        for col_name, prob in zip(prob_cols, probs):
            row[col_name] = prob


def main() -> None:
    args = parse_args()

    topic_id = str(args.topic_id)
    speech_file = Path(args.speech_file)
    propositions_file = Path(args.propositions_file)
    raw_json_file = Path(args.raw_json_file)

    if args.num_stances != len(LETTERS):
        raise ValueError(
            f"This script currently expects {len(LETTERS)} stances (A-E). "
            f"Got num-stances={args.num_stances}"
        )

    if args.output_file:
        output_file = Path(args.output_file)
    else:
        output_file = speech_file.with_name(f"{speech_file.stem}_filtered.xlsx")

    proposition = load_proposition(propositions_file, topic_id)
    rows = parse_speech_rows(
        speech_file=speech_file,
        reps_per_stance=args.reps_per_stance,
        letters=LETTERS,
    )
    tensor = load_raw_tensor(raw_json_file)
    validate_tensor_shape(
        tensor=tensor,
        reps_per_stance=args.reps_per_stance,
        num_stances=args.num_stances,
    )
    enrich_with_probabilities(rows=rows, tensor=tensor, letters=LETTERS)

    for row in rows:
        row["topic_id"] = topic_id
        row["proposition"] = proposition

    df = pd.DataFrame(rows)
    filtered_df = df[df["rep"] <= args.filter_max_rep].copy()

    expected_filtered = (args.filter_max_rep + 1) * args.num_stances
    if len(filtered_df) != expected_filtered:
        raise ValueError(
            f"Filtered rows should be {expected_filtered}, got {len(filtered_df)}. "
            f"Check rep values and source data ordering."
        )

    prob_cols = [f"prob_AI_extract_{letter}" for letter in LETTERS]
    output_cols = ["proposition", "argument", "initial stance", "rep"] + prob_cols

    output_file.parent.mkdir(parents=True, exist_ok=True)
    try:
        filtered_df.to_excel(output_file, index=False, columns=output_cols)
    except ImportError as exc:
        raise ImportError(
            "Excel export failed because an engine is missing. "
            "Please install openpyxl before running this script."
        ) from exc

    print(
        f"Done. Parsed {len(df)} rows, filtered {len(filtered_df)} rows, "
        f"saved to {output_file}"
    )


if __name__ == "__main__":
    main()
