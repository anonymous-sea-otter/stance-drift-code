"""
Collect additional encode votes for existing generated arguments and recompute
AI extraction probabilities/majority from 9 total votes (1 existing + N new).

Single encode, batched, no option-order shuffle
----------------------------------------------
Like ``utils.create_encode_batch_requests`` in the *non-multiple* branch, each
request uses a fixed option order: ``letters_shuffled = LETTERS`` (natural A–E).
There is **no** ``random.shuffle``; we do **not** use the
``multiple_summarization`` path.

We submit ``extra_encodes`` **separate** single-encode calls per row (same
prompt each time) in one batch API job so the model can vary only via sampling
(temperature / stochasticity), not via shuffled labels.

Display labels still map through ``letters_shuffled_encode`` the same way as
``utils.process_encode_batch_results`` (here identity: display letter = canonical).

Use --raw-results-json to persist API output plus request metadata, then
--from-raw-json to rebuild the xlsx without re-running the batch.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from openai import OpenAI

# Ensure project root is importable when script is run via file path.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils import generate_prompt, submit_batch_and_wait

LETTERS = ["A", "B", "C", "D", "E"]
AI_PROB_COLS = [f"prob_AI_extract_{l}" for l in LETTERS]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Collect additional encode results via batch API and recompute AI "
            "extraction proportions + majority vote."
        )
    )
    p.add_argument(
        "--input-xlsx",
        type=Path,
        default=Path("human_extraction/debata_speeches_3234_filtered_with_human.xlsx"),
        help="Input workbook with argument and existing AI extraction columns.",
    )
    p.add_argument(
        "--output-xlsx",
        type=Path,
        default=Path(
            "human_extraction/debata_speeches_3234_filtered_with_human_9encode.xlsx"
        ),
        help="Output workbook path (must not equal input path).",
    )
    p.add_argument(
        "--model-name",
        type=str,
        default="gpt-5.4",
        help="Model name for encode calls.",
    )
    p.add_argument(
        "--extra-encodes",
        type=int,
        default=8,
        help="Number of additional encodes per row.",
    )
    p.add_argument(
        "--prompt-choice",
        type=int,
        default=1,
        help="Prompt choice used by utils.generate_prompt(task='encode', ...).",
    )
    p.add_argument(
        "--sep",
        type=str,
        default="=",
        help="Separator token for prompt rendering.",
    )
    p.add_argument(
        "--batch",
        action="store_true",
        help="Use batch API for encode collection (recommended).",
    )
    p.add_argument(
        "--raw-results-json",
        type=Path,
        default=None,
        help=(
            "Path to write API state after a successful batch: batch_results plus "
            "requests metadata (for --from-raw-json)."
        ),
    )
    p.add_argument(
        "--from-raw-json",
        type=Path,
        default=None,
        help=(
            "Skip the API: load a file written by --raw-results-json (batch_results "
            "+ requests) and only merge into the xlsx."
        ),
    )
    return p.parse_args()


def validate_input(df: pd.DataFrame) -> None:
    required = ["proposition", "argument"] + AI_PROB_COLS
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    if len(df) != 100:
        raise ValueError(f"Expected 100 rows, got {len(df)}")
    if df["proposition"].nunique(dropna=False) != 1:
        raise ValueError("Expected a single proposition across all rows.")
    # Row position 0..99 must align with additional_counts[i]; non-default index breaks that.
    if not df.index.equals(pd.RangeIndex(len(df))):
        raise ValueError(
            "DataFrame index must be a default RangeIndex 0..n-1 after reset_index(drop=True)."
        )


def resolve_letter_with_tie_info(probs: np.ndarray) -> tuple[str, bool, list[str]]:
    max_val = float(np.max(probs))
    tied = [LETTERS[i] for i in range(len(LETTERS)) if abs(float(probs[i]) - max_val) < 1e-12]
    return min(tied, key=LETTERS.index), len(tied) > 1, tied


def create_encode_requests(
    df: pd.DataFrame,
    model_name: str,
    extra_encodes: int,
    sep: str,
    prompt_choice: int,
) -> list[dict]:
    """
    Build many single-encode requests (fixed natural A–E order), same spirit as
    ``utils.create_encode_batch_requests`` when *not* using multiple_summarization:
    ``letters_shuffled`` is fixed (here always natural order, no permutation shuffle).
    Each of ``extra_encodes`` rows is an independent encode with identical prompt text.
    """
    # Natural order only; display label A–E matches canonical stance letter.
    letters_shuffled_encode = LETTERS.copy()
    requests = []
    for row_idx, row in df.iterrows():
        proposition = str(row["proposition"])
        argument = str(row["argument"])
        for rep_idx in range(extra_encodes):
            prompt = generate_prompt(
                task="encode",
                proposition=proposition,
                letters=LETTERS,
                letters_shuffled=letters_shuffled_encode,
                sep=sep,
                argument=argument,
                prompt_choice=prompt_choice,
            )
            body = {
                "model": model_name,
                "messages": [{"role": "user", "content": prompt}],
            }
            if "gpt-5" not in model_name.lower():
                body["max_tokens"] = 10
                body["temperature"] = 0.7
                body["logprobs"] = True
                body["top_logprobs"] = 8

            req = {
                "custom_id": f"encode_row_{row_idx}_rep_{rep_idx}",
                "method": "POST",
                "url": "/v1/chat/completions",
                "body": body,
            }
            requests.append(
                {
                    "request": req,
                    "row_idx": int(row_idx),
                    "rep_idx": int(rep_idx),
                    "letters_shuffled_encode": letters_shuffled_encode,
                }
            )
    return requests


def extract_vote_letter(result: dict) -> str | None:
    choice = result["response"]["body"]["choices"][0]
    logprobs = choice.get("logprobs")
    if logprobs and logprobs.get("content"):
        top_logprobs = logprobs["content"][0]["top_logprobs"]
        for item in top_logprobs:
            token = item["token"].strip()
            if token in LETTERS:
                return token
    text = choice["message"]["content"]
    match = re.search(r"([A-E])", text)
    if match:
        return match.group(1)
    return None


def display_vote_to_canonical_letter(
    display_letter: str, letters_shuffled_encode: list[str]
) -> str:
    """
    Map display label to canonical stance (``utils.process_encode_batch_results``).
    With natural order, this is identity; kept for one code path if option order changes.
    """
    idx = LETTERS.index(display_letter.strip().upper())
    return letters_shuffled_encode[idx]


def parse_batch_votes(batch_results: dict, requests: list[dict], extra_encodes: int) -> np.ndarray:
    n_rows = 100
    additional_counts = np.zeros((n_rows, 5), dtype=int)
    parsed_per_row = np.zeros(n_rows, dtype=int)

    for req in requests:
        custom_id = req["request"]["custom_id"]
        row_idx = req["row_idx"]
        letters_shuffled_encode = req["letters_shuffled_encode"]
        if custom_id not in batch_results:
            continue
        vote_display = extract_vote_letter(batch_results[custom_id])
        if vote_display is None:
            continue
        # Critical: vote_display is the *display* key (A–E position), not canonical stance.
        canonical = display_vote_to_canonical_letter(vote_display, letters_shuffled_encode)
        additional_counts[row_idx, LETTERS.index(canonical)] += 1
        parsed_per_row[row_idx] += 1

    bad_rows = np.where(parsed_per_row != extra_encodes)[0].tolist()
    if bad_rows:
        raise ValueError(
            f"Some rows do not have exactly {extra_encodes} parsed new votes: {bad_rows}"
        )
    return additional_counts


def recompute_ai_columns(
    df: pd.DataFrame, additional_counts: np.ndarray, total_votes_n: int
) -> pd.DataFrame:
    out = df.copy()

    ai_probs = out[AI_PROB_COLS].to_numpy(dtype=float)
    base_letters = []
    base_counts = np.zeros((len(out), 5), dtype=int)

    for i in range(len(out)):
        base_letter, _, _ = resolve_letter_with_tie_info(ai_probs[i])
        base_letters.append(base_letter)
        base_counts[i, LETTERS.index(base_letter)] = 1

    total_counts = base_counts + additional_counts
    if not np.all(np.sum(total_counts, axis=1) == total_votes_n):
        raise ValueError(
            f"Total votes per row must be {total_votes_n} after merge (1 + extra)."
        )

    new_probs = total_counts.astype(float) / float(total_votes_n)
    if not np.allclose(np.sum(new_probs, axis=1), 1.0):
        raise ValueError("Recomputed probabilities do not sum to 1.0 for all rows.")

    for j, letter in enumerate(LETTERS):
        out[f"prob_AI_extract_{letter}"] = new_probs[:, j]
        out[f"ai_additional_count_{letter}"] = additional_counts[:, j]
        out[f"ai_total_count_{letter}"] = total_counts[:, j]

    maj = []
    tie = []
    tied_letters = []
    for i in range(len(out)):
        m, t, tied = resolve_letter_with_tie_info(new_probs[i])
        maj.append(m)
        tie.append(t)
        tied_letters.append(",".join(tied))

    out["majority_AI_extract"] = maj
    out["tie_AI_extract"] = tie
    out["tied_letters_AI"] = tied_letters
    out["ai_base_letter"] = base_letters
    out["ai_total_votes_n"] = total_votes_n
    return out


STATE_VERSION = 1


def save_batch_state(
    path: Path,
    batch_results: dict,
    requests: list[dict],
    extra_encodes: int,
    model_name: str,
) -> None:
    """Persist raw API output and request metadata (must match for --from-raw-json)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    state = {
        "version": STATE_VERSION,
        "extra_encodes": extra_encodes,
        "model_name": model_name,
        "batch_results": batch_results,
        "requests": requests,
    }
    with path.open("w", encoding="utf-8") as f:
        json.dump(state, f, ensure_ascii=False, indent=2)


def load_batch_state(path: Path) -> tuple[dict, list[dict], int, str]:
    with path.open(encoding="utf-8") as f:
        state = json.load(f)
    if isinstance(state, dict) and "batch_results" in state and "requests" in state:
        ver = state.get("version", 0)
        if ver != STATE_VERSION:
            print(f"Warning: state file version {ver}, expected {STATE_VERSION}.")
        extra = state.get("extra_encodes")
        model = state.get("model_name", "")
        if extra is None:
            raise ValueError("State file missing extra_encodes.")
        return state["batch_results"], state["requests"], int(extra), str(model)
    raise ValueError(
        "JSON must be a full state object with 'batch_results' and 'requests'. "
        "Re-run with --raw-results-json to capture both."
    )


def main() -> None:
    args = parse_args()

    if args.extra_encodes != 8:
        print(f"Warning: extra-encodes is {args.extra_encodes} (plan default is 8).")
    if args.output_xlsx.resolve() == args.input_xlsx.resolve():
        raise ValueError("Refusing to overwrite input file; use a different --output-xlsx.")
    if not args.batch:
        raise ValueError("This workflow requires --batch.")
    if args.from_raw_json and args.raw_results_json:
        raise ValueError("Use only one of --from-raw-json or --raw-results-json for a given run.")
    if args.from_raw_json and not args.from_raw_json.is_file():
        raise FileNotFoundError(f"--from-raw-json not found: {args.from_raw_json}")

    df = pd.read_excel(args.input_xlsx, engine="openpyxl")
    df = df.reset_index(drop=True)
    validate_input(df)

    if args.from_raw_json:
        batch_results, requests, state_extra, state_model = load_batch_state(args.from_raw_json)
        if state_extra != args.extra_encodes:
            raise ValueError(
                f"State file extra_encodes={state_extra} != --extra-encodes={args.extra_encodes}"
            )
        if state_model and state_model != args.model_name:
            print(
                f"Warning: state model was {state_model!r}, CLI has --model-name={args.model_name!r}."
            )
    else:
        requests = create_encode_requests(
            df=df,
            model_name=args.model_name,
            extra_encodes=args.extra_encodes,
            sep=args.sep,
            prompt_choice=args.prompt_choice,
        )
        expected_requests = 100 * args.extra_encodes
        if len(requests) != expected_requests:
            raise ValueError(
                f"Expected {expected_requests} requests, built {len(requests)}."
            )

        if "OPENAI_API_KEY" not in os.environ:
            raise ValueError("OPENAI_API_KEY is not set.")
        client = OpenAI(api_key=os.environ["OPENAI_API_KEY"], timeout=30, max_retries=2)

        batch_results = submit_batch_and_wait(
            client=client,
            requests=requests,
            batch_description=(
                f"Collect additional encode votes for 3234 workbook "
                f"(rows=100, extra_encodes={args.extra_encodes}, model={args.model_name})"
            ),
        )
        if args.raw_results_json is not None:
            save_batch_state(
                args.raw_results_json,
                batch_results,
                requests,
                args.extra_encodes,
                args.model_name,
            )
            print(f"Batch state (results + request metadata) saved to: {args.raw_results_json.resolve()}")
        else:
            print(
                "Tip: next time use --raw-results-json PATH to save API output + request metadata, "
                "then regenerate the xlsx with --from-raw-json PATH without re-calling the API."
            )

    additional_counts = parse_batch_votes(
        batch_results=batch_results,
        requests=requests,
        extra_encodes=args.extra_encodes,
    )
    out = recompute_ai_columns(
        df=df, additional_counts=additional_counts, total_votes_n=1 + args.extra_encodes
    )

    args.output_xlsx.parent.mkdir(parents=True, exist_ok=True)
    out.to_excel(args.output_xlsx, index=False, engine="openpyxl")

    print(f"Input: {args.input_xlsx.resolve()}")
    print(f"Output: {args.output_xlsx.resolve()}")
    print(f"Model: {args.model_name}")
    print(f"Rows: {len(out)}")
    print(f"Additional encodes per row: {args.extra_encodes}")
    print("Done.")


if __name__ == "__main__":
    main()
