import matplotlib.pyplot as plt
from openai import OpenAI
from together import Together
import os
import json
import argparse
import sys

from utils import estimate_tran_mat, visualize_transition_matrices, NumpyArrayEncoder, load_debate_speeches_dataset
from chat_client import UnifiedChatClient
import pdb

# Available models (for reference)
# models_available = [
#     "gpt-3.5-turbo",
#     "gpt-4o-mini",
#     "gpt-4o",
#     "meta-llama/Meta-Llama-3.1-8B-Instruct",
#     "mistralai/Mistral-7B-Instruct-v0.3",
# ]

MODEL_NAME_MAP = {
    "gpt-4o-mini": "gpt_4o_mini",
    "gpt-4.1": "gpt_4_1",
    "gpt-5.4": "gpt_5_4",
    "gpt-5.2": "gpt_5_2",
    "gpt-3.5-turbo": "gpt_3_5_turbo",
    "gpt-4o": "gpt_4o",
    "Qwen/Qwen3-Next-80B-A3B-Instruct": "qwen3_a3b",
    "meta-llama/Llama-3-8b-chat-hf": "llama3_8b",
    "google/gemma-3n-E4B-it": "gemma_3n_e4b",
    "meta-llama/Llama-3.3-70B-Instruct-Turbo": "llama3_3_70b",
    "meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo": "llama3_1_8b",
    "meta-llama/Llama-4-Maverick-17B-128E-Instruct-FP8": "llama4_maverick",
    "gemini-2.5-pro": "gemini_2_5_pro",
    "gemini-2.5-flash": "gemini_2_5_flash",
    "gemini-2.5-flash-lite": "gemini_2_5_flash_lite",
}

REPITITION_EST_MAT = 100  # Number of times to repeat the experiment of "Est" setting
letter_list = ["A", "B", "C", "D", "E"]
SEP = "="
PROMPT_CHOICE = 1


def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description="Run LLM predisposition analysis for debate speeches dataset topics"
    )
    parser.add_argument(
        "model_name",
        type=str,
        help="Model name to evaluate (e.g., gpt-4o-mini, gpt-3.5-turbo, etc.)",
    )
    parser.add_argument(
        "output_dir",
        type=str,
        help="Output directory to save results",
    )
    parser.add_argument(
        "--start_idx",
        type=int,
        default=0,
        help="Start index for debate speech topics (default: 0)",
    )
    parser.add_argument(
        "--end_idx",
        type=int,
        default=None,
        help="End index for debate speech topics (default: None, meaning all topics from start_idx)",
    )
    parser.add_argument(
        "--batch",
        action="store_true",
        help="Use batch API for faster processing (default: False)",
    )
    parser.add_argument(
        "--multiple_summarization",
        action="store_true",
        help="Use multiple summarization (encode multiple times with randomly shuffled option orders, average probabilities)",
    )
    parser.add_argument(
        "--summarization_count",
        type=int,
        default=5,
        help="Number of encodings with different shuffled option orders when using multiple summarization (default: 5)",
    )
    parser.add_argument(
        "--prompt_choice",
        type=str,
        default="1",
        help=(
            "Prompt choice for the experiment (default: 1). Numeric variants 1-9 "
            "or a named variant: 'reflection', 'reflection_third_person'."
        ),
    )
    parser.add_argument(
        "--reasoning_effort",
        type=str,
        default=None,
        choices=["none", "minimal", "low", "medium", "high"],
        help=(
            "Reasoning effort for reasoning models (gpt-5*). Default: omit the flag "
            "to use the OpenAI default (no parameter sent). 'none' is treated "
            "identically to omitting it (no parameter sent, no filename suffix); "
            "only minimal/low/medium/high are sent to the API and labeled in output "
            "filenames as _effort_<value>."
        ),
    )

    args = parser.parse_args()
    model_name = args.model_name
    output_dir = args.output_dir
    start_idx = args.start_idx
    end_idx = args.end_idx
    use_batch = args.batch
    multiple_summarization = args.multiple_summarization
    summarization_count = args.summarization_count
    prompt_choice = args.prompt_choice
    # generate_prompt compares numeric variants as ints (== 1 ... == 9) and named
    # variants as strings, so normalize digit strings from argparse back to int.
    if prompt_choice.isdigit():
        prompt_choice = int(prompt_choice)
    reasoning_effort = args.reasoning_effort

    # Load debate speeches dataset
    print("Loading debate speeches dataset...")
    df_unique_topics = load_debate_speeches_dataset()
    print(f"Loaded {len(df_unique_topics)} unique debate speech topics")

    # Validate range indices
    if start_idx < 0 or start_idx >= len(df_unique_topics):
        print(f"Error: start_idx {start_idx} is out of range [0, {len(df_unique_topics)-1}]")
        sys.exit(1)

    if end_idx is None:
        end_idx = len(df_unique_topics)
    elif end_idx <= start_idx or end_idx > len(df_unique_topics):
        print(f"Error: end_idx {end_idx} should be > start_idx and <= {len(df_unique_topics)}")
        sys.exit(1)

    print(f"Processing topics from index {start_idx} to {end_idx-1} (total: {end_idx - start_idx} topics)")

    # Validate batch API is only used with OpenAI models
    if use_batch and not model_name.startswith("gpt"):
        print("Error: Batch API is only supported with OpenAI models (gpt-*).")
        print("Either use an OpenAI model or remove the --batch flag.")
        sys.exit(1)

    # Validate reasoning_effort is only used with reasoning models (gpt-5*).
    # 'none'/omitted means "send nothing", so they are allowed for any model.
    if reasoning_effort not in (None, "none") and "gpt-5" not in model_name.lower():
        print(
            f"Error: --reasoning_effort is only valid for reasoning models (gpt-5*), "
            f"not '{model_name}'."
        )
        sys.exit(1)

    # Create output directory structure
    debate_speeches_output_dir = os.path.join(output_dir, MODEL_NAME_MAP[model_name])
    print("Creating output directory:", debate_speeches_output_dir)
    os.makedirs(debate_speeches_output_dir, exist_ok=True)

    print(f"Processing debate speech topics from index {start_idx} to {end_idx-1}")
    print(f"Using model: {model_name}")
    print(f"Using batch API: {use_batch}")
    print(f"Using multiple summarization: {multiple_summarization}")
    if multiple_summarization:
        print(f"Summarization count: {summarization_count}")
    print(f"Prompt choice: {prompt_choice}")
    print(f"Output directory: {debate_speeches_output_dir}")

    # Process each selected debate speech topic
    selected_topics = df_unique_topics.iloc[start_idx:end_idx]

    for idx, row in selected_topics.iterrows():
        proposition = row['topic']
        topic_id = row['topic_id']
        TOPIC = f"debate_speech_{topic_id}_"

        # Evaluate the model
        print(f"Evaluating {model_name} on topic_id: {topic_id}")

        # Create unified client for regular API calls
        unified_client = UnifiedChatClient(
            openai_api_key=os.environ.get("OPENAI_API_KEY"),
            together_api_key=os.environ.get("TOGETHER_API_KEY"),
            gemini_api_key=os.environ.get("GEMINI_API_KEY"),
            timeout=30,
            max_retries=2,
        )

        # For batch API, we still need the original OpenAI client
        if use_batch:
            if "gpt" in model_name:
                batch_client = OpenAI(
                    api_key=os.environ["OPENAI_API_KEY"], timeout=30, max_retries=2
                )
            else:
                print("Error: Batch API is only supported with OpenAI models (gpt-*).")
                sys.exit(1)
            client = batch_client
        else:
            client = unified_client
        # Create filename with batch indicator
        batch_suffix = "_batch" if use_batch else ""
        # Reasoning-effort suffix: only non-baseline efforts are labeled. Omitting the
        # flag or passing 'none' yields no suffix, keeping baseline filenames unchanged.
        effort_suffix = (
            f"_effort_{reasoning_effort}"
            if reasoning_effort not in (None, "none")
            else ""
        )
        file_name = (
            TOPIC
            + MODEL_NAME_MAP[model_name]
            + "_prompt_"
            + str(prompt_choice)
            + effort_suffix
            #   + batch_suffix # no need to distinguish between batch and non-batch runs
        )

        # Create full file paths with the debate speeches output directory
        log_file_path = os.path.join(debate_speeches_output_dir, file_name + ".log")
        json_file_path = os.path.join(debate_speeches_output_dir, file_name + ".json")
        raw_json_file_path = os.path.join(debate_speeches_output_dir, file_name + "_raw.json")

        if use_batch:
            print("Note: Batch processing may take several minutes to complete.")

        try:
            results_tranmat, results_probs = estimate_tran_mat(
                model=model_name,
                client=client,
                proposition=proposition,
                letters=letter_list,
                sep=SEP,
                repitition=REPITITION_EST_MAT,
                log_filename=log_file_path,
                max_tokens=200,
                max_argument_words=100,
                prompt_choice=prompt_choice,
                use_batch=use_batch,
                multiple_summarization=multiple_summarization,
                summarization_count=summarization_count,
                reasoning_effort=reasoning_effort,
            )
        except Exception as e:
            import traceback
            print(f"Error processing topic_id {topic_id} with model {model_name}: {type(e).__name__}: {e}")
            traceback.print_exc()
            print("Please check the log file for details:", log_file_path)
            continue

        try:
            with open(json_file_path, "w") as file_path:
                json.dump(results_tranmat, file_path, cls=NumpyArrayEncoder, indent=4)
            with open(raw_json_file_path, "w") as file_path:
                json.dump(results_probs, file_path, cls=NumpyArrayEncoder, indent=4)
        except:
            pdb.set_trace()

        # Save visualization to the debate speeches output directory
        plot_file_path = os.path.join(debate_speeches_output_dir, file_name)
        visualize_transition_matrices(
            results_tranmat, letter_list, proposition, plot_file_path
        )

        print(f"Completed processing for topic_id: {topic_id}")
        print(f"Results saved to: {json_file_path}")
        print(f"Visualization saved to: {plot_file_path}.png")
        print("-" * 50)


if __name__ == "__main__":
    main()
