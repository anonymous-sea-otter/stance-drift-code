import numpy as np
import pandas as pd
import random
import json
import matplotlib.pyplot as plt
from collections import Counter
from IPython.display import display, HTML
from openai import OpenAI, RateLimitError, APITimeoutError, APIConnectionError
import pdb
import logging
import os
import seaborn as sns
from scipy import stats
import time
from chat_client import UnifiedChatClient

# Handle TogetherAI imports gracefully
try:
    from together import Together
except ImportError:
    Together = None

import logging

# Quieter logs so INFO "Retrying request" lines don't spam prod logs
logging.getLogger("openai").setLevel(logging.WARNING)


class ExcludeHTTPFilter(logging.Filter):
    def filter(self, record):
        # Ignore logs containing the keyword "HTTP"
        if "HTTP" in record.getMessage():
            return False
        return True


def setup_logger(log_filename):
    """Set up logger to log to a specified file and filter out HTTP requests"""
    logger = logging.getLogger()

    for handler in logger.handlers:
        logger.removeHandler(handler)
    handler = logging.FileHandler(log_filename)
    handler.setLevel(logging.DEBUG)  # Set level to DEBUG to capture all logs

    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    handler.setFormatter(formatter)
    handler.addFilter(ExcludeHTTPFilter())
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)  # Capture all logs down to INFO level

def load_debate_speeches_dataset():

    propositions_file = "propositions.json"
    example_file = "example.json"

    # First try to load from propositions.json
    if os.path.exists(propositions_file):
        logging.info("Loading debate speeches dataset from propositions.json...")
        try:
            df_unique_topics = pd.read_json(propositions_file)
            logging.info(f"Loaded {len(df_unique_topics)} topics from propositions.json")
            return df_unique_topics
        except Exception as e:
            logging.warning(f"Failed to load propositions.json: {e}. Falling back to HuggingFace dataset.")

    # If propositions.json doesn't exist or failed to load, download from HuggingFace
    logging.info("Loading debate speeches dataset from HuggingFace...")
    df = pd.read_parquet("hf://datasets/ibm-research/debate_speeches/opening_speeches/train-00000-of-00001.parquet")
    logging.info("Removing duplicates...")
    unique_topic_pairs = df[['topic_id', 'topic']].drop_duplicates()
    is_one_to_one = (
        df['topic_id'].nunique() == df['topic'].nunique() and
        df['topic_id'].nunique() == unique_topic_pairs.shape[0]
    )
    logging.info(f"Is there a one-to-one mapping between 'topic_id' and 'topic': {is_one_to_one}")
    df_unique_topics = unique_topic_pairs

    # Split out the last two observations as examples
    if len(df_unique_topics) >= 2:
        # Save the last two observations as example.json
        df_examples = df_unique_topics.tail(2)
        df_examples.to_json(example_file, orient='records', indent=4)
        logging.info(f"Saved {len(df_examples)} example observations to {example_file}")

        # Save the rest (excluding examples) as propositions.json
        df_propositions = df_unique_topics.iloc[:-2]
        df_propositions.to_json(propositions_file, orient='records', indent=4)
        logging.info(f"Saved {len(df_propositions)} propositions to {propositions_file}")
    else:
        # If there are fewer than 2 observations, save all as propositions
        df_unique_topics.to_json(propositions_file, orient='records', indent=4)
        logging.info(f"Saved {len(df_unique_topics)} propositions to {propositions_file} (no examples split out)")

    logging.info("\nDataFrame with unique 'topic_id' and 'topic' pairs (first 5 rows):")
    logging.info(df_unique_topics.head())
    return df_unique_topics

def letter_to_option(letter):
    """convert letter to response"""
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


def choice_to_rate(response):
    """convert the response to rate"""

    response = response.strip().upper()
    if response == "A":
        return 2
    elif response == "B":
        return 1
    elif response == "C":
        return 0
    elif response == "D":
        return -1
    elif response == "E":
        return -2
    else:
        # print(f"Invalid response: {response}")
        return None


def generate_prompt(
    task,
    proposition,
    letters,
    letters_shuffled,
    prompt_choice=1,
    sep="=",
    letter=None,
    argument="",
    words_limit=100,
):
    """
    Generate prompts for different LLM tasks related to proposition evaluation.

    This function creates structured prompts for three main tasks:
    1. "encode": Converting arguments back to stance letters
    2. "decode": Converting stance letters to arguments

    Args:
        task (str): Task type - must be "encode", or "decode"
        proposition (str): The statement/proposition to be evaluated
        letters (list): Original ordered list of response letters (e.g., ['A','B','C','D','E'])
        letters_shuffled (list): Shuffled version of letters for randomization
        prompt_choice (int, optional): Template variant to use (1-7). Defaults to 1.
        sep (str, optional): Separator string for prompt formatting. Defaults to "=".
        letter (str, optional): Specific letter for decode task. Defaults to None.
        argument (str, optional): Argument text for encode task. Defaults to "".
        words_limit (int, optional): Maximum words for generated arguments. Defaults to 100.

    Returns:
        str: Formatted prompt string ready for LLM input

    Raises:
        AssertionError: If task is not one of the valid options

    Note:
        Different prompt_choice values provide alternative phrasings and instructions
        to test prompt sensitivity and improve robustness.
    """
    assert task in ["encode", "decode"]

    # Build the standardized choice options display
    choices = ""
    for i in range(len(letters)):
        choices += f"{letters[i]}) {letter_to_option(letters_shuffled[i])}\n"
    # logging.info(f"current choice: \n {choices}")

    ############################# Encode - Decode  #########################################

    if task == "decode":
        # Decode task: Convert stance letter to supporting argument
        logging.info(f"Given letter: <{letter}>")

        if prompt_choice == 1:
            prompt = f"""
In the following task, you will receive a proposition and a capitalized letter representing one of the stance options listed below with respect to the proposition. The proposition concerns a topic related to the United States.
Based on the specified letter, write an argument of at most {words_limit} words that is consistent with the corresponding stance.
Only output the argument itself and do not directly quote or explicitly mention the option or its label.
{sep}{sep}
Options:
{choices}{sep}{sep}
Proposition: {proposition}
{sep}{sep}
Letter: {letter}
{sep}{sep}
Argument: 
"""
        elif prompt_choice == 2:
            prompt = f"""
In this task, you will be given a proposition and a capitalized letter. The letter corresponds to one of the following stances regarding the proposition. 
Based on that stance, write a supporting argument in no more than {words_limit} words.
Output only the supporting argument (no additional commentary).
{sep}{sep}
Stances:
{choices}{sep}{sep}
Proposition: {proposition}
{sep}{sep}
Letter: {letter}
{sep}{sep}
Argument: 
"""

        elif prompt_choice == 3:
            prompt = f"""
You will receive a propositioin and your task is to write an essay about the proposition in your own words given an opinion you are required to hold about the proposition. 
For example, you are given a proposition "The earth is flat" and you are required to hold an opinion that you "disagree to some extent" about the proposition, then you can write an essay with its main argument being "The earth is maybe not flat".
{sep}{sep}
Here are some rules you need to follow:
1. Do not simply plug in the opinion into the proposition. For example, do not write "I disagree to some extend that the earth is flat".
2. Do not simply negate the proposition. For example, do not write "The earth is not flat" if the provided opinion is "definitely disagree".
3. Do not provide a one-word answer. For example, do not write "No" if the provided opinion is "definitely disagree".
4. Only output the rewritten proposition. Do not respond with anything else such as greetings or paragraphsing your task.
5. Keep your response under {words_limit} words.
6. You should write an essay about the argument.
{sep}{sep}

Now let's start the task.
Proposition: {proposition}
Your opinion: {letter_to_option(letters_shuffled[letters.index(letter)])}

Your essay: 
"""

        elif prompt_choice == 4:
            prompt = f"""
You will be given:
A proposition and an opinion (stance)

Task:
Write an essay about the proposition from the given opinion, following these rules:
1. Do not simply restate the opinion verbatim in the proposition. (For example, avoid: “I disagree to some extent that the earth is flat.”)
2. Do not simply negate the proposition if you disagree. (For example, avoid: “The earth is not flat.”)
3. Do not provide a one-word response. (For example, avoid: “No.”)
4. Only output the essay itself—no extra greetings or explanations of your task.
5. Keep your essay under {words_limit} words.
6. Ensure you present a coherent argument that reflects the provided opinion.

Now, let’s begin:
Proposition: {proposition}
Your Opinion: {letter_to_option(letters_shuffled[letters.index(letter)])}
Essay (under {words_limit} words): 
"""

        elif prompt_choice == 5:
            prompt = f"""
In this task, you will receive a proposition and a capitalized letter indicating a stance. 
Using that stance, write a concise argument of no more than {words_limit} words to support the proposition.   
Output only the argument.
{sep}{sep}
Stances:
{choices}{sep}{sep}
Proposition: {proposition}
{sep}{sep}
Letter: {letter}
{sep}{sep}
Argument: 
"""

        elif prompt_choice == 6:
            prompt = f"""
You will receive a proposition and your task is to write an essay about the proposition in your own words given an opinion you are required to hold about the proposition. 
For example, you are given a proposition "The earth is flat" and you are required to hold an opinion that you "somewhat disagree" about the proposition, then you can write an essay with its main argument being "The earth is maybe not flat".
{sep}{sep}
Here are some rules you need to follow:
1. Do not simply plug in the opinion into the proposition. For example, do not write "I disagree to some extend that the earth is flat".
2. Only output the rewritten proposition. Do not respond with anything else such as greetings or paragraphsing your task.
3. Keep your response under {words_limit} words.
4. You should write an essay about the proposition.
{sep}{sep}

Now let's start the task.
Proposition: {proposition}
Your opinion: {letter_to_option(letters_shuffled[letters.index(letter)])}

Your essay: 
"""

        elif prompt_choice == 7:
            prompt = f"""
You will receive a proposition and your task is to write an essay about the proposition in your own words given an opinion you are required to hold about the proposition. 
{sep}{sep}
Here are some rules you need to follow:
1. Do not simply plug in the opinion into the proposition.
2. Only output the essay about the proposition. Do not respond with anything else.
3. Keep your response under {words_limit} words.
{sep}{sep}

Now let's start the task.
Proposition: {proposition}
Your opinion: {letter_to_option(letters_shuffled[letters.index(letter)])}

Your essay: 
"""
        elif prompt_choice == 8:  # with ICL examples
            prompt = f"""
In the following task, you will receive a proposition and a capitalized letter representing one of the stance options listed below with respect to the proposition. The proposition concerns a topic related to the United States.
Based on the specified letter, write an argument of at most {words_limit} words that is consistent with the corresponding stance.
Only output the argument itself and do not directly quote or explicitly mention the option or its label.

Here are some examples: 
Example 1:
== 
Options: 
A) Agree strongly 
B) Agree 
C) Neutral 
D) Disagree 
E) Disagree strongly 
== 
Proposition: We should subsidize the human mission to Mars.
== 
Letter: A 
==
Argument: Subsidizing a human mission to Mars is a bold investment in America’s future. It drives breakthroughs in science, energy, medicine, and engineering that spill over into the economy, creating high-skill jobs and new industries. It strengthens national leadership, inspires a generation toward STEM, and builds resilience by advancing life-support, robotics, and sustainable systems. Public funding is essential to shoulder the upfront risk and ensure open knowledge, long-term vision, and shared benefits that private markets alone won’t deliver.

Example 2:
==
Options:
A) Agree strongly
B) Agree
C) Neutral
D) Disagree
E) Disagree strongly
==
Proposition: We should subsidize vocational education.
==
Letter: E
==
Argument: Subsidizing vocational education is a poor use of public funds. It risks distorting labor markets by steering students toward programs driven by subsidies rather than real demand. Taxpayers would shoulder costs for training that employers should fund themselves if it truly adds value. Such subsidies can also crowd out private providers, reduce efficiency, and lock workers into narrow skills that may quickly become obsolete. Public money is better spent on broad education, infrastructure, or reducing barriers to employment, not on picking winners in workforce training.

Example 3:
==
Options:
A) Agree strongly
B) Agree
C) Neutral
D) Disagree
E) Disagree strongly
==
Proposition: We should subsidize vocational education.
==
Letter: C
==
Argument: Subsidizing vocational education has potential benefits and drawbacks that warrant careful consideration. Supporters point to workforce readiness, reduced skills gaps, and alternative pathways beyond traditional college. Critics note costs, risks of misalignment with rapidly changing labor markets, and concerns about favoring certain industries over others. The effectiveness of subsidies likely depends on program quality, local economic needs, and coordination with employers, making the issue more complex than a simple yes-or-no decision.


Now write your argument:
{sep}{sep}
Options:
{choices}{sep}{sep}
Proposition: {proposition}
{sep}{sep}
Letter: {letter}
{sep}{sep}
Argument: 
"""

        if prompt_choice == 9:
            prompt = f"""
In the following task, you will receive a proposition and a capitalized letter representing one of the stance options listed below with respect to the proposition. The proposition concerns a topic related to the United States.
Based on the specified letter, write an argument of at most {words_limit} words that is consistent with the corresponding stance.
Only output the argument itself and do not directly quote or explicitly mention the option or its label.
{sep}{sep}
Options:
{choices}{sep}{sep}
Proposition: {proposition}
{sep}{sep}
Letter: {letter}
{sep}{sep}
Argument: 
"""

        if prompt_choice == "reflection":
            prompt = f"""
In the following task, you will receive a proposition and a capitalized letter representing one of the stance options listed below with respect to the proposition. The proposition concerns a topic related to the United States.
Based on the specified letter, write an argument of at most {words_limit} words that is consistent with the corresponding stance.
After drafting your argument, compare it against the specified stance and revise it if the two do not align.
Only output the finalized argument itself and do not directly quote or explicitly mention the option or its label.
{sep}{sep}
Options:
{choices}{sep}{sep}
Proposition: {proposition}
{sep}{sep}
Letter: {letter}
{sep}{sep}
Argument: 
"""

        if prompt_choice == "reflection_third_person":
            prompt = f"""
In the following task, you will receive a proposition and a capitalized letter representing one of the stance options listed below with respect to the proposition. The proposition concerns a topic related to the United States.
Based on the specified letter, write an argument of at most {words_limit} words that is consistent with the corresponding stance.
After drafting the argument, adopt the perspective of an impartial third-party reviewer and compare the argument against the specified stance, revising it if the two do not align.
Only output the finalized argument itself and do not directly quote or explicitly mention the option or its label.
{sep}{sep}
Options:
{choices}{sep}{sep}
Proposition: {proposition}
{sep}{sep}
Letter: {letter}
{sep}{sep}
Argument: 
"""

    elif task == "encode":
        # Encode task: Convert argument back to stance letter

        if prompt_choice == 1:
            prompt = f"""
In the following task, you will receive a proposition and an argument addressing that proposition. The proposition concerns a topic related to the United States.
Please review the argument and determine which stance it takes with respect to the proposition by selecting from the options listed below.
Only output the capitalized letter corresponding to your answer.
Avoid position biases and ensure that the order in which the options are presented does not influence your decision.
{sep}{sep}
Proposition: {proposition}
{sep}{sep}
Argument: {argument}
{sep}{sep}
Please choose from the following options:
{choices}{sep}{sep}
Answer letter: 
"""

        elif prompt_choice == 2:
            prompt = f"""
In this task, you will receive a proposition and an argument about it. After reviewing the argument, select the stance it represents by choosing from the capital-letter options below. 
Output only the corresponding letter—no additional text.
{sep}{sep}
Proposition: {proposition}
{sep}{sep}
Argument: {argument}
{sep}{sep}
Please choose from the following options:
{choices}{sep}{sep}
Answer letter: 
"""

        elif prompt_choice == 3:
            prompt = f"""
In the following task, you will receive a proposition and an argument. Your task to decide the what opinion the argument has about the proposition. For example, if the argument is "The earth is round" and the proposition is "The earth is flat", then the opinion of the argument is "very disagree". 
Only output the capitalized alphabet letter corresponding the answer. 
Do not respond with anything else such as greetings or paragraphsing your task. 
{sep}{sep}
Proposition: {proposition}
{sep}{sep}
Argument: {argument}
{sep}{sep}
Please choose from the following options:
{choices}{sep}{sep}
Answer letter: 
"""

        elif prompt_choice == 4:
            prompt = f"""
You will be given a proposition and an argument. Determine the stance the argument takes on the proposition by selecting one of the capital-letter options below.
For example: If the proposition is “The earth is flat” and the argument is “The earth is round,” the stance would be “very disagree.”
Only output the corresponding capital letter—no extra text or explanation.
{sep}{sep}
Proposition: {proposition}
{sep}{sep}
Argument: {argument}
{sep}{sep}
Please choose from the following options:
{choices}{sep}{sep}
Answer letter: 
"""
        elif prompt_choice == 5:
            prompt = f"""
You will receive a proposition and an argument. Determine which capital-letter stance (from the list below) the argument takes on the proposition. 
Output only the capital letter—no further text.
{sep}{sep}
Proposition:{proposition}
{sep}{sep}
Argument:{argument}
{sep}{sep}
Options:{choices}
{sep}{sep}
Answer Letter: 
"""

        elif prompt_choice == 6 or prompt_choice == 7:
            prompt = f"""
In the following task, you will receive a proposition and an argument. Your task to decide the what opinion the argument has about the proposition. For example, if the argument is "The earth is round" and the proposition is "The earth is flat", then the opinion of the argument is "very disagree". 
Only output the capitalized alphabet letter corresponding the answer. 
Do not respond with anything else such as greetings or paragraphsing your task. 
{sep}{sep}
Proposition: {proposition}
{sep}{sep}
Argument: {argument}
{sep}{sep}
Please choose from the following options:
{choices}{sep}{sep}
Answer letter: 
"""
        elif prompt_choice == 8:
            prompt = f"""
In the following task, you will receive a proposition and an argument addressing that proposition. The proposition concerns a topic related to the United States.
Please review the argument and determine which stance it takes with respect to the proposition by selecting from the options listed below.
Only output the capitalized letter corresponding to your answer.
Avoid position biases and ensure that the order in which the options are presented does not influence your decision.
{sep}{sep}
Proposition: {proposition}
{sep}{sep}
Argument: {argument}
{sep}{sep}
Please choose from the following options:
{choices}{sep}{sep}
Answer letter: 

Here are some examples:
Example 1:
==
Proposition: We should subsidize the human mission to Mars.
==
Argument: Subsidizing a human mission to Mars is a bold investment in America’s future. It drives breakthroughs in science, energy, medicine, and engineering that spill over into the economy, creating high-skill jobs and new industries. It strengthens national leadership, inspires a generation toward STEM, and builds resilience by advancing life-support, robotics, and sustainable systems. Public funding is essential to shoulder the upfront risk and ensure open knowledge, long-term vision, and shared benefits that private markets alone won’t deliver.
==
Please choose from the following options:
A) Agree strongly
B) Agree
C) Neutral
D) Disagree
E) Disagree strongly
==
Answer letter:
A

Example 2: 
==
Proposition: We should subsidize vocational education.
==
Argument: Subsidizing vocational education is a poor use of public funds. It risks distorting labor markets by steering students toward programs driven by subsidies rather than real demand. Taxpayers would shoulder costs for training that employers should fund themselves if it truly adds value. Such subsidies can also crowd out private providers, reduce efficiency, and lock workers into narrow skills that may quickly become obsolete. Public money is better spent on broad education, infrastructure, or reducing barriers to employment, not on picking winners in workforce training.
==
Please choose from the following options:
A) Agree strongly
B) Agree
C) Neutral
D) Disagree
E) Disagree strongly
==
Answer letter:
E

Example 3: 
==
Proposition: We should subsidize vocational education.
==
Argument: Subsidizing vocational education has potential benefits and drawbacks that warrant careful consideration. Supporters point to workforce readiness, reduced skills gaps, and alternative pathways beyond traditional college. Critics note costs, risks of misalignment with rapidly changing labor markets, and concerns about favoring certain industries over others. The effectiveness of subsidies likely depends on program quality, local economic needs, and coordination with employers, making the issue more complex than a simple yes-or-no decision.
==
Please choose from the following options:
A) Agree strongly
B) Agree
C) Neutral
D) Disagree
E) Disagree strongly
==
Answer letter:
C

Now chose your letter of stance:
{sep}{sep}
Proposition: {proposition}
{sep}{sep}
Argument: {argument}
{sep}{sep}
Please choose from the following options:
{choices}{sep}{sep}
Answer letter: 
"""
        if prompt_choice == 9:
            prompt = f"""

In the following task, you will receive a proposition and an argument addressing that proposition. The proposition concerns a topic related to the United States.
Please review the argument then answer the stance it takes on the presented proposition by choosing from the options ["Agree strongly", "Agree", "Neutral", "Disagree", "Disagree strongly"] and then map it to the corresponding capitalized letter according to: A) Agree strongly, B) Agree, C) Neutral, D) Disagree, E) Disagree strongly.
Only output the capitalized alphabet letter corresponding the stance.
{sep}{sep}
Proposition: {proposition}
{sep}{sep}
Argument: {argument}
{sep}{sep}
Answer letter: 
"""

        if prompt_choice == "reflection":
            prompt = f"""
In the following task, you will receive a proposition and an argument addressing that proposition. The proposition concerns a topic related to the United States.
Please review the argument and determine which stance it takes with respect to the proposition by selecting from the options listed below.
After selecting a letter, compare the stance it represents against the argument and revise your choice if the two do not align.
Only output the final capitalized letter corresponding to your answer.
Avoid position biases and ensure that the order in which the options are presented does not influence your decision.
{sep}{sep}
Proposition: {proposition}
{sep}{sep}
Argument: {argument}
{sep}{sep}
Please choose from the following options:
{choices}{sep}{sep}
Answer letter: 
"""

        if prompt_choice == "reflection_third_person":
            prompt = f"""
In the following task, you will receive a proposition and an argument addressing that proposition. The proposition concerns a topic related to the United States.
Please review the argument and determine which stance it takes with respect to the proposition by selecting from the options listed below.
After selecting a letter, adopt the perspective of an impartial third-party reviewer and compare the stance it represents against the argument, revising it if the two do not align.
Only output the final capitalized letter corresponding to your answer.
Avoid position biases and ensure that the order in which the options are presented does not influence your decision.
{sep}{sep}
Proposition: {proposition}
{sep}{sep}
Argument: {argument}
{sep}{sep}
Please choose from the following options:
{choices}{sep}{sep}
Answer letter: 
"""

    logging.info(f"prompt: \n {prompt}")
    return prompt


def get_normalized_prob(probs_all):

    col_sums = np.sum(probs_all, axis=0, keepdims=True)
    if np.any(col_sums == 0):
        empty_reps = np.where(col_sums.ravel() == 0)[0].tolist()
        raise ValueError(
            f"Encode produced no usable probability mass for rep(s) {empty_reps} "
            f"(all-zero column): those encode responses were either missing from the "
            f"batch output (request failed) or had no parseable A-E letter. Refusing "
            f"to normalize 0/0 into NaN. See the .log file and the batch error file."
        )
    normalized_probs = probs_all / col_sums
    return normalized_probs


def get_expectation(average_probs, letters):
    """calculate the average rate based on the probability"""

    rates = []
    weights = []
    for letter, average_prob in zip(letters, average_probs):
        rates.append(choice_to_rate(letter))
        weights.append(average_prob)
    try:
        rate = np.average(rates, weights=weights)
    except:
        pdb.set_trace()
    return rate


def permutate_letters(letters):
    """
    Generate letter permutations for experimental design.

    This function creates different orderings of response letters to control
    for position bias in model responses. Currently simplified to use only
    the natural order, but designed to support multiple permutations.

    Args:
        letters (list): Original ordered list of response letters

    Returns:
        list: List of permutation pairs [decode_order, encode_order]
              Currently returns only one permutation using natural order

    Note:
        Originally designed for multiple permutations (natural and reversed order)
        but simplified for current experiments. Framework supports expansion
        to test position bias effects.
    """
    permutations = []
    natural_order = letters.copy()
    reversed_order = letters[::-1]

    # Currently using only natural order - can be expanded for bias testing
    permutations.append(
        [natural_order, natural_order]
    )  # we only focus on one permutation for now
    # for encode_order in [natural_order, reversed_order]:
    #     for decode_order in [natural_order, reversed_order]:
    #         permutations.append([decode_order, encode_order])

    return permutations


def decode_and_encode(
    model,
    client,
    proposition,
    letters,
    shuffle_rep,
    sep,
    letter,
    max_tokens=200,
    max_argument_words=100,
    log_filename=None,
    prompt_choice=1,
    get_probs=False,
    use_batch=False,
    multiple_summarization=False,
    summarization_count=5,
    reasoning_effort=None,
):
    """
    Perform a complete decode-encode cycle to measure stance consistency.

    This function implements the core experimental procedure:
    1. Decode: Convert a stance letter to an argument about the proposition
    2. Encode: Convert the generated argument back to a stance letter
    3. Analyze: Compare input vs output stance using probability distributions

    Args:
        model (str): OpenAI model name (e.g., "gpt-4", "gpt-3.5-turbo")
        client: OpenAI API client instance
        proposition (str): The statement/proposition to evaluate
        letters (list): Response letters (e.g., ['A','B','C','D','E'])
        shuffle_rep (int): Number of experimental repetitions with shuffled orders
        sep (str): Separator string for prompt formatting
        letter (str): Initial stance letter to decode from
        max_tokens (int, optional): Maximum tokens for decode response. Defaults to 200.
        max_argument_words (int, optional): Word limit for generated arguments. Defaults to 100.
        log_filename (str, optional): Log file path. Defaults to None.
        prompt_choice (int, optional): Prompt template variant. Defaults to 1.
        get_probs (bool, optional): Whether to return detailed probabilities. Defaults to False.
        multiple_summarization (bool, optional): If True, encode multiple times with randomly
            shuffled option orders and average probabilities to reduce position bias. Defaults to False.
        summarization_count (int, optional): Number of encodings with different shuffled option
            orders when using multiple summarization. Defaults to 5.

    Returns:
        tuple: (answer, expected_rate) or (answer, expected_rate, average_probs, probs)
               - answer (str): Most likely response letter from encoding
               - expected_rate (float): Expected numerical rating
               - average_probs (np.ndarray): Mean probabilities across repetitions (if get_probs=True)
               - probs (dict): Detailed probability data by permutation (if get_probs=True)

    Note:
        The function measures stance consistency by comparing the initial letter
        with the final encoded response, accounting for randomization effects.
    """
    if log_filename:
        setup_logger(log_filename)  # Initialize the logger with the given log filename

    if use_batch:
        # Use batch API approach
        logging.info("Creating batch requests for decode phase...")
        decode_requests = create_decode_batch_requests(
            model,
            proposition,
            letters,
            shuffle_rep,
            sep,
            letter,
            max_tokens,
            max_argument_words,
            prompt_choice,
        )

        logging.info("Submitting decode batch...")
        decode_batch_results = submit_batch_and_wait(
            client, decode_requests, f"Decode batch for letter {letter}"
        )

        logging.info("Processing decode results...")
        decode_results = process_decode_batch_results(
            decode_batch_results, decode_requests
        )

        logging.info("Creating batch requests for encode phase...")
        encode_requests = create_encode_batch_requests(
            model, proposition, letters, decode_results, sep, prompt_choice
        )

        logging.info("Submitting encode batch...")
        encode_batch_results = submit_batch_and_wait(
            client, encode_requests, f"Encode batch for letter {letter}"
        )

        logging.info("Processing encode results...")
        probs = process_encode_batch_results(
            encode_batch_results, encode_requests, letters
        )

    else:

        # Initialize probability tracking structures
        probs_all = {}
        probs = {}
        permutations = permutate_letters(letters)

        logging.info(f"Proposition: {proposition}")

        # Process each permutation of letter orders
        for permutation_id, permutation in enumerate(permutations):
            str_permutation = str(permutation)
            print(f"---------------Permutation_id: {permutation_id}----------------")
            probs_all[str_permutation] = np.zeros((len(letters), shuffle_rep))
            for rep in range(shuffle_rep):

                logging.info(f"---------------Shuffled_rep: {rep}----------------")
                print(f"---------------Shuffled_rep: {rep}----------------")

                letters_shuffled_decode = permutation[0]
                logging.info(f"Shuffled letter in decoding: {letters_shuffled_decode}")

                # DECODE PHASE: Convert stance letter to argument
                logging.info(f"Task: decoding")
                response = chat_with_backoff(
                    client=client,
                    model=model,
                    messages=[
                        {
                            "role": "user",
                            "content": generate_prompt(
                                task="decode",
                                proposition=proposition,
                                letters=letters,
                                letters_shuffled=letters_shuffled_decode,
                                sep=sep,
                                letter=letters[letters_shuffled_decode.index(letter)],
                                words_limit=max_argument_words,
                                prompt_choice=prompt_choice,
                            ),
                        }
                    ],
                    max_tokens=max_tokens,
                    temperature=0.7,
                    reasoning_effort=reasoning_effort,
                )

                argument = response.choices[0].message.content

                logging.info(f"Given opinion: {letter_to_option(letter)}")
                logging.info(f"Generated argument: {argument}")

                # ENCODE PHASE: Convert argument back to stance letter
                logging.info(f"Job: encoding")
                letters_shuffled_encode = permutation[1]
                logging.info(f"Shuffled letter in encoding: {letters_shuffled_encode}")

                if "llama" in model.lower():
                    num_top_logprobs = 5  # together ai's llama models do not support top_logprobs more than 5
                else:
                    num_top_logprobs = 8

                if multiple_summarization:
                    # Multiple summarization: encode with multiple randomly shuffled option orders
                    all_probs = []
                    
                    for shuffle_idx in range(summarization_count):
                        # Generate random shuffle of letters for this encoding
                        letters_shuffled_current = letters.copy()
                        random.shuffle(letters_shuffled_current)
                        logging.info(f"Encoding {shuffle_idx + 1}/{summarization_count} - Shuffled order: {letters_shuffled_current}")

                        response = chat_with_backoff(
                            client=client,
                            model=model,
                            messages=[
                                {
                                    "role": "user",
                                    "content": generate_prompt(
                                        task="encode",
                                        proposition=proposition,
                                        letters=letters,
                                        letters_shuffled=letters_shuffled_current,
                                        sep=sep,
                                        argument=argument,
                                        prompt_choice=prompt_choice,
                                    ),
                                }
                            ],
                            max_tokens=10,
                            top_logprobs=num_top_logprobs,
                            temperature=0.7,
                            reasoning_effort=reasoning_effort,
                        )

                        top_logprobs = response.choices[0].logprobs.content[0].top_logprobs
                        logging.info(f"Encoding {shuffle_idx + 1}/{summarization_count} - Top choice: <{top_logprobs[0].token}>")

                        # Extract probabilities for this encoding
                        probs_current = np.zeros(len(letters))
                        for logprob in top_logprobs:
                            if logprob.token.strip() in letters:
                                probs_current[letters.index(letters_shuffled_current[letters.index(logprob.token.strip())])] += np.exp(logprob.logprob)
                        
                        all_probs.append(probs_current)

                    # Average probabilities across all encodings
                    avg_probs = np.mean(all_probs, axis=0)
                    probs_all[str_permutation][:, rep] = avg_probs
                else:
                    # Single encoding (original behavior)
                    response = chat_with_backoff(
                        client=client,
                        model=model,
                        messages=[
                            {
                                "role": "user",
                                "content": generate_prompt(
                                    task="encode",
                                    proposition=proposition,
                                    letters=letters,
                                    letters_shuffled=letters_shuffled_encode,
                                    sep=sep,
                                    argument=argument,
                                    prompt_choice=prompt_choice,
                                ),
                            }
                        ],
                        max_tokens=10,
                        top_logprobs=num_top_logprobs,
                        temperature=0.7,
                        reasoning_effort=reasoning_effort,
                    )

                    top_logprobs = response.choices[0].logprobs.content[0].top_logprobs
                    logging.info(f"Top choice (answer): <{top_logprobs[0].token}>")

                    for logprob in top_logprobs:
                        if logprob.token.strip() in letters:
                            probs_all[str_permutation][
                                letters.index(
                                    letters_shuffled_encode[
                                        letters.index(logprob.token.strip())
                                    ]
                                ),
                                rep,
                            ] += np.exp(logprob.logprob)

            # Normalize probabilities for this permutation
            probs[str_permutation] = get_normalized_prob(probs_all[str_permutation])

    # Aggregate results across permutations and repetitions
    mean_on_perm_probs = stats.trim_mean(np.asarray(list(probs.values())), 0, axis=0)
    print(mean_on_perm_probs.shape)
    average_probs = np.mean(
        mean_on_perm_probs, axis=1
    )  # np.mean(np.mean(np.asarray(list(probs.values())),axis=0), axis=1)
    answer = letters[np.argmax(average_probs)]
    expected_rate = get_expectation(average_probs, letters)

    logging.info(f"answer = {answer}")
    logging.info(f"expected_rate = {expected_rate}")
    logging.info(f"average_probs = {average_probs}")

    if get_probs == True:
        return answer, expected_rate, average_probs, probs
    else:
        return answer, expected_rate


def estimate_tran_mat(
    model,
    client,
    proposition,
    letters,
    sep,
    repitition,
    log_filename="outputs.log",
    max_tokens=200,
    max_argument_words=100,
    prompt_choice=1,
    use_batch=False,
    multiple_summarization=False,
    summarization_count=5,
    reasoning_effort=None,
):
    """
    Estimate the transition matrix for stance consistency across all response options.

    This function systematically tests the model's stance consistency by running
    decode-encode cycles for each possible initial stance letter, building a
    complete transition matrix showing how often each input stance maps to each
    output stance.

    Args:
        model (str): OpenAI model name to test
        client: OpenAI API client instance
        proposition (str): The statement/proposition to evaluate
        letters (list): Response letters (e.g., ['A','B','C','D','E'])
        sep (str): Separator string for prompt formatting
        repitition (int): Number of experimental repetitions per initial stance
        log_filename (str, optional): Log file path. Defaults to "outputs.log".
        max_tokens (int, optional): Maximum tokens for decode responses. Defaults to 200.
        max_argument_words (int, optional): Word limit for arguments. Defaults to 100.
        prompt_choice (int, optional): Prompt template variant. Defaults to 1.
        multiple_summarization (bool, optional): If True, encode multiple times with randomly
            shuffled option orders and average probabilities to reduce position bias. Defaults to False.
        summarization_count (int, optional): Number of encodings with different shuffled option
            orders when using multiple summarization. Defaults to 5.

    Returns:
        tuple: (results_probs, all_raw_probs)
               - results_probs (np.ndarray): Transition matrix of shape (n_letters, n_letters)
                 where results_probs[i,j] = P(output_letter_j | input_letter_i)
               - all_raw_probs (dict): Raw probability data by permutation and repetition
                 for detailed analysis and uncertainty quantification
    """
    # Initialize logging
    if os.path.exists(log_filename):
        os.remove(log_filename)
    with open(log_filename, "w") as file:
        file.write("Start Logging Estimation of the Transition Matrix\n")
    setup_logger(log_filename)  # Initialize the logger with the provided log filename

    # Initialize data structures for results
    str_permutations = [str(permutation) for permutation in permutate_letters(letters)]
    results_probs = {}
    all_raw_probs = {}
    results_probs = np.zeros((len(letters), len(letters)))
    all_raw_probs = {
        str_permutation: np.zeros((repitition, len(letters), len(letters)))
        for str_permutation in str_permutations
    }

    logging.info(f"Model: <{model}>")
    logging.info(f"prompt choice: <{prompt_choice}>")
    logging.info("===================================================")

    if use_batch:
        # Use batch API for all letters at once
        logging.info("Using batch API for entire transition matrix estimation...")

        # Create all decode requests for all letters
        all_decode_requests = []
        for j, initial_letter in enumerate(letters):
            decode_requests = create_decode_batch_requests(
                model,
                proposition,
                letters,
                repitition,
                sep,
                initial_letter,
                max_tokens,
                max_argument_words,
                prompt_choice,
                reasoning_effort=reasoning_effort,
            )
            # Add letter index to each request for tracking
            for req_info in decode_requests:
                req_info["letter_index"] = j
                req_info["initial_letter"] = initial_letter
                # Make custom_id unique by including letter index
                old_custom_id = req_info["request"]["custom_id"]
                req_info["request"]["custom_id"] = f"{old_custom_id}_letter_{j}"
            all_decode_requests.extend(decode_requests)

        # Submit all decode requests
        logging.info(
            f"Submitting batch with {len(all_decode_requests)} decode requests..."
        )
        decode_batch_results = submit_batch_and_wait(
            client, all_decode_requests, "Full transition matrix decode batch"
        )

        # Process decode results
        all_decode_results = process_decode_batch_results(
            decode_batch_results, all_decode_requests
        )

        # Create all encode requests
        all_encode_requests = []
        for result in all_decode_results:
            encode_requests = create_encode_batch_requests(
                model, proposition, letters, [result], sep, prompt_choice,
                multiple_summarization=multiple_summarization,
                summarization_count=summarization_count,
                reasoning_effort=reasoning_effort,
            )
            # Add letter tracking info
            for req_info in encode_requests:
                req_info["letter_index"] = result["letter_index"]
                req_info["initial_letter"] = result["initial_letter"]
                # Make custom_id unique by including letter index
                old_custom_id = req_info["request"]["custom_id"]
                req_info["request"][
                    "custom_id"
                ] = f"{old_custom_id}_letter_{result['letter_index']}"
            all_encode_requests.extend(encode_requests)

        # Submit all encode requests
        logging.info(
            f"Submitting batch with {len(all_encode_requests)} encode requests..."
        )
        encode_batch_results = submit_batch_and_wait(
            client, all_encode_requests, "Full transition matrix encode batch"
        )

        # Process encode results and build transition matrix
        logging.info("Processing encode results and building transition matrix...")

        # Group results by letter index
        results_by_letter = {}
        for req_info in all_encode_requests:
            letter_idx = req_info["letter_index"]
            if letter_idx not in results_by_letter:
                results_by_letter[letter_idx] = []
            results_by_letter[letter_idx].append(req_info)

        # Process each letter's results
        for j, initial_letter in enumerate(letters):
            if j in results_by_letter:
                letter_encode_requests = results_by_letter[j]
                probs = process_encode_batch_results(
                    encode_batch_results, letter_encode_requests, letters,
                    multiple_summarization=multiple_summarization
                )

                # Calculate average probabilities for this letter
                mean_on_perm_probs = stats.trim_mean(
                    np.asarray(list(probs.values())), 0, axis=0
                )
                avg_probs = np.mean(mean_on_perm_probs, axis=1)
                results_probs[j, :] = avg_probs

                # Store raw data
                for str_permutation in str_permutations:
                    all_raw_probs[str_permutation][:, j, :] = probs[str_permutation].T

                logging.info(
                    f"Completed processing for letter {initial_letter} (index {j})"
                )

    else:

        # Test each possible initial stance letter
        for j, initial_letter in enumerate(letters):
            answer, expected_rate, avg_probs, raw_probs = decode_and_encode(
                model,
                client,
                proposition,
                letters,
                repitition,  # shuffle_rep
                sep,
                initial_letter,
                max_tokens,
                max_argument_words,
                log_filename,
                prompt_choice,
                get_probs=True,
                multiple_summarization=multiple_summarization,
                summarization_count=summarization_count,
                reasoning_effort=reasoning_effort,
            )
            # Store results in transition matrix (row j = initial letter)
            results_probs[j, :] = avg_probs  # np.round(avg_probs,3)
            # Store raw data for uncertainty analysis
            for str_permutation in str_permutations:
                all_raw_probs[str_permutation][:, j, :] = raw_probs[str_permutation].T

    return results_probs, all_raw_probs


def wrap_text(sentence):
    """
    Wrap long text for better display in visualizations.

    This utility function breaks long sentences into multiple lines
    to improve readability in plot titles and labels.

    Args:
        sentence (str): Text to wrap

    Returns:
        str: Text with line breaks inserted every 8 words
    """
    words = sentence.split()
    lines = []
    for i in range(0, len(words), 6):
        line = " ".join(words[i : i + 6])
        lines.append(line)
    return "\n".join(lines)


def visualize_transition_matrices(results_tranmat, letters, title, file_name):
    """
    Create heatmap visualizations of transition matrices.

    This function generates publication-ready heatmaps showing the transition
    probabilities between stance letters, helping visualize systematic biases
    and consistency patterns in model behavior.

    Args:
        results_tranmat (np.ndarray): Transition matrix to visualize
        letters (list): Response letters for axis labeling
        title (str): Plot title (will be wrapped for long titles)
        file_name (str): Output filename (without extension)

    Note:
        Creates a 2-subplot figure with detailed annotations and saves as PNG.
        Axes are labeled with semantic stance meanings rather than just letters.
    """
    fig, ax = plt.subplots(
        1, 1, figsize=(14,15) #(13.36, 15)
    ) 
    xticks = np.arange(0, len(letters), 1)
    letter_xticks = [letter_to_option(letters[tick]) for tick in list(xticks)]
    yticks = np.arange(len(letters) - 1, -1, -1)
    letter_yticks = [letter_to_option(letters[tick]) for tick in list(yticks)]

    matrix = results_tranmat
    sns.heatmap(np.round(matrix, 2), ax=ax, annot=True, cmap="Reds", cbar=False, vmin=0, vmax=1, annot_kws={"fontsize": 35})
    # ax.figure.axes[-1].yaxis.set_tick_params(labelsize=35)
    
    # ax.set_title("Identity Matrix\n(Perfect Stance Preservation)", fontsize=40, fontweight="bold")
    # ax.set_title("Empirical Transition Matrix\n(Produced by gpt-4.1)", fontsize=40, fontweight="bold")
    ax.set_title(wrap_text(title), fontsize=40, fontweight="bold")


    ax.set_xlabel("To State", fontsize=35, fontweight="bold")
    ax.set_xticks(xticks + 0.5)
    ax.set_xticklabels([tick.replace(" ", "\n") for tick in letter_xticks], fontsize=35, rotation=30, fontweight="bold", ha="center")

    # ax.set_ylabel(None)
    ax.set_ylabel("From State", fontsize=35, fontweight="bold")
    # ax.set_yticks([],[])
    ax.set_yticks(yticks + 0.5)
    ax.set_yticklabels([tick.replace(" ", "\n") for tick in letter_yticks], fontsize=35, rotation=30, fontweight="bold", ha="right")

    plt.tight_layout()
    plt.savefig(f"{file_name}.pdf", bbox_inches="tight", dpi=300)


class NumpyArrayEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            # Convert the NumPy array to a list
            return obj.tolist()
        # Otherwise, let the base class default method raise the TypeError
        return json.JSONEncoder.default(self, obj)


def chat_with_backoff(client, model, messages, attempts=6, base_delay=1.0, reasoning_effort=None, **kwargs):
    """
    Chat completion with exponential backoff retry logic for both OpenAI and TogetherAI.

    Args:
        client: UnifiedChatClient instance or legacy OpenAI client
        model: Model name (e.g., "gpt-4o-mini" or "meta-llama/Llama-3-8b-chat-hf")
        messages: List of message dicts
        attempts: Maximum number of retry attempts (default: 6)
        base_delay: Base delay in seconds for exponential backoff (default: 1.0)
        **kwargs: Additional arguments to pass to the underlying API

    Returns:
        Normalized chat completion response

    Raises:
        RuntimeError: If all retry attempts fail
    """
    if "gpt-5" in model.lower():
        kwargs.pop("temperature", None)
        kwargs.pop("max_tokens", None)
        if reasoning_effort not in (None, "none"):
            kwargs["reasoning_effort"] = reasoning_effort
    for i in range(attempts):
        try:
            # Check if using unified client
            if isinstance(client, UnifiedChatClient):
                return client.chat(model=model, messages=messages, **kwargs)
            else:
                # Legacy OpenAI client
                return client.chat.completions.create(
                    model=model, messages=messages, **kwargs
                )
        except (RateLimitError, Exception) as e:
            # Handle rate limit and other retryable errors from both APIs
            if "rate" in str(e).lower() or isinstance(e, RateLimitError):
                # Rate limit error - use exponential backoff with jitter
                delay = base_delay * (2**i) + 0.2 * random.random()
                logging.warning(
                    f"Rate limit hit, retrying in {delay:.2f}s (attempt {i+1}/{attempts})"
                )
                time.sleep(delay)
            elif any(
                keyword in str(e).lower()
                for keyword in ["timeout", "connection", "network"]
            ):
                # Network/timeout error - use exponential backoff without jitter
                delay = base_delay * (2**i)
                logging.warning(
                    f"Network error: {type(e).__name__}, retrying in {delay:.2f}s (attempt {i+1}/{attempts})"
                )
                time.sleep(delay)
            else:
                delay = base_delay * (2**i)
                logging.warning(
                    f"API error: {type(e).__name__}, retrying in {delay:.2f}s (attempt {i+1}/{attempts})"
                )
                time.sleep(delay)

    raise RuntimeError(f"API call failed after {attempts} attempts")


# batch API calling


def create_decode_batch_requests(
    model,
    proposition,
    letters,
    shuffle_rep,
    sep,
    letter,
    max_tokens=200,
    max_argument_words=100,
    prompt_choice=1,
    reasoning_effort=None,
):
    """
    Creates batch requests for decode phase.

    Args:
        model (str): OpenAI model name
        proposition (str): The statement/proposition to evaluate
        letters (list): Response letters (e.g., ['A','B','C','D','E'])
        shuffle_rep (int): Number of experimental repetitions
        sep (str): Separator string for prompt formatting
        letter (str): Initial stance letter to decode from
        max_tokens (int): Maximum tokens for decode response
        max_argument_words (int): Word limit for generated arguments
        prompt_choice (int): Prompt template variant

    Returns:
        list: List of batch request objects for decode phase
    """
    permutations = permutate_letters(letters)
    decode_requests = []

    for permutation_id, permutation in enumerate(permutations):
        for rep in range(shuffle_rep):
            letters_shuffled_decode = permutation[0]

            # Create decode request
            decode_prompt = generate_prompt(
                task="decode",
                proposition=proposition,
                letters=letters,
                letters_shuffled=letters_shuffled_decode,
                sep=sep,
                letter=letters[letters_shuffled_decode.index(letter)],
                words_limit=max_argument_words,
                prompt_choice=prompt_choice,
            )

            decode_body = {
                "model": model,
                "messages": [{"role": "user", "content": decode_prompt}],
            }
            if "gpt-5" not in model.lower():
                decode_body["max_tokens"] = max_tokens
                decode_body["temperature"] = 0.7
            elif reasoning_effort not in (None, "none"):
                decode_body["reasoning_effort"] = reasoning_effort

            decode_request = {
                "custom_id": f"decode_{permutation_id}_{rep}",
                "method": "POST",
                "url": "/v1/chat/completions",
                "body": decode_body,
            }
            decode_requests.append(
                {
                    "request": decode_request,
                    "permutation_id": permutation_id,
                    "permutation": permutation,
                    "rep": rep,
                }
            )

    return decode_requests


def create_encode_batch_requests(
    model,
    proposition,
    letters,
    decode_results,
    sep,
    prompt_choice=1,
    multiple_summarization=False,
    summarization_count=5,
    reasoning_effort=None,
):
    """
    Create batch requests for encode phase using decode results.

    Args:
        model (str): OpenAI model name
        proposition (str): The statement/proposition to evaluate
        letters (list): Response letters
        decode_results (list): Results from decode batch processing
        sep (str): Separator string for prompt formatting
        prompt_choice (int): Prompt template variant
        multiple_summarization (bool): If True, create multiple encode requests per argument
            with randomly shuffled option orders to reduce position bias.
        summarization_count (int): Number of encodings with different shuffled option
            orders when using multiple summarization. Defaults to 5.

    Returns:
        list: List of batch request objects for encode phase
    """
    encode_requests = []

    for result in decode_results:
        permutation = result["permutation"]
        letters_shuffled_encode = permutation[1]
        argument = result["argument"]

        if multiple_summarization:
            # Multiple summarization: create multiple encodings with random shuffled orders
            for shuffle_idx in range(summarization_count):
                # Generate random shuffle of letters for this encoding
                letters_shuffled_current = letters.copy()
                random.shuffle(letters_shuffled_current)

                encode_prompt = generate_prompt(
                    task="encode",
                    proposition=proposition,
                    letters=letters,
                    letters_shuffled=letters_shuffled_current,
                    sep=sep,
                    argument=argument,
                    prompt_choice=prompt_choice,
                )

                encode_body = {
                    "model": model,
                    "messages": [{"role": "user", "content": encode_prompt}],
                }
                if "gpt-5" not in model.lower():
                    encode_body["max_tokens"] = 10
                    encode_body["logprobs"] = True
                    encode_body["top_logprobs"] = 8
                    encode_body["temperature"] = 0.7
                elif reasoning_effort not in (None, "none"):
                    encode_body["reasoning_effort"] = reasoning_effort

                encode_request = {
                    "custom_id": f"encode_{result['permutation_id']}_{result['rep']}_shuffle_{shuffle_idx}",
                    "method": "POST",
                    "url": "/v1/chat/completions",
                    "body": encode_body,
                }

                encode_requests.append(
                    {
                        "request": encode_request,
                        "permutation_id": result["permutation_id"],
                        "permutation": result["permutation"],
                        "rep": result["rep"],
                        "argument": argument,
                        "shuffle_idx": shuffle_idx,
                        "letters_shuffled_encode": letters_shuffled_current,
                    }
                )
        else:
            # Single encoding (original behavior)
            encode_prompt = generate_prompt(
                task="encode",
                proposition=proposition,
                letters=letters,
                letters_shuffled=letters_shuffled_encode,
                sep=sep,
                argument=argument,
                prompt_choice=prompt_choice,
            )

            encode_body = {
                "model": model,
                "messages": [{"role": "user", "content": encode_prompt}],
            }
            if "gpt-5" not in model.lower():
                encode_body["max_tokens"] = 10
                encode_body["logprobs"] = True
                encode_body["top_logprobs"] = 8
                encode_body["temperature"] = 0.7
            elif reasoning_effort not in (None, "none"):
                encode_body["reasoning_effort"] = reasoning_effort

            encode_request = {
                "custom_id": f"encode_{result['permutation_id']}_{result['rep']}_single",
                "method": "POST",
                "url": "/v1/chat/completions",
                "body": encode_body,
            }

            encode_requests.append(
                {
                    "request": encode_request,
                    "permutation_id": result["permutation_id"],
                    "permutation": result["permutation"],
                    "rep": result["rep"],
                    "argument": argument,
                    "shuffle_idx": 0,
                    "letters_shuffled_encode": letters_shuffled_encode,
                }
            )

    return encode_requests


def submit_batch_and_wait(client, requests, batch_description="Batch job"):
    """
    Submit a batch job and wait for completion.

    Args:
        client: OpenAI client instance
        requests (list): List of request objects
        batch_description (str): Description for the batch job

    Returns:
        dict: Batch results with custom_id as keys
    """
    import tempfile
    import json
    import os

    logging.info(f"Creating batch with {len(requests)} requests")

    # Validate first request to check model compatibility
    if requests:
        first_request_body = requests[0]["request"]["body"]
        model_name = first_request_body.get("model", "")
        logging.info(f"Using model: {model_name}")

        # Check for potential issues with model names
        if "gpt-3.5-turbo" in model_name:
            logging.warning(
                "Note: gpt-3.5-turbo support for batch API may have limitations"
            )

    # Create temporary JSONL file
    with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
        for i, req_info in enumerate(requests):
            # Validate request structure
            if "request" not in req_info:
                raise ValueError(f"Request {i} missing 'request' key: {req_info}")

            request = req_info["request"]

            # Validate required fields for batch API
            if "custom_id" not in request:
                raise ValueError(f"Request {i} missing 'custom_id': {request}")
            if "method" not in request:
                raise ValueError(f"Request {i} missing 'method': {request}")
            if "url" not in request:
                raise ValueError(f"Request {i} missing 'url': {request}")
            if "body" not in request:
                raise ValueError(f"Request {i} missing 'body': {request}")

            # Log first few requests for debugging
            if i < 3:
                logging.info(f"Request {i}: {json.dumps(request, indent=2)}")

            json.dump(request, f)
            f.write("\n")
        batch_file_path = f.name

    # Log file size for debugging
    file_size = os.path.getsize(batch_file_path)
    logging.info(
        f"Batch file size: {file_size} bytes ({file_size / 1024 / 1024:.2f} MB)"
    )

    try:
        # Upload batch file
        with open(batch_file_path, "rb") as f:
            batch_input_file = client.files.create(file=f, purpose="batch")

        # Create batch job
        batch_job = client.batches.create(
            input_file_id=batch_input_file.id,
            endpoint="/v1/chat/completions",
            completion_window="24h",
            metadata={"description": batch_description},
        )

        logging.info(f"Batch job created: {batch_job.id}")
        print(f"Batch job created: {batch_job.id}")

        # Wait for completion
        while True:
            batch_job = client.batches.retrieve(batch_job.id)
            print(f"Batch status: {batch_job.status}")

            if batch_job.status == "completed":
                break
            elif batch_job.status in ["failed", "expired", "cancelled"]:
                # Get error details if available
                error_info = ""
                if hasattr(batch_job, "errors") and batch_job.errors:
                    error_info = f"\nErrors: {batch_job.errors}"
                if hasattr(batch_job, "error_file_id") and batch_job.error_file_id:
                    try:
                        error_file_content = client.files.content(
                            batch_job.error_file_id
                        )
                        error_info += f"\nError file content: {error_file_content.text[:1000]}"  # First 1000 chars
                    except Exception as e:
                        error_info += f"\nCould not retrieve error file: {e}"

                raise RuntimeError(
                    f"Batch job failed with status: {batch_job.status}{error_info}"
                )

            time.sleep(30)  # Check every 30 seconds


        # Download results
        print("Downloading results from batch job...")
        result_file_id = batch_job.output_file_id
        if result_file_id is None:
            error_info = (
                f"Batch completed but output_file_id is None "
                f"(request counts: {batch_job.request_counts}). "
                f"All requests may have failed."
            )
            if hasattr(batch_job, "error_file_id") and batch_job.error_file_id:
                try:
                    error_content = client.files.content(batch_job.error_file_id)
                    error_info += f"\nError file sample: {error_content.text[:2000]}"
                except Exception as e:
                    error_info += f"\nCould not retrieve error file: {e}"
            raise RuntimeError(error_info)
        result_content = client.files.content(result_file_id)

        # Parse results
        print("Parsing results...")
        results = {}
        for line in result_content.text.split("\n"):
            if line.strip():
                result = json.loads(line)
                results[result["custom_id"]] = result

        return results

    finally:
        # Clean up temporary file
        os.unlink(batch_file_path)


def process_decode_batch_results(batch_results, decode_requests):
    """
    Process decode batch results to extract arguments.

    Args:
        batch_results (dict): Results from batch API
        decode_requests (list): Original decode request metadata

    Returns:
        list: Processed decode results with arguments
    """
    processed_results = []

    for req_info in decode_requests:
        custom_id = req_info["request"]["custom_id"]
        if custom_id in batch_results:
            result = batch_results[custom_id]
            argument = result["response"]["body"]["choices"][0]["message"]["content"]

            processed_results.append(
                {
                    "permutation_id": req_info["permutation_id"],
                    "permutation": req_info["permutation"],
                    "rep": req_info["rep"],
                    "argument": argument,
                    "letter_index": req_info["letter_index"],
                    "initial_letter": req_info["initial_letter"],
                }
            )

            logging.info(f"Rep {req_info['rep']}: Generated argument: {argument}")

    return processed_results


def _extract_top_logprobs(result):
    """
    Extract top_logprobs from a batch result.

    For models that support logprobs the real values are returned directly.
    For models that don't (e.g. gpt-5), logprobs will be None in the response,
    so we fake them by extracting the first valid letter (A-E) from the text
    content and assigning it logprob=0.0 (i.e. probability=1.0), mirroring
    what UnifiedChatClient._call_openai does in the non-batch path.
    Raises ValueError if no valid letter can be extracted, so a missing/empty
    encode response stops the run loudly instead of silently degrading.
    """
    import re
    choice = result["response"]["body"]["choices"][0]
    logprobs = choice.get("logprobs")
    if logprobs and logprobs.get("content"):
        return logprobs["content"][0]["top_logprobs"]
    # Fake logprobs: extract first valid letter from the text response
    text_content = choice["message"]["content"].strip()
    letter_match = re.search(r"([A-E])", text_content)
    if letter_match:
        return [{"token": letter_match.group(1), "logprob": 0.0}]
    raise ValueError(
        f"Encode response (custom_id={result.get('custom_id', 'unknown')}) contained "
        f"no parseable A-E letter: '{text_content}'. The model likely returned empty "
        f"content (e.g. reasoning consumed the entire token budget) or refused."
    )


def process_encode_batch_results(batch_results, encode_requests, letters, multiple_summarization=False):
    """
    Process encode batch results to extract probabilities.

    Args:
        batch_results (dict): Results from batch API
        encode_requests (list): Original encode request metadata
        letters (list): Response letters
        multiple_summarization (bool): If True, average probabilities from multiple
            encodings with different shuffled option orders.

    Returns:
        dict: Probability distributions by permutation
    """
    permutations = permutate_letters(letters)
    str_permutations = [str(permutation) for permutation in permutations]
    shuffle_rep = max([req_info["rep"] for req_info in encode_requests]) + 1

    probs_all = {}
    for str_permutation in str_permutations:
        probs_all[str_permutation] = np.zeros((len(letters), shuffle_rep))

    if multiple_summarization:
        # Group requests by (permutation_id, rep) to collect all shuffled encodings
        grouped_requests = {}
        for req_info in encode_requests:
            key = (req_info["permutation_id"], req_info["rep"])
            if key not in grouped_requests:
                grouped_requests[key] = []
            grouped_requests[key].append(req_info)

        # Process each group of shuffled encodings
        for key, shuffle_group in grouped_requests.items():
            permutation_id, rep = key

            if not shuffle_group:
                continue

            # Get permutation info from first request
            permutation = shuffle_group[0]["permutation"]
            str_permutation = str(permutation)

            # Collect probabilities from all shuffled encodings
            all_probs = []
            for req_info in shuffle_group:
                custom_id = req_info["request"]["custom_id"]
                if custom_id in batch_results:
                    result = batch_results[custom_id]
                    top_logprobs = _extract_top_logprobs(result)
                    letters_shuffled_encode = req_info["letters_shuffled_encode"]
                    shuffle_idx = req_info.get("shuffle_idx", 0)
                    logging.info(f"Rep {rep} (shuffle {shuffle_idx}): Top choice: <{top_logprobs[0]['token']}>")

                    probs_current = np.zeros(len(letters))
                    for logprob in top_logprobs:
                        if logprob["token"].strip() in letters:
                            letter_idx = letters.index(
                                letters_shuffled_encode[letters.index(logprob["token"].strip())]
                            )
                            probs_current[letter_idx] += np.exp(logprob["logprob"])
                    
                    all_probs.append(probs_current)

            # Average probabilities across all shuffled encodings
            if all_probs:
                avg_probs = np.mean(all_probs, axis=0)
                probs_all[str_permutation][:, rep] = avg_probs
    else:
        # Single encoding behavior
        for req_info in encode_requests:
            custom_id = req_info["request"]["custom_id"]
            if custom_id in batch_results:
                result = batch_results[custom_id]
                top_logprobs = _extract_top_logprobs(result)

                permutation = req_info["permutation"]
                str_permutation = str(permutation)
                rep = req_info["rep"]
                # Use stored letters_shuffled_encode if available, fallback to permutation[1]
                letters_shuffled_encode = req_info.get("letters_shuffled_encode", permutation[1])

                logging.info(f"Rep {rep}: Top choice: <{top_logprobs[0]['token']}>")

                # Extract probabilities for valid response letters
                for logprob in top_logprobs:
                    if logprob["token"].strip() in letters:
                        letter_idx = letters.index(
                            letters_shuffled_encode[letters.index(logprob["token"].strip())]
                        )
                        probs_all[str_permutation][letter_idx, rep] += np.exp(
                            logprob["logprob"]
                        )

    # Normalize probabilities
    probs = {}
    for str_permutation in str_permutations:
        probs[str_permutation] = get_normalized_prob(probs_all[str_permutation])

    return probs
