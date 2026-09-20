"""
Shared model name mappings for LaTeX tables and plot labels.
"""

# LaTeX-formatted names (used in .tex table output)
MODEL_NAME_LATEX = {
    "gpt_4o_mini": r"\texttt{GPT-4o-mini}",
    "gpt_4_1": r"\texttt{GPT-4.1}",
    "gpt_5_4": r"\texttt{GPT-5.4}",
    "gpt_3_5_turbo": r"\texttt{GPT-3.5-turbo}",
    "gemma_3n_e4b": r"\texttt{gemma-3n-E4B}",
    "llama3_3_70b": r"\texttt{Llama-3.3-70B}",
    "llama3_1_8b": r"\texttt{Llama-3.1-8B}",
    "llama4_maverick": r"\texttt{Llama-4-Maverick}",
    "qwen3_a3b": r"\texttt{Qwen3-A3B}",
    "gpt_4o_mini_reversed": r"\texttt{GPT-4o-mini (reversed)}",
    "gpt_4o_mini_temp0": r"\texttt{GPT-4o-mini (temp0)}",
    "gpt_4o_mini_multiple_summarization": r"\texttt{GPT-4o-mini (multiple extraction)}",
    "gpt_4o_mini_in_context": r"\texttt{GPT-4o-mini (in-context)}",
    "gpt_4o_mini_assert": r"\texttt{GPT-4o-mini (assert)}",
    "gpt_4o_mini_reflection": r"\texttt{GPT-4o-mini (reflection)}",
    "gpt_5_4_reflection": r"\texttt{GPT-5.4 (reflection)}",
    "gpt_5_4_reflection_third_person": r"\texttt{GPT-5.4 (third-person reflection)}",
    "gpt_5_4_reflection_third_person_medium": r"\texttt{GPT-5.4 (third-person reflection, medium reasoning)}",
}

# Plot-formatted names (used in matplotlib xtick labels, with \n for line breaks)
MODEL_NAME_PLOT = {
    "gpt_4o_mini": "GPT-4o-mini",
    "gpt_4_1": "GPT-4.1",
    "gpt_5_4": "GPT-5.4",
    "gpt_3_5_turbo": "GPT-3.5-turbo",
    "gemma_3n_e4b": "gemma-3n-\nE4B",
    "llama3_3_70b": "Llama-3.3-70B",
    "llama3_1_8b": "Llama-3.1-8B",
    "llama4_maverick": "Llama-4-\nMaverick",
    "qwen3_a3b": "Qwen3-A3B",
    "gpt_4o_mini_reversed": "GPT-4o-mini\n(reversed)",
    "gpt_4o_mini_temp0": "GPT-4o-mini\n(temp0)",
    "gpt_4o_mini_multiple_summarization": "GPT-4o-mini\n(multiple\nextraction)",
    "gpt_4o_mini_in_context": "GPT-4o-mini\n(in-context)",
    "gpt_4o_mini_assert": "GPT-4o-mini\n(assert)",
    "gpt_4o_mini_reflection": "GPT-4o-mini\n(reflection)",
    "gpt_5_4_reflection": "GPT-5.4\n(reflection)",
    "gpt_5_4_reflection_third_person": "GPT-5.4\n(third-person\nreflection)",
    "gpt_5_4_reflection_third_person_medium": "GPT-5.4\n(third-person\nreflection,\nmedium reasoning)",
}


def get_latex_name(model_key):
    """Get LaTeX-formatted model name, with fallback to escaped raw name."""
    return MODEL_NAME_LATEX.get(model_key, model_key.replace("_", r"\_"))


def get_plot_name(model_key):
    """Get plot-formatted model name, with fallback to raw name."""
    return MODEL_NAME_PLOT.get(model_key, model_key)


def get_plot_name_single_line(model_key):
    """Plot label on one line (e.g. horizontal bar y-ticks); strips embedded newlines."""
    return get_plot_name(model_key).replace("\n", " ")
