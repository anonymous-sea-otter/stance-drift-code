"""
Export supplementary Data S1 as a multi-sheet Excel file.

Each sheet contains the tabulated data underlying one or more figure/table panels
in machine-readable format (numeric columns, no formatted strings).

Sheets:
    Fig2A_S3A   - Transition matrix (topic 2401, gpt_5_4): Mean & SE
    Fig2B_S3B   - Transition matrix (topic 3234, gpt_5_4): Mean & SE
    Table1A     - Success rate by initial stance          (9 subset models)
    Table1B     - Bonferroni-Corrected CI Inclusion Rate  (9 subset models)
    Fig5        - SPR bar chart values (9 subset models)
    TableS1     - Success rate by initial stance          (18 comprehensive runs)
    TableS2     - Bonferroni-Corrected CI Inclusion Rate  (18 comprehensive runs)
    TableS3     - CI Inclusion Rate                       (18 comprehensive runs)
    TableS4     - Pass rate at threshold 0.95             (18 comprehensive runs)
    FigS4       - SPR bar chart values (18 comprehensive runs)
    FigS6A      - Pass rate vs threshold curves           (18 comprehensive runs)
    FigS6B      - AUC of the pass-rate curve (normalized) (18 comprehensive runs)
"""

import os
import numpy as np
import pandas as pd
from visualization import load_and_compute, get_raw_json_path
from name_maps import get_plot_name

RESULTS_DIR = "postprocess_results"
DETAILED_CSV = "postprocess_detailed_results.csv"
MODEL_COMPARISON_CSV = "model_comparison.csv"
AUC_CSV = os.path.join("figures", "auc_results_all_letters.csv")
OUTPUT_FILE = os.path.join("figures", "data_S1.xlsx")

STANCE_ORDER = ["A", "B", "C", "D", "E"]
STANCE_LABELS = {
    "A": "Agree strongly",
    "B": "Agree",
    "C": "Neutral",
    "D": "Disagree",
    "E": "Disagree strongly",
}

SUBSET_MODELS = [
    "gpt_4o_mini",
    "gpt_4_1",
    "gpt_5_4",
    "gpt_3_5_turbo",
    "gemma_3n_e4b",
    "llama3_3_70b",
    "llama3_1_8b",
    "llama4_maverick",
    "qwen3_a3b",
]

COMPREHENSIVE_MODELS = [
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

HEATMAP_CONFIGS = [
    {"topic_id": "2401", "model": "gpt_5_4", "sheet": "Fig2A_S3A"},
    {"topic_id": "3234", "model": "gpt_5_4", "sheet": "Fig2B_S3B"},
]

TABLE_METRICS = [
    ("Mean_mean", "Mean_standard_error"),
    ("Includes_1_bonferroni_mean", "Includes_1_bonferroni_standard_error"),
    ("Includes_1_mean", "Includes_1_standard_error"),
    ("Success_rate_mean", "Success_rate_standard_error"),
]

THRESHOLDS = np.round(np.arange(0.50, 1.05, 0.05), 2)


def display_name(model_key):
    """Human-readable model name for Excel columns/cells."""
    return get_plot_name(model_key).replace("\n", " ")


def build_transition_matrix_df(topic_id, model, prompt_id=1):
    """5x5 transition matrix as long-form DataFrame with Mean and SE."""
    json_path, _ = get_raw_json_path(topic_id, model, prompt_id)
    if not os.path.exists(json_path):
        print(f"  Warning: {json_path} not found, skipping.")
        return None
    mean, se = load_and_compute(json_path)
    rows = []
    for i, fl in enumerate(STANCE_ORDER):
        for j, tl in enumerate(STANCE_ORDER):
            rows.append({
                "From_State": STANCE_LABELS[fl],
                "To_State": STANCE_LABELS[tl],
                "Mean": round(float(mean[i, j]), 6),
                "SE": round(float(se[i, j]), 6),
            })
    return pd.DataFrame(rows)


def build_table_df(model_list, mean_col, se_col):
    """Model x Stance table with Mean and SE columns."""
    comparison_file = os.path.join(RESULTS_DIR, MODEL_COMPARISON_CSV)
    df = pd.read_csv(comparison_file)
    df = df[df["Model"].isin(model_list)]
    rows = []
    for model in model_list:
        for letter in STANCE_ORDER:
            cell = df[(df["Model"] == model) & (df["Letter"] == letter)]
            if len(cell) == 0:
                continue
            rows.append({
                "Model": display_name(model),
                "Stance": STANCE_LABELS[letter],
                "Mean": round(float(cell[mean_col].values[0]), 6),
                "SE": round(float(cell[se_col].values[0]), 6),
            })
    return pd.DataFrame(rows)


def build_spr_df(model_list):
    """Per-model average Stance Preservation Rate (bar chart values)."""
    rows = []
    for model in model_list:
        csv_path = os.path.join(RESULTS_DIR, model, DETAILED_CSV)
        if not os.path.exists(csv_path):
            continue
        df = pd.read_csv(csv_path)
        rows.append({
            "Model": display_name(model),
            "SPR_Mean": round(float(df["Mean"].mean()), 6),
        })
    return pd.DataFrame(rows)


def build_threshold_curves_df(model_list):
    """Pass rate vs threshold (Fig. S6A): models as rows, thresholds as columns."""
    records = []
    for model in model_list:
        csv_path = os.path.join(RESULTS_DIR, model, DETAILED_CSV)
        if not os.path.exists(csv_path):
            continue
        df = pd.read_csv(csv_path)
        row = {"Model": display_name(model)}
        for t in THRESHOLDS:
            t_eff = t if t <= 1.0 else 1.0 - 1e-6
            rate = float((df["Mean"] > t_eff).mean())
            row[f"Rate_{t:.2f}"] = round(rate, 6)
        records.append(row)
    return pd.DataFrame(records)


def build_auc_df():
    """AUC bar chart values (already computed by success_rate_metric.py)."""
    df = pd.read_csv(AUC_CSV)
    df["Model"] = df["Model"].apply(display_name)
    df["AUC"] = df["AUC"].round(6)
    return df


def main():
    os.makedirs("figures", exist_ok=True)
    comparison_file = os.path.join(RESULTS_DIR, MODEL_COMPARISON_CSV)
    if not os.path.exists(comparison_file):
        print(f"Error: {comparison_file} not found. Run postprocess.py first.")
        return

    sheet_count = 0
    with pd.ExcelWriter(OUTPUT_FILE, engine="openpyxl") as writer:

        # Transition matrices (Fig 2A+S3A, Fig 2B+S3B)
        for cfg in HEATMAP_CONFIGS:
            df = build_transition_matrix_df(cfg["topic_id"], cfg["model"])
            if df is not None:
                df.to_excel(writer, sheet_name=cfg["sheet"], index=False)
                sheet_count += 1

        # Table 1A & 1B (subset models)
        subset_sheets = [
            ("Table1A", *TABLE_METRICS[0]),
            ("Table1B", *TABLE_METRICS[1]),
        ]
        for sheet_name, mean_col, se_col in subset_sheets:
            build_table_df(SUBSET_MODELS, mean_col, se_col).to_excel(
                writer, sheet_name=sheet_name, index=False
            )
            sheet_count += 1

        # Fig 5 (subset SPR bar)
        build_spr_df(SUBSET_MODELS).to_excel(
            writer, sheet_name="Fig5", index=False
        )
        sheet_count += 1

        # Table S1–S4 (comprehensive models)
        for i, (mean_col, se_col) in enumerate(TABLE_METRICS, start=1):
            build_table_df(COMPREHENSIVE_MODELS, mean_col, se_col).to_excel(
                writer, sheet_name=f"TableS{i}", index=False
            )
            sheet_count += 1

        # Fig S4 (comprehensive SPR bar)
        build_spr_df(COMPREHENSIVE_MODELS).to_excel(
            writer, sheet_name="FigS4", index=False
        )
        sheet_count += 1

        # Fig S6A (pass-rate curves)
        build_threshold_curves_df(COMPREHENSIVE_MODELS).to_excel(
            writer, sheet_name="FigS6A", index=False
        )
        sheet_count += 1

        # Fig S6B (AUC bar)
        if os.path.exists(AUC_CSV):
            build_auc_df().to_excel(
                writer, sheet_name="FigS6B", index=False
            )
            sheet_count += 1
        else:
            print(f"  Warning: {AUC_CSV} not found, skipping FigS6B.")

    print(f"Data S1 saved to: {OUTPUT_FILE}")
    print(f"  Sheets written: {sheet_count}")


if __name__ == "__main__":
    main()
