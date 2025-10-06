
"""
OCR Visualization Utilities
===========================

This module creates clean, publication-ready grouped boxplots for comparing
OCR methods (traditional vs LLM) across tasks (Ingredients vs Nutrition).
It also includes t-test helpers to assess statistical significance.

Key features
------------
- Load multiple CSV "runs" per method.
- Dynamic metric selection (WER, CER, F1, GRITS, time, RAM, etc.).
- Cold-start control (drop first N rows per run).
- Failure filtering (e.g., drop rows with F1 == 0).
- Grouped boxplots with spacing and slim boxes (Matplotlib only).
- Combine runs into one box per task, or show each run as its own box.
- Optional dataset filtering if a column exists (e.g., 'dataset' or 'task').
- Paired/unpaired t-tests between methods per task.

Assumptions
-----------
- CSV files use ';' as separator.
- Metric column names are consistent across files.
- Method type is inferred from columns:
  - LLM files contain columns ending with '_llm'
  - Traditional OCR files contain columns ending with '_ocr' or '_trad'
- For Ingredients vs Nutrition filtering, if present we try columns named:
  ['dataset', 'task', 'type'] and match values (case-insensitive) against
  {'ingredients', 'nutrition'}.

Example usage
-------------
from ocr_viz import grouped_boxplot_two_methods_two_tasks, paired_ttests_two_tasks

files_trad_ing = ["tesseract_ingredients_run1.csv", "tesseract_ingredients_run2.csv", "tesseract_ingredients_run3.csv"]
files_trad_nut = ["tesseract_nutrition_run1.csv",  "tesseract_nutrition_run2.csv",  "tesseract_nutrition_run3.csv"]
files_llm_ing  = ["results_gpt4o_ingredients_run1.csv","results_gpt4o_ingredients_run2.csv","results_gpt4o_ingredients_run3.csv"]
files_llm_nut  = ["gpt4o_nutrition_run1.csv","gpt4o_nutrition_run2.csv","gpt4o_nutrition_run3.csv"]

# One box per task (runs combined), spaced & slim
grouped_boxplot_two_methods_two_tasks(
    method1_runs={"Ingredients": files_trad_ing, "Nutrition": files_trad_nut},
    method2_runs={"Ingredients": files_llm_ing,  "Nutrition": files_llm_nut},
    metric="f1", method1_name="Traditional OCR", method2_name="LLM (GPT-4)",
    combine_runs=True, remove_first_n=5, exclude_failures=True,
    figsize=(8,5), method_gap=1.5, box_width=0.25, showfliers=True,
    ylabel="F1-Score", title="Traditional vs LLM — F1 by Task"
)

# Paired t-tests per task
paired_ttests_two_tasks(
    method1_runs={"Ingredients": files_trad_ing, "Nutrition": files_trad_nut},
    method2_runs={"Ingredients": files_llm_ing,  "Nutrition": files_llm_nut},
    metric="f1", remove_first_n=5, exclude_failures=True
)

"""

from __future__ import annotations
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

plt.rcParams.update({
    "figure.dpi": 120,
    "axes.grid": True,
    "grid.linestyle": "--",
    "grid.alpha": 0.35,
    "axes.axisbelow": True,
    "font.size": 11
})

# ----------------------------- helpers ---------------------------------

_POSSIBLE_DATASET_COLS = ["dataset", "task", "type"]  # case-insensitive search


def _infer_method_type(df: pd.DataFrame) -> str:
    cols = [c.lower() for c in df.columns]
    if any(c.endswith("_llm") for c in cols):
        return "llm"
    if any(c.endswith("_ocr") or c.endswith("_trad") for c in cols):
        return "trad"
    return "unknown"


# Map generic metric names to typical column names per method type.
# Extend here if you have more metrics.
_METRIC_MAP = {
        "time_total":   {"llm": "time_llm_s",        "trad": "time_trad_s"},   # gesamte Verarbeitungszeit pro Bild
        "time_yolo":    {"trad": "time_yolo_s"},                               # Zeit für YOLO-Detektion
        "time_ocr":     {"trad": "time_ocr_s"},                                # Zeit für OCR (Tesseract)
        "time_api":     {"llm": "time_api_llm_s"},                             # Zeit für LLM-API-Aufruf
        "time_preproc": {"llm": "time_preproc_llm_s","trad": "time_preproc_s"},# Vorverarbeitungszeit
        "time_postproc":{"llm": "time_postproc_llm_s","trad": "time_postproc_s"},# Nachverarbeitungszeit
        "cpu_time":     {"llm": "cpu_llm_s",        "trad": "cpu_trad_s"},     # CPU-Zeit
        "mem_peak":     {"llm": "mem_llm_peak",     "trad": "mem_trad_peak"},  # Spitzen-Speicherverbrauch
        "wer":          {"llm": "wer_llm",          "trad": "wer_ocr"},        # Word Error Rate
        "cer":          {"llm": "cer_llm",          "trad": "cer_ocr"},        # Character Error Rate
        "precision_overall":    {"llm": "precision_overall_llm", "trad": "precision_overall_ocr"},
        "recall_overall":       {"llm": "recall_overall_llm",    "trad": "recall_overall_ocr"},
        "f1_overall":           {"llm": "f1_overall_llm",       "trad": "f1_overall_ocr"},
        "total_tokens": {"llm": "llm_total_tokens"},                        # Gesamte Tokenanzahl (LLM),
        "total_cost":   {"llm": "llm_cost_usd"},                        # Gesamtkosten (LLM)
        "overall_table_score": {"llm": "overall_table_score", "trad": "overall_table_score"}, # Gesamttabellen-Score
        "grits_content_overall_f1": {"llm": "grits_content_overall_f1", "trad": "grits_content_overall_f1"}, # GRITS Inhalts-F1
        "grits_content_overall_recall": {"llm": "grits_content_overall_recall", "trad": "grits_content_overall_recall"}, # GRITS Inhalts-Recall
        "grits_content_overall_precision": {"llm": "grits_content_overall_precision", "trad": "grits_content_overall_precision"}, # GRITS Inhalts-Precision
        "grits_topology_f1": {"llm": "grits_topology_f1", "trad": "grits_topology_f1"}, # GRITS Topologie-F1
        "grits_topology_recall": {"llm": "grits_topology_recall", "trad": "grits_topology_recall"}, # GRITS Topologie-Recall
        "grits_topology_precision": {"llm": "grits_topology_precision", "trad": "grits_topology_precision"}, # GRITS Topologie-Precision
        "precision_elements": {"llm": "precision_elements_llm", "trad": "precision_elements_ocr"}, # Element-Precision
        "recall_elements":    {"llm": "recall_elements_llm",    "trad": "recall_elements_ocr"},    # Element-Recall
        "f1_elements":        {"llm": "f1_elements_llm",        "trad": "f1_elements_ocr"},        # Element-F1
        "precision_values":   {"llm": "precision_values_llm",   "trad": "precision_values_ocr"},   # Wert-Precision
        "recall_values":      {"llm": "recall_values_llm",      "trad": "recall_values_ocr"},      # Wert-Recall
        "f1_values":          {"llm": "f1_values_llm",          "trad": "f1_values_ocr"},          # Wert-F1
        "precision_pairs":    {"llm": "precision_pairs_llm",    "trad": "precision_pairs_ocr"},    # Paar-Precision
        "recall_pairs":       {"llm": "recall_pairs_llm",       "trad": "recall_pairs_ocr"},       # Paar-Recall
        "f1_pairs":           {"llm": "f1_pairs_llm",           "trad": "f1_pairs_ocr"},           # Paar-F1
        "precision_triples":  {"llm": "precision_triples_llm",  "trad": "precision_triples_ocr"},  # Triple-Precision
        "recall_triples":     {"llm": "recall_triples_llm",     "trad": "recall_triples_ocr"},     # Triple-Recall
        "f1_triples":         {"llm": "f1_triples_llm",         "trad": "f1_triples_ocr"},         # Triple-F1
        "cell_accuracy":      {"llm": "cell_accuracy",      "trad": "cell_accuracy"},      # Zellen-Genauigkeit
        "grits_header_f1": {"llm": "grits_content_header_f1", "trad": "grits_content_header_f1"}, # GRITS Header-F1
        "grits_header_recall": {"llm": "grits_content_header_recall", "trad": "grits_content_header_recall"}, # GRITS Header-Recall
        "grits_header_precision": {"llm": "grits_content_header_precision", "trad": "grits_content_header_precision"}, # GRITS Header-Precision
        "grits_labels_f1": {"llm": "grits_content_labels_f1", "trad": "grits_content_labels_f1"}, # GRITS Labels-F1
        "grits_labels_recall": {"llm": "grits_content_labels_recall", "trad": "grits_content_labels_recall"}, # GRITS Labels-Recall
        "grits_labels_precision": {"llm": "grits_content_labels_precision", "trad": "grits_content_labels_precision"}, # GRITS Labels-Precision
        "grits_values_f1": {"llm": "grits_content_values_f1", "trad": "grits_content_values_f1"}, # GRITS Values-F1
        "grits_values_recall": {"llm": "grits_content_values_recall", "trad": "grits_content_values_recall"}, # GRITS Values-Recall
        "grits_values_precision": {"llm": "grits_content_values_precision", "trad": "grits_content_values_precision"}, # GRITS Values-Precision
        "title_similarity": {"llm": "title_similarity_llm", "trad": "title_similarity_ocr"}, # Ähnlichkeit Titel
        "footnote_similarity": {"llm": "footnote_similarity_llm", "trad": "footnote_similarity_ocr"}, # Ähnlichkeit Fußnoten
}


def _resolve_metric_column(df: pd.DataFrame, metric: str) -> Tuple[str, str]:
    """Return (method_type, column_name) for given metric in df."""
    mtype = _infer_method_type(df)
    if mtype not in _METRIC_MAP or metric not in _METRIC_MAP:
        # If unknown or unmapped, try direct column presence
        if metric in df.columns:
            return mtype, metric
        # Try case-insensitive match
        lower_map = {c.lower(): c for c in df.columns}
        if metric.lower() in lower_map:
            return mtype, lower_map[metric.lower()]
        raise KeyError(f"Cannot resolve metric '{metric}' in columns: {list(df.columns)[:6]}...")
    col = _METRIC_MAP[metric].get(mtype)
    if col is None:
        # Fallback: if user passed a direct column that exists
        if metric in df.columns:
            return mtype, metric
        raise KeyError(f"Metric '{metric}' not available for method type '{mtype}'.")
    # ensure exists (case-insensitive)
    lower_map = {c.lower(): c for c in df.columns}
    if col.lower() not in lower_map:
        raise KeyError(f"Expected column '{col}' not found. Available: {list(df.columns)[:6]}...")
    return mtype, lower_map[col.lower()]


def _filter_dataset(df: pd.DataFrame, dataset: Optional[str]) -> pd.DataFrame:
    """Filter by dataset/task if a suitable column exists."""
    if not dataset:
        return df
    dtarget = dataset.strip().lower()
    for col in _POSSIBLE_DATASET_COLS:
        for dfcol in df.columns:
            if dfcol.lower() == col:
                return df[df[dfcol].astype(str).str.lower() == dtarget]
    return df  # no matching column; return as-is


def _load_runs(files: List[str],
               metric: str,
               remove_first_n: int = 0,
               exclude_failures: bool = False,
               dataset: Optional[str] = None) -> Tuple[str, List[np.ndarray], pd.DataFrame]:
    """Load multiple CSVs and return method_type, list of 1D arrays (values per run), and merged df for t-tests."""
    runs = []
    merged = None
    mtype_global = None

    for i, f in enumerate(files):
        df = pd.read_csv(f, sep=";")
        df = _filter_dataset(df, dataset)
        if remove_first_n > 0 and len(df) > remove_first_n:
            df = df.iloc[remove_first_n:]

        # determine f1 column for failure filtering
        try:
            mtype_i, metric_col = _resolve_metric_column(df, metric)
        except KeyError as e:
            raise

        if mtype_global is None:
            mtype_global = mtype_i

        if exclude_failures:
            f1_col = None
            # pick correct overall f1 per method
            try:
                _, f1_col = _resolve_metric_column(df, "f1")
            except KeyError:
                # if no f1, leave as None
                pass
            if f1_col is not None:
                df = df[df[f1_col] > 0]

        values = df[metric_col].dropna().to_numpy()
        runs.append(values)

        # For paired t-tests we try to keep an id column if present
        # We guess typical id column names:
        id_col = None
        for c in df.columns:
            if c.lower() in ("product_id", "image_id", "doc_id", "id"):
                id_col = c
                break
        if id_col is not None:
            tmp = df[[id_col, metric_col]].copy()
            tmp.columns = ["_id", f"run{i+1}"]
            merged = tmp if merged is None else merged.merge(tmp, on="_id", how="outer")

    return mtype_global or "unknown", runs, merged


# -------------------------- plotting API --------------------------------

def grouped_boxplot_two_methods_two_tasks(
    method1_runs: Dict[str, List[str]],
    method2_runs: Dict[str, List[str]],
    metric: str,
    method1_name: str = "Traditional",
    method2_name: str = "LLM",
    tasks_order: Tuple[str, str] = ("Ingredients", "Nutrition"),
    combine_runs: bool = True,
    remove_first_n: int = 0,
    exclude_failures: bool = False,
    dataset_filter: Optional[str] = None,
    figsize: Tuple[float, float] = (9, 5),
    method_gap: float = 1.6,
    box_width: float = 0.28,
    showfliers: bool = True,
    ylabel: Optional[str] = None,
    title: Optional[str] = None,
    ax: Optional[plt.Axes] = None,
):
    """
    Create a grouped boxplot with spacing:
    X-axis has two groups (method1, method2). Within each, two boxes (tasks).
    - If combine_runs=True: each task box merges all runs into one distribution.
    - If combine_runs=False: each task shows one box per run, with legend.

    method?_runs: dict with keys == task names and values == list of CSV file paths (3 runs).
    metric: string metric name (generic or direct column).
    dataset_filter: if provided, try to filter rows for given dataset name.
    method_gap: spacing between the two method groups.
    box_width: set slimmer boxes.
    """
    # Load values for each (method, task)
    data = []
    legends = set()
    for m_idx, (m_name, runs_dict) in enumerate([(method1_name, method1_runs), (method2_name, method2_runs)]):
        for t_idx, task in enumerate(tasks_order):
            files = runs_dict.get(task, [])
            if not files:
                raise ValueError(f"No files provided for {m_name} / {task}")
            mtype, runs, _ = _load_runs(files, metric, remove_first_n, exclude_failures, dataset_filter)
            if combine_runs:
                # Combine all runs into one array
                combined = np.concatenate(runs) if len(runs) else np.array([])
                data.append((m_idx, t_idx, m_name, task, [combined]))
            else:
                data.append((m_idx, t_idx, m_name, task, runs))
                for i in range(len(runs)):
                    legends.add(f"Run {i+1}")

    # Prepare positions with spacing
    # Base positions for method groups
    n_methods = 2
    n_tasks = len(tasks_order)
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    ax.grid(True, axis="y", linestyle="--", alpha=0.4)
    ax.set_axisbelow(True)

    positions = []
    labels = []
    box_data = []  # list of list-of-arrays for plt.boxplot

    # Determine per-task offset so that boxes within a method don't touch
    inner_gap = 0.4  # space between tasks within the same method
    for (m_idx, t_idx, m_name, task, runs_or_combined) in data:
        base = m_idx * (n_tasks + method_gap)
        # center two tasks around base+ (n_tasks-1)/2
        # place tasks at base + t_idx*(1+inner_gap)
        pos = base + t_idx * (1.0 + inner_gap)
        positions.append(pos)
        labels.append(f"{task}")
        # runs_or_combined is a list-of-arrays (either 1 array combined or list for each run)
        if len(runs_or_combined) == 1:
            box_data.append(runs_or_combined[0])
        else:
            # If not combining runs, we will create multiple narrow boxes around 'pos'
            # We'll slightly jitter positions for the run-wise boxes
            jitter = np.linspace(-0.25, 0.25, num=len(runs_or_combined))
            for k, arr in enumerate(runs_or_combined):
                box_data.append(arr)
                positions.append(pos + jitter[k])
                labels.append(f"{task}\nRun {k+1}")

    # Create boxes
    bp = ax.boxplot(
        box_data,
        positions=positions,
        widths=box_width,
        showfliers=showfliers,
        patch_artist=False  # keep default style; no explicit colors for publication neutrality
    )

    # X ticks: group by method
    # Compute centers for each method to place group labels underneath
    method_centers = []
    method_ticklabels = [method1_name, method2_name]
    for m_idx in range(n_methods):
        group_positions = [p for (p, (mm, tt, *_)) in zip(positions, data* ( (len(positions)//len(data)) or 1)) if mm == m_idx]
        # Fallback: compute approximate center
        left = m_idx * (n_tasks + method_gap)
        right = left + (n_tasks - 1) * (1.0 + inner_gap)
        method_centers.append((left + right) / 2.0)

    ax.set_xticks(method_centers)
    ax.set_xticklabels(method_ticklabels)

    # Secondary tick labels for tasks under each group (optional)
    # We'll add small text annotations for task positions
    for (m_idx, t_idx, m_name, task, _) in data:
        base = m_idx * (n_tasks + method_gap)
        pos = base + t_idx * (1.0 + inner_gap)
        ax.text(pos, ax.get_ylim()[0], task, ha='center', va='bottom', rotation=0, fontsize=9)

    # Y label & title
    if ylabel is None:
        ylabel = metric
    ax.set_ylabel(ylabel)
    if title:
        ax.set_title(title)

    plt.tight_layout()
    plt.show()


# --------------------------- stats API ----------------------------------

def paired_ttests_two_tasks(
    method1_runs: Dict[str, List[str]],
    method2_runs: Dict[str, List[str]],
    metric: str,
    remove_first_n: int = 0,
    exclude_failures: bool = False,
    dataset_filter: Optional[str] = None,
) -> None:
    """
    Perform paired t-tests per task between two methods.
    The pairing is done by merging on an ID column if available.
    """
    for task in ("Ingredients", "Nutrition"):
        files1 = method1_runs.get(task, [])
        files2 = method2_runs.get(task, [])
        if not files1 or not files2:
            print(f"[t-test] Skip task '{task}' (missing files).")
            continue
        # Average over runs per image/document
        def avg_per_id(files):
            merged_all = None
            for i, f in enumerate(files):
                df = pd.read_csv(f, sep=';')
                df = _filter_dataset(df, dataset_filter)
                if remove_first_n > 0 and len(df) > remove_first_n:
                    df = df.iloc[remove_first_n:]
                _, metric_col = _resolve_metric_column(df, metric)
                # id column guess
                id_col = None
                for c in df.columns:
                    if c.lower() in ("product_id", "image_id", "doc_id", "id"):
                        id_col = c
                        break
                if id_col is None:
                    # create an index id if missing (will be less precise for pairing)
                    df = df.reset_index().rename(columns={"index": "id_idx"})
                    id_col = "id_idx"
                df = df[[id_col, metric_col]].copy()
                df.columns = ["_id", f"run{i+1}"]
                merged_all = df if merged_all is None else merged_all.merge(df, on="_id", how="outer")
            merged_all["avg"] = merged_all[[c for c in merged_all.columns if c.startswith("run")]].mean(axis=1)
            return merged_all[["_id", "avg"]]

        avg1 = avg_per_id(files1)
        avg2 = avg_per_id(files2)
        merged = avg1.merge(avg2, on="_id", suffixes=("_m1", "_m2")).dropna()
        if len(merged) < 2:
            print(f"[t-test] Not enough paired samples for task '{task}'.")
            continue
        t_stat, p_val = stats.ttest_rel(merged["avg_m1"], merged["avg_m2"])
        print(f"[t-test] Task: {task:11s} | n={len(merged)} | t={t_stat:.3f} | p={p_val:.3e}")
