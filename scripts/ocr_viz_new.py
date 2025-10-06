
"""
OCR Visualization Utilities
===========================

This module creates clean, publication-ready grouped boxplots for comparing
OCR methods (traditional vs LLM) across tasks (Ingredients vs Nutrition).
It also includes t-test helpers to assess statistical significance
and utilities for multi-metric plots, time-per-image analysis, and
simple confusion matrices based on metric thresholds.

Key features
------------
- Load multiple CSV "runs" per method.
- Dynamic metric selection (WER, CER, F1, GRITS, time, RAM, etc.).
- Cold-start control (drop first N rows per run).
- Failure filtering (e.g., drop rows with F1 == 0).
- Grouped boxplots with spacing and slim boxes (Matplotlib only).
- Combine runs into one box per task, or show each run as its own box.
- Optional dataset filtering if a column exists (e.g., 'dataset' or 'task').
- Paired t-tests between methods per task.
- Single-metric plots: (a) three runs of one method; (b) two-method comparison for one task.
- Multi-metric plots per task (e.g., WER, CER, F1, Precision, Recall).
- Time-per-image analysis if a table with number of images per product is provided.
- Confusion matrix from thresholding metrics (e.g., F1 ≥ 0.9 = "correct").

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
from ocr_viz import (
    boxplot_metric_over_runs,
    boxplot_two_methods_one_task,
    grouped_boxplot_two_methods_two_tasks,
    paired_ttests_two_tasks,
    multi_metric_by_task_single_method,
    multi_metric_by_task_two_methods,
    time_per_image_boxplot,
    confusion_matrix_from_threshold
)

# Define file lists elsewhere and call functions as needed.
"""

from __future__ import annotations
from typing import Dict, List, Optional, Tuple, Sequence
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from matplotlib.patches import Patch


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

def pastel_palette(n: int, name="Set2"):
    cmap = plt.get_cmap(name)
    return [cmap(i) for i in range(n)]

def _infer_method_type(df: pd.DataFrame) -> str:
    cols = [c.lower() for c in df.columns]
    if any(c.endswith("_llm") for c in cols):
        return "llm"
    if any(c.endswith("_ocr") or c.endswith("_trad") for c in cols):
        return "trad"
    return "unknown"


# Map generic metric names to typical column names per method type.
# Extend here if you have more metrics.

_DEFAULT_METRIC_LABELS = {
    "wer": "WER",
    "cer": "CER",
    "precision": "Precision",
    "recall": "Recall",
    "f1": "F1-Score",
    "precision_overall": "Precision",
    "recall_overall": "Recall",
    "f1_overall": "F1-Score",
    "overall_table_score": "GriTS Overall Table Score",
    "grits_content_overall_f1": "GriTS Content Score",
    "grits_content_overall_recall": "GriTS Content Recall",
    "grits_content_overall_precision": "GriTS Content Precision",
    "grits_topology_f1": "GriTS Topology Score",
    "grits_topology_recall": "GriTS Topology Recall",
    "grits_topology_precision": "GriTS Topology Precision",
    "cell_accuracy": "Cell Accuracy",
    "title_similarity": "Title Similarity",
    "footnote_similarity": "Footnote Similarity",
    "grits_header_precision": "GriTS Header Precision",
    "grits_header_recall": "GriTS Header Recall",
    "grits_header_f1": "GriTS Header F1-Score",
    "grits_labels_precision": "GriTS Labels Precision",
    "grits_labels_recall": "GriTS Labels Recall",
    "grits_labels_f1": "GriTS Labels F1-Score",
    "grits_values_precision": "GriTS Values Precision",
    "grits_values_recall": "GriTS Values Recall",
    "grits_values_f1": "GriTS Values F1-Score",
    "precision_triples": "Triple Precision",
    "recall_triples": "Triple Recall",
    "f1_triples": "Triple F1-Score",
    "time_preproc": "Preprocessing Time",
    "time_api": "API Time",
    "time_postproc": "Postprocessing Time",
    "time_total": "Total Time",
}


_METRIC_MAP = {
        "iou":          {"llm": "iou",          "trad": "iou"},        # Intersection over Union
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


def _pretty_metric_name(name: str, metric_display_map: Optional[dict] = None) -> str:
    """Gibt einen hübschen Anzeigenamen zurück (Mapping > Defaults > Original)."""
    if metric_display_map and name in metric_display_map:
        return metric_display_map[name]
    return _DEFAULT_METRIC_LABELS.get(name, name)


def _resolve_metric_column(df: pd.DataFrame, metric: str) -> Tuple[str, str]:
    """Return (method_type, column_name) for given metric in df."""
    mtype = _infer_method_type(df)
    if metric in df.columns:
        return mtype, metric
    # Try mapping
    if metric in _METRIC_MAP:
        col = _METRIC_MAP[metric].get(mtype)
        if col is None:
            raise KeyError(f"Metric '{metric}' unavailable for method type '{mtype}'.")
        lower_map = {c.lower(): c for c in df.columns}
        if col.lower() not in lower_map:
            raise KeyError(f"Expected column '{col}' not found. Available: {list(df.columns)[:8]}...")
        return mtype, lower_map[col.lower()]
    # Case-insensitive fallback
    lower_map = {c.lower(): c for c in df.columns}
    if metric.lower() in lower_map:
        return mtype, lower_map[metric.lower()]
    raise KeyError(f"Cannot resolve metric '{metric}'.")


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


def _guess_id_col(df: pd.DataFrame) -> Optional[str]:
    for c in df.columns:
        if c.lower() in ("product_id", "image_id", "doc_id", "id"):
            return c
    return None


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

        # ermitteln, welche Spalte zur Metrik gehört
        mtype_i, metric_col = _resolve_metric_column(df, metric)
        if mtype_global is None:
            mtype_global = mtype_i

        if exclude_failures:
            # 1) Zeilen mit leerem Text verwerfen (methodenspezifisch)
            #    - LLM:   'llm_text'
            #    - Trad.: 'ocr_text'
            #    Falls beide vorkommen (selten), genügt es, wenn einer nicht leer ist.
            llm_col = _get_col_case_insensitive(df, "llm_text")
            ocr_col = _get_col_case_insensitive(df, "ocr_text")

            if llm_col or ocr_col:
                masks = []
                if llm_col:
                    masks.append(df[llm_col].notna() & (df[llm_col].astype(str).str.strip() != ""))
                if ocr_col:
                    masks.append(df[ocr_col].notna() & (df[ocr_col].astype(str).str.strip() != ""))
                # Wenn beide Spalten existieren: behalte Zeile, wenn mind. EINE nicht leer ist.
                # Wenn nur eine existiert: verwende deren Maske.
                if masks:
                    keep_mask = masks[0]
                    for m in masks[1:]:
                        keep_mask = keep_mask | m
                    df = df[keep_mask]

            # 2) Optional zusätzlich F1==0 raus (wie bisher)
            # try:
            #     _, f1_col = _resolve_metric_column(df, "f1")
            #     df = df[df[f1_col] > 0]
            # except KeyError:
            #     pass

        # Werte extrahieren
        values = df[metric_col].dropna().to_numpy()
        runs.append(values)

        # Für pairing/t-tests zusammenführen (falls ID existiert)
        id_col = _guess_id_col(df)
        if id_col is not None:
            tmp = df[[id_col, metric_col]].copy()
            tmp.columns = ["_id", f"run{i+1}"]
            merged = tmp if merged is None else merged.merge(tmp, on="_id", how="outer")

    return mtype_global or "unknown", runs, merged

def _get_col_case_insensitive(df: pd.DataFrame, name: str) -> Optional[str]:
    for c in df.columns:
        if c.lower() == name.lower():
            return c
    return None

# -------------------------- plotting API --------------------------------

def boxplot_metric_over_runs(
    files: List[str],
    metric: str,
    method_name: str,
    remove_first_n: int = 0,
    exclude_failures: bool = False,
    dataset_filter: Optional[str] = None,
    figsize: Tuple[float, float] = (7, 5),
    box_width: float = 0.25,
    showfliers: bool = True,
    ylabel: Optional[str] = None,
    title: Optional[str] = None,
):
    """Show 3 slim boxplots (one per run) for a single method & task."""
    _, runs, _ = _load_runs(files, metric, remove_first_n, exclude_failures, dataset_filter)
    data = runs
    positions = np.arange(len(data))
    fig, ax = plt.subplots(figsize=figsize)
    ax.boxplot(data, positions=positions, widths=box_width, showfliers=showfliers)
    _draw_mean_lines(ax, positions, data, box_width)
    ax.set_xticks(positions)
    ax.set_xticklabels([f"Run {i+1}" for i in range(len(data))])
    ax.set_xlabel(method_name)
    if ylabel is None: ylabel = metric
    ax.set_ylabel(ylabel)
    if title: ax.set_title(title)
    plt.tight_layout()
    plt.show()


def boxplot_two_methods_one_task(
    method1_files: List[str],
    method2_files: List[str],
    metric: str,
    method1_name: str = "Traditional",
    method2_name: str = "LLM",
    remove_first_n: int = 0,
    exclude_failures: bool = False,
    dataset_filter: Optional[str] = None,
    figsize: Tuple[float, float] = (8, 5),
    method_gap: float = 1.6,
    box_width: float = 0.25,
    showfliers: bool = True,
    ylabel: Optional[str] = None,
    title: Optional[str] = None,
    combine_runs: bool = True,
):
    """Two groups on x-axis (method1 vs method2). Inside each: one box (combined runs) or 3 boxes (per run)."""
    data = []
    labels = []
    for m_idx, (name, files) in enumerate([(method1_name, method1_files), (method2_name, method2_files)]):
        _, runs, _ = _load_runs(files, metric, remove_first_n, exclude_failures, dataset_filter)
        if combine_runs:
            data.append(np.concatenate(runs))
            labels.append(name)
        else:
            for i, arr in enumerate(runs):
                data.append(arr)
                labels.append(f"{name}\nRun {i+1}")
    # positions with spacing
    fig, ax = plt.subplots(figsize=figsize)
    positions = []
    x = 0.0
    for i, arr in enumerate(data):
        positions.append(x)
        x += (method_gap if ("\nRun" not in labels[i]) else 0.35)
    ax.boxplot(data, positions=positions, widths=box_width, showfliers=showfliers)
    ax.set_xticks(positions)
    ax.set_xticklabels(labels)
    if ylabel is None: ylabel = metric
    ax.set_ylabel(ylabel)
    if title: ax.set_title(title)
    plt.tight_layout()
    plt.show()


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
    Two groups (method1, method2) on x-axis; within each, two boxes (Ingredients, Nutrition).
    If combine_runs=False, each task shows 3 slim boxes (one per run).
    """
    data = []
    for m_idx, (m_name, runs_dict) in enumerate([(method1_name, method1_runs), (method2_name, method2_runs)]):
        for t_idx, task in enumerate(tasks_order):
            files = runs_dict.get(task, [])
            if not files:
                raise ValueError(f"No files provided for {m_name} / {task}")
            _, runs, _ = _load_runs(files, metric, remove_first_n, exclude_failures, dataset_filter)
            data.append((m_idx, t_idx, m_name, task, runs))

    # Prepare plot
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    inner_gap = 0.4
    positions = []
    box_data = []
    for (m_idx, t_idx, m_name, task, runs) in data:
        base = m_idx * (2 + method_gap)
        pos = base + t_idx * (1.0 + inner_gap)
        if combine_runs:
            positions.append(pos)
            box_data.append(np.concatenate(runs))
        else:
            jitter = np.linspace(-0.25, 0.25, num=len(runs))
            for k, arr in enumerate(runs):
                positions.append(pos + jitter[k])
                box_data.append(arr)
    ax.boxplot(box_data, positions=positions, widths=box_width, showfliers=showfliers)
    # method tick centers
    centers = []
    for m_idx in range(2):
        left = m_idx * (2 + method_gap)
        right = left + (2 - 1) * (1.0 + inner_gap)
        centers.append((left + right) / 2.0)
    ax.set_xticks(centers)
    ax.set_xticklabels([method1_name, method2_name])
    # annotate task labels
    ymin = ax.get_ylim()[0]
    for (m_idx, t_idx, m_name, task, runs) in data:
        base = m_idx * (2 + method_gap)
        pos = base + t_idx * (1.0 + inner_gap)
        ax.text(pos, ymin, task, ha='center', va='bottom', fontsize=9)
    if ylabel is None: ylabel = metric
    ax.set_ylabel(ylabel)
    if title: ax.set_title(title)
    plt.tight_layout()
    plt.show()


# --------------------------- multi-metric API ----------------------------

def _collect_metric_arrays(files: List[str], metrics: Sequence[str],
                           remove_first_n: int, exclude_failures: bool, dataset_filter: Optional[str]):
    arrays_per_metric = []
    for metric in metrics:
        _, runs, _ = _load_runs(files, metric, remove_first_n, exclude_failures, dataset_filter)
        arrays_per_metric.append(np.concatenate(runs))
    return arrays_per_metric


def multi_metric_by_task_single_method(
    method_runs: Dict[str, List[str]],
    metrics: Sequence[str],
    method_name: str,
    tasks_order: Tuple[str, str] = ("Ingredients", "Nutrition"),
    remove_first_n: int = 0,
    exclude_failures: bool = False,
    dataset_filter: Optional[str] = None,
    figsize: Tuple[float, float] = (9, 5),
    task_gap: float = 1.6,
    box_width: float = 0.22,
    showfliers: bool = True,
    ylabel: str = "Wert (0–1)",
    title: Optional[str] = None,
    xtitle: Optional[str] = None,  # <— NEU: Übergeordnete X-Achsen-Beschriftung
    show_task_labels: bool = False
):
    """
    X-axis: tasks (Ingredients, Nutrition). Within each: one slim box per metric (e.g., CER, WER, F1, Precision, Recall).
    Useful when you want multiple metrics in one figure for a single method.
    """
    # Collect data
    data = []
    for t_idx, task in enumerate(tasks_order):
        files = method_runs.get(task, [])
        if not files:
            raise ValueError(f"No files provided for {method_name} / {task}")
        arrays = _collect_metric_arrays(files, metrics, remove_first_n, exclude_failures, dataset_filter)
        data.append((t_idx, task, arrays))

    # Prepare plot
    fig, ax = plt.subplots(figsize=figsize)
    positions = []
    box_data = []
    for (t_idx, task, arrays) in data:
        base = t_idx * (len(arrays) + task_gap)
        # place each metric box with slight unit spacing
        for m_i, arr in enumerate(arrays):
            positions.append(base + m_i * 1.0)
            box_data.append(arr)
    ax.boxplot(box_data, positions=positions, widths=box_width, showfliers=showfliers)

    # Tick-Positionen (Zentren je Task)
    centers = []
    for t_idx in range(len(tasks_order)):
        left = t_idx * (len(metrics) + task_gap)
        right = left + (len(metrics) - 1) * 1.0
        centers.append((left + right) / 2.0)

    ax.set_xticks(centers)

    if show_task_labels:
        ax.set_xticklabels(list(tasks_order))
    else:
        # keine Task-Namen auf der x-Achse
        ax.set_xticklabels([])
        ax.tick_params(axis="x", length=0)  # kurze Ticks ausblenden

    # metric labels unter den Boxen (so wie gehabt)
    ymin = ax.get_ylim()[0]
    for t_idx, task, arrays in data:
        base = t_idx * (len(arrays) + task_gap)
        for m_i in range(len(arrays)):
            ax.text(base + m_i * 1.0, ymin, _pretty_metric_name(metrics[m_i]),
                    ha='center', va='bottom', fontsize=8)

    ax.set_ylabel(ylabel)

    # nur dein xtitle anzeigen
    if xtitle is not None:
        ax.set_xlabel(xtitle, labelpad=8)

    if title is None:
        title = f"{method_name} — {', '.join(metrics)}"
    ax.set_title(title)
    plt.tight_layout()
    plt.show()

def multi_metric_by_task_two_methods(
    method1_runs: Dict[str, List[str]],
    method2_runs: Dict[str, List[str]],
    metrics: Sequence[str],
    method1_name: str = "Traditional",
    method2_name: str = "LLM",
    tasks_order: Tuple[str, str] = ("Ingredients", "Nutrition"),
    remove_first_n: int = 0,
    exclude_failures: bool = False,
    dataset_filter: Optional[str] = None,
    figsize: Tuple[float, float] = (10, 5),
    group_gap: float = 2.0,
    box_width: float = 0.2,
    showfliers: bool = True,
    ylabel: str = "Wert (0–1)",
    title: Optional[str] = None
):
    """
    X-axis shows two groups per task: (method1, method2). Inside each group, several metric boxes.
    E.g., for Nutrition: method1 CER/WER/F1/Precision/Recall, then method2 CER/WER/F1/Precision/Recall.
    """
    # Data collection
    packed = []  # list of (task_idx, task, [(method_name, arrays_per_metric)])
    for t_idx, task in enumerate(tasks_order):
        m1_files = method1_runs.get(task, [])
        m2_files = method2_runs.get(task, [])
        if not m1_files or not m2_files:
            raise ValueError(f"Missing files for task '{task}'.")
        arrs1 = _collect_metric_arrays(m1_files, metrics, remove_first_n, exclude_failures, dataset_filter)
        arrs2 = _collect_metric_arrays(m2_files, metrics, remove_first_n, exclude_failures, dataset_filter)
        packed.append((t_idx, task, [(method1_name, arrs1), (method2_name, arrs2)]))

    # Plot
    fig, ax = plt.subplots(figsize=figsize)
    positions = []
    box_data = []
    # spacing: for each task, allocate len(metrics) boxes for method1, a small inner gap, then len(metrics) for method2
    inner_gap = 0.6
    for t_idx, task, method_arrs in packed:
        base = t_idx * (2*len(metrics) + inner_gap + group_gap)
        # method1 metrics
        for m_i, arr in enumerate(method_arrs[0][1]):
            positions.append(base + m_i * 1.0)
            box_data.append(arr)
        # method2 metrics
        shift = len(metrics) + inner_gap
        for m_i, arr in enumerate(method_arrs[1][1]):
            positions.append(base + shift + m_i * 1.0)
            box_data.append(arr)
    ax.boxplot(box_data, positions=positions, widths=box_width, showfliers=showfliers)

    # x ticks at task centers
    centers = []
    for t_idx in range(len(tasks_order)):
        left = t_idx * (2*len(metrics) + inner_gap + group_gap)
        right = left + (2*len(metrics) + inner_gap - 1)
        centers.append((left + right) / 2.0)
    ax.set_xticks(centers)
    ax.set_xticklabels(list(tasks_order))

    # annotate metric names under each subgroup
    ymin = ax.get_ylim()[0]
    idx = 0
    for t_idx, task, method_arrs in packed:
        base = t_idx * (2*len(metrics) + inner_gap + group_gap)
        # labels for method1
        for m_i, _ in enumerate(method_arrs[0][1]):
            ax.text(base + m_i * 1.0, ymin, metrics[m_i], ha='center', va='bottom', fontsize=8)
        # labels for method2
        shift = len(metrics) + inner_gap
        for m_i, _ in enumerate(method_arrs[1][1]):
            ax.text(base + shift + m_i * 1.0, ymin, metrics[m_i], ha='center', va='bottom', fontsize=8)

    ax.set_ylabel(ylabel)
    if title is None:
        title = f"{method1_name} vs {method2_name} — {', '.join(metrics)}"
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
                id_col = _guess_id_col(df) or df.reset_index().rename(columns={"index":"id_idx"}).columns[-1]
                if id_col not in df.columns:
                    df = df.reset_index().rename(columns={"index":"id_idx"})
                    id_col = "id_idx"
                tmp = df[[id_col, metric_col]].copy()
                tmp.columns = ["_id", f"run{i+1}"]
                merged_all = tmp if merged_all is None else merged_all.merge(tmp, on="_id", how="outer")
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


# --------------------- time-per-image / scatter --------------------------

def time_per_image_boxplot(
    method_runs: Dict[str, List[str]],
    images_per_product: pd.DataFrame,
    metric_time: str = "time_total",
    id_col_images: str = "product_id",
    n_images_col: str = "n_images",
    tasks_order: Tuple[str, str] = ("Ingredients", "Nutrition"),
    method_name: str = "Method",
    remove_first_n: int = 0,
    exclude_failures: bool = False,
    dataset_filter: Optional[str] = None,
    figsize: Tuple[float, float] = (9,5),
    box_width: float = 0.25,
    showfliers: bool = True,
    ylabel: str = "Zeit pro Bild (s)",
    title: Optional[str] = None
):
    """
    Requires a DataFrame images_per_product with columns [product_id, n_images].
    Computes (time_total / n_images) per product and plots boxes per task.
    """
    if id_col_images not in images_per_product.columns or n_images_col not in images_per_product.columns:
        raise ValueError(f"images_per_product must contain columns '{id_col_images}' and '{n_images_col}'.")

    fig, ax = plt.subplots(figsize=figsize)
    positions = []
    box_data = []
    task_gap = 1.6
    for t_idx, task in enumerate(tasks_order):
        files = method_runs.get(task, [])
        if not files:
            continue
        _, runs, merged = _load_runs(files, metric_time, remove_first_n, exclude_failures, dataset_filter)
        # average time per product across runs
        if merged is None or "_id" not in merged.columns:
            continue
        merged["avg_time"] = merged[[c for c in merged.columns if c.startswith("run")]].mean(axis=1)
        df = merged.merge(images_per_product[[id_col_images, n_images_col]], left_on="_id", right_on=id_col_images, how="left")
        df = df.dropna(subset=[n_images_col, "avg_time"])
        df["time_per_image"] = df["avg_time"] / df[n_images_col].replace(0, np.nan)
        positions.append(t_idx * (1.0 + task_gap))
        box_data.append(df["time_per_image"].dropna().to_numpy())

    ax.boxplot(box_data, positions=positions, widths=box_width, showfliers=showfliers)
    ax.set_xticks(positions)
    ax.set_xticklabels(list(tasks_order))
    ax.set_ylabel(ylabel)
    if title is None:
        title = f"{method_name} — Zeit pro Bild"
    ax.set_title(title)
    plt.tight_layout()
    plt.show()


# ------------------------- confusion matrix ------------------------------

def confusion_matrix_from_threshold(
    method1_runs: Dict[str, List[str]],
    method2_runs: Dict[str, List[str]],
    metric: str = "f1",
    threshold: float = 0.9,
    task: str = "Ingredients",
    remove_first_n: int = 0,
    exclude_failures: bool = False,
    dataset_filter: Optional[str] = None,
    method1_name: str = "Traditional",
    method2_name: str = "LLM",
    figsize: Tuple[float, float] = (5,4),
    title: Optional[str] = None
):
    """
    Build a simple 2x2 confusion table comparing methods on a given task:
    Pass if metric >= threshold, otherwise Fail.
    Uses average over the three runs per product.
    """
    def avg_per_id(files):
        merged_all = None
        for i, f in enumerate(files):
            df = pd.read_csv(f, sep=';')
            df = _filter_dataset(df, dataset_filter)
            if remove_first_n > 0 and len(df) > remove_first_n:
                df = df.iloc[remove_first_n:]
            _, metric_col = _resolve_metric_column(df, metric)
            id_col = _guess_id_col(df) or df.reset_index().rename(columns={"index":"id_idx"}).columns[-1]
            if id_col not in df.columns:
                df = df.reset_index().rename(columns={"index":"id_idx"})
                id_col = "id_idx"
            tmp = df[[id_col, metric_col]].copy()
            tmp.columns = ["_id", f"run{i+1}"]
            merged_all = tmp if merged_all is None else merged_all.merge(tmp, on="_id", how="outer")
        merged_all["avg"] = merged_all[[c for c in merged_all.columns if c.startswith("run")]].mean(axis=1)
        return merged_all[["_id", "avg"]]

    m1 = avg_per_id(method1_runs.get(task, []))
    m2 = avg_per_id(method2_runs.get(task, []))
    merged = m1.merge(m2, on="_id", suffixes=("_m1", "_m2")).dropna()
    if merged.empty:
        print("No paired data for confusion matrix.")
        return

    y1 = (merged["avg_m1"] >= threshold).astype(int)
    y2 = (merged["avg_m2"] >= threshold).astype(int)

    # confusion counts:
    # rows: method1 (0/1), cols: method2 (0/1)
    cm = np.zeros((2,2), dtype=int)
    for a, b in zip(y1, y2):
        cm[a, b] += 1

    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(cm, cmap="Blues")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center", color="black")
    ax.set_xticks([0,1]); ax.set_yticks([0,1])
    ax.set_xticklabels([f"{method2_name} < {threshold}", f"{method2_name} ≥ {threshold}"])
    ax.set_yticklabels([f"{method1_name} < {threshold}", f"{method1_name} ≥ {threshold}"])
    if title is None:
        title = f"Confusion Matrix ({task}) — threshold {metric} ≥ {threshold}"
    ax.set_title(title)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    plt.tight_layout()
    plt.show()

# ======================= descriptive statistics API =======================

def _box_stats_from_array(arr: np.ndarray) -> dict:
    """Compute boxplot statistics for a 1D array (matching matplotlib's whisker logic)."""
    arr = np.asarray(arr, dtype=float)
    arr = arr[~np.isnan(arr)]
    if arr.size == 0:
        return dict(n=0, mean=np.nan, std=np.nan, median=np.nan,
                    q1=np.nan, q3=np.nan, whisker_low=np.nan, whisker_high=np.nan,
                    data_min=np.nan, data_max=np.nan)

    n = arr.size
    mean = float(np.mean(arr))
    std = float(np.std(arr, ddof=1)) if n > 1 else 0.0
    q1 = float(np.percentile(arr, 25))
    median = float(np.percentile(arr, 50))
    q3 = float(np.percentile(arr, 75))
    iqr = q3 - q1
    low_bound = q1 - 1.5 * iqr
    high_bound = q3 + 1.5 * iqr
    data_min = float(np.min(arr))
    data_max = float(np.max(arr))
    # whiskers extend to the most extreme data points within the IQR fences
    whisker_low = float(np.min(arr[arr >= low_bound])) if np.any(arr >= low_bound) else data_min
    whisker_high = float(np.max(arr[arr <= high_bound])) if np.any(arr <= high_bound) else data_max

    return dict(n=n, mean=mean, std=std, median=median,
                q1=q1, q3=q3, whisker_low=whisker_low, whisker_high=whisker_high,
                data_min=data_min, data_max=data_max)


def stats_metric_over_runs(
    files: List[str],
    metric: str,
    remove_first_n: int = 0,
    exclude_failures: bool = False,
    dataset_filter: Optional[str] = None,
    include_combined: bool = True
) -> pd.DataFrame:
    """
    Return descriptive stats per run (and optional combined) for a single method & task.
    Columns: run_label, n, mean, std, median, q1, q3, whisker_low, whisker_high, data_min, data_max
    """
    _, runs, _ = _load_runs(files, metric, remove_first_n, exclude_failures, dataset_filter)
    rows = []
    for i, arr in enumerate(runs):
        stats_d = _box_stats_from_array(arr)
        stats_d["run_label"] = f"Run {i+1}"
        rows.append(stats_d)
    if include_combined and runs:
        combined = np.concatenate(runs)
        stats_d = _box_stats_from_array(combined)
        stats_d["run_label"] = "All runs"
        rows.append(stats_d)
    df = pd.DataFrame(rows)[["run_label","n","mean","std","median","q1","q3","whisker_low","whisker_high","data_min","data_max"]]
    return df


def stats_two_methods_one_task(
    method1_files: List[str],
    method2_files: List[str],
    metric: str,
    method1_name: str = "Traditional",
    method2_name: str = "LLM",
    remove_first_n: int = 0,
    exclude_failures: bool = False,
    dataset_filter: Optional[str] = None,
    combine_runs: bool = True
) -> pd.DataFrame:
    """
    Return stats for a two-method comparison on a single task.
    If combine_runs=True: one row per method ("All runs").
    Else: one row per method-run.
    """
    rows = []
    for mname, files in [(method1_name, method1_files), (method2_name, method2_files)]:
        _, runs, _ = _load_runs(files, metric, remove_first_n, exclude_failures, dataset_filter)
        if combine_runs:
            arr = np.concatenate(runs) if runs else np.array([])
            d = _box_stats_from_array(arr); d["method"] = mname; d["run_label"] = "All runs"
            rows.append(d)
        else:
            for i, arr in enumerate(runs):
                d = _box_stats_from_array(arr); d["method"] = mname; d["run_label"] = f"Run {i+1}"
                rows.append(d)
    cols = ["method","run_label","n","mean","std","median","q1","q3","whisker_low","whisker_high","data_min","data_max"]
    return pd.DataFrame(rows)[cols]


def stats_two_methods_two_tasks(
    method1_runs: Dict[str, List[str]],
    method2_runs: Dict[str, List[str]],
    metric: str,
    method1_name: str = "Traditional",
    method2_name: str = "LLM",
    tasks_order: Tuple[str, str] = ("Ingredients", "Nutrition"),
    remove_first_n: int = 0,
    exclude_failures: bool = False,
    dataset_filter: Optional[str] = None,
    combine_runs: bool = True
) -> pd.DataFrame:
    """
    Return stats mirroring grouped_boxplot_two_methods_two_tasks.
    Rows correspond 1:1 to boxes in that plot (per task & method; per-run if combine_runs=False).
    """
    rows = []
    for mname, runs_dict in [(method1_name, method1_runs), (method2_name, method2_runs)]:
        for task in tasks_order:
            files = runs_dict.get(task, [])
            if not files:
                raise ValueError(f"No files for {mname} / {task}")
            _, runs, _ = _load_runs(files, metric, remove_first_n, exclude_failures, dataset_filter)
            if combine_runs:
                arr = np.concatenate(runs) if runs else np.array([])
                d = _box_stats_from_array(arr); d.update({"method": mname, "task": task, "run_label": "All runs"})
                rows.append(d)
            else:
                for i, arr in enumerate(runs):
                    d = _box_stats_from_array(arr); d.update({"method": mname, "task": task, "run_label": f"Run {i+1}"})
                    rows.append(d)
    cols = ["method","task","run_label","n","mean","std","median","q1","q3","whisker_low","whisker_high","data_min","data_max"]
    return pd.DataFrame(rows)[cols]


def stats_multi_metric_by_task_two_methods(
    method1_runs: Dict[str, List[str]],
    method2_runs: Dict[str, List[str]],
    metrics: Sequence[str],
    method1_name: str = "Traditional",
    method2_name: str = "LLM",
    tasks_order: Tuple[str, str] = ("Ingredients", "Nutrition"),
    remove_first_n: int = 0,
    exclude_failures: bool = False,
    dataset_filter: Optional[str] = None
) -> pd.DataFrame:
    """
    Compute stats for multiple metrics for both methods and tasks.
    Rows: method | task | metric | run_label=All runs | stats
    (Runs are combined to match the default multi-metric plots.)
    """
    rows = []
    for task in tasks_order:
        for (mname, runs_dict) in [(method1_name, method1_runs), (method2_name, method2_runs)]:
            files = runs_dict.get(task, [])
            if not files:
                raise ValueError(f"Missing files for {mname}/{task}")
            for metric in metrics:
                _, runs, _ = _load_runs(files, metric, remove_first_n, exclude_failures, dataset_filter)
                arr = np.concatenate(runs) if runs else np.array([])
                d = _box_stats_from_array(arr)
                d.update({"method": mname, "task": task, "metric": metric, "run_label": "All runs"})
                rows.append(d)
    cols = ["method","task","metric","run_label","n","mean","std","median","q1","q3","whisker_low","whisker_high","data_min","data_max"]
    return pd.DataFrame(rows)[cols]

def _draw_mean_lines(ax, positions, arrays, box_width, **kwargs):
    """
    Zeichnet pro Box eine horizontale gestrichelte Linie auf Höhe des Mittelwerts.
    - positions: x-Position jeder Box (Liste/Folge von floats)
    - arrays:    Datenarrays pro Box (gleiche Reihenfolge wie positions)
    - box_width: selbe Breite wie im boxplot-Aufruf
    - kwargs:    z.B. {'linestyle':'--','linewidth':1,'alpha':0.9}
    """
    if kwargs is None:
        kwargs = {}
    if "linestyle" not in kwargs:
        kwargs["linestyle"] = "--"
    if "linewidth" not in kwargs:
        kwargs["linewidth"] = 1.0
    if "alpha" not in kwargs:
        kwargs["alpha"] = 0.9

    half = box_width * 0.5
    for x, arr in zip(positions, arrays):
        arr = np.asarray(arr)
        if arr.size == 0 or np.all(np.isnan(arr)):
            continue
        m = float(np.nanmean(arr))
        ax.hlines(m, x - half, x + half, **kwargs)

def boxplot_two_methods_metrics_grouped_by_method_one_task(
    method1_runs: Dict[str, List[str]],
    method2_runs: Dict[str, List[str]],
    metrics: Sequence[str],                 # z. B. ["wer","cer"] oder ["precision_overall","recall_overall","f1_overall"]
    task: str = "Ingredients",
    method1_name: str = "Traditional",
    method2_name: str = "LLM",
    remove_first_n: int = 0,
    exclude_failures: bool = False,
    dataset_filter: Optional[str] = None,
    combine_runs: bool = True,
    figsize: Tuple[float, float] = (10, 5),
    method_gap: float = 2.2,
    inner_sep: float = 1.0,
    box_width: float = 0.25,
    showfliers: bool = True,
    colors: Optional[Sequence[str]] = None,
    ylabel: Optional[str] = None,
    title: Optional[str] = None,
    ylimit: Optional[Tuple[float, float]] = None,
    metric_display_map: Optional[Dict[str, str]] = None,  # <— NEU
):
    files1 = method1_runs.get(task, [])
    files2 = method2_runs.get(task, [])
    if not files1 or not files2:
        raise ValueError(f"Fehlende Dateien für Task '{task}'.")

    if colors is None or len(colors) < len(metrics):
        base = ["tab:blue","tab:orange","tab:green","tab:red","tab:purple",
                "tab:brown","tab:pink","tab:gray","tab:olive","tab:cyan"]
        colors = base[:len(metrics)]

    # Daten sammeln
    data_m1, data_m2 = [], []
    for metric in metrics:
        _, runs1, _ = _load_runs(files1, metric, remove_first_n, exclude_failures, dataset_filter)
        _, runs2, _ = _load_runs(files2, metric, remove_first_n, exclude_failures, dataset_filter)
        data_m1.append(runs1)
        data_m2.append(runs2)

    M = len(metrics)
    group_width = (M - 1) * inner_sep
    base1 = 0.0
    base2 = base1 + group_width + method_gap

    positions, box_data, color_per_box = [], [], []

    # Methode 1
    for m_idx in range(M):
        x = base1 + m_idx * inner_sep
        runs = data_m1[m_idx]
        if combine_runs:
            arr = np.concatenate(runs) if runs else np.array([])
            positions.append(x); box_data.append(arr); color_per_box.append(colors[m_idx])
        else:
            jitter = np.linspace(-0.25, 0.25, num=len(runs))
            for j, arr in enumerate(runs):
                positions.append(x + jitter[j]); box_data.append(arr); color_per_box.append(colors[m_idx])

    # Methode 2
    for m_idx in range(M):
        x = base2 + m_idx * inner_sep
        runs = data_m2[m_idx]
        if combine_runs:
            arr = np.concatenate(runs) if runs else np.array([])
            positions.append(x); box_data.append(arr); color_per_box.append(colors[m_idx])
        else:
            jitter = np.linspace(-0.25, 0.25, num=len(runs))
            for j, arr in enumerate(runs):
                positions.append(x + jitter[j]); box_data.append(arr); color_per_box.append(colors[m_idx])

    # Plot
    fig, ax = plt.subplots(figsize=figsize)
    bp = ax.boxplot(box_data, positions=positions, widths=box_width, showfliers=showfliers, patch_artist=True)
    for box, col in zip(bp["boxes"], color_per_box):
        box.set_facecolor(col); box.set_edgecolor(col); box.set_alpha(0.6)

    # Mean-Linien (optional)
    try:
        _draw_mean_lines(ax, positions, box_data, box_width, linestyle="--", linewidth=1.0, alpha=0.9)
    except NameError:
        pass

    # X-Ticks = Methoden
    center1 = base1 + group_width / 2.0
    center2 = base2 + group_width / 2.0
    ax.set_xticks([center1, center2])
    ax.set_xticklabels([method1_name, method2_name])

    # Y
    if ylabel is None: ylabel = "Wert"
    ax.set_ylabel(ylabel)
    if ylimit is not None: ax.set_ylim(ylimit)

    # Titel
    pretty_metrics = ", ".join(_pretty_metric_name(m, metric_display_map) for m in metrics)
    if title is None:
        title = f"{task} — {method1_name} vs {method2_name} ({pretty_metrics})"
    ax.set_title(title)

    # Legende: Farben = Metriken (schöne Namen)
    handles = [Patch(facecolor=c, edgecolor=c, alpha=0.6, label=_pretty_metric_name(m, metric_display_map))
               for m, c in zip(metrics, colors)]
    ax.legend(handles=handles, title="Metrik", loc="best")

    plt.tight_layout()
    plt.show()

from typing import Sequence, Dict, Tuple, Optional, List
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def metric_distribution_one_method(
    files: List[str],
    metric: str,
    method_name: str = "Method",
    task_label: Optional[str] = None,
    remove_first_n: int = 0,
    exclude_failures: bool = False,
    dataset_filter: Optional[str] = None,
    # Histogram-Parametrisierung:
    nbins: int = 20,
    bin_edges: Optional[Sequence[float]] = None,  # alternativ zu nbins
    normalize: bool = False,     # True => Prozent
    cumulative: bool = False,    # kumulative Verteilung anzeigen/berechnen
    # Punkt- & Schwellenzählung:
    value_points: Optional[Sequence[float]] = None,  # z.B. [0.0]
    eps: float = 1e-12,           # Toleranz für exakte Punktwerte
    thresholds: Optional[Sequence[float]] = None,  # z.B. [0.1, 0.2]
    # Darstellung:
    combine_runs: bool = True,    # True => ein Histogramm (alle 3 Runs kombiniert), False => gruppierte Balken pro Run
    figsize: Tuple[float, float] = (8, 5),
    color: str = "tab:blue",
    alpha: float = 0.75,
    edgecolor: Optional[str] = None,
    title: Optional[str] = None,
    xlabel: Optional[str] = None,
    ylabel: Optional[str] = None,
    ylimit: Optional[Tuple[float, float]] = None,
    metric_display_map: Optional[Dict[str, str]] = None,  # hübsche Namen
):
    """
    Erstellt ein Histogramm (Matplotlib) und liefert Tabellen mit Bins/Counts zurück.
    Rückgabe: dict mit 'hist' (DataFrame), optional 'points' und 'thresholds' (DataFrames).
    """
    # --- Helper für pretty metric name ---
    def _pretty(name: str) -> str:
        try:
            return _pretty_metric_name(name, metric_display_map)  # aus deinem Modul
        except NameError:
            return metric_display_map.get(name, name) if metric_display_map else name

    # --- Daten laden (nutzt deinen Loader) ---
    _, runs, _ = _load_runs(files, metric, remove_first_n, exclude_failures, dataset_filter)
    if not runs:
        raise ValueError("Keine Daten gefunden.")

    # kombiniertes Array (für Bins/Counts)
    combined = np.concatenate(runs) if runs else np.array([])
    combined = combined[~np.isnan(combined)]

    # Falls nichts da ist, sauber beenden
    if combined.size == 0:
        fig, ax = plt.subplots(figsize=figsize)
        ax.set_title(title or f"{method_name} — {task_label or ''} — {_pretty(metric)}")
        ax.text(0.5, 0.5, "Keine Daten", ha="center", va="center", transform=ax.transAxes)
        ax.axis("off")
        plt.tight_layout(); plt.show()
        return {"hist": pd.DataFrame()}

    # --- Bin-Kanten bestimmen ---
    if bin_edges is None:
        # Freedman–Diaconis als Default, falls möglich
        q1, q3 = np.percentile(combined, [25, 75])
        iqr = q3 - q1
        if iqr > 0:
            h = 2 * iqr * (combined.size ** (-1/3))
            if h > 0:
                nbins_fd = int(np.ceil((combined.max() - combined.min()) / h)) or nbins
                nbins = np.clip(nbins_fd, 5, 100)
        bin_edges = np.linspace(combined.min(), combined.max(), nbins + 1)
    else:
        bin_edges = np.asarray(bin_edges, dtype=float)
        nbins = len(bin_edges) - 1

    # --- Histogrammzählen ---
    def _hist(arr):
        arr = arr[~np.isnan(arr)]
        counts, edges = np.histogram(arr, bins=bin_edges)
        if cumulative:
            counts = np.cumsum(counts)
        if normalize:
            total = arr.size if arr.size > 0 else 1
            counts = counts / total * 100.0  # Prozent
        return counts

    # DataFrame bauen: pro Run oder kombiniert
    records = []
    if combine_runs:
        counts = _hist(combined)
        for i in range(nbins):
            records.append({
                "run_label": "All runs",
                "bin_left": bin_edges[i],
                "bin_right": bin_edges[i+1],
                "bin_center": 0.5*(bin_edges[i]+bin_edges[i+1]),
                "count": counts[i]
            })
    else:
        for ri, arr in enumerate(runs, 1):
            counts = _hist(arr)
            for i in range(nbins):
                records.append({
                    "run_label": f"Run {ri}",
                    "bin_left": bin_edges[i],
                    "bin_right": bin_edges[i+1],
                    "bin_center": 0.5*(bin_edges[i]+bin_edges[i+1]),
                    "count": counts[i]
                })
    hist_df = pd.DataFrame.from_records(records)

    # Prozentspalte, wenn normalize=True
    if normalize:
        hist_df.rename(columns={"count": "percent"}, inplace=True)

    # --- Punktwerte zählen (z. B. WER == 0) ---
    points_df = None
    if value_points:
        point_records = []
        if combine_runs:
            N = combined.size
            for v in value_points:
                mask = np.isfinite(combined) & (np.abs(combined - v) <= eps)
                c = int(mask.sum())
                row = {"run_label": "All runs", "value": float(v), "count": c}
                if normalize and N > 0:
                    row["percent"] = c / N * 100.0
                point_records.append(row)
        else:
            for ri, arr in enumerate(runs, 1):
                arr = arr[~np.isnan(arr)]
                N = arr.size
                for v in value_points:
                    mask = np.isfinite(arr) & (np.abs(arr - v) <= eps)
                    c = int(mask.sum())
                    row = {"run_label": f"Run {ri}", "value": float(v), "count": c}
                    if normalize and N > 0:
                        row["percent"] = c / N * 100.0
                    point_records.append(row)
        points_df = pd.DataFrame.from_records(point_records)

    # --- Schwellenwerte (<= t) zählen ---
    thr_df = None
    if thresholds:
        thr_records = []
        if combine_runs:
            N = combined.size
            for t in thresholds:
                c = int((combined <= t).sum())
                row = {"run_label": "All runs", "threshold_le": float(t), "count_le": c}
                if normalize and N > 0:
                    row["percent_le"] = c / N * 100.0
                thr_records.append(row)
        else:
            for ri, arr in enumerate(runs, 1):
                arr = arr[~np.isnan(arr)]
                N = arr.size
                for t in thresholds:
                    c = int((arr <= t).sum())
                    row = {"run_label": f"Run {ri}", "threshold_le": float(t), "count_le": c}
                    if normalize and N > 0:
                        row["percent_le"] = c / N * 100.0
                    thr_records.append(row)
        thr_df = pd.DataFrame.from_records(thr_records)

    # --- Plot ---
    fig, ax = plt.subplots(figsize=figsize)
    centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    width = (bin_edges[1] - bin_edges[0]) * (0.9 if combine_runs else 0.35)

    if combine_runs:
        y = hist_df["percent" if normalize else "count"].to_numpy()
        ax.bar(centers, y, width=width, color=color, alpha=alpha, edgecolor=edgecolor)
        label_runs = "All runs"
        ax.legend([label_runs], loc="best")
    else:
        # gruppierte Balken je Run
        n = len(runs)
        offsets = np.linspace(- (n-1)/2, (n-1)/2, n) * width
        for ri in range(n):
            sub = hist_df[hist_df["run_label"] == f"Run {ri+1}"]
            y = sub["percent" if normalize else "count"].to_numpy()
            ax.bar(centers + offsets[ri], y, width=width, alpha=alpha, label=f"Run {ri+1}", edgecolor=edgecolor)
        ax.legend(loc="best", title="Runs")

    # Achsen & Titel
    if xlabel is None:
        xlabel = _pretty(metric)
    if ylabel is None:
        ylabel = "Prozent" if normalize else ("kumulierte Anzahl" if cumulative else "Anzahl")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if ylimit is not None:
        ax.set_ylim(ylimit)

    if title is None:
        parts = [method_name]
        if task_label:
            parts.append(task_label)
        parts.append(_pretty(metric))
        title = " — ".join(parts) + (" (kumulativ)" if cumulative else "")
    ax.set_title(title)

    plt.tight_layout()
    plt.show()

    out = {"hist": hist_df}
    if points_df is not None:
        out["points"] = points_df
    if thr_df is not None:
        out["thresholds"] = thr_df
    return out

def metric_distribution_two_methods(
    method1_files: List[str],
    method2_files: List[str],
    metric: str,
    method1_name: str = "Traditional",
    method2_name: str = "LLM",
    task_label: Optional[str] = None,
    remove_first_n: int = 0,
    exclude_failures: bool = False,
    dataset_filter: Optional[str] = None,
    # Binning:
    nbins: int = 20,
    bin_edges: Optional[Sequence[float]] = None,
    normalize: bool = False,
    cumulative: bool = False,
    # Exaktwerte / Schwellen:
    value_points: Optional[Sequence[float]] = None,
    eps: float = 1e-12,
    thresholds: Optional[Sequence[float]] = None,
    # Darstellung:
    layout: str = "overlay",      # "overlay", "grouped", NEU: "overlay_ordered"
    figsize: Tuple[float, float] = (8, 5),
    colors: Tuple[str, str] = ("tab:blue", "tab:orange"),
    alpha: Tuple[float, float] = (0.55, 0.55),   # nur für "overlay" & "grouped"
    edgecolor: Optional[str] = None,
    title: Optional[str] = None,
    xlabel: Optional[str] = None,
    ylabel: Optional[str] = None,
    ylimit: Optional[Tuple[float, float]] = None,
    metric_display_map: Optional[Dict[str, str]] = None,
    # NEU – nur für overlay_ordered:
    front_shrink: float = 0.92    # vorderer Balken etwas schmaler zeichnen
):
    def _pretty(name: str) -> str:
        try:
            return _pretty_metric_name(name, metric_display_map)
        except NameError:
            return metric_display_map.get(name, name) if metric_display_map else name

    # Daten laden
    _, runs1, _ = _load_runs(method1_files, metric, remove_first_n, exclude_failures, dataset_filter)
    _, runs2, _ = _load_runs(method2_files, metric, remove_first_n, exclude_failures, dataset_filter)
    arr1 = np.concatenate(runs1) if runs1 else np.array([])
    arr2 = np.concatenate(runs2) if runs2 else np.array([])
    arr1 = arr1[~np.isnan(arr1)]
    arr2 = arr2[~np.isnan(arr2)]

    if arr1.size == 0 and arr2.size == 0:
        fig, ax = plt.subplots(figsize=figsize)
        ax.text(0.5, 0.5, "Keine Daten", ha="center", va="center", transform=ax.transAxes)
        ax.set_title(title or f"{method1_name} vs {method2_name} — {_pretty(metric)}")
        ax.axis("off")
        plt.tight_layout(); plt.show()
        return {"hist": pd.DataFrame()}

    # Gemeinsame Bins (Freedman–Diaconis oder vorgegeben)
    all_vals = np.concatenate([arr1, arr2]) if arr1.size and arr2.size else (arr1 if arr1.size else arr2)
    if bin_edges is None:
        q1, q3 = np.percentile(all_vals, [25, 75])
        iqr = q3 - q1
        if iqr > 0:
            h = 2 * iqr * (all_vals.size ** (-1/3))
            if h > 0:
                nbins_fd = int(np.ceil((all_vals.max() - all_vals.min()) / h)) or nbins
                nbins = int(np.clip(nbins_fd, 5, 100))
        bin_edges = np.linspace(all_vals.min(), all_vals.max(), nbins + 1)
    else:
        bin_edges = np.asarray(bin_edges, dtype=float)
        nbins = len(bin_edges) - 1

    def _hist(arr):
        counts, _ = np.histogram(arr, bins=bin_edges)
        if cumulative:
            counts = np.cumsum(counts)
        if normalize:
            total = arr.size if arr.size > 0 else 1
            counts = counts / total * 100.0
        return counts

    counts1 = _hist(arr1)
    counts2 = _hist(arr2)
    centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])

    # Plot
    fig, ax = plt.subplots(figsize=figsize)
    width = (bin_edges[1] - bin_edges[0]) * (0.9 if layout.startswith("overlay") else 0.42)

    if layout == "overlay":
        ax.bar(centers, counts1, width=width, color=colors[0], alpha=alpha[0],
               edgecolor=edgecolor, label=method1_name)
        ax.bar(centers, counts2, width=width, color=colors[1], alpha=alpha[1],
               edgecolor=edgecolor, label=method2_name)

    elif layout == "grouped":
        ax.bar(centers - width/2, counts1, width=width, color=colors[0], alpha=alpha[0],
               edgecolor=edgecolor, label=method1_name)
        ax.bar(centers + width/2, counts2, width=width, color=colors[1], alpha=alpha[1],
               edgecolor=edgecolor, label=method2_name)

    elif layout == "overlay_ordered":
        # Kein Alpha → keine Farbmischung; Reihenfolge je Bin nach Höhe
        labelled1 = False
        labelled2 = False
        for xc, h1, h2 in zip(centers, counts1, counts2):
            if h1 >= h2:
                # method1 hinten, method2 vorn
                ax.bar(xc, h1, width=width, color=colors[0], alpha=1.0, zorder=1,
                       edgecolor=edgecolor, label=(method1_name if not labelled1 else None))
                ax.bar(xc, h2, width=width*front_shrink, color=colors[1], alpha=1.0, zorder=2,
                       edgecolor=edgecolor, label=(method2_name if not labelled2 else None))
            else:
                # method2 hinten, method1 vorn
                ax.bar(xc, h2, width=width, color=colors[1], alpha=1.0, zorder=1,
                       edgecolor=edgecolor, label=(method2_name if not labelled2 else None))
                ax.bar(xc, h1, width=width*front_shrink, color=colors[0], alpha=1.0, zorder=2,
                       edgecolor=edgecolor, label=(method1_name if not labelled1 else None))
            labelled1 = True
            labelled2 = True
    else:
        raise ValueError("layout muss 'overlay', 'grouped' oder 'overlay_ordered' sein.")

    # Achsen & Titel
    if xlabel is None:
        xlabel = _pretty(metric)
    if ylabel is None:
        ylabel = ("Prozent" if normalize else ("kumulierte Anzahl" if cumulative else "Anzahl"))
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if ylimit is not None:
        ax.set_ylim(ylimit)

    parts = [method1_name, "vs", method2_name, "—", _pretty(metric)]
    if task_label:
        parts.insert(0, f"{task_label} —")
    if title is None:
        title = " ".join(parts)
    ax.set_title(title)
    ax.legend(loc="best", title="Methode")

    plt.tight_layout()
    plt.show()

    # Tabellen (optional, wie gehabt):
    records = []
    for i in range(nbins):
        records.append({"method": method1_name, "bin_left": bin_edges[i], "bin_right": bin_edges[i+1],
                        "bin_center": centers[i], "value": counts1[i]})
        records.append({"method": method2_name, "bin_left": bin_edges[i], "bin_right": bin_edges[i+1],
                        "bin_center": centers[i], "value": counts2[i]})
    hist_df = pd.DataFrame.from_records(records)

    points_df = None
    if value_points:
        rows = []
        for name, arr in [(method1_name, arr1), (method2_name, arr2)]:
            N = arr.size
            for v in value_points:
                c = int(np.sum(np.isfinite(arr) & (np.abs(arr - v) <= eps)))
                row = {"method": name, "value": float(v), "count": c}
                if normalize and N > 0:
                    row["percent"] = c / N * 100.0
                rows.append(row)
        points_df = pd.DataFrame(rows)

    thr_df = None
    if thresholds:
        rows = []
        for name, arr in [(method1_name, arr1), (method2_name, arr2)]:
            N = arr.size
            for t in thresholds:
                c = int(np.sum(arr <= t))
                row = {"method": name, "threshold_le": float(t), "count_le": c}
                if normalize and N > 0:
                    row["percent_le"] = c / N * 100.0
                rows.append(row)
        thr_df = pd.DataFrame(rows)

    out = {"hist": hist_df}
    if points_df is not None: out["points"] = points_df
    if thr_df is not None: out["thresholds"] = thr_df
    return out

def boxplot_time_with_without_failures_two_methods(
    method1_files: List[str],
    method2_files: List[str],
    metric_time: str = "time_total",      # z. B. "time_total", "time_api", "time_ocr"
    method1_name: str = "Traditional",
    method2_name: str = "LLM",
    task_label: Optional[str] = None,     # nur für den Titel
    remove_first_n: int = 0,
    dataset_filter: Optional[str] = None,
    combine_runs: bool = True,            # True: je Subgruppe 1 Box (Runs kombiniert); False: je Subgruppe 3 Boxen (Runs)
    figsize: Tuple[float, float] = (9, 5),
    method_gap: float = 2.2,              # Abstand der beiden Methodengruppen
    inner_sep: float = 1.0,               # Abstand der Subgruppen (inkl/ohne) innerhalb einer Methode
    box_width: float = 0.24,
    showfliers: bool = True,
    colors: Tuple[str, str] = ("tab:blue", "tab:orange"),   # (inkl. Failures, ohne Failures)
    legend_labels: Tuple[str, str] = ("inkl. Failures", "ohne Failures"),
    ylabel: str = "Zeit (s)",
    title: Optional[str] = None,
    ylimit: Optional[Tuple[float, float]] = None,
    edgecolor: Optional[str] = None,      # z. B. "#555" für dezente Kanten
):
    """
    Vergleicht Laufzeiten beider Methoden als Boxplot:
    X-Achse: zwei Gruppen (method1, method2).
    Jede Gruppe: zwei Subgruppen (inkl. Failures, ohne Failures).
    Farben kodieren den Failure-Filter.
    """
    # --- Daten laden: mit und ohne Failures ---
    # Methode 1
    _, runs1_with, _  = _load_runs(method1_files, metric_time, remove_first_n, exclude_failures=False, dataset=dataset_filter)
    _, runs1_wo, _    = _load_runs(method1_files, metric_time, remove_first_n, exclude_failures=True,  dataset=dataset_filter)
    # Methode 2
    _, runs2_with, _  = _load_runs(method2_files, metric_time, remove_first_n, exclude_failures=False, dataset=dataset_filter)
    _, runs2_wo, _    = _load_runs(method2_files, metric_time, remove_first_n, exclude_failures=True,  dataset=dataset_filter)

    # --- Positionen & Daten aufbauen ---
    positions: List[float] = []
    box_data: List[np.ndarray] = []
    color_per_box: List[str] = []

    # pro Methode reservieren wir zwei Subgruppen: [inkl., ohne]
    group_width = inner_sep  # Distanz von inkl. zu ohne innerhalb einer Methode
    base1 = 0.0
    base2 = base1 + group_width + method_gap

    def add_subgroup(base_x: float, runs_list: List[np.ndarray], color: str):
        if combine_runs:
            arr = np.concatenate(runs_list) if runs_list else np.array([])
            positions.append(base_x)
            box_data.append(arr)
            color_per_box.append(color)
        else:
            jitter = np.linspace(-0.25, 0.25, num=len(runs_list)) if runs_list else np.array([0.0])
            for j, arr in enumerate(runs_list):
                positions.append(base_x + (jitter[j] if len(jitter) > j else 0.0))
                box_data.append(arr)
                color_per_box.append(color)

    # Methode 1 (links)
    add_subgroup(base1,              runs1_with, colors[0])  # inkl. Failures
    add_subgroup(base1 + inner_sep,  runs1_wo,   colors[1])  # ohne Failures
    # Methode 2 (rechts)
    add_subgroup(base2,              runs2_with, colors[0])
    add_subgroup(base2 + inner_sep,  runs2_wo,   colors[1])

    # --- Plot ---
    fig, ax = plt.subplots(figsize=figsize)
    bp = ax.boxplot(
        box_data,
        positions=positions,
        widths=box_width,
        showfliers=showfliers,
        patch_artist=True
    )

    # Boxen einfärben
    for b, col in zip(bp["boxes"], color_per_box):
        b.set_facecolor(col)
        b.set_edgecolor(edgecolor if edgecolor else col)
        b.set_alpha(0.75 if edgecolor is None else 0.9)
        if edgecolor:
            b.set_linewidth(1.0)

    # Mittelwert-Linien (falls vorhanden)
    try:
        _draw_mean_lines(ax, positions, box_data, box_width, linestyle="--", linewidth=1.0, alpha=0.9)
    except NameError:
        pass

    # X-Ticks: Zentren der Methodengruppen
    center1 = base1 + group_width / 2.0
    center2 = base2 + group_width / 2.0
    ax.set_xticks([center1, center2])
    ax.set_xticklabels([method1_name, method2_name])

    # Subgruppen-Beschriftung klein unter die Boxen (optional)
    # ymin = ax.get_ylim()[0]
    # for x, lbl in [(base1, legend_labels[0]), (base1 + inner_sep, legend_labels[1]),
    #                (base2, legend_labels[0]), (base2 + inner_sep, legend_labels[1])]:
    #     ax.text(x, ymin, lbl, ha="center", va="bottom", fontsize=9)

    # Y-Achse & Titel
    ax.set_ylabel(ylabel)
    if ylimit is not None:
        ax.set_ylim(ylimit)

    if title is None:
        title_parts = [method1_name, "vs", method2_name, "—", metric_time]
        if task_label:
            title_parts.insert(0, f"{task_label} —")
        title = " ".join(title_parts)
    ax.set_title(title)

    # Legende (Farben = Filterstatus)
    handles = [Patch(facecolor=colors[0], edgecolor=edgecolor or colors[0], alpha=0.75, label=legend_labels[0]),
               Patch(facecolor=colors[1], edgecolor=edgecolor or colors[1], alpha=0.75, label=legend_labels[1])]
    ax.legend(handles=handles, title="Datenbasis", loc="best")

    plt.tight_layout()
    plt.show()

def _cumpeak_runs(files: List[str],
                  metric_mem: str = "mem_peak",
                  remove_first_n: int = 0,
                  exclude_failures: bool = False,
                  dataset_filter: Optional[str] = None) -> List[np.ndarray]:
    """
    Liefert pro Run die kumulative Peak-Kurve (max. bisheriger Speicherbedarf)
    als 1D-Array (Länge = Anzahl Iterationen nach Filtern).
    """
    _, runs, _ = _load_runs(files, metric_mem, remove_first_n, exclude_failures, dataset_filter)
    curves = []
    for arr in runs:
        arr = np.asarray(arr, dtype=float)
        arr = arr[~np.isnan(arr)]
        if arr.size == 0:
            curves.append(np.array([], dtype=float))
        else:
            curves.append(np.maximum.accumulate(arr))
    return curves


def memory_peak_over_iterations_one_method(
    files: List[str],
    metric_mem: str = "mem_peak",       # mappt auf mem_llm_peak / mem_trad_peak
    method_name: str = "Method",
    task_label: Optional[str] = None,   # nur im Titel
    remove_first_n: int = 0,
    exclude_failures: bool = False,
    dataset_filter: Optional[str] = None,
    combine_runs: bool = True,          # True: Runs mitteln + Band; False: Kurven/Bars je Run
    style: str = "line",                # "line" | "bar" | "both"
    figsize: Tuple[float, float] = (9, 5),
    color: str = "tab:blue",
    edgecolor: Optional[str] = None,
    band: str = "std",                  # "std" | "minmax" | None (nur bei combine_runs=True)
    alpha_band: float = 0.15,
    ylabel: str = "Memory Peak (MB)",
    xlabel: str = "Anzahl verarbeiteter Produkte (Iterationen)",
    title: Optional[str] = None,
    ylimit: Optional[Tuple[float, float]] = None,
):
    """
    Zeigt den kumulativen Memory-Peak über den Iterationen für EINE Methode.
    - style="line": Linie(n); "bar": Säulen; "both": Säulen + Linie.
    - combine_runs=True: Ø-Kurve über alle Runs + Band (Std oder Min–Max).
      (Runs werden auf die gemeinsame MIN-Länge getrimmt, um pro Iteration zu mitteln.)
    """
    curves = _cumpeak_runs(files, metric_mem, remove_first_n, exclude_failures, dataset_filter)
    curves = [c for c in curves if c.size > 0]
    if len(curves) == 0:
        fig, ax = plt.subplots(figsize=figsize)
        ax.text(0.5, 0.5, "Keine Daten", ha="center", va="center", transform=ax.transAxes)
        ax.set_axis_off()
        plt.tight_layout(); plt.show()
        return

    fig, ax = plt.subplots(figsize=figsize)

    if combine_runs:
        L = min(len(c) for c in curves)
        if L == 0:
            ax.text(0.5, 0.5, "Keine Daten (nach Trimmen)", ha="center", va="center", transform=ax.transAxes)
            ax.set_axis_off(); plt.tight_layout(); plt.show(); return
        M = np.vstack([c[:L] for c in curves])  # shape (runs, L)
        mean_curve = M.mean(axis=0)
        x = np.arange(1, L+1)

        if style in ("bar", "both"):
            ax.bar(x, mean_curve, width=0.85, color=color, edgecolor=edgecolor or color, alpha=0.7)
        if style in ("line", "both"):
            ax.plot(x, mean_curve, linewidth=2.0, color=edgecolor or color)

        if band in ("std", "minmax"):
            if band == "std":
                std = M.std(axis=0, ddof=1) if M.shape[0] > 1 else np.zeros_like(mean_curve)
                lower, upper = mean_curve - std, mean_curve + std
            else:
                lower, upper = M.min(axis=0), M.max(axis=0)
            ax.fill_between(x, lower, upper, color=color, alpha=alpha_band, linewidth=0)

    else:
        # pro Run
        n = len(curves)
        Lmax = max(len(c) for c in curves)
        x = np.arange(1, Lmax+1)
        if style == "bar":
            # gruppierte Balken pro Iteration (schmal), kann bei vielen Iterationen dicht werden
            width = 0.85 / n
            offsets = np.linspace(-(n-1)/2, (n-1)/2, n) * width
            for i, c in enumerate(curves):
                xi = np.arange(1, len(c)+1) + offsets[i]
                ax.bar(xi, c, width=width, alpha=0.7, label=f"Run {i+1}",
                       color=color, edgecolor=edgecolor or color)
        elif style == "both":
            # dünne Balken + Linien obendrauf
            width = 0.7 / n
            offsets = np.linspace(-(n-1)/2, (n-1)/2, n) * width
            for i, c in enumerate(curves):
                xi = np.arange(1, len(c)+1) + offsets[i]
                ax.bar(xi, c, width=width, alpha=0.5, color=color, edgecolor=edgecolor or color)
                ax.plot(np.arange(1, len(c)+1), c, linewidth=1.6, label=f"Run {i+1}",
                        color=edgecolor or color)
        else:  # "line"
            for i, c in enumerate(curves):
                ax.plot(np.arange(1, len(c)+1), c, linewidth=1.8, label=f"Run {i+1}",
                        alpha=0.9)

        if style in ("line", "both"):
            ax.legend(loc="best", title="Runs")

    # Achsen & Titel
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if ylimit is not None:
        ax.set_ylim(ylimit)
    if title is None:
        title = f"{method_name} — Memory Peak über Iterationen"
        if task_label:
            title = f"{task_label} — " + title
    ax.set_title(title)
    ax.grid(True, axis="y", linestyle="--", alpha=0.35)
    plt.tight_layout()
    plt.show()


def _series_runs_generic(files: List[str],
                         metric: str,
                         remove_first_n: int = 0,
                         exclude_failures: bool = False,
                         dataset_filter: Optional[str] = None,
                         cumulative: bool = True) -> List[np.ndarray]:
    """Lädt die Serie(n) pro Run und macht daraus optional die kumulative Peak-Kurve."""
    _, runs, _ = _load_runs(files, metric, remove_first_n, exclude_failures, dataset_filter)
    out = []
    for arr in runs:
        a = np.asarray(arr, dtype=float)
        a = a[~np.isnan(a)]
        if a.size == 0:
            out.append(np.array([], dtype=float))
        else:
            out.append(np.maximum.accumulate(a) if cumulative else a)
    return out


def memory_peak_over_iterations_three_methods(
    method1_files: List[str],
    method2_files: List[str],
    method3_files: List[str],
    metric_mem: str = "mem_peak",           # mappt via _METRIC_MAP auf mem_*_peak
    method1_name: str = "Traditional",
    method2_name: str = "LLM",
    method3_name: str = "Other",
    task_label: Optional[str] = None,
    remove_first_n: int = 0,
    exclude_failures: bool = False,         # gilt für ALLE
    dataset_filter: Optional[str] = None,
    combine_runs: bool = True,
    cumulative: bool = True,                # NEU: kumulativer Peak (True) vs. aktuelle Werte (False)
    style: str = "line",                    # "line" | "bar" | "both"
    figsize: Tuple[float, float] = (9, 5),
    colors: Tuple[str, str, str] = ("tab:blue", "tab:orange", "tab:green"),
    edgecolors: Optional[Tuple[Optional[str], Optional[str], Optional[str]]] = None,  # (edge1, edge2, edge3)
    band: Optional[str] = "std",            # "std" | "minmax" | None  (nur combine_runs=True)
    alpha_band: float = 0.15,
    ylabel: str = "Memory Peak (MB)",
    xlabel: str = "Anzahl verarbeiteter Produkte (Iterationen)",
    title: Optional[str] = None,
    ylimit: Optional[Tuple[float, float]] = None,
):
    """
    Vergleicht (kumulative) Memory-Serien von DREI Methoden.
    - combine_runs=True: mittelt je Methode über Runs auf gemeinsame X-Länge (minimale Länge über alle drei).
    - style: Linien, Balken oder beides. Bei Balken werden pro Iteration drei Gruppen-Balken gezeichnet.
    - band: Streuungsband um die gemittelte Kurve je Methode (std oder minmax).
    - cumulative=False: zeigt die MOMENTANwerte; cumulative=True: bisherige Peaks.
    """
    # --- Daten laden ---
    curves1 = _series_runs_generic(method1_files, metric_mem, remove_first_n, exclude_failures, dataset_filter, cumulative)
    curves2 = _series_runs_generic(method2_files, metric_mem, remove_first_n, exclude_failures, dataset_filter, cumulative)
    curves3 = _series_runs_generic(method3_files, metric_mem, remove_first_n, exclude_failures, dataset_filter, cumulative)

    curves1 = [c for c in curves1 if c.size > 0]
    curves2 = [c for c in curves2 if c.size > 0]
    curves3 = [c for c in curves3 if c.size > 0]

    if len(curves1) == 0 and len(curves2) == 0 and len(curves3) == 0:
        fig, ax = plt.subplots(figsize=figsize)
        ax.text(0.5, 0.5, "Keine Daten", ha="center", va="center", transform=ax.transAxes)
        ax.set_axis_off(); plt.tight_layout(); plt.show(); return

    fig, ax = plt.subplots(figsize=figsize)

    # Farben/Kanten
    ec1 = (edgecolors[0] if edgecolors else None) or colors[0]
    ec2 = (edgecolors[1] if edgecolors else None) or colors[1]
    ec3 = (edgecolors[2] if edgecolors else None) or colors[2]

    if combine_runs:
        # Gemeinsame X-Länge bestimmen (für sauberes Alignment über ALLE drei Methoden)
        lens = []
        if curves1: lens.append(min(len(c) for c in curves1))
        if curves2: lens.append(min(len(c) for c in curves2))
        if curves3: lens.append(min(len(c) for c in curves3))
        L = min(lens) if lens else 0

        if L == 0:
            ax.text(0.5, 0.5, "Keine gemeinsamen Iterationen", ha="center", va="center", transform=ax.transAxes)
            ax.set_axis_off(); plt.tight_layout(); plt.show(); return

        # Matritzen bauen (runs x L), Mittelkurven & Bänder
        def _mean_and_band(curves):
            M = np.vstack([c[:L] for c in curves])  # (R, L)
            mean = M.mean(axis=0)
            if band == "std" and M.shape[0] > 1:
                sd = M.std(axis=0, ddof=1)
                low, up = mean - sd, mean + sd
            elif band == "minmax":
                low, up = M.min(axis=0), M.max(axis=0)
            else:
                low = up = None
            return mean, low, up, M

        mean1 = low1 = up1 = M1 = None
        mean2 = low2 = up2 = M2 = None
        mean3 = low3 = up3 = M3 = None
        x = np.arange(1, L+1)

        if curves1:
            mean1, low1, up1, M1 = _mean_and_band(curves1)
        if curves2:
            mean2, low2, up2, M2 = _mean_and_band(curves2)
        if curves3:
            mean3, low3, up3, M3 = _mean_and_band(curves3)

        # Zeichnen
        if style in ("bar", "both"):
            width = 0.8 / 3.0
            offs = (-width, 0.0, width)
            if mean1 is not None: ax.bar(x + offs[0], mean1, width=width, color=colors[0], edgecolor=ec1, alpha=0.75, label=f"{method1_name} (Ø)")
            if mean2 is not None: ax.bar(x + offs[1], mean2, width=width, color=colors[1], edgecolor=ec2, alpha=0.75, label=f"{method2_name} (Ø)")
            if mean3 is not None: ax.bar(x + offs[2], mean3, width=width, color=colors[2], edgecolor=ec3, alpha=0.75, label=f"{method3_name} (Ø)")

        if style in ("line", "both"):
            if mean1 is not None: ax.plot(x, mean1, linewidth=2.0, color=ec1, label=(None if style=="both" else f"{method1_name} (Ø)"))
            if mean2 is not None: ax.plot(x, mean2, linewidth=2.0, color=ec2, label=(None if style=="both" else f"{method2_name} (Ø)"))
            if mean3 is not None: ax.plot(x, mean3, linewidth=2.0, color=ec3, label=(None if style=="both" else f"{method3_name} (Ø)"))

        if band in ("std", "minmax"):
            if low1 is not None: ax.fill_between(x, low1, up1, color=colors[0], alpha=alpha_band, linewidth=0)
            if low2 is not None: ax.fill_between(x, low2, up2, color=colors[1], alpha=alpha_band, linewidth=0)
            if low3 is not None: ax.fill_between(x, low3, up3, color=colors[2], alpha=alpha_band, linewidth=0)

    else:
        # pro Run (je Methode)
        if curves1:
            for i, c in enumerate(curves1):
                ax.plot(np.arange(1, len(c)+1), c, linewidth=1.6, color=colors[0], alpha=0.9,
                        label=(f"{method1_name} Run {i+1}" if i == 0 else None))
        if curves2:
            for j, c in enumerate(curves2):
                ax.plot(np.arange(1, len(c)+1), c, linewidth=1.6, color=colors[1], alpha=0.9,
                        label=(f"{method2_name} Run {j+1}" if j == 0 else None))
        if curves3:
            for k, c in enumerate(curves3):
                ax.plot(np.arange(1, len(c)+1), c, linewidth=1.6, color=colors[2], alpha=0.9,
                        label=(f"{method3_name} Run {k+1}" if k == 0 else None))

    # Achsen & Titel
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if ylimit is not None:
        ax.set_ylim(ylimit)
    if title is None:
        base = f"{method1_name} vs {method2_name} vs {method3_name} — {'kumulativer ' if cumulative else ''}Memory über Iterationen"
        title = f"{task_label} — {base}" if task_label else base
    ax.set_title(title)
    ax.grid(True, axis="y", linestyle="--", alpha=0.35)
    ax.legend(loc="best", title="Methode")
    plt.tight_layout()
    plt.show()

def memory_peak_over_iterations_two_methods(
    method1_files: List[str],
    method2_files: List[str],
    metric_mem: str = "mem_peak",
    method1_name: str = "Traditional",
    method2_name: str = "LLM",
    task_label: Optional[str] = None,
    remove_first_n: int = 0,
    exclude_failures: bool = False,      # wird für BEIDE Methoden angewandt
    dataset_filter: Optional[str] = None,
    combine_runs: bool = True,
    style: str = "line",                 # "line" | "bar" | "both"
    figsize: Tuple[float, float] = (9, 5),
    colors: Tuple[str, str] = ("tab:blue", "tab:orange"),
    edgecolors: Optional[Tuple[Optional[str], Optional[str]]] = None,  # (edge1, edge2)
    band: str = "std",                  # "std" | "minmax" | None (nur combine_runs=True)
    alpha_band: float = 0.15,
    ylabel: str = "Memory Peak (MB)",
    xlabel: str = "Anzahl verarbeiteter Produkte (Iterationen)",
    title: Optional[str] = None,
    ylimit: Optional[Tuple[float, float]] = None,
):
    """
    Vergleicht den kumulativen Memory-Peak zweier Methoden im selben Plot.
    Bei combine_runs=True werden die Runs je Methode gemittelt (auf MIN-Länge getrimmt).
    """
    curves1 = _cumpeak_runs(method1_files, metric_mem, remove_first_n, exclude_failures, dataset_filter)
    curves2 = _cumpeak_runs(method2_files, metric_mem, remove_first_n, exclude_failures, dataset_filter)
    curves1 = [c for c in curves1 if c.size > 0]
    curves2 = [c for c in curves2 if c.size > 0]
    if len(curves1) == 0 and len(curves2) == 0:
        fig, ax = plt.subplots(figsize=figsize)
        ax.text(0.5, 0.5, "Keine Daten", ha="center", va="center", transform=ax.transAxes)
        ax.set_axis_off(); plt.tight_layout(); plt.show(); return

    fig, ax = plt.subplots(figsize=figsize)

    if combine_runs:
        # Methode 1
        L1 = min((len(c) for c in curves1), default=0)
        L2 = min((len(c) for c in curves2), default=0)
        if L1 > 0:
            M1 = np.vstack([c[:L1] for c in curves1])
            mean1 = M1.mean(axis=0); x1 = np.arange(1, L1+1)
        else:
            M1 = None
        if L2 > 0:
            M2 = np.vstack([c[:L2] for c in curves2])
            mean2 = M2.mean(axis=0); x2 = np.arange(1, L2+1)
        else:
            M2 = None

        # Zeichnen je nach style
        ec1 = (edgecolors[0] if edgecolors else None) or colors[0]
        ec2 = (edgecolors[1] if edgecolors else None) or colors[1]

        if style in ("bar", "both"):
            if L1 > 0:
                ax.bar(x1 - 0.15, mean1, width=0.3, color=colors[0], edgecolor=ec1, alpha=0.7, label=f"{method1_name} (Ø)")
            if L2 > 0:
                ax.bar(x2 + 0.15, mean2, width=0.3, color=colors[1], edgecolor=ec2, alpha=0.7, label=f"{method2_name} (Ø)")
        if style in ("line", "both"):
            if L1 > 0:
                ax.plot(x1, mean1, linewidth=2.0, color=ec1, label=f"{method1_name} (Ø)")
            if L2 > 0:
                ax.plot(x2, mean2, linewidth=2.0, color=ec2, label=f"{method2_name} (Ø)")

        if band in ("std", "minmax"):
            if L1 > 0:
                if band == "std":
                    std1 = M1.std(axis=0, ddof=1) if M1.shape[0] > 1 else np.zeros_like(mean1)
                    low1, up1 = mean1 - std1, mean1 + std1
                else:
                    low1, up1 = M1.min(axis=0), M1.max(axis=0)
                ax.fill_between(x1, low1, up1, color=colors[0], alpha=alpha_band, linewidth=0)
            if L2 > 0:
                if band == "std":
                    std2 = M2.std(axis=0, ddof=1) if M2.shape[0] > 1 else np.zeros_like(mean2)
                    low2, up2 = mean2 - std2, mean2 + std2
                else:
                    low2, up2 = M2.min(axis=0), M2.max(axis=0)
                ax.fill_between(x2, low2, up2, color=colors[1], alpha=alpha_band, linewidth=0)

    else:
        # pro Run
        for i, c in enumerate(curves1):
            ax.plot(np.arange(1, len(c)+1), c, linewidth=1.6, color=colors[0], alpha=0.9,
                    label=(f"{method1_name} Run {i+1}" if i == 0 else None))
        for j, c in enumerate(curves2):
            ax.plot(np.arange(1, len(c)+1), c, linewidth=1.6, color=colors[1], alpha=0.9,
                    label=(f"{method2_name} Run {j+1}" if j == 0 else None))

    # Achsen & Titel
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if ylimit is not None:
        ax.set_ylim(ylimit)
    if title is None:
        base = f"{method1_name} vs {method2_name} — Memory Peak über Iterationen"
        title = f"{task_label} — {base}" if task_label else base
    ax.set_title(title)
    ax.grid(True, axis="y", linestyle="--", alpha=0.35)
    ax.legend(loc="best", title="Methode")
    plt.tight_layout()
    plt.show()