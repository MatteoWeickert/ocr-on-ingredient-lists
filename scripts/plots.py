import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

from ocr_viz_new import (
    boxplot_metric_over_runs,
    boxplot_two_methods_one_task,
    grouped_boxplot_two_methods_two_tasks,
    paired_ttests_two_tasks,
    multi_metric_by_task_single_method,
    multi_metric_by_task_two_methods,
    time_per_image_boxplot,
    confusion_matrix_from_threshold,
    stats_metric_over_runs,
    stats_two_methods_one_task,
    stats_two_methods_two_tasks,
    stats_multi_metric_by_task_two_methods,
    boxplot_two_methods_metrics_grouped_by_method_one_task,
    metric_distribution_one_method,
    metric_distribution_two_methods,
    boxplot_time_with_without_failures_two_methods,
    memory_peak_over_iterations_three_methods,
    memory_peak_over_iterations_two_methods
)

WORKING_DIR = "C:\\Users\\maweo\\OneDrive - Universität Münster\\Dokumente\\Studium\\Semester 6\\Bachelorarbeit\\eval_results"
OUTPUT_DIR = "C:\\Users\\maweo\\OneDrive - Universität Münster\\Dokumente\\Studium\\Semester 6\\Bachelorarbeit\\eval_results\\plots"

def pastel_palette(n: int, name="Set2"):
    cmap = plt.get_cmap(name)
    return [cmap(i) for i in range(n)]
def plot_comparison(files_method1, files_method2, metric, method1_name="Methode A", method2_name="Methode B",
                    remove_first_n=0, exclude_failures=False):
    """
    Stellt die Verteilung einer Kennzahl (Metrik) für zwei Methoden/Versuchsvarianten mittels Boxplots gegenüber.
    
    Parameter:
    - files_method1, files_method2: Listen der CSV-Dateipfade für die Läufe der beiden Methoden.
    - metric: Name der darzustellenden Kennzahl (z.B. 'time_total', 'wer', 'f1' usw.).
    - method1_name, method2_name: Beschriftungen für die beiden verglichenen Methoden (werden in der Grafik angezeigt).
    - remove_first_n: Wenn >0, werden die ersten n Einträge jedes Laufs (Kaltstart-Phase) entfernt.
    - exclude_failures: Wenn True, werden Datensätze mit Totalausfällen (F1 = 0) ausgeschlossen.
    """
    # Bestimme den Methodentyp (LLM oder traditionell) anhand der Spalten des ersten Laufes
    df1_sample = pd.read_csv(files_method1[0], sep=';')
    df2_sample = pd.read_csv(files_method2[0], sep=';')
    def get_method_type(df):
        cols = df.columns
        if any(col.endswith('_llm') for col in cols):
            return 'llm'
        elif any(col.endswith('_ocr') or col.endswith('_trad') for col in cols):
            return 'trad'
        else:
            return 'other'
    type1 = get_method_type(df1_sample)
    type2 = get_method_type(df2_sample)
    # Mapping: allgemeiner Metrikname -> spezifische Spalte je nach Methode
    metric_map = {
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
    if metric not in metric_map:
        raise ValueError(f"Unbekannte Metrik '{metric}'. Verfügbare Metriken: {list(metric_map.keys())}.")
    col1 = metric_map[metric].get(type1)
    col2 = metric_map[metric].get(type2)
    if col1 is None or col2 is None:
        raise ValueError(f"Metrik '{metric}' ist für einen der Methodentypen nicht vorhanden (Typ1: {type1}, Typ2: {type2}).")
    # Daten aus allen Läufen beider Methoden sammeln
    combined_df_list = []
    # Methode 1: alle Lauf-Dateien einlesen
    for i, file in enumerate(files_method1, start=1):
        df = pd.read_csv(file, sep=';')
        if remove_first_n:
            df = df.iloc[remove_first_n:]        # Kaltstart-Daten entfernen
        if exclude_failures:
            # Totalausfälle (F1 = 0) ausschließen
            f1_col = "f1_overall_llm" if type1 == 'llm' else "f1_overall_ocr"
            if f1_col in df.columns:
                df = df[df[f1_col] > 0]
        # Werte der gewünschten Metrik auswählen
        if col1 not in df.columns:
            raise ValueError(f"Spalte {col1} fehlt in Datei {file}.")
        values = df[col1].dropna()
        # DataFrame mit Kennzahl und Labels für Methode und Lauf erstellen
        combined_df_list.append(pd.DataFrame({
            'Gruppe': method1_name,
            'Lauf': f'Lauf {i}',
            'Wert': values
        }))
    # Methode 2: alle Läufe einlesen
    for j, file in enumerate(files_method2, start=1):
        df = pd.read_csv(file, sep=';')
        if remove_first_n:
            df = df.iloc[remove_first_n:]
        if exclude_failures:
            f1_col = "f1_overall_llm" if type2 == 'llm' else "f1_overall_ocr"
            if f1_col in df.columns:
                df = df[df[f1_col] > 0]
        if col2 not in df.columns:
            raise ValueError(f"Spalte {col2} fehlt in Datei {file}.")
        values = df[col2].dropna()
        combined_df_list.append(pd.DataFrame({
            'Gruppe': method2_name,
            'Lauf': f'Lauf {j}',
            'Wert': values
        }))
    # Alle Ergebnisse zu einem DataFrame kombinieren
    combined_df = pd.concat(combined_df_list, ignore_index=True)
    # Boxplot erstellen
    plt.figure(figsize=(8,6))
    sns.set_theme(style="whitegrid", font_scale=1.1)
    ax = sns.boxplot(data=combined_df, x='Gruppe', y='Wert', hue='Lauf', showfliers=True)
    # Achsenbeschriftungen und Titel
    nice_labels = {
        "time_total":  "Verarbeitungszeit (s)",
        "time_yolo":   "YOLO-Erkennungszeit (s)",
        "time_ocr":    "OCR-Verarbeitungszeit (s)",
        "time_api":    "LLM-API-Roundtrip (s)",
        "time_preproc":"Preprocessing-Time (s)",
        "time_postproc":"Postprocessing-Time (s)",
        "cpu_time":    "CPU-Zeit (s)",
        "mem_peak":    "Spitzenspeicherbedarf (MB)",
        "wer":         "Word Error Rate",
        "cer":         "Character Error Rate",
        "precision":   "Precision",
        "recall":      "Recall",
        "f1":          "F1-Score",
        "total_tokens":"Gesamte Tokenanzahl",
        "total_cost":  "Gesamtkosten (USD)",
        "overall_table_score": "GriTS Gesamtscore",
        "grits_content_overall_f1": "GriTS Inhalt-F1",
        "grits_content_overall_recall": "GriTS Inhalt-Recall",
        "grits_content_overall_precision": "GriTS Inhalt-Precision",
        "grits_topology_f1": "GriTS Tabellenstruktur-F1",
        "grits_topology_recall": "GriTS Tabellenstruktur-Recall",
        "grits_topology_precision": "GriTS Tabellenstruktur-Precision",
        "precision_elements": "Precision der erkannten Nährwertelemente",
        "recall_elements":    "Recall der erkannten Nährwertelemente",
        "f1_elements":        "F1-Score der erkannten Nährwertelemente",
        "precision_values":   "Precision der erkannten Mengenwerte",
        "recall_values":      "Recall der erkannten Mengenwerte",
        "f1_values":          "F1-Score der erkannten Mengenwerte",
        "precision_pairs":    "Precision der erkannten Nährwert-Mengenpaare",
        "recall_pairs":       "Recall der erkannten Nährwert-Mengenpaare",
        "f1_pairs":           "F1-Score der erkannten Nährwert-Mengenpaare",
        "precision_triples":  "Precision der erkannten Nährwert-Mengen-Spalten-Triples",
        "recall_triples":     "Recall der erkannten Nährwert-Mengen-Spalten-Triples",
        "f1_triples":         "F1-Score der erkannten Nährwert-Mengen-Spalten-Triples",
        "cell_accuracy":      "Zellen-Genauigkeit",
        "grits_header_f1":    "GriTS Spaltenüberschriften-F1",
        "grits_header_recall":    "GriTS Spaltenüberschriften-Recall",
        "grits_header_precision":    "GriTS Spaltenüberschriften-Precision",
        "grits_labels_f1":      "GriTS Nährwertelemente-F1",
        "grits_labels_recall":      "GriTS Nährwertelemente-Recall",
        "grits_labels_precision":      "GriTS Nährwertelemente-Precision",
        "grits_values_f1":      "GriTS Mengenwerte-F1", 
        "grits_values_recall":      "GriTS Mengenwerte-Recall",
        "grits_values_precision":      "GriTS Mengenwerte-Precision",
        "title_similarity": "Ähnlichkeit des erkannten Titels",
        "footnote_similarity": "Ähnlichkeit der erkannten Fußnote",
    }
    ylabel = nice_labels.get(metric, metric)
    ax.set_xlabel('')  # X-Achse (Gruppe) ist selbsterklärend, keine Label nötig
    ax.set_ylabel(ylabel)
    # Titel: Methoden und Metrik
    title_metric = ylabel.split('(')[0].strip()  # ohne Einheit für Kürze
    ax.set_title(f'{method1_name} vs. {method2_name} – {title_metric}')
    ax.legend(title='Lauf', loc='best')
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/comparison_{method1_name}_{method2_name}_{title_metric}.png", dpi=300)  # Speichert den Plot als PNG-Datei
    plt.show()

# Dateilisten für LLM (GPT-4o) vs. traditionelle OCR Pipeline (je 3 Läufe)
files_llm_main_ingredients = [f"{WORKING_DIR}\\main\\ingredients_llm_main\\results_gpt4o_ingredients_run1.csv", f"{WORKING_DIR}\\main\\ingredients_llm_main\\results_gpt4o_ingredients_run2.csv", f"{WORKING_DIR}\\main\\ingredients_llm_main\\results_gpt4o_ingredients_run3.csv"]
files_ocr_main_ingredients = [f"{WORKING_DIR}\\main\\ingredients_trad_main\\ingredients_trad_main_run1.csv", f"{WORKING_DIR}\\main\\ingredients_trad_main\\ingredients_trad_main_run2.csv", f"{WORKING_DIR}\\main\\ingredients_trad_main\\ingredients_trad_main_run3.csv"]

files_llm_main_nutrition = [f"{WORKING_DIR}\\main\\nutrition_llm_main\\nutrition_llm_main_run1.csv", f"{WORKING_DIR}\\main\\nutrition_llm_main\\nutrition_llm_main_run2.csv", f"{WORKING_DIR}\\main\\nutrition_llm_main\\nutrition_llm_main_run3.csv"]
files_ocr_main_nutrition = [f"{WORKING_DIR}\\main\\nutrition_trad_main\\nutrition_trad_main_run1.csv", f"{WORKING_DIR}\\main\\nutrition_trad_main\\nutrition_trad_main_run2.csv", f"{WORKING_DIR}\\main\\nutrition_trad_main\\nutrition_trad_main_run3.csv"]

files_ocr_no_preprocessing_nutrition = [f"{WORKING_DIR}\\Ablations\\no-preprocessing\\nutrition_nopreprocessing\\eval_results_2025-09-17_14-40-32_traditional.csv", f"{WORKING_DIR}\\Ablations\\no-preprocessing\\nutrition_nopreprocessing\\eval_results_2025-09-18_02-21-59_traditional.csv", f"{WORKING_DIR}\\Ablations\\no-preprocessing\\nutrition_nopreprocessing\\nutrition_no_preprocess_lstm.csv"]
files_ocr_no_preprocessing_ingredients = [f"{WORKING_DIR}\\Ablations\\no-preprocessing\\ingredients_nopreprocessing\\eval_results_2025-09-17_14-43-13_traditional.csv", f"{WORKING_DIR}\\Ablations\\no-preprocessing\\ingredients_nopreprocessing\\eval_results_2025-09-18_02-24-42_traditional.csv", f"{WORKING_DIR}\\Ablations\\no-preprocessing\\ingredients_nopreprocessing\\tesseract_without_preprocess.csv"]

files_llm_on_yolo_ingredients = [f"{WORKING_DIR}\\Ablations\\llm-on-yolo-crop\\ingredients_llm_on_yolo\\eval_results_2025-09-17_22-03-27_llm.csv", f"{WORKING_DIR}\\Ablations\\llm-on-yolo-crop\\ingredients_llm_on_yolo\\eval_results_2025-09-18_03-51-59_llm.csv", f"{WORKING_DIR}\\Ablations\\llm-on-yolo-crop\\ingredients_llm_on_yolo\\eval_results_2025-09-19_00-59-25_llm.csv"]
files_llm_on_yolo_nutrition = [f"{WORKING_DIR}\\Ablations\\llm-on-yolo-crop\\nutrition_llm_on_yolo\\eval_results_2025-09-17_21-43-45_llm.csv", f"{WORKING_DIR}\\Ablations\\llm-on-yolo-crop\\nutrition_llm_on_yolo\\eval_results_2025-09-18_03-34-52_llm.csv", f"{WORKING_DIR}\\Ablations\\llm-on-yolo-crop\\nutrition_llm_on_yolo\\eval_results_2025-09-19_00-40-19_llm.csv"]

files_gpt4omini_ingredients = [f"{WORKING_DIR}\\Ablations\\mini-modell\\gpt4omini\\eval_results_2025-09-19_00-09-29_llm.csv", f"{WORKING_DIR}\\Ablations\\gpt-4o-mini\\ingredients_gpt4omini\\eval_results_2025-09-18_19-57-06_llm.csv", f"{WORKING_DIR}\\Ablations\\mini-modell\\gpt4omini\\eval_results_2025-09-19_00-12-06_llm.csv"]

files_ocr_psm7_ingredients = [f"C:\\Users\\maweo\\OneDrive - Universität Münster\\Dokumente\\Studium\\Semester 6\\Bachelorarbeit\\eval_results\\Ablations\\psm-modes\\ingredients_psm7_run1\\eval_results_2025-09-18_00-07-31_traditional.csv"]
files_ocr_psm11_ingredients = [f"C:\\Users\\maweo\\OneDrive - Universität Münster\\Dokumente\\Studium\\Semester 6\\Bachelorarbeit\\eval_results\\Ablations\\psm-modes\\ingredients_psm11_run1\\eval_results_2025-09-18_00-45-26_traditional.csv"]
files_ocr_psm6_nutrition = [f"C:\\Users\\maweo\\OneDrive - Universität Münster\\Dokumente\\Studium\\Semester 6\\Bachelorarbeit\\eval_results\\Ablations\\psm-modes\\nutrition_psm6_run1\\eval_results_2025-09-18_01-09-38_traditional.csv"]

# # F1-Score Vergleich (LLM vs OCR) – ohne Totalausfälle, mit Kaltstart-Entfernung
# plot_comparison(files_llm, files_ocr, metric="f1", 
#                 method1_name="LLM (GPT-4o)", method2_name="OCR (Tesseract)", 
#                 remove_first_n=0, exclude_failures=False)

# boxplot_metric_over_runs(
#     files = files_llm_main_nutrition,
#     metric = "overall_table_score",
#     method_name = "mLLM (GPT-4o)",
#     remove_first_n = 0,
#     exclude_failures = False,
#     ylabel = "Overall Table Score",
#     title = "mLLM (GPT-4o) — Overall Table Score für Nährwerttabellen",
# )

# result = stats_metric_over_runs(
#     files = files_llm_on_yolo_ingredients,
#     metric = "wer",
#     remove_first_n = 0,
#     exclude_failures = False
# )

# print("Ergebnisse WER LLM on YOLO:")
# print(result)

# result = stats_metric_over_runs(
#     files = files_llm_on_yolo_ingredients,
#     metric = "cer",
#     remove_first_n = 0,
#     exclude_failures = False
# )

# print("Ergebnisse CER LLM on YOLO:")
# print(result)

# result = stats_metric_over_runs(
#     files = files_llm_main_ingredients,
#     metric = "wer",
#     remove_first_n = 0,
#     exclude_failures = False
# )

# print("Ergebnisse WER LLM on Main Ingredients:")
# print(result)

result = stats_metric_over_runs(
    files = files_llm_main_ingredients,
    metric = "time_total",
    remove_first_n = 0,
    exclude_failures = True
)

print("Ergebnisse CER LLM on Main Ingredients:")
print(result)

# result = stats_metric_over_runs(
#     files = files_llm_on_yolo_nutrition,
#     metric = "grits_content_overall_f1",
#     remove_first_n = 0,
#     exclude_failures = True
# )

# print("Ergebnisse GriTS Content F1 LLM on YOLO Nutrition:")
# print(result)

# result = stats_metric_over_runs(
#     files = files_llm_on_yolo_nutrition,
#     metric = "grits_content_overall_f1",
#     remove_first_n = 0,
#     exclude_failures = False
# )

# print("Ergebnisse GriTS Content F1 LLM on YOLO Nutrition MIT Totalausfälle:")
# print(result)

# result = stats_metric_over_runs(
#     files = files_llm_on_yolo_nutrition,
#     metric = "grits_topology_f1",
#     remove_first_n = 0,
#     exclude_failures = True
# )

# print("Ergebnisse GriTS Topology F1 LLM on YOLO Nutrition:")
# print(result)

# result = stats_metric_over_runs(
#     files = files_llm_on_yolo_nutrition,
#     metric = "grits_topology_f1",
#     remove_first_n = 0,
#     exclude_failures = False
# )

# print("Ergebnisse GriTS Topology F1 LLM on YOLO Nutrition MIT Totalausfälle:")
# print(result)

# result = stats_metric_over_runs(
#     files = files_llm_main_nutrition,
#     metric = "grits_content_overall_f1",
#     remove_first_n = 0,
#     exclude_failures = True
# )

# print("Ergebnisse GriTS Content F1 Main Nutrition:")
# print(result)

# result = stats_metric_over_runs(
#     files = files_llm_main_nutrition,
#     metric = "grits_topology_f1",
#     remove_first_n = 0,
#     exclude_failures = True
# )

# print("Ergebnisse GriTS Topology F1 Main Nutrition:")
# print(result)

# result = stats_metric_over_runs(
#     files = files_llm_on_yolo_nutrition,
#     metric = "time_total",
#     remove_first_n = 0,
#     exclude_failures = True
# )

# print("Ergebnisse Time total LLM on YOLO Nutrition:")
# print(result)

# result = stats_metric_over_runs(
#     files = files_llm_main_nutrition,
#     metric = "time_total",
#     remove_first_n = 0,
#     exclude_failures = True
# )

# print("Ergebnisse Time total Main Nutrition:")
# print(result)

# result = stats_metric_over_runs(
#     files = files_llm_on_yolo_nutrition,
#     metric = "total_cost",
#     remove_first_n = 0,
#     exclude_failures = True
# )

# print("Ergebnisse Total Cost LLM on YOLO Nutrition:")
# print(result)


# result = stats_metric_over_runs(
#     files = files_llm_main_nutrition,
#     metric = "total_cost",
#     remove_first_n = 0,
#     exclude_failures = False
# )

# print("Ergebnisse TTotal Cost Main Nutrition:")
# print(result)

# result = stats_metric_over_runs(
#     files = files_llm_on_yolo_ingredients,
#     metric = "grits_content_overall_f1",
#     remove_first_n = 0,
#     exclude_failures = False
# )

# print("Ergebnisse CV With Preprocessing Content:")
# print(result)

# result = stats_metric_over_runs(
#     files = files_ocr_main_nutrition,
#     metric = "iou",
#     remove_first_n = 0,
#     exclude_failures = False
# )

# print("Ergebnisse iou Nutrition:")
# print(result)

# result = stats_metric_over_runs(
#     files = files_ocr_main_ingredients,
#     metric = "iou",
#     remove_first_n = 0,
#     exclude_failures = False
# )

# print("Ergebnisse iou Ingredients:")
# print(result)

# grouped_boxplot_two_methods_two_tasks(
#     method1_runs={"Ingredients": files_ocr_main_ingredients, "Nutrition": files_ocr_main_nutrition},
#     method2_runs={"Ingredients": files_llm_main_ingredients,  "Nutrition": files_llm_main_nutrition},
#     metric="f1_overall", method1_name="Traditional OCR", method2_name="LLM (GPT-4)",
#     combine_runs=True, remove_first_n=5, exclude_failures=True,
#     figsize=(8,5), method_gap=1.5, box_width=0.25, showfliers=True,
#     ylabel="F1-Score", title="Traditional vs LLM — F1 by Task"
# )

# # Verarbeitungszeit-Verteilung Vergleich (inkl. Kaltstart, da remove_first_n=0)
# plot_comparison(files_llm, files_ocr, metric="time_total", 
#                 method1_name="LLM (GPT-4o)", method2_name="OCR (Tesseract)", 
#                 remove_first_n=0, exclude_failures=False)

# boxplot_two_methods_metrics_grouped_by_method_one_task(
#     method1_runs={"Ingredients": files_ocr_main_nutrition, "Nutrition": files_ocr_main_nutrition},
#     method2_runs={"Ingredients": files_ocr_psm6_nutrition, "Nutrition": files_ocr_psm6_nutrition},
#     metrics=["grits_content_overall_f1", "grits_topology_f1", "overall_table_score"],
#     task="Nutrition",
#     method1_name="psm 4", method2_name="psm 6",
#     remove_first_n=0, exclude_failures=False,
#     combine_runs=True,
#     title="Vergleich der OCR-PSM-Modi (Nährwerttabellen)",
#     ylabel="Score (0-1)",
#     ylimit=(0,1.1)
# )


bin_edges = np.linspace(0, 1, 11)  # 11 gleiche Bins

# out = metric_distribution_one_method(
#     files=files_ocr_main_ingredients,
#     metric="wer",
#     bin_edges=bin_edges,
#     method_name="CV-Pipeline",
#     task_label="Ingredients",
#     remove_first_n=0, exclude_failures=False,  # Totalausfälle ggf. drinlassen
#     normalize=False, cumulative=False,
#     title="Absolute Verteilung der WER-Werte (CV-Pipeline)",
#     value_points=[0.0],    # <- explizit WER==0 zählen
# )

# print(out["points"])       # Tabelle mit Count (und Prozent, falls normalize=True)
cols = pastel_palette(2, "Pastel1")

# metric_distribution_two_methods(
#     method1_files=files_ocr_main_nutrition,
#     method2_files=files_llm_main_nutrition,
#     metric="overall_table_score",
#     method1_name="CV-Pipeline",
#     method2_name="mLLM-Pipeline",
#     task_label="Ingredients",
#     remove_first_n=0, exclude_failures=True,   # Totalausfälle ggf. drinlassen
#     normalize=True, layout="overlay_ordered",
#     title="Relative Verteilung des Overall Table Scores nach Methode",
#     bin_edges=bin_edges,
#     front_shrink=0.9,
#     cumulative=True,
#     colors=cols)

# boxplot_time_with_without_failures_two_methods(
#     method1_files=files_ocr_main_nutrition,   # Traditional – Nutrition (3 CSVs)
#     method2_files=files_llm_main_nutrition,   # LLM – Nutrition (3 CSVs)
#     metric_time="time_total",
#     method1_name="CV-Pipeline",
#     method2_name="mLLM-Pipeline",
#     task_label="Nutrition",
#     remove_first_n=0,      # Kaltstart entfernen
#     combine_runs=True,           # je Subgruppe eine Box
#     colors=cols,
#     ylabel="Wall-Clock-Time (s)",
#     title="Wall-Clock-Time mit und ohne Totalausfällen",
# )

# multi_metric_by_task_single_method(
#     method_runs={"Nutrition": files_ocr_main_nutrition, "Ingredients": files_ocr_main_ingredients},
#     metrics=["time_yolo", "time_preproc", "time_ocr", "time_postproc", "time_total"],
#     method_name="CV-Pipeline",
#     tasks_order=("Nutrition",),
#     exclude_failures=True,
#     remove_first_n=0,
#     ylabel="Zeit (s)",
#     title="Zeitaufschlüsselung der CV-Pipeline für Nährwerttabellen",
#     xtitle="Schritte der CV-Pipeline"
# )

# memory_peak_over_iterations_three_methods(
#     method1_files=files_ocr_main_nutrition,
#     method2_files=files_llm_main_nutrition,
#     method3_files=files_ocr_no_preprocessing_nutrition,
#     method1_name="CV-Pipeline",
#     method2_name="mLLM-Pipeline",
#     method3_name="CV-Pipeline no Preprocessing",
#     task_label="Nutrition",
#     remove_first_n=0,
#     exclude_failures=False,
#     cumulative=True,
#     combine_runs=True,
#     style="line",
#     band="minmax",
#     colors=("tab:blue","tab:orange","tab:green"),
#     ylabel="Memory Peak (MB)", 
#     title="Running Maximum des Spitzen-Speicherbedarfs"
# )

# memory_peak_over_iterations_two_methods(
#     method1_files=files_ocr_main_nutrition,
#     method2_files=files_llm_main_nutrition,
#     method1_name="CV-Pipeline",
#     method2_name="mLLM-Pipeline",
#     task_label="Nutrition",
#     remove_first_n=0,
#     exclude_failures=False,
#     combine_runs=True,
#     style="line",
#     band="minmax",
#     colors=("tab:blue","tab:orange"),
#     ylabel="Memory Peak (MB)",
#     title="Running Maximum des Spitzen-Speicherbedarfs"
# )