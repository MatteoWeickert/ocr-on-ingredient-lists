# evaluate.py

import os
import csv
import json
import time, resource, psutil
import re
from typing import List, Dict, Tuple, Any, Set, Counter
from pathlib import Path
from collections import defaultdict
from datetime import datetime
import pandas as pd
from ultralytics import YOLO
from jiwer import wer, cer
from contextlib import contextmanager
import difflib

# Importieren der ausgelagerten Pipeline-Funktionen
from traditional_pipeline import run_traditional_pipeline
from llm_pipeline import run_llm_pipeline

@contextmanager
def timer(stats: dict, key: str):
    t0 = time.perf_counter()
    try:
        yield
    finally:
        stats[key] = stats.get(key, 0.0) + (time.perf_counter() - t0)

def extract_json_from_llm_output(llm_str):
    """
    Extrahiert JSON-Inhalt aus einer LLM-Antwort, die in Markdown-Code-Blöcken eingebettet ist.
    Unterstützt verschiedene Formate wie ```json oder ``` ohne Sprach-Tag.
    """
    if not llm_str or not isinstance(llm_str, str):
        return llm_str
    
    # Pattern für ```json ... ``` oder ``` ... ```
    pattern = r'```(?:json)?\s*\n?(.*?)\n?```'
    match = re.search(pattern, llm_str, re.DOTALL | re.IGNORECASE)
    
    if match:
        return match.group(1).strip()
    
    # Fallback: Wenn keine Code-Blöcke gefunden werden, den Original-String zurückgeben
    return llm_str.strip()

def compact_json_str(json_str):
    """
    Konvertiert einen JSON-String zu einem kompakten Format.
    Behandelt sowohl rohe JSON-Strings als auch Code-Block-eingebettete JSON-Strings.
    """
    if isinstance(json_str, dict):
        return json.dumps(json_str, ensure_ascii=False, separators=(',', ':'))
    try:
        raw = extract_json_from_llm_output(json_str)
        if not raw or raw.strip() == "":
            return ""
        
        # Versuche JSON zu parsen
        dict_obj = json.loads(raw)
        return json.dumps(dict_obj, ensure_ascii=False, separators=(',', ':'))
    except json.JSONDecodeError as e:
        print(f"WARN: JSON-Parse-Fehler: {e}")
        print(f"Roher Inhalt: '{raw[:100]}...' (erste 100 Zeichen)")
        return ""
    except Exception as e:
        print(f"WARN: Allgemeiner Fehler beim JSON-Processing: {e}")
        return ""

def json_for_cell(obj: Any) -> str:
    """
    Gibt obj als kompaktes JSON (eine Zeile) zurück.
    Keine Newlines, Unicode bleibt erhalten.
    """
    try:
        return json.dumps(obj, ensure_ascii=False, separators=(',', ':'))
    except Exception:
        return ""

def evaluate_nutrition_table_structure(gt_table: Dict, ocr_table: Dict, method_name: str):
    """
    Führt einen strukturbasierten Vergleich zwischen einer Ground-Truth- und einer OCR-Nährwerttabelle durch.

    Args:
        gt_table: Das Ground-Truth-Dict für die Nährwerttabelle.
        ocr_table: Das von der OCR-Pipeline extrahierte Dict.

    Returns:
        Ein Dict mit detaillierten Evaluationsmetriken.
    """

    def _normalize_text(text: str) -> str:
        """Bereinigt und normalisiert einen String für den Vergleich."""
        if not isinstance(text, str):
            return ""
        # Kleinschreibung, Entfernung von Leerzeichen am Rand und Reduzierung von Leerzeichenfolgen
        return re.sub(r'\s+', ' ', text).lower().strip()

    # Sicherstellen, dass die Eingabedaten gültige Diktionäre sind
    gt = gt_table if isinstance(gt_table, dict) else {}
    ocr = ocr_table if isinstance(ocr_table, dict) else {}

    metrics = {}

    # 1. Metadaten-Vergleich (Titel & Fußnote)
    gt_title = _normalize_text(gt.get("title", ""))
    ocr_title = _normalize_text(ocr.get("title", ""))
    metrics[f"title_similarity_{method_name}"] = difflib.SequenceMatcher(None, gt_title, ocr_title).ratio()

    gt_footnote = _normalize_text(gt.get("footnote", ""))
    ocr_footnote = _normalize_text(ocr.get("footnote", ""))
    metrics[f"footnote_similarity_{method_name}"] = difflib.SequenceMatcher(None, gt_footnote, ocr_footnote).ratio()

    # 2. Spalten-Vergleich
    gt_columns = [_normalize_text(col) for col in gt.get("columns", [])]
    ocr_columns = [_normalize_text(col) for col in ocr.get("columns", [])]
    metrics[f"column_counts_{method_name}"] = {"gt": len(gt_columns), f"{method_name}": len(ocr_columns)}

    # A. Struktureller Abgleich der Spaltenlisten via Sequenz-Matcher
    list_matcher = difflib.SequenceMatcher(None, gt_columns, ocr_columns)
    metrics[f"column_structural_similarity_{method_name}"] = list_matcher.ratio()

    # Berechne die Ähnlichkeit der Spaltennamen
    matched_column_similarities = []
    for block in list_matcher.get_matching_blocks():
        if block.size == 0: continue # Ignoriere den finalen Dummy-Block
        
        # Extrahiere die Paare von übereinstimmenden Spaltennamen
        gt_matched_cols = gt_columns[block.a : block.a + block.size] # Übereinstimmung in GT
        ocr_matched_cols = ocr_columns[block.b : block.b + block.size] # Übereinstimmung in OCR

        for gt_col, ocr_col in zip(gt_matched_cols, ocr_matched_cols):
            string_similarity = difflib.SequenceMatcher(None, gt_col, ocr_col).ratio()
            matched_column_similarities.append(string_similarity)

    if matched_column_similarities:
        metrics[f"avg_matched_column_name_similarity_{method_name}"] = sum(matched_column_similarities) / len(matched_column_similarities)
    else:
        metrics[f"avg_matched_column_name_similarity_{method_name}"] = 0.0

    # 3. Zeilen-Vergleich
    gt_rows = gt.get("rows", [])
    ocr_rows = ocr.get("rows", [])

    # Erstelle Maps für einen einfachen Zugriff über das normalisierte Label
    gt_row_map = {_normalize_text(row.get("label", "")): row.get("values", []) for row in gt_rows if _normalize_text(row.get("label", ""))}
    ocr_row_map = {_normalize_text(row.get("label", "")): row.get("values", []) for row in ocr_rows if _normalize_text(row.get("label", ""))}

    gt_labels = set(gt_row_map.keys())
    ocr_labels = set(ocr_row_map.keys())

    # Finde übereinstimmende, fehlende und zusätzliche Labels
    matched_labels = gt_labels.intersection(ocr_labels)
    missing_labels = gt_labels.difference(ocr_labels) # In GT, aber nicht in OCR
    extra_labels = ocr_labels.difference(gt_labels)   # In OCR, aber nicht in GT

    metrics[f"row_counts_{method_name}"] = {
    "gt": len(gt_labels),
    "ocr": len(ocr_labels),
    "matched": len(matched_labels),
    "missing_in_pred": len(missing_labels),
    "extra_in_pred": len(extra_labels)
    }

    metrics[f"missing_labels_{method_name}"] = sorted(list(missing_labels))
    metrics[f"extra_labels_{method_name}"] = sorted(list(extra_labels))

    # Detail-Analyse für übereinstimmende Zeilen
    row_details = []
    row_content_score = []

    for label in sorted(list(matched_labels)):
        gt_values = [_normalize_text(v) for v in gt_row_map[label]]
        ocr_values = [_normalize_text(v) for v in ocr_row_map[label]]
        
        row_eval = {
            "label": label,
            "gt_value_count": len(gt_values),
            "pred_value_count": len(ocr_values) 
        }

        # Vergleiche Zelleninhalte nur, wenn die Spaltenanzahl übereinstimmt
        if len(gt_values) == len(ocr_values):
            # Fall 1: Spaltenanzahl stimmt überein -> Zelle-für-Zelle-Vergleich
            row_eval["comparison_method"] = "cell_by_cell"
            sims = [difflib.SequenceMatcher(None, gt_v, ocr_v).ratio() for gt_v, ocr_v in zip(gt_values, ocr_values)]
            row_eval["cell_similarities"] = sims
            # Der Score für diese Zeile ist der Durchschnitt der Zell-Ähnlichkeiten
            if sims:
                row_content_score.append(sum(sims) / len(sims))

        else:
            # Fall 2: Spaltenanzahl stimmt NICHT überein -> String-Verkettungs-Fallback
            row_eval["comparison_method"] = "string_concatenation_fallback"
            gt_row_string = " ".join(gt_values)
            ocr_row_string = " ".join(ocr_values)
            similarity = difflib.SequenceMatcher(None, gt_row_string, ocr_row_string).ratio()
            row_eval["content_similarity_fallback"] = similarity
            # Der Score für diese Zeile ist die Ähnlichkeit des Gesamtstrings
            row_content_score.append(similarity)

        row_details.append(row_eval)

    metrics[f"row_details_{method_name}"] = json_for_cell(row_details)
    if row_content_score:
        metrics[f"avg_content_similarity_matched_rows_{method_name}"] = sum(row_content_score) / len(row_content_score)
    else:
        metrics[f"avg_content_similarity_matched_rows_{method_name}"] = 0.0

    # 4. Gesamtergebnis berechnen
    weights = {
        "title": 0.05,
        "columns": 0.15,
        "rows_structure": 0.30,
        "rows_content": 0.50
    }

    # Score für die Zeilenstruktur (Precision/Recall der Labels)
    total_unique_labels = len(gt_labels.union(ocr_labels))
    row_structure_score = len(matched_labels) / total_unique_labels if total_unique_labels > 0 else 1.0

    # Score für die Spalten
    column_score = (metrics[f"column_structural_similarity_{method_name}"] * 0.5) + (metrics[f"avg_matched_column_name_similarity_{method_name}"] * 0.5)

    overall_score = (
        metrics[f"title_similarity_{method_name}"] * weights["title"] +
        column_score * weights["columns"] +
        row_structure_score * weights["rows_structure"] +
        metrics[f"avg_content_similarity_matched_rows_{method_name}"] * weights["rows_content"]
    )
    metrics[f"overall_structural_score_{method_name}"] = overall_score

    return metrics

def transform_dict_to_string(dict_obj: dict, class_filter: str) -> str:
    """
    Konvertiert ein Dictionary in einen normalisierten String für Metriken wie WER/CER.
    """

    if isinstance(dict_obj, str):
        if dict_obj.strip() == "":
            return ""
        dict_obj = json.loads(dict_obj)

    if not isinstance(dict_obj, dict):
        return str(dict_obj)

    if class_filter == "ingredients":
        # Extrahiert den reinen Zutaten-Text
        return dict_obj.get("ingredients_text", "").strip()

    elif class_filter == "nutrition":
        # Baut aus der Nährwerttabelle einen sinnvollen, vergleichbaren String
        if "nutrition_table" in dict_obj:
            dict_obj = dict_obj.get("nutrition_table", {})
        
        parts = []
        if dict_obj.get("title"):
            parts.append(dict_obj["title"])
        
        # Spaltenüberschriften hinzufügen
        if dict_obj.get("columns"):
            parts.append(" ".join(dict_obj["columns"]))

        # Jede Zeile als "Label Wert1 Wert2..." hinzufügen
        if "rows" in dict_obj:
            for row in dict_obj.get("rows", []):
                label = row.get("label", "")
                values = " ".join(row.get("values", []))
                parts.append(f"{label} {values}")

        if dict_obj.get("footnote"):
            parts.append(dict_obj["footnote"])

        # Alles zu einem einzigen, normalisierten String verbinden
        return re.sub(r"\s+", " ", " ".join(filter(None, parts))).strip()

    return "" # Fallback für unbekannte Klassen

def load_config() -> Dict:
    """Lädt die Konfiguration aus der config.json."""

    cfg_path = Path(__file__).with_name("config.json")
    if not cfg_path.exists():
        raise FileNotFoundError(f"Konfigurationsdatei nicht gefunden: {cfg_path}")
    
    with cfg_path.open("r", encoding="utf-8") as f:
        cfg = json.load(f)

    required_keys = ["img_dir", "gt_json", "model", "class_filter", "out_dir", "traditional_script", "llm_script"]
    for k in required_keys:
        if k not in cfg or not cfg[k]:
            raise ValueError(f"Schlüssel '{k}' fehlt oder ist leer in der config.json")
        
    return cfg

def group_images_by_product(image_paths: List[Path]) -> Dict[str, List[Path]]:
    """Gruppiert Bilder basierend auf der Produkt-ID im Dateinamen."""
    groups = defaultdict(list)
    for image_path in image_paths:
        product_id = image_path.name.split('_')[0]
        groups[product_id].append(image_path)
    return groups

def load_ground_truth(gt_json: Path) -> Dict:
    """Lädt die Ground-Truth-Daten aus einer JSON-Datei."""
    if not gt_json.exists():
        raise FileNotFoundError(f"Ground-Truth-Datei nicht gefunden: {gt_json}")
    with gt_json.open("r", encoding="utf-8") as f:
        return json.load(f)

def load_images(img_dir: Path) -> List[Path]:
    """Lädt alle Bildpfade aus einem Verzeichnis."""
    return sorted([img_dir / f for f in os.listdir(img_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))])

def calculate_iou(box_gt: List[Dict], box_pred: List[int]) -> float:
    """Berechnet die Intersection over Union (IoU) für zwei Bounding Boxes."""
    if not box_gt or not box_pred:
        return 0.0

    max_iou = 0.0

    x1_pred, y1_pred, x2_pred, y2_pred = box_pred

    for box in box_gt:
        x1_gt = box.get("left")
        y1_gt = box.get("top")
        x2_gt = x1_gt + box.get("width")
        y2_gt = y1_gt + box.get("height")

        # Koordinaten der Schnittmenge berechnen
        x1_inter = max(x1_gt, x1_pred)
        y1_inter = max(y1_gt, y1_pred)
        x2_inter = min(x2_gt, x2_pred)
        y2_inter = min(y2_gt, y2_pred)

        inter_width = max(0, x2_inter - x1_inter)
        inter_height = max(0, y2_inter - y1_inter)
        inter_area = inter_width * inter_height

        # Flächen der beiden Boxen berechnen
        gt_area = (x2_gt - x1_gt) * (y2_gt - y1_gt)
        pred_area = (x2_pred - x1_pred) * (y2_pred - y1_pred)

        union_area = gt_area + pred_area - inter_area

        iou = inter_area / union_area if union_area > 0 else 0.0

        if iou > max_iou:
            max_iou = iou

    return max_iou

def calculate_word_level_metrics(gt_text: str, pred_text: str, gt_object: Dict, pred_object: Dict, class_filter: str) -> Dict[str, float | str]:
    """Berechnet Precision, Recall und F1-Score auf Wortebene für Zutatenlisten auf allen Wörtern und für Nährwerttabellen auf allen Wörtern,
    auf Elementen, auf Werten, auf (Element, Wert)-Paaren und auf (Element, Spaltenname, Wert)-Tripeln. """

    if (not gt_text and not pred_text) or (not gt_object and not pred_object):
        return {"precision": 1.0, "recall": 1.0, "f1": 1.0}
    if (not gt_words or not pred_words) or (not gt_object or not pred_object):
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0}
    
    if class_filter == "ingredients":

        gt_words = Counter(gt_text.split())
        pred_words = Counter(pred_text.split())

        # Berechnung der Metriken auf Wortebene
        metrics = _calculate_precision_recall_f1_multiset(gt_words, pred_words)
        return metrics
    
    elif class_filter == "nutrition":
        # Extrahiere die relevanten Mengen aus den Dictionaries
        elements_gt = _elements_from_table(gt_object)
        elements_pred = _elements_from_table(pred_object)
        values_gt = _values_from_table(gt_object)
        values_pred = _values_from_table(pred_object)
        pairs_gt = _element_value_pairs(gt_object)
        pairs_pred = _element_value_pairs(pred_object)
        triples_gt = _element_column_value_triplets(gt_object)
        triples_pred = _element_column_value_triplets(pred_object)

        # Berechnung der Metriken auf verschiedenen Ebenen
        metrics_elements = _calculate_precision_recall_f1_set(elements_gt, elements_pred)
        metrics_values = _calculate_precision_recall_f1_multiset(values_gt, values_pred)
        metrics_pairs = _calculate_precision_recall_f1_set(pairs_gt, pairs_pred)
        metrics_triples = _calculate_precision_recall_f1_set(triples_gt, triples_pred)

        # Gesamtmetriken auf Wortebene
        gt_words = Counter(gt_text.split())
        pred_words = Counter(pred_text.split())
        metrics_words = _calculate_precision_recall_f1_multiset(gt_words, pred_words)

        # Alle Metriken zusammenführen
        all_metrics = {
            "precision_elements": metrics_elements["precision"],
            "recall_elements": metrics_elements["recall"],
            "f1_elements": metrics_elements["f1"],
            "precision_values": metrics_values["precision"],
            "recall_values": metrics_values["recall"],
            "f1_values": metrics_values["f1"],
            "precision_pairs": metrics_pairs["precision"],
            "recall_pairs": metrics_pairs["recall"],
            "f1_pairs": metrics_pairs["f1"],
            "precision_triples": metrics_triples["precision"],
            "recall_triples": metrics_triples["recall"],
            "f1_triples": metrics_triples["f1"],
            "precision_words": metrics_words["precision"],
            "recall_words": metrics_words["recall"],
            "f1_words": metrics_words["f1"]
        }
        return all_metrics

    else:   
        return {"precision": "NaN", "recall": "NaN", "f1": "NaN"}

def _calculate_precision_recall_f1_set(gt: Set, pred: Set) -> Dict[str, float]:
    """Berechnet Precision, Recall und F1-Score aus True Positives, False Positives und False Negatives."""
    tp = len(gt & pred)
    fp = len(pred - gt)
    fn = len(gt - pred)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    return {"precision": precision, "recall": recall, "f1": f1}

def _calculate_precision_recall_f1_multiset(gt: Counter, pred: Counter) -> Dict[str, float]:
    """Berechnet Precision, Recall und F1-Score aus True Positives, False Positives und False Negatives."""
    tp = sum((gt & pred).values())
    fp = sum((pred - gt).values())
    fn = sum((gt - pred).values())
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    return {"precision": precision, "recall": recall, "f1": f1}

def _elements_from_table(tbl: Dict) -> Set[str]:
    """Menge der Elemente (Zeilen-Labels)."""
    rows = tbl.get("rows", []) if isinstance(tbl, dict) else []
    return {r.get("label","") for r in rows if r.get("label","")}

def _values_from_table(table: Dict) -> Counter[str]:
    """Hier Multimenge, da Values doppelt vorkommen können - Menge der Werte (aus allen Zeilen)."""
    rows = table.get("rows", []) if isinstance(table, dict) else []
    vals = []
    for row in rows:
        for v in (row.get("values") or []):
            if v:
                vals.append(v)
    return Counter(vals)

def _element_value_pairs(table: Dict) -> Set[Tuple[str, str]]:
    """Menge der (Element, Wert)-Paare (alle Werte über alle Spalten)."""
    pairs = set()
    rows = table.get("rows", []) if isinstance(table, dict) else []
    for r in rows:
        label = r.get("label","")
        if not label: 
            continue
        for value in r.get("values", []) or []:
            v = value
            if v:
                pairs.add((label, v))
    return pairs

def _element_column_value_triplets(table: Dict) -> Set[Tuple[str, str, str]]:
    """Menge der (Element, Spaltenname, Wert)-Tripel - falls Spalten vorhanden."""
    trips = set()
    if not isinstance(table, dict):
        return trips
    cols = [c for c in (table.get("columns") or [])]
    rows = table.get("rows", []) or []
    for row in rows:
        label = row.get("label","")
        vals = [v for v in (row.get("values") or [])]
        if not label or not cols or not vals:
            continue
        for i in range(len(cols)):
            if vals[i]:
                trips.add((label, cols[i], vals[i]))
    return trips


def main():
    """
    Hauptfunktion zur Steuerung des Evaluationsprozesses.
    """

    cfg = load_config()
    print("Konfiguration geladen:", cfg)

    img_dir = Path(cfg["img_dir"])
    gt_json = Path(cfg["gt_json"])
    model_path = Path(cfg["model"])
    class_filter = cfg["class_filter"]

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    out_base = Path(cfg["out_dir"])
    new_out_dir = out_base / f"{timestamp}_{class_filter}"

    new_out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Starte Evaluation für Klasse: '{class_filter}'...")

    gt_data = load_ground_truth(gt_json)
    image_paths = load_images(img_dir)
    grouped_images = group_images_by_product(image_paths)
    
    model = YOLO(str(model_path))
    class_name_to_id = {name: i for i, name in model.names.items()}
    if class_filter not in class_name_to_id:
        raise ValueError(f"Klasse '{class_filter}' nicht im Modell gefunden. Verfügbare Klassen: {list(model.names.values())}")
    target_id = class_name_to_id[class_filter]

    results = []

    for product_id, paths in grouped_images.items():
        print(f"\n--- Verarbeite Produkt: {product_id} ---")

        # Ground Truth für dieses Produkt extrahieren
        gt_item = next((item for item in gt_data if str(item.get("produktnr")) == product_id), None)
        gt_text = ""
        gt_bbox = []
        gt_object = {}
        dict_metrics_ocr = {}
        dict_metrics_llm = {}
        ocr_string = ""
        llm_string = ""

        if gt_item:
            gt_object = gt_item.get("text", {})
            gt_object_compact = compact_json_str(gt_object)
            gt_text = transform_dict_to_string(gt_object, class_filter)
            gt_bbox = gt_item.get("bbox", [])
        else:
             print(f"Warnung: Keine GT-Daten für Produkt {product_id} gefunden.")

        ###################################################
        # --- Traditionelle Pipeline ---
        ###################################################
        process = psutil.Process(os.getpid())
        cpu_start = process.cpu_times()
        time_start_trad = time.perf_counter()

        trad_result = run_traditional_pipeline(model, paths, target_id, new_out_dir, product_id)
        
        time_end_trad = time.perf_counter()
        cpu_end = process.cpu_times()

        # Metriken sammeln
        end_to_end_time_trad = time_end_trad - time_start_trad
        cpu_trad_time = (cpu_end.user - cpu_start.user) + (cpu_end.system - cpu_start.system) # tatsächliche CPU-Zeit
        # Peak-RAM (Windows): peak working set
        mi_full = process.memory_full_info()
        peak_mb = getattr(mi_full, "peak_wset", 0) / (1024*1024)

        # Text auslesen
        if trad_result["structured_data"]:
            ocr_string = transform_dict_to_string(trad_result["structured_data"], class_filter)
        if trad_result["structured_data"] and trad_result["structured_data"].get("nutrition_table"):
            dict_metrics_ocr = evaluate_nutrition_table_structure(gt_object, trad_result["structured_data"]["nutrition_table"], "ocr")
            trad_result_compact = compact_json_str(trad_result["structured_data"].get("nutrition_table", {}))

        if trad_result["structured_data"] and trad_result["structured_data"].get("raw_text"):
            ocr_raw = trad_result["structured_data"].get("raw_text", "")
        else:
            ocr_raw = ""
        if trad_result["yolo_result"]:
            yolo_res = trad_result["yolo_result"]
        if trad_result["times"]:
            trad_times = trad_result["times"]
            time_yolo = trad_times.get("yolo_total", 0.0)
            time_ocr = trad_times.get("ocr", 0.0)
            time_postproc = trad_times.get("postprocessing", 0.0)
            time_preproc = trad_times.get("crop-preprocessing", 0.0)
        print(f"Zeit (Traditionell): {end_to_end_time_trad:.2f}s")
        
        ###################################################
        # --- LLM-Pipeline ---
        ###################################################
        cpu_start_llm = process.cpu_times()
        time_start_llm = time.perf_counter()

        llm_res_data = run_llm_pipeline([str(p) for p in paths], class_filter)

        time_end_llm = time.perf_counter()
        cpu_end_llm = process.cpu_times()

        # Metriken sammeln
        end_to_end_time_llm = time_end_llm - time_start_llm
        cpu_llm_time = (cpu_end_llm.user - cpu_start_llm.user) + (cpu_end_llm.system - cpu_start_llm.system) # tatsächliche CPU-Zeit
        mi_full = process.memory_full_info()
        peak_mb_llm = getattr(mi_full, "peak_wset", 0) / (1024*1024)

        if llm_res_data["text"]:
            llm_res = llm_res_data["text"]
        if llm_res_data["times"]:
            llm_times = llm_res_data["times"]
            time_preproc_llm = llm_times.get("preprocessing", 0.0)
            time_api_llm = llm_times.get("api_roundtrip", 0.0)
            time_postproc_llm = llm_times.get("postprocessing", 0.0)
        if class_filter == "nutrition":
            llm_res_compact = compact_json_str(llm_res)
            print("LLM-Antwort (kompakt):", llm_res_compact)
            print(f"Typ von llm_res_compact: {type(llm_res_compact)}")
            llm_string = transform_dict_to_string(llm_res_compact, class_filter)
            try:
                if isinstance(json.loads(llm_res_compact), dict):
                    dict_metrics_llm = evaluate_nutrition_table_structure(gt_object, json.loads(llm_res_compact), "llm")
            except json.JSONDecodeError as e:
                print(f"Fehler beim Parsen der LLM-Antwort: {e}")
                dict_metrics_llm = {}
        else:
            llm_string = llm_res
            
        print(f"Zeit (LLM): {end_to_end_time_llm:.2f}s")

        # --- Metriken berechnen ---
        pred_bbox = yolo_res["box"] if yolo_res else []
        iou = calculate_iou(gt_bbox, pred_bbox)

        metrics_ocr = calculate_word_level_metrics(gt_text, ocr_string, gt_object, trad_result.get("structured_data", {}), class_filter)
        # metrics_llm = calculate_word_level_metrics(gt_text, llm_string, gt_object, llm_res if isinstance(llm_res, dict) else {}, class_filter)

        row = {
            "product_id": product_id,
            "class_requested": class_filter,
            "gt_text": gt_text,
            
            # Traditionelle Pipeline
            "ocr_text": ocr_string,
            "ocr_raw": ocr_raw,
            "yolo_confidence": yolo_res["confidence"] if yolo_res else None,
            "iou": iou,
            "time_trad_s": end_to_end_time_trad,
            "cpu_trad_s": cpu_trad_time,
            "mem_trad_peak": peak_mb,
            "time_yolo_s": time_yolo,
            "time_ocr_s": time_ocr,
            "time_preproc_s": time_preproc,
            "time_postproc_s": time_postproc,
            "wer_ocr": wer(gt_text, ocr_string),
            "cer_ocr": cer(gt_text, ocr_string),
            "precision_ocr": metrics_ocr["precision"],
            "recall_ocr": metrics_ocr["recall"],
            "f1_ocr": metrics_ocr["f1"],
            
            # LLM Pipeline
            # "llm_text": llm_string,
            # "time_llm_s": time_llm,
            # "cpu_llm_s": cpu_llm_time,
            # "mem_llm_peak": peak_mb_llm,
            # "time_preproc_llm_s": time_preproc_llm,
            # "time_api_llm_s": time_api_llm,
            # "time_postproc_llm_s": time_postproc_llm,
            # "wer_llm": wer(gt_text, llm_string),
            # "cer_llm": cer(gt_text, llm_string),
            # "precision_llm": metrics_llm["precision"],
            # "recall_llm": metrics_llm["recall"],
            # "f1_llm": metrics_llm["f1"],
            # "llm_total_tokens": llm_res_data["total_tokens"],
            # "llm_cost_usd": llm_res_data["cost_usd"],
            
            "error_notes": "" if yolo_res else "yolo_no_box_found"
        }
        results.append(row)

        if class_filter == "nutrition":
            row["gt_table_json"]  = gt_object_compact
            row["ocr_table_json"] = trad_result_compact
            # row["llm_table_json"] = llm_res_compact

        if dict_metrics_ocr:
            row.update(dict_metrics_ocr)
        if dict_metrics_llm:
            row.update(dict_metrics_llm)

        print(f"Ergebnis: {row}")

    # --- Ergebnisse in CSV-Datei speichern ---
    df = pd.DataFrame(results)
    csv_path = new_out_dir / f"eval_results_{timestamp}.csv"

    df.to_csv(csv_path, index=False, encoding='utf-8-sig', sep=";", quotechar='"', quoting=csv.QUOTE_ALL, escapechar='\\')
    print(f"\nEvaluation abgeschlossen. Ergebnisse gespeichert in: {csv_path}")
    print(f"Visuelle Ausgaben (BBoxen, Crops) gespeichert in: {new_out_dir}")

if __name__ == "__main__":
    main()