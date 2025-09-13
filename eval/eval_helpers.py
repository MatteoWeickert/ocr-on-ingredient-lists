import os
import numpy as np
import re
import json
import time
import difflib
from pathlib import Path
from collections import defaultdict, Counter
from contextlib import contextmanager
from typing import List, Dict, Tuple, Any, Set
from jiwer import wer, cer

################################
# Timing-Helper
################################
@contextmanager
def timer(stats: dict, key: str):
    t0 = time.perf_counter()
    try:
        yield
    finally:
        elapsed = time.perf_counter() - t0
        # Nur die aktuelle Zeit für den Key speichern, nicht aufaddieren
        stats[key] = elapsed

################################
# JSON-Handling
################################

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

################################
# Config-Handling
################################
# def load_config_from_file(path: str | Path) -> Dict:
#     p = Path(path)
#     if not p.exists():
#         raise FileNotFoundError(f"Konfigurationsdatei nicht gefunden: {p}")
#     cfg = json.loads(p.read_text(encoding="utf-8"))
#     _validate_config(cfg)
#     return cfg

def _validate_config(cfg: Dict):
    required = ["img_dir", "gt_json", "model", "class_filter", "prefered_analysis_method", "out_dir", "traditional_script", "llm_script"]
    for k in required:
        if k not in cfg or not cfg[k]:
            raise ValueError(f"Schlüssel '{k}' fehlt oder ist leer in der config")

################################
# Evaluation preparation 
################################
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

################################
# Evaluation Metrics
################################

def evaluate_nutrition_table_structure(gt_table: Dict, ocr_table: Dict, method_name: str):
    """
    Führt einen strukturbasierten Vergleich zwischen einer Ground-Truth- und einer OCR-Nährwerttabelle durch.

    Args:
        gt_table: Das Ground-Truth-Dict für die Nährwerttabelle.
        ocr_table: Das von der OCR-Pipeline extrahierte Dict.

    Returns:
        Ein Dict mit detaillierten Evaluationsmetriken.
    """

    # Sicherstellen, dass die Eingabedaten gültige Diktionäre sind
    gt = gt_table if isinstance(gt_table, dict) else {}
    ocr = ocr_table if isinstance(ocr_table, dict) else {}

    metrics = {}

    # TITEL
    gt_title = gt.get("title", "")
    ocr_title = ocr.get("title", "")
    metrics[f"title_similarity_{method_name}"] = 1- cer(gt_title, ocr_title)


    # FUßNOTE
    gt_footnote = gt.get("footnote", "")
    ocr_footnote = ocr.get("footnote", "")
    metrics[f"footnote_similarity_{method_name}"] = 1 - cer(gt_footnote, ocr_footnote) 

    # SPALTEN
    gt_columns = [col for col in gt.get("columns", [])]
    ocr_columns = [col for col in ocr.get("columns", [])]
    metrics[f"column_counts_{method_name}"] = {"gt": len(gt_columns), f"{method_name}": len(ocr_columns)}
    metrics[f"column_count_match_{method_name}"] = (len(gt_columns) == len(ocr_columns))

    column_wer = wer(gt_columns, ocr_columns) if gt_columns and ocr_columns else 1.0 if gt_columns or ocr_columns else 0.0
    column_cer = cer(" ".join(gt_columns), " ".join(ocr_columns)) if gt_columns and ocr_columns else 1.0 if gt_columns or ocr_columns else 0.0
    metrics[f"column_structural_similarity_cer_{method_name}"] = 1 - column_cer
    metrics[f"column_structural_similarity_wer_{method_name}"] = 1 - column_wer
    
    # 3. Zeilen-Vergleich
    gt_rows = gt.get("rows", [])
    ocr_rows = ocr.get("rows", [])

    # Erstelle Maps für einen einfachen Zugriff über das normalisierte Label
    gt_row_map = {row.get("label", ""): row.get("values", []) for row in gt_rows if row.get("label", "")}
    ocr_row_map = {row.get("label", ""): row.get("values", []) for row in ocr_rows if row.get("label", "")}

    gt_labels = set(gt_row_map.keys())
    ocr_labels = set(ocr_row_map.keys())

    # Finde übereinstimmende, fehlende und zusätzliche Labels
    matched_labels = gt_labels.intersection(ocr_labels)
    missing_labels = gt_labels.difference(ocr_labels) # In GT, aber nicht in OCR
    extra_labels = ocr_labels.difference(gt_labels)   # In OCR, aber nicht in GT

    metrics["row_count_match"] = (len(gt_labels) == len(ocr_labels))

    metrics[f"row_counts_{method_name}"] = {
    "gt": len(gt_labels),
    "ocr": len(ocr_labels),
    "matched": len(matched_labels),
    "missing_in_pred": len(missing_labels),
    "extra_in_pred": len(extra_labels)
    }

    metrics[f"missing_labels_{method_name}"] = sorted(list(missing_labels))
    metrics[f"extra_labels_{method_name}"] = sorted(list(extra_labels))


    overall_errors = []

    for label in sorted(list(matched_labels)):
        gt_values = gt_row_map[label]
        ocr_values = ocr_row_map[label]

        # Vergleiche Zelleninhalte nur, wenn die Spaltenanzahl übereinstimmt
        if len(gt_values) == len(ocr_values):
            errors = [cer(gt_v, ocr_v) for gt_v, ocr_v in zip(gt_values, ocr_values)]
            error_mean = np.mean(errors) if errors else 0.0
            overall_errors.append(error_mean)
        
        else:
            # Fall 2: Spaltenanzahl stimmt NICHT überein -> String-Verkettungs-Fallback
            gt_row_string = " ".join(gt_values)
            ocr_row_string = " ".join(ocr_values)
            error = cer(gt_row_string, ocr_row_string)
            overall_errors.append(error)

    metrics[f"avg_content_similarity_matched_rows_{method_name}"] = 1 - (np.mean(overall_errors) if overall_errors else 0.0)

    # Benutze CER zur Berechnung des Zeileninhaltscores
    gt_rows_text = ""
    ocr_rows_text = ""

    for row in gt_rows:
        row_text = row.get("label", "") + " " + " ".join(row.get("values", []))
        gt_rows_text += row_text + " "

    for row in ocr_rows:
        row_text = row.get("label", "") + " " + " ".join(row.get("values", []))
        ocr_rows_text += row_text + " "

    row_content_error = cer(gt_rows_text.strip(), ocr_rows_text.strip())
    metrics[f"row_structural_similarity_{method_name}"] = 1 - row_content_error

    # 4. Gesamtergebnis berechnen
    weights = {
        "title": 0.10,
        "columns": 0.40,
        "rows": 0.40,
        "footnote": 0.10
    }

    overall_score = (
        metrics[f"title_similarity_{method_name}"] * weights["title"] +
        metrics[f"column_structural_similarity_{method_name}"] * weights["columns"] +
        metrics[f"row_structural_similarity_{method_name}"] * weights["rows"] +
        metrics[f"footnote_structural_similarity_{method_name}"] * weights["footnote"]
    )
    metrics[f"overall_structural_score_{method_name}"] = overall_score

    return metrics

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

def calculate_word_level_metrics(gt_text: str, pred_text: str, gt_object: Dict, pred_object: Dict, class_filter: str, method: str) -> Dict[str, float | str]:
    """Berechnet Precision, Recall und F1-Score auf Wortebene für Zutatenlisten auf allen Wörtern und für Nährwerttabellen auf allen Wörtern,
    auf Elementen, auf Werten, auf (Element, Wert)-Paaren und auf (Element, Spaltenname, Wert)-Tripeln. """

    if (not gt_text and not pred_text) or (not gt_object and not pred_object):
        return {f"precision_overall_{method}": 1.0, f"recall_overall_{method}": 1.0, f"f1_overall_{method}": 1.0}
    if (not gt_text or not pred_text) or (not gt_object or not pred_object):
        return {f"precision_overall_{method}": 0.0, f"recall_overall_{method}": 0.0, f"f1_overall_{method}": 0.0}
    
    if class_filter == "ingredients":

        gt_words = Counter(gt_text.split())
        pred_words = Counter(pred_text.split())

        # Berechnung der Metriken auf Wortebene
        metrics = _calculate_precision_recall_f1_multiset(gt_words, pred_words)
        output = {
            f"precision_overall_{method}": metrics["precision"],
            f"recall_overall_{method}": metrics["recall"],
            f"f1_overall_{method}": metrics["f1"]
        }
        return output

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
            f"precision_elements_{method}": metrics_elements["precision"],
            f"recall_elements_{method}": metrics_elements["recall"],
            f"f1_elements_{method}": metrics_elements["f1"],
            f"precision_values_{method}": metrics_values["precision"],
            f"recall_values_{method}": metrics_values["recall"],
            f"f1_values_{method}": metrics_values["f1"],
            f"precision_pairs_{method}": metrics_pairs["precision"],
            f"recall_pairs_{method}": metrics_pairs["recall"],
            f"f1_pairs_{method}": metrics_pairs["f1"],
            f"precision_triples_{method}": metrics_triples["precision"],
            f"recall_triples_{method}": metrics_triples["recall"],
            f"f1_triples_{method}": metrics_triples["f1"],
            f"precision_overall_{method}": metrics_words["precision"],
            f"recall_overall_{method}": metrics_words["recall"],
            f"f1_overall_{method}": metrics_words["f1"]
        }
        return all_metrics

    else:   
        return {"precision": "NaN", "recall": "NaN", "f1": "NaN"}

def _calculate_precision_recall_f1_set(gt: Set, pred: Set) -> Dict[str, float]:
    """Berechnet Precision, Recall und F1-Score aus True Positives, False Positives und False Negatives."""
    print(f"DEBUG: gt set: {gt}")
    print(f"DEBUG: pred set: {pred}")
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

def _elements_from_table(table: Dict) -> Set[str]:
    """Menge der Elemente (Zeilen-Labels)."""
    print(f"DEBUG: table type: {type(table)}")
    rows = table.get("rows", []) if isinstance(table, dict) else []
    elements = set()
    for r in rows:
        if r.get("label",""):
            elements.add(r.get("label",""))
    return elements

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
    cols = list(cols[1:])
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