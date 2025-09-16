from pathlib import Path
from datetime import datetime
import csv
import json
import psutil
import pandas as pd
from ultralytics import YOLO
from jiwer import wer, cer
import time
import os
from symspellpy.symspellpy import SymSpell

from eval_helpers import (
    load_ground_truth, load_images, group_images_by_product, compact_json_str,
    transform_dict_to_string, calculate_iou, evaluate_nutrition_table_structure,
    calculate_word_level_metrics, calculate_grits_metric, measure_ram_peak, calculate_composite_indicator_nutrition, calculate_composite_indicator_ingredients
)
from traditional_pipeline import run_traditional_pipeline

def run_traditional(cfg: dict):
    """
    Führt die Evaluation mit der übergebenen Konfiguration aus.
    Erwartete Keys in cfg:
        - img_dir, gt_json, model, class_filter, out_dir, traditional_script, llm_script
    """

    print("Konfiguration:", cfg)

    img_dir = Path(cfg.get("img_dir"))
    gt_json = Path(cfg.get("gt_json"))
    model_path = Path(cfg.get("model"))
    class_filter = cfg.get("class_filter")

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    out_base = Path(cfg.get("out_dir"))
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
    target_id = class_name_to_id.get(class_filter)

    # Initialisiere SymSpell für die Rechtschreibkorrektur
    sym_spell = SymSpell(max_dictionary_edit_distance=2, prefix_length=7)
    dictionary_path = Path(__file__).parent / "de-100k.txt"
    sym_spell.load_dictionary(dictionary_path, term_index=0, count_index=1)

    results = []
    for product_id, paths in grouped_images.items():
        print(f"\n--- Verarbeite Produkt: {product_id} ---")

        # Ground Truth für dieses Produkt extrahieren
        gt_item = next((item for item in gt_data if str(item.get("produktnr")) == product_id), None)
        gt_text = ""
        gt_bbox = []
        gt_object = {}
        structure_score_ocr = {}
        grits_metrics = {}
        ocr_string = ""

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

        trad_result, mem_peak = measure_ram_peak(run_traditional_pipeline, model, paths, target_id, new_out_dir, product_id, sym_spell)
        
        time_end_trad = time.perf_counter()
        cpu_end = process.cpu_times()

        # Metriken sammeln
        end_to_end_time_trad = time_end_trad - time_start_trad
        cpu_trad_time = (cpu_end.user - cpu_start.user) + (cpu_end.system - cpu_start.system) # tatsächliche CPU-Zeit
       
        if trad_result.get("yolo_result"):
            yolo_res = trad_result.get("yolo_result")
        trad_times = trad_result.get("times", {})
        time_yolo = trad_times.get("yolo_total", None)
        time_ocr = trad_times.get("ocr", None)
        time_postproc = trad_times.get("postprocessing", None)
        time_preproc = trad_times.get("crop_preprocess", None)
        print(f"Zeit (Traditionell): {end_to_end_time_trad:.2f}s")
        
        # --- Metriken berechnen ---
        pred_bbox = yolo_res["box"] if yolo_res else []
        iou = calculate_iou(gt_bbox, pred_bbox)

        trad_result_compact = ""
        if trad_result.get("structured_data"):
            ocr_string = transform_dict_to_string(trad_result.get("structured_data", {}), class_filter)
            ocr_raw = trad_result.get("structured_data", {}).get("raw_text", "") or ""

        if class_filter == "nutrition" and trad_result.get("structured_data", {}).get("nutrition_table") is not None:
            structure_score_ocr = evaluate_nutrition_table_structure(gt_object, trad_result.get("structured_data", {}).get("nutrition_table", {}), "ocr")
            trad_result_compact = compact_json_str(trad_result.get("structured_data", {}).get("nutrition_table", {}))
            metrics_ocr = calculate_word_level_metrics(gt_text, ocr_string, gt_object, trad_result.get("structured_data", {}).get("nutrition_table", {}), class_filter, method = "ocr")
            grits_metrics = calculate_grits_metric(gt_object, trad_result.get("structured_data", {}).get("nutrition_table", {}))
            if end_to_end_time_trad and grits_metrics and metrics_ocr and mem_peak:
                composite_score = calculate_composite_indicator_nutrition(end_to_end_time_trad, mem_peak, grits_metrics.get("overall_table_score"), api_cost=0.0)

        if class_filter == "ingredients" and trad_result.get("structured_data", {}).get("ingredients_text") is not None:
            metrics_ocr = calculate_word_level_metrics(gt_text, ocr_string, gt_object, trad_result.get("structured_data", {}).get("ingredients_text", {}), class_filter, method = "ocr")
            if end_to_end_time_trad and metrics_ocr and mem_peak:
                composite_score = calculate_composite_indicator_ingredients(end_to_end_time_trad, mem_peak, wer(gt_text, ocr_string), cer(gt_text, ocr_string), metrics_ocr.get("f1_overall_ocr"), api_cost=0.0)

        row = {
            "product_id": product_id,
            "class_requested": class_filter,
            "gt_text": gt_text,
            
            # Traditionelle Pipeline
            "ocr_text": ocr_string,
            "ocr_raw": ocr_raw,
            "yolo_confidence": yolo_res["confidence"] if yolo_res else None,
            "pred_bbox": pred_bbox,
            "iou": iou,
            "time_trad_s": end_to_end_time_trad,
            "cpu_trad_s": cpu_trad_time,
            "mem_trad_peak": mem_peak,
            "time_yolo_s": time_yolo,
            "time_ocr_s": time_ocr,
            "time_preproc_s": time_preproc,
            "time_postproc_s": time_postproc,
            "wer_ocr": wer(gt_text, ocr_string) if wer(gt_text, ocr_string) <= 1.0 else 1.0,
            "cer_ocr": cer(gt_text, ocr_string) if cer(gt_text, ocr_string) <= 1.0 else 1.0,
            "error_notes": "" if yolo_res else "yolo_no_box_found",
            "composite_indicator": composite_score
        }
        
        if class_filter == "nutrition":
            row["gt_table_json"]  = gt_object_compact
            row["ocr_table_json"] = trad_result_compact

        if structure_score_ocr:
            row.update(structure_score_ocr)
        if metrics_ocr:
            row.update(metrics_ocr)
        if grits_metrics:
            row.update(grits_metrics)
        

        results.append(row)

        print(f"Ergebnis: {row}")

    # --- Ergebnisse in CSV-Datei speichern ---
    df = pd.DataFrame(results)
    csv_path = new_out_dir / f"eval_results_{timestamp}_traditional.csv"

    df.to_csv(csv_path, index=False, encoding='utf-8-sig', sep=";", quotechar='"', quoting=csv.QUOTE_ALL, escapechar='\\')
    print(f"\nEvaluation abgeschlossen. Ergebnisse gespeichert in: {csv_path}")
    print(f"Visuelle Ausgaben (BBoxen, Crops) gespeichert in: {new_out_dir}")