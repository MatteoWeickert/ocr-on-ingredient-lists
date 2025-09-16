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

from eval_helpers import (
    load_ground_truth, load_images, group_images_by_product, compact_json_str,
    transform_dict_to_string, evaluate_nutrition_table_structure,
    calculate_word_level_metrics, calculate_grits_metric, measure_ram_peak, calculate_composite_indicator_nutrition, calculate_composite_indicator_ingredients
)
from llm_pipeline import run_llm_pipeline

def run_llm(cfg: dict, temperature: float, llm_model: str):
    """
    Führt die Evaluation mit der übergebenen Konfiguration aus.
    Erwartete Keys in cfg:
        - img_dir, gt_json, model, class_filter, out_dir, traditional_script, llm_script
    """

    print("Konfiguration:", cfg)

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
        llm_string = ""
        structure_score_llm = {}
        grits_metrics = {}
        llm_json = {}

        if gt_item:
            gt_object = gt_item.get("text", {})
            gt_object_compact = compact_json_str(gt_object)
            gt_text = transform_dict_to_string(gt_object, class_filter)
            gt_bbox = gt_item.get("bbox", [])
        else:
             print(f"Warnung: Keine GT-Daten für Produkt {product_id} gefunden.")
        
        ###################################################
        # --- LLM-Pipeline ---
        ###################################################
        process = psutil.Process(os.getpid())
        cpu_start_llm = process.cpu_times()
        time_start_llm = time.perf_counter()

        llm_res_data, mem_peak = measure_ram_peak(run_llm_pipeline, [str(p) for p in paths], class_filter, temperature, llm_model)

        time_end_llm = time.perf_counter()
        cpu_end_llm = process.cpu_times()

        # Metriken sammeln
        end_to_end_time_llm = time_end_llm - time_start_llm
        cpu_llm_time = (cpu_end_llm.user - cpu_start_llm.user) + (cpu_end_llm.system - cpu_start_llm.system) # tatsächliche CPU-Zeit

        if llm_res_data.get("text") is not None:
            llm_res = llm_res_data.get("text", "")
        llm_times = llm_res_data.get("times", {})
        time_preproc_llm = llm_times.get("preprocessing", None)
        time_api_llm = llm_times.get("api_roundtrip", None)
        time_postproc_llm = llm_times.get("postprocessing", None)
        if class_filter == "nutrition":
            llm_res_compact = compact_json_str(llm_res)
            print("LLM-Antwort (kompakt):", llm_res_compact)
            print(f"Typ von llm_res_compact: {type(llm_res_compact)}")
            llm_string = transform_dict_to_string(llm_res_compact, class_filter)
            try:
                if isinstance(json.loads(llm_res_compact), dict):
                    metrics_llm = calculate_word_level_metrics(gt_text, llm_string, gt_object, json.loads(llm_res_compact), class_filter, method = "llm")
                    structure_score_llm = evaluate_nutrition_table_structure(gt_object, json.loads(llm_res_compact), "llm")
                    grits_metrics = calculate_grits_metric(gt_object, json.loads(llm_res_compact))
                    if end_to_end_time_llm and grits_metrics and metrics_llm and mem_peak:
                        composite_score = calculate_composite_indicator_nutrition(end_to_end_time_llm, mem_peak, grits_metrics.get("overall_table_score"), llm_res_data.get("cost_usd", 0.0))
            except json.JSONDecodeError as e:
                print(f"Fehler beim Parsen der LLM-Antwort: {e}")
                metrics_llm = {}
                structure_score_llm = {}
                grits_metrics = {}
                composite_score = None
        else:
            llm_string = llm_res or ""
            metrics_llm = calculate_word_level_metrics(gt_text, llm_string, gt_object, llm_res_data.get("text", {}), class_filter, method = "llm")
            if end_to_end_time_llm and metrics_llm and mem_peak:
                composite_score = calculate_composite_indicator_ingredients(end_to_end_time_llm, mem_peak, wer(gt_text, llm_string), cer(gt_text, llm_string), metrics_llm.get("f1_overall_ocr"), llm_res_data.get("cost_usd", 0.0))

        print(f"Zeit (LLM): {end_to_end_time_llm:.2f}s")

        # --- Metriken berechnen ---

        row = {
            "product_id": product_id,
            "class_requested": class_filter,
            "gt_text": gt_text,
            
            # LLM Pipeline
            "llm_text": llm_string,
            "time_llm_s": end_to_end_time_llm,
            "cpu_llm_s": cpu_llm_time,
            "mem_llm_peak": mem_peak,
            "time_preproc_llm_s": time_preproc_llm,
            "time_api_llm_s": time_api_llm,
            "time_postproc_llm_s": time_postproc_llm,
            "wer_llm": wer(gt_text, llm_string),
            "cer_llm": cer(gt_text, llm_string),
            "composite_indicator": composite_score,
            "llm_total_tokens": llm_res_data["total_tokens"],
            "llm_cost_usd": llm_res_data["cost_usd"],
        }
        
        if class_filter == "nutrition":
            row["gt_table_json"]  = gt_object_compact
            row["llm_table_json"] = llm_res_compact

        if structure_score_llm:
            row.update(structure_score_llm)
        if metrics_llm:
            row.update(metrics_llm)
        if grits_metrics:
            row.update(grits_metrics)

        results.append(row)

        print(f"Ergebnis: {row}")

    # --- Ergebnisse in CSV-Datei speichern ---
    df = pd.DataFrame(results)
    csv_path = new_out_dir / f"eval_results_{timestamp}_llm.csv"

    df.to_csv(csv_path, index=False, encoding='utf-8-sig', sep=";", quotechar='"', quoting=csv.QUOTE_ALL, escapechar='\\')
    print(f"\nEvaluation abgeschlossen. Ergebnisse gespeichert in: {csv_path}")
    print(f"Visuelle Ausgaben (BBoxen, Crops) gespeichert in: {new_out_dir}")