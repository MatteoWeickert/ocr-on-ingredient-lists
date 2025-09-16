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
    transform_dict_to_string, measure_ram_peak, calculate_iou
)
from llm_pipeline import run_llm_pipeline

def run_llm(cfg: dict):
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

        llm_res_data, mem_peak = measure_ram_peak(run_llm_pipeline, [str(p) for p in paths], class_filter)

        time_end_llm = time.perf_counter()
        cpu_end_llm = process.cpu_times()

        # Metriken sammeln
        end_to_end_time_llm = time_end_llm - time_start_llm
        cpu_llm_time = (cpu_end_llm.user - cpu_start_llm.user) + (cpu_end_llm.system - cpu_start_llm.system) # tatsächliche CPU-Zeit

        if llm_res_data.get("text") is not None:
            llm_res = llm_res_data.get("text", "")
        llm_times = llm_res_data.get("times", {})
        time_preprocessing = llm_times.get("preprocessing", None)
        time_api_llm = llm_times.get("api_roundtrip", None)
        time_postproc_llm = llm_times.get("postprocessing", None)

        llm_res_compact = compact_json_str(llm_res)
        bbox_coordinates = json.loads(llm_res_compact).get("box", {})
        if bbox_coordinates:
            x1, y1, x2, y2 = (bbox_coordinates.get("x1"), bbox_coordinates.get("y1"), bbox_coordinates.get("x2"), bbox_coordinates.get("y2")) if bbox_coordinates else (None, None, None, None)
            bbox_array = [x1, y1, x2, y2]

            iou = calculate_iou(gt_bbox, bbox_array)

        else:
            bbox_array = []
            iou = 0.0 if gt_bbox else 1.0  # Wenn keine GT und keine Vorhersage, perfekte Übereinstimmung

        print(f"Zeit (LLM): {end_to_end_time_llm:.2f}s")

        # --- Metriken berechnen ---

        row = {
            "product_id": product_id,
            "class_requested": class_filter,
            "gt_text": gt_text,
            "predicted_bbox": bbox_array,
            "iou": iou,
            
            # LLM Pipeline
            "time_llm_s": end_to_end_time_llm,
            "cpu_llm_s": cpu_llm_time,
            "mem_llm_peak": mem_peak,
            "time_preprocessing": time_preprocessing,
            "time_api_llm_s": time_api_llm,
            "time_postproc_llm_s": time_postproc_llm
            #"llm_total_tokens": llm_res_data["total_tokens"],
            #"llm_cost_usd": llm_res_data["cost_usd"],
        }

        results.append(row)

        print(f"Ergebnis: {row}")

    # --- Ergebnisse in CSV-Datei speichern ---
    df = pd.DataFrame(results)
    csv_path = new_out_dir / f"eval_results_{timestamp}_llm.csv"

    df.to_csv(csv_path, index=False, encoding='utf-8-sig', sep=";", quotechar='"', quoting=csv.QUOTE_ALL, escapechar='\\')
    print(f"\nEvaluation abgeschlossen. Ergebnisse gespeichert in: {csv_path}")
    print(f"Visuelle Ausgaben (BBoxen, Crops) gespeichert in: {new_out_dir}")