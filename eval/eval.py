# this scripts evaluates the traditional model vs the AI model using different metrics

import os 
import json
import time
from typing import List, Tuple, Dict
from pathlib import Path
from collections import defaultdict
from ultralytics import YOLO
import cv2
import pandas as pd
import pytesseract
from dotenv import load_dotenv
import base64
from io import BytesIO
from openai import OpenAI
from PIL import Image
import numpy as np
import shutil
from datetime import datetime

def load_config():
    cfg_path = Path(__file__).with_name("config.json")

    if not cfg_path.exists():
        raise FileNotFoundError(f"Config file not found: {cfg_path}")
    
    with cfg_path.open("r") as f:
        cfg = json.load(f)

    # Pflichtfelder prüfen
    required_keys = ["img_dir", "gt_json", "model", "class_filter", "out_csv", "out_dir", "llm_script"]
    for k in required_keys:
        if k not in cfg or cfg[k] is None:
            raise ValueError(f"'{k}' fehlt in der config.json")
        
    return cfg

def load_llm_api_key():
        # Laden des .env-Files mit dem API-Keys
        load_dotenv()


def group_images_by_product(image_paths):
    groups = defaultdict(list)
    for image_path in image_paths:
            product_id = Path(image_path).name.split('_')[0] 
            groups[product_id].append(image_path)

    return groups

def run_llm_script(image_paths, class_filter):
    """
    Runs the LLM script for evaluation.
    """
    
    encoded_images = []

    ############## Bildverarbeitung & Encoding ################
    for image_path in image_paths:
        img = cv2.imread(str(image_path))
        
        if img is None:
            print(f"Bild konnte nicht geladen werden: {image_path}")
            continue

        processed_img = process_cv2_picture(img)
        
        encoded_img = encode_image(processed_img)
        encoded_images.append(encoded_img)


    load_llm_api_key()

    client = OpenAI(api_key=os.getenv('api_key')) # Zugriff auf den API-Key

    # prepare API request
    api_request = []
    # case 1: ingredients
    if class_filter == "ingredients":

        api_request.append(
        {
            "type": "input_text",
            # "text": "Die Bilder zeigen die Verpackung eines Produkts aus verschiedenen Perspektiven. Bitte extrahiere die Zutatenliste und die Nährwerttabelle des Produkts. Wenn du keine Zutatenliste oder Nährwerttabelle findest, gib bitte an, dass diese nicht vorhanden sind."
            #"text": f"Die Bilder zeigen die Verpackung eines Produkts (Produktnummer: ${product_id}) aus verschiedenen Perspektiven. Extrahiere den Text der Zutatenliste und die Nährwerttabelle des Produkts. Wenn du keine Zutatenliste oder Nährwerttabelle findest, gib bitte an, dass diese nicht vorhanden sind. Rückgabeformat: 'Produktnummer: [Produktnummer]', \n'Zutatenliste: [Text Zutatenliste]', \n'Nährwerttabelle: [Text Nährwerttabelle]'."
            "text": f""
        }
        )
    # case 2: nutrition
    elif class_filter == "nutrition":
        api_request.append(
            {
                "type": "input_text",
                # "text": "Die Bilder zeigen die Verpackung eines Produkts aus verschiedenen Perspektiven. Bitte extrahiere die Zutatenliste und die Nährwerttabelle des Produkts. Wenn du keine Zutatenliste oder Nährwerttabelle findest, gib bitte an, dass diese nicht vorhanden sind."
                #"text": f"Die Bilder zeigen die Verpackung eines Produkts (Produktnummer: ${product_id}) aus verschiedenen Perspektiven. Extrahiere den Text der Zutatenliste und die Nährwerttabelle des Produkts. Wenn du keine Zutatenliste oder Nährwerttabelle findest, gib bitte an, dass diese nicht vorhanden sind. Rückgabeformat: 'Produktnummer: [Produktnummer]', \n'Zutatenliste: [Text Zutatenliste]', \n'Nährwerttabelle: [Text Nährwerttabelle]'."
                "text": f""
            }
        )
        
    else:
        raise ValueError(f"Unbekannte Klasse: {class_filter}")

             

    for encoded_img in encoded_images:
        api_request.append({
            "type": "input_image",
            "image_url": encoded_img
        })

    # Erstellen der Anfrage an die API für dieses Produkt
    response = client.responses.create(
    model="gpt-4.1-mini",
    input=[
        {
        "role": "user",
        "content": api_request
        }
    ]
    )

    # Return result
    return(response.output_text)


def run_yolo_script(model, paths: list[Path] , target_id: int):
    """
    Runs the YOLO model on the given image paths.
    """
    images = [cv2.imread(str(path)) for path in paths]
    results = model(images)

    best_box = None
    best_conf = -1
    best_image_path = None

    for img_path, result in zip(paths, results):
            
        for box in result.boxes:
            cls_id = int(box.cls[0])
            conf = box.conf[0]

            if cls_id == target_id and conf > best_conf and conf > 0.5:
                best_conf = conf
                best_box = box
                best_image_path = img_path

    print(type(target_id))

    best = {
        "image_path": best_image_path,
        "box": best_box.xyxy[0].tolist() if best_box else None,
        "confidence": best_conf,
        "detected_class": model.names[target_id] if target_id in model.names else None
    }

    return best if best_box is not None else None

def run_ocr_script(image):
    """
    Runs the OCR script for evaluation.
    """
    config = r'--oem 3 --psm 6'  # OCR Engine Mode + Page Segmentation Mode
    text = pytesseract.image_to_string(image, lang='deu', config=config)
    return text.strip() if text else ""

def load_ground_truth(gt_json: Path) -> Dict:
    """
    Loads the ground truth data from a JSON file.
    """
    if not gt_json.exists():
        raise FileNotFoundError(f"Ground truth file not found: {gt_json}")
    
    with gt_json.open("r", encoding="utf-8") as f:
        gt_data = json.load(f)
    
    return gt_data

def load_images(img_dir: Path) -> List[Path]:
    """
    Loads all images from the specified directory.
    """
    return [img_dir / img for img in os.listdir(img_dir) if img.lower().endswith(('.jpg', '.jpeg', '.png'))]

def draw_bounding_box(image_path, box: List[float], label: str, color=(0, 255, 0)):
    """
    Draws a bounding box on the image.
    """
    img = cv2.imread(image_path)
    x1, y1, x2, y2 = map(int, box)
    cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
    cv2.putText(img, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 2, color, 6)

    return img

def process_cv2_picture(img):

        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY) # RGB zu GRAY, weil PIL das Bild in RGB speichert
        #blurred = cv2.GaussianBlur(gray, (3, 3), 0) # Bild wird unscharf gemacht, um Rauschen zu entfernen
        inverted = cv2.bitwise_not(gray) # Bildinvertierung: helle Pixel werden dunkel und umgekehrt -> Schrift hell, Background dunkel
        _, binary = cv2.threshold(inverted, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU) # OTSU berechnet optimalen Schwellenwert für das gesamte Bild -> alle darunter werden schwarz, darüber (also der Text) weiß
        #binary = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 51, 2)
        processed = cv2.bitwise_not(binary) # Umkehrung Binärbild: Text wird schwarz, Hintergrund weiß

        return processed

def encode_image(img):
        """
        Encodes an image to a Base64 string for loading to API.
        """               

        # Check if the input is a valid image format
        if isinstance(img, np.ndarray):
            img = Image.fromarray(img)
        elif not isinstance(img, Image.Image):
            raise ValueError("Input must be a PIL Image or a NumPy array representing an image.")

        # Bild → BytesIO (Speicherpuffer) → Bytes → Base64-Text → Data-URL        

        buffered = BytesIO() # BytesIO-Objekt zum Speichern des Bildes im Arbeitsspeicher
        img.save(buffered, format="PNG") # Bild wird in PNG Format gespeichert
        img_bytes= buffered.getvalue() # Bild wird in Bytes umgewandelt
        base64_str = base64.b64encode(img_bytes).decode('utf-8') # Bilddaten werden in einen Base64-String umgewandelt, sodass sie als Text gespeichert werden können
        return f"data:image/png;base64,{base64_str}" # Erzeugug der Data-URL für das Bild

def main():


    ### LOAD CONFIGURATION ###

    cfg = load_config()
    print("Config geladen:", cfg)

    # Pfade laden

    img_dir = Path(cfg["img_dir"])
    gt_json = Path(cfg["gt_json"])
    model_path = Path(cfg["model"])
    out_csv = Path(cfg["out_csv"])
    out_dir = Path(cfg["out_dir"])
    llm_script = Path(cfg["llm_script"])

    class_filter = cfg["class_filter"]

    # CREATE FOLDER IF NOT EXISTS
    if out_dir.exists():
        shutil.rmtree(out_dir)  # empty folder
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Start der Evaluation mit folgenden Parametern:\n"
          f"Bildverzeichnis: {img_dir}\n"
          f"GT JSON: {gt_json}\n"
          f"YOLO-Modell: {model_path}\n"
          f"Klassenfilter: {class_filter}\n")
    
    ### LOAD DATA ###

    # LOAD GROUND TRUTH
    gt_data = load_ground_truth(gt_json)
    

    # LOAD EVALUATION PICTURES
    image_paths = load_images(img_dir)
    
    # GROUP IMAGES BY PRODUCT-ID
    grouped_images = group_images_by_product(image_paths)
    
    # LOAD YOLO MODEL
    model = YOLO(str(model_path))
    
    class_names = model.names # {0: 'ingredients', 1: 'nutrition'}
    class_name_to_id = {}
    for class_id, name in class_names.items():
        class_name_to_id[name] = class_id

    print(f"Klassen im Modell: {class_names}")
    print(f"Klassen-IDs: {class_name_to_id}")
    print(f"Klassenfilter: {class_filter}")
    if class_filter not in class_name_to_id:
        raise ValueError(f"Klasse '{class_filter}' nicht im Modell gefunden.")
    target_id = class_name_to_id[class_filter]

    rows = []

    ### RUN EVALUATION FOR EACH PRODUCT GROUP ###
    for product_id, paths in grouped_images.items():

        gt_text = ""

        for gt_item in gt_data:
            if gt_item["ProduktNR"] == product_id:
                text_field = gt_item["Text"]
                if not isinstance(text_field, dict):
                    print(f"'Text' is not a dict at ProduktNR {gt_item.get('ProduktNR')}, but: {type(text_field)}")
                    raise AttributeError(f"'Text' field is not a dict at ProduktNR {gt_item.get('ProduktNR')}")
                gt_text = gt_item["Text"].get("rows", [])
                break

        # Run YOLO script
        best = run_yolo_script(model, paths, target_id)

        if best is None: # No object of the target class detected

            llm_res = run_llm_script(paths, class_filter)

            rows.append({
                "product_id": product_id,
                "class_requested": class_filter,
                "yolo_confidence": "",
                "bbox_xyxy": "",
                "ocr_text": "",
                "llm_result": llm_res,
                "gt_text": gt_text,
                "error_notes": "yolo_no_box"
            })

        # draw visuals
        drawn = draw_bounding_box(str(best["image_path"]), best["box"], f"{best['detected_class']}, {best['confidence']:.2f}", color=(0, 255, 0))
        out_bbox_path = out_dir / f"{product_id}_bbox.jpg"
        cv2.imwrite(str(out_bbox_path), drawn)

        x1, y1, x2, y2 = best["box"]
        crop = cv2.imread(str(best["image_path"]))[int(y1):int(y2), int(x1):int(x2)]
        processed = process_cv2_picture(crop)
        out_crop_path = out_dir / f"{product_id}_cropProcessed.jpg"
        cv2.imwrite(str(out_crop_path), processed)

        # run OCR on the cropped image
        ocr_text = run_ocr_script(processed)
        print(f"OCR Text for {product_id}: {ocr_text}")

        # run LLM script
        llm_res = run_llm_script(paths, class_filter)
        print(f"LLM Result for {product_id}: {llm_res}")

        # Calculate metrics
            ### add metrics ###

        rows.append({
            "product_id": product_id,
            "class_requested": class_filter,
            "yolo_confidence": best["confidence"],
            "bbox_xyxy": best["box"],
            "ocr_text": ocr_text,
            "llm_result": llm_res,
            "gt_text": gt_text,
            "error_notes": ""
        })

    # Save results to CSV

    df = pd.DataFrame(rows)
    date_str = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")

    if not out_csv.suffix:
        out_csv = out_csv.with_suffix('.csv')
        
    out_csv = out_csv.with_name(f"{out_csv.stem}_{date_str}{out_csv.suffix}")

    df.to_csv(out_csv, index=False, encoding='utf-8')

    print(f"Evaluation completed. Results saved in: {out_csv}")
    print(f"Images saved in: {out_dir}")


if __name__ == "__main__":
    main()