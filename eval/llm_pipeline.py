# llm_pipeline.py

import json
import os
import base64
from io import BytesIO
from typing import List, Dict, Any
import cv2
import numpy as np
from PIL import Image
from openai import OpenAI
from dotenv import load_dotenv
from ultralytics import YOLO
from pathlib import Path

from eval_helpers import timer

times = {} # Dict zum Speichern einiger Zwischenzeiten

# ==============================================================================
# KONSTANTEN (Preise pro Million Token)
# ==============================================================================
# Stand: August 2024 (GPT-4o)
MODEL_PRICING = {
    "gpt-4o": {"input": 2.50 / 1_000_000, "output": 10.00 / 1_000_000},
    "gpt-4.1-mini": {"input": 0.40 / 1_000_000, "output": 1.60 / 1_000_000}
}

# ==============================================================================
# HAUPTFUNKTION
# ==============================================================================

def run_llm_pipeline(model: YOLO, image_paths: List[Path], target_id: int, out_dir: Path, product_id: str, class_filter: str) -> str:
    """
    Öffentliche Hauptfunktion, die die LLM-Pipeline für eine Gruppe von Bildern ausführt.
    Diese Funktion wird vom Evaluationsskript aufgerufen.

    Args:
        image_paths: Eine Liste von Pfaden zu den Bildern EINES Produkts.
        class_filter: Die zu extrahierende Klasse ('ingredients' oder 'nutrition').

    Returns:
        Ein Dictionary mit dem extrahierten Text, Token-Verbrauch und geschätzten Kosten.
    """

    with timer(times, "preprocessing"):
        # 1. API-Key laden
        _load_llm_api_key()
        try:
            client = OpenAI(api_key=os.getenv('api_key'))
        except Exception as e:
            print(f"Fehler bei der Initialisierung des OpenAI-Clients: {e}")
            return {"text": "API_CLIENT_INIT_ERROR", "total_tokens": 0, "cost_usd": 0.0}

        encoded_images = []
        for image_path in image_paths:
            try:
                # Lade Bild mit cv2 für die Vorverarbeitung
                img = cv2.imread(image_path)
                if img is None:
                    print(f"Warnung: Bild konnte nicht geladen werden: {image_path}")
                    continue
                
                # Bild für die API enkodieren
                encoded_images.append(_encode_image(img))
            except Exception as e:
                print(f"Fehler bei der Bildverarbeitung für {image_path}: {e}")

        if not encoded_images:
            return {"text": {}, "prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0, "cost_usd": 0.0}

        # 3. Prompt basierend auf dem Filter erstellen
        prompt_text = _create_prompt(class_filter)

        # 4. API-Anfrage zusammenbauen und senden
        api_request_content = [{"type": "input_text", "text": prompt_text}]
        for encoded_img in encoded_images:
            api_request_content.append({
                "type": "input_image",
                "image_url": encoded_img
            })

        model_to_use = "gpt-4o"

    try:
        with timer(times, "api_roundtrip"):
            response = client.responses.create(
                model=model_to_use,
                input=[
                    {
                    "role": "user", 
                    "content": api_request_content
                    }
                ],
                temperature=0.2, # Kreativität des Modells niedrig, um keine unerwarteten Ergebnisse zu erzeugen
            )

        with timer(times, "postprocessing"):
            output_text = response.output_text
            print(f"LLM-Antwort erhalten: {output_text}") 
            usage = response.usage
            cost = _calculate_llm_cost(model_to_use, usage.input_tokens, usage.output_tokens)

        return {
            "text": output_text, # Der rohe JSON-String vom Modell
            "prompt_tokens": usage.input_tokens, # Bilder werden zu Token umgewandelt und in input_tokens gezählt
            "completion_tokens": usage.output_tokens,
            "total_tokens": usage.total_tokens,
            "cost_usd": cost,
            "times": times
        }
    
    except Exception as e:
        print(f"Fehler bei der Anfrage an die OpenAI API: {e}")
        return {"text": "API_ERROR", "prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0, "cost_usd": 0.0, "times": times}

# ==============================================================================
# HELFERFUNKTIONEN
# ==============================================================================

def _load_llm_api_key():
    """Lädt den API-Key aus der .env-Datei."""
    load_dotenv()

def _encode_image(img: np.ndarray) -> str:
    """
    Kodiert ein Bild (NumPy-Array) in einen für die API geeigneten Base64-String.
    """
    pil_img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    buffered = BytesIO()
    pil_img.save(buffered, format="PNG")
    img_bytes = buffered.getvalue()
    base64_str = base64.b64encode(img_bytes).decode('utf-8')
    return f"data:image/png;base64,{base64_str}"

def _create_prompt(class_filter: str) -> str:
    """Erstellt den spezifischen Prompt für die API-Anfrage."""
    if class_filter == "nutrition":
        return (
            """
            YOU ARE A HYPER-FOCUSED IMAGE ANALYSIS ENGINE. YOUR ONLY TASK IS TO IDENTIFY THE SINGLE BEST IMAGE FROM A SET THAT CLEARLY SHOWS A NUTRITION TABLE AND RETURN ITS BOUNDING BOX COORDINATES. FAILURE IS NOT AN OPTION.

            INPUT: You will receive multiple images of a food product.
            
            YOUR SINGLE, NON-NEGOTIABLE TASK:
            1.  ANALYZE ALL provided images.
            2.  SELECT the ONE image that shows the MOST COMPLETE AND CLEARLY READABLE nutrition table. This is the "best" image.
            3.  IDENTIFY the precise bounding box that encloses this nutrition table on the "best" image. The box should be as tight as possible around the target text block.

            STRICT JSON OUTPUT FORMAT - NO EXCEPTIONS:
            - YOUR OUTPUT MUST BE A SINGLE, VALID JSON OBJECT. NO MARKDOWN, NO EXPLANATIONS, NO TEXT BEFORE OR AFTER THE JSON.
            - THE JSON MUST START WITH { AND END WITH }.
            - IF NO NUTRITION TABLE CAN BE CLEARLY IDENTIFIED ON ANY IMAGE, YOU MUST RETURN AN EMPTY JSON OBJECT: {}.

            JSON STRUCTURE (STRICTLY NOTHING ELSE):
            {
              "box": {
                "x1": integer,
                "y1": integer,
                "x2": integer,
                "y2": integer
              }
            }
            
            COORDINATE SYSTEM RULES:
            - The origin (0,0) is the TOP-LEFT corner of the image.
            - "x1" and "y1" are the coordinates of the TOP-LEFT corner of the bounding box.
            - "x2" and "y2" are the coordinates of the BOTTOM-RIGHT corner of the bounding box.
            - All coordinates must be integer pixel values.

            FINAL COMMAND: ANALYZE, SELECT, LOCATE, AND RESPOND WITH THE JSON. ZERO DEVIATION.
            """
        )
    elif class_filter == "ingredients":
        return (
            """
            YOU ARE A HYPER-FOCUSED IMAGE ANALYSIS ENGINE. YOUR ONLY TASK IS TO IDENTIFY THE SINGLE BEST IMAGE FROM A SET THAT CLEARLY SHOWS A INGREDIENT LIST AND RETURN ITS BOUNDING BOX COORDINATES. FAILURE IS NOT AN OPTION.

            INPUT: You will receive multiple images of a food product.
            
            YOUR SINGLE, NON-NEGOTIABLE TASK:
            1.  ANALYZE ALL provided images.
            2.  SELECT the ONE image that shows the MOST COMPLETE AND CLEARLY READABLE ingredient list. This is the "best" image.
            3.  IDENTIFY the precise bounding box that encloses this ingredient list on the "best" image. The box should be as tight as possible around the target text block.

            STRICT JSON OUTPUT FORMAT - NO EXCEPTIONS:
            - YOUR OUTPUT MUST BE A SINGLE, VALID JSON OBJECT. NO MARKDOWN, NO EXPLANATIONS, NO TEXT BEFORE OR AFTER THE JSON.
            - THE JSON MUST START WITH { AND END WITH }.
            - IF NO INGREDIENT LIST CAN BE CLEARLY IDENTIFIED ON ANY IMAGE, YOU MUST RETURN AN EMPTY JSON OBJECT: {}.

            JSON STRUCTURE (STRICTLY NOTHING ELSE):
            {
              "box": {
                "x1": integer,
                "y1": integer,
                "x2": integer,
                "y2": integer
              }
            }
            
            COORDINATE SYSTEM RULES:
            - The origin (0,0) is the TOP-LEFT corner of the image.
            - "x1" and "y1" are the coordinates of the TOP-LEFT corner of the bounding box.
            - "x2" and "y2" are the coordinates of the BOTTOM-RIGHT corner of the bounding box.
            - All coordinates must be integer pixel values.

            FINAL COMMAND: ANALYZE, SELECT, LOCATE, AND RESPOND WITH THE JSON. ZERO DEVIATION.
            """
        )

def _calculate_llm_cost(model: str, prompt_tokens: int, completion_tokens: int) -> float:
    """Berechnet die Kosten für einen API-Aufruf basierend auf vordefinierten Preisen."""
    pricing = MODEL_PRICING.get(model)
    if not pricing:
        print(f"Warnung: Keine Preisinformationen für Modell '{model}' gefunden.")
        return 0.0
    
    input_cost = prompt_tokens * pricing["input"]
    output_cost = completion_tokens * pricing["output"]
    return input_cost + output_cost