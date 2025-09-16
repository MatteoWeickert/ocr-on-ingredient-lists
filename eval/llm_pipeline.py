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

def run_llm_pipeline(image_paths: List[str], class_filter: str) -> Dict[str, Any]:
    """
    Führt die LLM-Pipeline aus, um die beste Bounding Box aus einer Bilderserie zu extrahieren.
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
        for i, image_path in enumerate(image_paths):
            try:
                # Lade Bild mit PIL für die Vorverarbeitung
                img = Image.open(image_path)
                if img is None:
                    print(f"Warnung: Bild konnte nicht geladen werden: {image_path}")
                    continue
                
                # Bild für die API enkodieren
                encoded_images.append({
                    "id": f"image_{i}",
                    "image_data": _encode_image(img)
                })
            except Exception as e:
                print(f"Fehler bei der Bildverarbeitung für {image_path}: {e}")

        if not encoded_images:
            return {"error": "IMAGE_ENCODING_FAILED"}

        # 3. Prompt basierend auf dem Filter erstellen
        image_ids = [img.get("id") for img in encoded_images]
        prompt_text = _create_prompt(class_filter, image_ids)

        # 4. API-Anfrage zusammenbauen und senden
        api_request_content = [{"type": "text", "text": prompt_text}]
        for encoded_img in encoded_images:
            api_request_content.append({
                "type": "image_url",
                "image_url": {
                    "url": encoded_img.get("image_data"),
                }
            })

        model_to_use = "gpt-4o"

    try:
        with timer(times, "api_roundtrip"):
            response = client.chat.completions.create(
                model=model_to_use,
                messages=[
                    {
                    "role": "user", 
                    "content": api_request_content
                    }
                ],
                temperature=0.2 # Kreativität des Modells niedrig, um keine unerwarteten Ergebnisse zu erzeugen
                # response_format={"type": "json_object"} # Reine Textantwort (kein JSON-Parsing durch die API
            )

        with timer(times, "postprocessing"):
            output_text = response.choices[0].message.content
            print(f"LLM-Antwort erhalten: {output_text}") 
            #usage = response.usage
            #cost = _calculate_llm_cost(model_to_use, usage.input_tokens, usage.output_tokens)

        return {
            "text": output_text, # Der rohe JSON-String vom Modell
            #"prompt_tokens": usage.input_tokens, # Bilder werden zu Token umgewandelt und in input_tokens gezählt
            #"completion_tokens": usage.output_tokens,
            #"total_tokens": usage.total_tokens,
            #"cost_usd": cost,
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

def _encode_image(img: Image.Image) -> str:
    """
    Kodiert ein Bild (PIL Image) in einen für die API geeigneten Base64-String.
    """
    buffered = BytesIO()
    img.save(buffered, format="PNG")
    img_bytes = buffered.getvalue()
    base64_str = base64.b64encode(img_bytes).decode('utf-8')
    return f"data:image/png;base64,{base64_str}"

def _create_prompt(class_filter: str, image_ids: List[str]) -> str:
    """Erstellt den spezifischen Prompt für die API-Anfrage."""
    if class_filter in ["nutrition", "ingredients"]:
        target_description = "Nährwerttabelle (nutrition table)" if class_filter == "nutrition" else "Zutatenliste (ingredient list)"
        
        # Erzeuge eine für das LLM lesbare Liste der Bild-IDs
        image_id_list_str = ", ".join(f"{id}" for id in image_ids)

        return (
        f"""
            YOU ARE A HYPER-PRECISE IMAGE LOCALIZATION ROBOT. YOUR SOLE TASK IS TO FIND THE BOUNDING BOX OF A '{target_description}' ACROSS AN IMAGE AND RETURN ITS LOCATION.

            INPUT: You will receive an image. I have assigned them the following unique identifiers: {image_id_list_str}.
            
            YOUR SINGLE, NON-NEGOTIABLE TASK:
            1.  ANALYZE THE PROVIDED IMAGE.
            2.  IDENTIFY the precise bounding box that encloses the '{target_description}' on that image.

            STRICT JSON OUTPUT FORMAT - NO EXCEPTIONS:
            - YOUR OUTPUT MUST BE A SINGLE, VALID JSON OBJECT.
            - IF a '{target_description}' CANNOT BE CLEARLY IDENTIFIED ON ANY IMAGE, YOU MUST RETURN AN EMPTY JSON OBJECT: {{}}.

            JSON STRUCTURE (STRICTLY NOTHING ELSE):
            {{
              "box": {{
                "x1": integer,
                "y1": integer,
                "x2": integer,
                "y2": integer
              }}
            }}
            
            KEY DEFINITIONS:
            - "box": The pixel coordinates of the bounding box on that selected image.
            
            COORDINATE SYSTEM RULES:
            - The origin (0,0) is the TOP-LEFT corner of the image.
            - "x1", "y1" are the TOP-LEFT corner of the box.
            - "x2", "y2" are the BOTTOM-RIGHT corner of the box.

            FINAL COMMAND: ANALYZE, SELECT, LOCATE, AND RESPOND WITH THE JSON. ZERO DEVIATION.
        """
        )
    return "Unsupported class_filter"

def _calculate_llm_cost(model: str, prompt_tokens: int, completion_tokens: int) -> float:
    """Berechnet die Kosten für einen API-Aufruf basierend auf vordefinierten Preisen."""
    pricing = MODEL_PRICING.get(model)
    if not pricing:
        print(f"Warnung: Keine Preisinformationen für Modell '{model}' gefunden.")
        return 0.0
    
    input_cost = prompt_tokens * pricing["input"]
    output_cost = completion_tokens * pricing["output"]
    return input_cost + output_cost