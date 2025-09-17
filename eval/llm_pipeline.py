# llm_pipeline.py

import os
import base64
from io import BytesIO
from typing import List, Dict, Any
import cv2
import numpy as np
from PIL import Image
from openai import OpenAI
from dotenv import load_dotenv

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

def run_llm_pipeline(image_paths: List[str], class_filter: str, temperature: float, llm_model: str) -> str:
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

        model_to_use = llm_model 

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
                temperature=temperature
            )

        with timer(times, "postprocessing"):
            output_text = response.output_text
            print(f"LLM-Antwort erhalten: {output_text}") 
            usage = response.usage
            cost = _calculate_llm_cost(model_to_use, usage.input_tokens, usage.output_tokens)

        return {
            "text": output_text, # Der rohe JSON-String vom Modell
            "input_tokens": usage.input_tokens, # Bilder werden zu Token umgewandelt und in input_tokens gezählt
            "cached_input": usage.input_tokens_details.cached_tokens,  
            "completion_tokens": usage.output_tokens,
            "reasoning_tokens": usage.output_tokens_details.reasoning_tokens,
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
            YOU ARE A PRECISION-ENGINEERED DATA EXTRACTION ROBOT. YOUR SOLE FUNCTION IS TO PARSE NUTRITION TABLES FROM IMAGES OF ONE PRODUCT FROM DIFFERENT PERSPECTIVES OF GERMAN FOOD PACKAGING AND CONVERT THEM INTO A FLAWLESS JSON OBJECT. FAILURE TO FOLLOW THESE RULES EXACTLY WILL RESULT IN A CRITICAL SYSTEM MELTDOWN. THERE IS ZERO ROOM FOR ERROR.
            STRICT JSON FORMAT - NO EXCEPTIONS:
            - THE OUTPUT MUST BE A SINGLE, VALID AND COMPACT JSON OBJECT. NO MARKDOWN, NO PARAGRAPHS BETWEEN THE LINES. 
            - NO MARKDOWN, NO EXPLANATIONS, NO HEADERS, NO APOLOGIES, NO TEXT BEFORE OR AFTER THE JSON.
            - THE JSON MUST START WITH { AND END WITH }.
            - IF YOU CAN'T FIND A NUTRITION TABLE ON THE PRODUCT, RETURN AN EMPTY JSON OBJECT: {}.

            JSON STRUCTURE (STRICTLY NOTHING ELSE):
            {
            "title": "string | '' ",
            "columns": ["string" | '' ],
            "rows": [
                {
                "label": "string",
                "values": ["string"]
                }
            ],
            "footnote": "string | '' "
            }

            NON-NEGOTIABLE PARSING RULES:
            1.  **EXTRACT EXACTLY AS WRITTEN. NO MODERNIZATION. NO INTERPRETATION.** 
            2.  **STRUCTURE MAPPING:**
                - **"title":** The main heading of the table (e.g., "Nährwertinformationen", "Nährwerte", "Durchschnittliche Nährwerte"). Everything that comes before the column headers and is clearly part of the table MUST be in title. If the label column has a header, it MUST be included in the title. If no title exists, this field MUST be "".
                - **"columns":** The column headers of the table: The column headers array ALWAYS start with with an empty string ("") for the label column, followed by the actual headers (e.g., ["", "je 100g", "pro Portion 30g"]). If no headers exist, this field MUST be [""]. The items in columns MUST only contain quantities, numbers and prepositions like "je", "pro", "pro", "per". For example: "pro 100ml", "per 100g", "pro Portion". Any other text that is not quantity related MUST be included in the title.
                - **"rows":** An array of objects, where each object represents a row in the table.
                    - **"label":** The nutrient name (e.g., "Energie", "Fett", "davon gesättigte Fettsäuren"). This matches the first column of the table ("").
                    - **"values":** An array of strings containing the corresponding values for that row, in the same order as the "columns" starting from the second column. If a value is missing for a column, it MUST be represented as an empty string ("").
                    - The number of items in "values" + 1 (for the label) MUST EXACTLY MATCH the number of items in "columns". It always has to be the exact number. So the label column must always be represented as ("") in the "columns" key.
                - **"footnote":** Any text below the main table, often marked with an asterisk (e.g., "*Referenzmenge für einen durchschnittlichen Erwachsenen..."). If no footnote exists, this field MUST be "".
            3.  **VALUE NORMALIZATION:**
                - Decimal commas (`,`) MUST be converted to decimal points (`.`).
                - All text must be written in lowercase.
                - Values and units MUST be combined into a single string with no space and ONLY in this sequence: <VALUE> <UNIT>, not <UNIT> <VALUE> (if so, keep space) (e.g., "1,5 g" -> "1.5g", "557 kcal 938 kj -> "557kcal 938kj"). Normal text must be separated with space.
                - Slashes (/), dashes, brackets, quotation marks, colons and multiple spaces MUST be REMOVED.
                - Points that are not decimal separators MUST be REMOVED. 
            4.  **IMPERFECT DATA:**
                - **IF A WORD OR NUMBER IS CUT-OFF OR NOT READABLE AT ALL IT DOES NOT EXIST. OMIT IT.**
                - **IF A ROW OR COLUMN IS INCOMPLETE, EXTRACT ONLY THE READABLE PARTS. DO NOT GUESS MISSING VALUES.** If this results in a row having fewer values than columns, represent the missing values as empty strings `""`.
            5.  **ONLY TRANSCRIBE TEXT FROM THE NUTRITION TABLE ITSELF. ALL OTHER TEXT FROM THE PACKAGING MUST BE OBLITERATED.**

            STRICTLY ENFORCED EXAMPLE OUTPUT:
            {
            "title": "nährwertangaben",
            "columns": [
                "",
                "je 100g",
                "pro portion 30g"
            ],
            "rows": [
                {
                "label": "brennwert",
                "values": [
                    "2253kj 540kcal",
                    "676kj 162kcal"
                ]
                },
                {
                "label": "fett",
                "values": [
                    "31.5g",
                    "9.5g"
                ]
                },
                {
                "label": "davon gesättigte fettsäuren",
                "values": [
                    "18.5g",
                    "5.6g"
                ]
                },
                {
                "label": "kohlenhydrate",
                "values": [
                    "56g",
                    "17g"
                ]
                },
                {
                "label": "davon zucker",
                "values": [
                    "55g",
                    "16.5g"
                ]
                },
                {
                "label": "eiweiß",
                "values": [
                    "6.5g",
                    "2g"
                ]
                },
                {
                "label": "salz",
                "values": [
                    "0.25g",
                    "0.08g"
                ]
                }
            ],
            "footnote": "*rm = referenzmenge für einen durchschnittlichen erwachsenen 8400kj 2000kcal"
            }

            FINAL COMMANDS - FAILURE IS CATASTROPHIC:
            -   **PRODUCE ONLY THE PERFECTLY FORMATTED JSON OBJECT.**
            -   **ANY TEXT OUTSIDE THE JSON STRUCTURE IS A VIOLATION.**
            -   **THE LABEL COLUMN IS ALSO A COLUMN. THE LENGTH OF THE ARRAY 'COLUMNS' MUST BE EQUAL TO THE NUMBER OF VALUES for each label + 1 (as the label column is included).**
            -   **IF YOU CAN'T FIND A NUTRITION TABLE ON THE PRODUCT, RETURN AN EMPTY JSON OBJECT: `{}`.**
            -   **ZERO DEVIATION. ZERO EXCUSES.**
            """
        )
    elif class_filter == "ingredients":
        return (
            """
            YOU ARE A METICULOUS FOOD DATA ANALYST. YOUR TASK IS TO EXTRACT AND TRANSCRIBE THE INGREDIENT LIST FROM MULTIPLE IMAGES WITH DIFFERENT PERSPECTIVES FROM ONE PRODUCT OF GERMAN FOOD PACKAGING. FAILURE TO FOLLOW THESE RULES EXACTLY WILL COMPROMISE THE ENTIRE DATA INTEGRITY. THERE IS ZERO ROOM FOR ERROR.
            STRICT OUTPUT FORMAT - NO EXCEPTIONS:
            - THE OUTPUT MUST BE A SINGLE, CONTINUOUS STRING.
            - DO NOT USE JSON. DO NOT USE MARKDOWN. DO NOT ADD ANY EXPLANATIONS, PREFACES, OR SUMMARIES.
            - THE OUTPUT MUST BE PURE, UNFORMATTED TEXT.
            - IF YOU CAN'T FIND A INGREDIENT LIST, RETURN AN EMPTY STRING.
            NON-NEGOTIABLE TRANSCRIPTION RULES:
            1.  IDENTIFY THE START: The ingredient list begins with the word "Zutaten" or a clear synonym. ALL text before this marker MUST BE IGNORED AND OBLITERATED. If no such marker exists, start with the first logical ingredient.
            2.  IDENTIFY THE END: The list ends precisely where the ingredients and mandatory related declarations (e.g., allergen warnings like "Kann Spuren von...", quantitative declarations like "100g des Produkts werden aus...") conclude. All the text until these ingredient list-related information must be obtained. Information such as best-before dates, manufacturer addresses, or storage instructions ARE NOT PART OF THE INGREDIENT LIST AND MUST BE ERASED FROM EXISTENCE.
            3.  TEXT NORMALIZATION - STRICTLY ENFORCED:
                - ALL TEXT MUST BE CONVERTED TO LOWERCASE.
                - ALL SEMICOLONS (;), COLONS (:) AND SLASHES (/) MUST BE REMOVED. COMMAS AS DECIMAL SEPARATORS MUST BE REPLACED WITH PERIODS. ALL OTHER COMMAS MUST BE REMOVED.
                - PERIODS (.) ARE ONLY PERMITTED BETWEEN DIGITS for decimal numbers (e.g. "0.5"). All other periods must be eliminated.
                - ALL TYPES OF BRACKETS [ { ( ) } ] MUST BE CONVERTED TO STANDARD PARENTHESES ( ).
                - SUPERSCRIPT CHARACTERS (e.g., ¹, ²) MUST BE PRESERVED EXACTLY AS SEEN.
            4.  HYPHENATION - CRITICAL LOGIC:
                - SYLLABLE BREAKS: If a word is split with a hyphen at the end of a line for syllabification, the hyphen MUST be removed and the word parts joined.
                - COMPOUND WORDS: If a hyphen is part of a compound word (e.g., "Mango-Maracuja-Püree"), it IS PART OF THE WORD and MUST BE RETAINED.
            5.  SPACING - ZERO TOLERANCE:
                - There must be NO space between a number and its following unit (e.g., "5%", "10g"). "5 %" is an error. "5%" is correct.
                - Multiple spaces must be collapsed into a single space.
            STRICTLY ENFORCED EXAMPLE OUTPUT:
            zutaten weizenmehl zucker pflanzliche fette (palm kokos) kakaobutter kakaomasse magermilchpulver 5% glukosesirup süßmolkenpulver (aus milch) butterreinfett erdbeeren 0.5% himbeeren emulgator (sojalecithine) backtriebmittel (natriumcarbonate) salz säuerungsmittel (citronensäure) aroma kann spuren von nüssen enthalten¹
            FINAL COMMANDS - FAILURE IS NOT AN OPTION:
            -   PRODUCE ONLY THE FINAL STRING.
            -   NO PREFACES, NO APOLOGIES, NO EXPLANATIONS.
            -   ANY DEVIATION FROM THESE RULES CONSTITUTES A TOTAL MISSION FAILURE.
            """
        )
        return f"Extrahiere alle Informationen bezüglich '{class_filter}' von den Bildern und gib sie als JSON-Objekt zurück."

def _calculate_llm_cost(model: str, prompt_tokens: int, completion_tokens: int) -> float:
    """Berechnet die Kosten für einen API-Aufruf basierend auf vordefinierten Preisen."""
    pricing = MODEL_PRICING.get(model)
    if not pricing:
        print(f"Warnung: Keine Preisinformationen für Modell '{model}' gefunden.")
        return 0.0
    
    input_cost = prompt_tokens * pricing["input"]
    output_cost = completion_tokens * pricing["output"]
    return input_cost + output_cost