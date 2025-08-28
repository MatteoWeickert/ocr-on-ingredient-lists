# traditional_pipeline.py

import cv2
import json
import numpy as np
import pytesseract
import re
import unicodedata
from pathlib import Path
from typing import List, Dict, Any

from rapidfuzz import process, fuzz
from sklearn.cluster import DBSCAN
from sklearn.cluster import AgglomerativeClustering
from ultralytics import YOLO
from pytesseract import Output

ABORT_PHRASES = [
    r"unter schutzatmosph[aä]re",
    r"nach dem [öo]ffnen",
    r"bei [0-9]+ ?[°º]*c",
    r"mindestens",
    r"k[üu]hl und",
    r"kühl lagern"
    r"trocken lagern",
    r"trocken und",
    r"bei diesem produkt",
    r"zubereitung",
    r"unge[öo]ffnet",
    r"hergestellt für"
]

GOLDEN_KEYS = [
    "energie", "brennwert", "durchschnittliche", "durchschnittlich", "nährwerte", "pro", "fett", "gesättigte", "fettsäuren", "nährwertinformationen",
    "kohlenhydrate", "davon", "-davon", "- davon", "zucker", "eiweiß", "salz", "g", "kcal", "kj",
    "%", "mg", "ml", "cal", "kj/kcal", "-zucker", "davon:", "-davon:", "- davon:", r"%rm", r"%rm*", "rm", "-gesättigte", "- gesättigte", "je"
]

# -----------------------------------------------------------
# Mapping: regex-Varianten  ➔  kanonischer Key
KEY_CORRECTIONS = [
    # energie (oft energqe, energ1e, etc.)
    (re.compile(r'en[e3]r[g9lq][1i!l][e3]', re.I | re.U), "energie"),
    
    # brennwert (brenwert, brennwett, brenwertt)
    (re.compile(r'br[e3]nn?w[ea3]r?tt?', re.I), "brennwert"),

    # durchschnittliche (durchschmtthche, durchscnittlich, etc.)
    (re.compile(r'd[uv]rch[a-z0-9]{6,12}', re.I), "durchschnittliche"),

    # nährwerte
    (re.compile(r'n[äa@][h|n]?[r]?w[e3]rt[e3]?', re.I | re.U), "nährwerte"),

    # pro (p0, pr0, etc.)
    (re.compile(r'pr[o0]', re.I), "pro"),

    # fett (fett, fet, fettf, fettl)
    (re.compile(r'f[e3][t7r]{1,3}', re.I), "fett"),

    # gesättigte (ges5tigete, ge5attigte, gessattigte)
    (re.compile(r'ge[s5]{1,2}[äa][t+]{1,2}[i1l!]g?[t+][e3]?', re.I | re.U), "gesättigte"),

    # fettsäuren (fettsaüren, fett5auren, fetztzauren)
    (re.compile(r'f[e3][t7]{1,2}s[äa][uvv][r]?[e3]n?', re.I | re.U), "fettsäuren"),

    # kohlenhydrate (koilenhydrat, koh1enhydrat, kohlenhdrate, etc.)
    (re.compile(r'k[o0][hl1][e3]n[h|n][yuv][dcl][r][aä][t+][e3]?', re.I | re.U), "kohlenhydrate"),

    # davon (dav0n, davon)
    (re.compile(r'd[a4uo][vw][o0][nm]', re.I), "davon"),

    # zucker (zvcker, zuccer, zucker)
    (re.compile(r'z[uuv][ck]{1,2}[e3]r', re.I), "zucker"),

    # eiweiß (eiweis, eiw3is, eiweiB, eiwe1ß)
    (re.compile(r'e[i1l][wvv][eo3][i1lr][sß]', re.I | re.U), "eiweiß"),

    # salz (sa1z, sa|z, sa!z)
    (re.compile(r's[aä@][l1|!][z2]', re.I), "salz"),

    # kJ (kJ, kj, k j, k.j., k-j.)
    (re.compile(r'^(k|kj|k j|k\.j\.|k-?j\.?)$', re.I), "kj"),

    # kcal (kcal, kcal., kcal-, k cal, k.cal, k-cal)
    (re.compile(r'^(kcal|k cal|k\.cal|k-?cal)$', re.I), "kcal"),

    # nährwertinformationen 
    (re.compile(r'n[äa@]hr?w[e3]rt.{5,15}[t7]ion[e3]n', re.I | re.U), "nährwertinformationen"),
    
    # zutaten
    (re.compile(
        r'\b[2z][\s\.\-–—_]*[uüv][\s\.\-–—_]*[t\+7][\s\.\-–—_]*[aä@][\s\.\-–—_]*[t\+7][\s\.\-–—_]*[e3][\s\.\-–—_]*[nm]\b',
        re.I | re.U
    ), "zutaten"),

    # kann  (kan, kann, kaan, kanm, etc.)
    (re.compile(r'\bka+n+[nm]?\b', re.I), "kann"),

    # enthalten  (enthalten, enthaten, entha1ten, entfalten -> optional f/h/l)
    (re.compile(r'\be[nm]t[hf]?[aä@][lt1i][e3]n\b', re.I | re.U), "enthalten"),

    # spuren  (spuren, spurn, spuren., spurrn)
    (re.compile(r'\bspu[rr]?[e3]n\b', re.I), "spuren"),

    # ei  (ei, eier, eii, el, li -> häufig verlesen)
    (re.compile(r'\b(e[i1l]{1,2})\b', re.I), "ei"),

    # glutenfrei  (gluten frei, glvtenfrei, glütenfrei, glu+enfrei)
    (re.compile(r'\bgl[uuv][t+][e3]n\s*fr[e3][i1l]\b', re.I | re.U), "glutenfrei")


]
# -----------------------------------------------------------

# ==============================================================================
# ÖFFENTLICHE METHODE ZUM ABRUFEN 
# ==============================================================================

def run_traditional_pipeline(model: YOLO, paths: List[Path], target_id: int, out_dir: Path, product_id: str) -> Dict[str, Any]:
    """
    Öffentliche Hauptfunktion, die die traditionelle Pipeline ausführt.
    Diese Funktion wird vom Evaluationsskript aufgerufen.
    """
    className_to_id = {"ingredients": 0, "nutrition": 1}
    id_to_className = {v: k for k, v in className_to_id.items()}
    
    target_class = id_to_className.get(target_id)
    if not target_class:
        raise ValueError(f"Ungültige Target-ID: {target_id}")

    # Die eigentliche Logik wird in einer internen Funktion ausgeführt
    result = _execute_extraction_logic(model, paths, target_class, target_id, out_dir, product_id)
    
    # Ergebnis für das Evaluationsskript formatieren
    structured_data = result.get("structured_data", {})
    
    # Konvertiere das strukturierte Ergebnis in einen String für einfache Metriken (CER/WER)
    ocr_text_for_metrics = json.dumps(structured_data, sort_keys=True, ensure_ascii=False) if structured_data else ""

    return {
        "structured_data": structured_data,
        "yolo_result": result.get("yolo_result")
    }

# ==============================================================================
# INTERNE KERNLOGIK (interne `main`-Methode)
# ==============================================================================

def _execute_extraction_logic(model: YOLO, image_paths: List[Path], target_class: str, target_class_id: int, out_dir: Path, product_id: str) -> Dict[str, Any]:
    """
    Führt den Hauptprozess von YOLO-Detektion bis zum Post-Processing durch.
    """

    # 1. YOLO-Detektion
    images = [cv2.imread(str(path)) for path in image_paths]
    results = model(images)

    best_box, best_conf, best_image_path = None, -1, None
    for img_path, result in zip(image_paths, results):
        for box in result.boxes:
            if int(box.cls[0]) == target_class_id and float(box.conf[0]) > best_conf:
                best_conf = float(box.conf[0])
                best_box = box
                best_image_path = img_path
    
    if best_box is None:
        return {"structured_data": {}, "yolo_result": None}

    yolo_result = {"box": best_box.xyxy[0].tolist(), "confidence": best_conf}

    # 2. Bild zuschneiden und vorverarbeiten
    best_image = cv2.imread(str(best_image_path))
    x1, y1, x2, y2 = map(int, yolo_result["box"])
    cropped = best_image[y1:y2, x1:x2]
    
    if cropped.size == 0:
        return {"structured_data": {"error": "Leere Bounding Box"}, "yolo_result": yolo_result}

    cv2.imwrite(str(out_dir / f"{product_id}_cropped.jpg"), cropped)

    preprocessed = _process_cv2_picture(cropped)
    cv2.imwrite(str(out_dir / f"{product_id}_crop_processed.jpg"), preprocessed)

    # 3. OCR ausführen und Ergebnis speichern
    config = r'--oem 1 --psm 4'
    ocr_raw_string = pytesseract.image_to_string(preprocessed, lang='deu', config=config)
    ocr_data = pytesseract.image_to_data(preprocessed, lang='deu', config=config, output_type=Output.DICT)

    ocr_visualize_path = out_dir / f"{product_id}_ocr_visualized.jpg"
    _visualize_ocr_boxes(preprocessed, ocr_data, ocr_visualize_path)

    # 4. Klassenspezifisches Post-Processing
    structured_data = {}
    if target_class == "ingredients":
        processed_text = _normalize_ingredients(ocr_raw_string)
        raw_text = _formate_raw_text(ocr_raw_string)
        structured_data = {"ingredients_text": processed_text, "raw_text": raw_text}

    elif target_class == "nutrition":
        processed_text = _process_nutrition_table(ocr_data)
        raw_text = _formate_raw_text(ocr_raw_string)
        structured_data = {"nutrition_table": processed_text, "raw_text": raw_text}

    return {"structured_data": structured_data, "yolo_result": yolo_result}

# ==============================================================================
# SPEZIFISCHE VERARBEITUNGS-PIPELINES
# ==============================================================================

def _process_nutrition_table(ocr_data: Dict) -> Dict:
    """
    Führt die komplette Nährwerttabellen-Analyse durch, von der Box-Extraktion
    bis zur Erstellung des finalen JSON-Objekts.
    """

    boxes = []

    for i, txt in enumerate(ocr_data["text"]):
        txt = txt.strip()
        if not txt:
            continue
        x1, y1 = ocr_data["left"][i],  ocr_data["top"][i]
        w , h  = ocr_data["width"][i], ocr_data["height"][i]

        # print(txt)
        
        normalized_txt = _normalize_nutrition(txt)

        # print(f"Normalized Text: {normalized_txt}")
        boxes.append((x1, y1, x1 + w, y1 + h, normalized_txt))

    heights = [box[3] - box[1] for box in boxes] # y2 - y1

    row_tolerance = 0.5 * np.median(heights) # halbe Zeilenhöhe

    # Sortiere Boxen nach mittlerer Höhe
    boxes.sort(key=lambda b: (b[1] + b[3]) / 2)       # center‑y
    
    rows = []  # Liste von Zeilen -> jede Zeile ist List[box]
    for box in boxes:
        cy = (box[1] + box[3]) / 2
        if not rows: # wenn rows leer
            rows.append([box])
            continue

        # Prüfe Abstand zur vorigen Zeile
        last_row_cy = np.mean([(box[1] + box[3]) / 2 for box in rows[-1]]) # bilde Mittelwert aus allen Boxmittelpunkten der letzten Zeile 
        if abs(cy - last_row_cy) <= row_tolerance:          # Box gehört zur selben Zeile
            rows[-1].append(box)
        else:                                         # neue Zeile beginnen
            rows.append([box])

    # Gib das vorläufige prozessierte Ergebnis aus
    print(f"\nErgebnis vor finalem Line - Postprocessing:")
    for i, row in enumerate(rows, 1):
        # Extrahiere nur den Text für die Anzeige
        line_text = " ".join(box[4] for box in row)
        print(f"{i:02d}: {line_text}")

    ### Processing der Zeilen basierend auf Kontextinformationen
    final_processed_rows = []

    for row in rows: # row ist ein List[Box]
        row.sort(key=lambda b: b[0])  # Sortiere Boxen in der Zeile nach x1 (linke Kante)
        row = _merge_num_unit_in_row(row)  # Mergen von Einheiten in der Zeile
        words = [box[4].strip() for box in row if box[4].strip()]

        # print(f"\nZeile: {words}")

        # Prüfe, ob es sich um eine Energie, Salz oder Nährwertzeile handelt
        is_energy_line = any(w in {"energie", "brennwert"} for w in words)
        is_salt_line = "salz" in words

        # Prüfe zuerst die aktuelle Zeile
        is_nutrition_line = any(w in {"nährwerte", "nährwertinformationen", "pro", "je"} for w in words)
        
        # Wenn nicht in der aktuellen Zeile gefunden, prüfe die vorherige
        if not is_nutrition_line:
            try:
                current_row_index = rows.index(row)
                if current_row_index > 0:
                    previous_row_boxes = rows[current_row_index - 1]
                    # Extrahiere die Wörter aus den Boxen der vorherigen Zeile
                    words_in_previous_row = [box[4] for box in previous_row_boxes]
                    
                    if any(w in {"nährwerte", "nährwertinformationen", "pro", "je"} for w in words_in_previous_row):
                        is_nutrition_line = True
            except ValueError:
                pass

        corrected_boxes = []
        for i, box in enumerate(row):
            word = box[4].strip()  # Extrahiere den Text aus der Box und entferne Leerzeichen

            # Heuristik 1: "739" -> "7,3g" (Fehler: Komma und Einheit fehlen)
            if re.fullmatch(r'<?[1-9]\d[90]', word) and is_energy_line == False and is_nutrition_line == False and '%' not in word and not (i + 1 < len(words) and re.match(r'^[\(\[\{]*%[\)\]\.,;:]*$', (words[i+1] or '').strip())):
                corrected_word = f"{word[0]}.{word[1]}g"
                print(f"Kontext-Korrektur (Heuristik 1): '{word}' -> '{corrected_word}'")
                new_box = (box[0], box[1], box[2], box[3], corrected_word)  # Ersetze das Wort in der Box
                corrected_boxes.append(new_box)
                continue

            # HEURISTIK 2: "289g" -> "28g"
            elif re.fullmatch(r'<?\d{2}9g', word) and is_energy_line == False and is_nutrition_line == False:
                corrected_word = f"{word[:2]}g"
                print(f"Kontext-Korrektur (Heuristik 2): '{word}' -> '{corrected_word}'")
                new_box = (box[0], box[1], box[2], box[3], corrected_word)  # Ersetze das Wort in der Box
                corrected_boxes.append(new_box)
                continue
            
            # HEURISTIK 3: "2360" -> "23,6g"
            elif re.fullmatch(r'<?\d{3}(g|0)', word) and is_energy_line == False and is_salt_line == False and is_nutrition_line == False :
                corrected_word = f"{word[:2]}.{word[2]}g"
                print(f"Kontext-Korrektur (Heuristik 3): '{word}' -> '{corrected_word}'")
                new_box = (box[0], box[1], box[2], box[3], corrected_word)  # Ersetze das Wort in der Box
                corrected_boxes.append(new_box)
                continue

            # HEURISTIK 4: "16.29" -> "16.2g"
            # Das Muster (Zahl, eine Ziffer, dann eine 9) ist extrem wahrscheinlich ein OCR-Fehler für "...g".
            elif re.fullmatch(r'<?\d+[.,]\d[90]', word) and is_salt_line == False:
                corrected_word = f"{word[:-1]}g"
                print(f"Kontext-Korrektur (Heuristik 4): '{word}' -> '{corrected_word}'")
                new_box = (box[0], box[1], box[2], box[3], corrected_word)  # Ersetze das Wort in der Box
                corrected_boxes.append(new_box)
                continue

            # HEURISTIK 5: "6.29 g" -> "6.2g"
            pattern = r'(\d+[.,]\d)[90]\s*g$' # $ = Ende des Strings
            match = re.match(pattern, word)
            if match and is_salt_line == False:
                base_number = match.group(1)  # z.B. "6.2"
                corrected_word = f"{base_number}g"
                print(f"Kontext-Korrektur (Heuristik 5): '{word}' -> '{corrected_word}'")
                new_box = (box[0], box[1], box[2], box[3], corrected_word)  # Ersetze das Wort in der Box
                corrected_boxes.append(new_box)
                continue

            # Heuristik 6: Salz "200" -> "2,00" (kontextbasiert)
            # ========================================================
            if is_salt_line:

                if re.fullmatch(r'\d{3}9', word):

                    corrected_word = f"{word[0]}.{word[1:3]}g"
                    print(f"Kontext-Korrektur (Salz): '{word}' -> '{corrected_word}'")
                    new_box = (box[0], box[1], box[2], box[3], corrected_word)  # Ersetze das Wort in der Box
                    corrected_boxes.append(new_box)
                    continue

                elif re.fullmatch(r'\d{3}', word):
                    # Wenn es eine 3-stellige Zahl ist, die KEIN "9" am Ende hat,
                    # dann ist es wahrscheinlich ein Fehler für "2,00g".
                    corrected_word = f"{word[0]}.{word[1:]}g"
                    print(f"Kontext-Korrektur (Salz): '{word}' -> '{corrected_word}'")
                    new_box = (box[0], box[1], box[2], box[3], corrected_word)  # Ersetze das Wort in der Box
                    corrected_boxes.append(new_box)
                    continue

                elif re.fullmatch(r'\d[.,]\d{2}9', word) or re.fullmatch(r'\d[.,]\d{2}9g', word):
                    # Wenn es eine Zahl mit 2 Nachkommastellen und einer 9 am Ende ist,
                    # dann ist es wahrscheinlich ein Fehler für "2,00g".
                    corrected_word = f"{word[0]}.{word[2:4]}g"
                    print(f"Kontext-Korrektur (Salz): '{word}' -> '{corrected_word}'")
                    new_box = (box[0], box[1], box[2], box[3], corrected_word)  # Ersetze das Wort in der Box
                    corrected_boxes.append(new_box)
                    continue

            # HEURISTIK 7: "1009" -> "100g"
            if is_nutrition_line and not is_energy_line and re.fullmatch(r'\d{3}(9|0)', word):
                corrected_word = f"{word[:3]}g"
                print(f"Kontext-Korrektur (Heuristik 7): '{word}' -> '{corrected_word}'")
                new_box = (box[0], box[1], box[2], box[3], corrected_word)  # Ersetze das Wort in der Box
                corrected_boxes.append(new_box)
                continue

            # Wenn keine kontextsensitive Regel gegriffen hat, behalte das Wort bei.
            corrected_boxes.append(box)

        final_processed_rows.append(corrected_boxes)

    # Gib das endgültig prozessierte Ergebnis aus
    # print(f"\nErgebnis nach finalem Line - Postprocessing:")
    for i, row in enumerate(final_processed_rows, 1):
        # Extrahiere nur den Text für die Anzeige
        line_text = " ".join(box[4] for box in row)
        print(f"{i:02d}: {line_text}")

    ### BEGINN SPALTENLOGIK ###

    body_start_index = _find_body_start_index(final_processed_rows)
    footer_start_index = _find_footer_start_index(final_processed_rows)
    header_rows = final_processed_rows[:body_start_index]  # Header-Zeilen vor dem Body
    body_rows = final_processed_rows[body_start_index:footer_start_index]  # Body-Zeilen
    footer_rows = final_processed_rows[footer_start_index:]  # Footer-Zeilen

    # Ermittle die Mittelpunkte der Spalten
    centers = _process_calculated_clusters(body_rows, min_row_coverage_ratio=0.2)
    centers = sorted(centers)

    if not centers:
        print("Keine Spaltenzentren gefunden.")
        return {
                "title": "",
                "columns": "",
                "rows": "",
                "footer": ""
            }

    ### BAUEN DES JSON-RESULTATS ###

    rows_raw_text = []
    rows_raw_boxes = []
    used_centers = set()  # Set, um genutzte Cluster-Mittelpunkte zu verfolgen

    for row in body_rows:
        cleaned_row = [box for box in row if box[4].strip()]  # Entferne leere Boxen
        words_per_center = {i: [] for i in range(len(centers))}  # Wörter nach Cluster-ID gruppieren

        for box in sorted(cleaned_row, key = lambda b: (b[0], b[1])):  # Sortiere Boxen nach x1, y1
            cid = int(np.argmin([abs((box[0] + box[2]) / 2 - center) for center in centers]))
            words_per_center[cid].append(box)  # Füge die Box in die entsprechende Cluster-ID ein

        # Zwischenergebnis printen ohne Migration
        #print("\nZwischenErgebnis:")
        # for cid, boxes in words_per_center.items():
        #     if boxes:
        #         #print(f"Spalte {cid}: " + " | ".join(f"{box[4]}" for box in boxes))
        #     else:
        #         #print(f"Spalte {cid}: (leer)")


        SEMANTIC_PAIRS = {"davon": {"gesättigte fettsäuren", "gesättigte", "zucker", "einfach", "mehrfach"}, "gesättigte": "fettsäuren", "einfach": "gesättigte", "mehrfach": "gesättigte", "brennwert": "kj kcal", "energie": "kj kcal"}  # Semantische Paare, die migriert werden können
        # Einheiten, die eine Zahl vor sich benötigen
        VALUE_UNITS = {"g", "mg", "ml", "%"} 

        # Iteriere durch die Spaltenpaare von links nach rechts
        for cid in range(len(centers) - 1):
            source_cid = cid
            target_cid = cid + 1
            
            source_boxes = words_per_center.get(source_cid, [])
            target_boxes = words_per_center.get(target_cid, [])

            if not source_boxes or not target_boxes:
                continue

            # --- Regel 1: Energie-Paar-Zusammenführung ---
            # Sucht nach Mustern wie "... 289 kj" | "273 kcal ..."
            
            source_content = " ".join(b[4] for b in source_boxes)
            target_content = " ".join(b[4] for b in target_boxes)

            # Regex, um einen numerischen Wert (optional mit Einheit kj) am Ende zu finden
            match_kj = re.search(r'(\d+([.,]\d+)?\s*kj)$', source_content)
            
            # Regex, um einen numerischen Wert mit kcal zu finden
            match_kcal = re.search(r'^(\d+([.,]\d+)?\s*kcal)', target_content)

            if match_kj and match_kcal:
                print(f"Energie-Paar-Migration: Füge '{match_kj.group(1)}' und '{match_kcal.group(1)}' zusammen.")
                
                # Finde die Boxen, die zum kj-Block gehören (kann mehr als ein Wort sein)
                kj_block_boxes = []
                temp_content = ""
                for box in reversed(source_boxes):
                    kj_block_boxes.insert(0, box)
                    temp_content = box[4] + " " + temp_content
                    if match_kj.group(1) in temp_content:  # erste Capture-Group des Regex (hier: ganzer Ausdruck)
                        break
                
                # Migriere den Block
                words_per_center[target_cid] = kj_block_boxes + words_per_center[target_cid]
                # Entferne die migrierten Boxen aus der Quelle
                words_per_center[source_cid] = [b for b in source_boxes if b not in kj_block_boxes]
                
                continue

            # Extrahiere die relevanten Wörter für die Prüfung
            last_word_in_source = source_boxes[-1][4].strip()
            first_word_in_target = target_boxes[0][4].strip()
            
            # --- Regel 2: Chirurgische Migration (Zahl zu Einheit / Wort zu Wort) ---
            
            # Fall 2a: Fehlende Zahl vor Einheit (inkl. kj/kcal)
            if any(token in VALUE_UNITS.union({"kj", "kcal"}) for token in first_word_in_target.split()) and re.fullmatch(r'\d+([.,]\d+)?', last_word_in_source):
                print(f"Zahl-zu-Einheit-Migration: Verschiebe '{last_word_in_source}' zu '{first_word_in_target}'.")
                # Migriere nur die letzte Box der Quelle an den Anfang des Ziels
                words_per_center[target_cid].insert(0, source_boxes.pop())
                continue

            # Fall 2b: Semantische Paare
            pair_value = SEMANTIC_PAIRS[last_word_in_source] if last_word_in_source in SEMANTIC_PAIRS else None

            if isinstance(pair_value, set):
                cond = first_word_in_target in pair_value
            else:
                cond = first_word_in_target == pair_value

            if cond:
                print(f"Semantische Paar-Migration: Ziehe '{first_word_in_target}' nach links.")
                # Migriere die erste Box des Ziels an das Ende der Quelle
                words_per_center[source_cid].append(target_boxes.pop(0))

                # Prüfe Sonderfall, dass wenn "gesättigte" rübergezogen wurde, auf "fettsäuren" in derselben Spalte geprüft werden soll und im Fall auch mit rübergezogen werden soll
                if any(box[4].strip() == "gesättigte" for box in words_per_center[source_cid]) and any(box[4].strip() == "fettsäuren" for box in words_per_center[target_cid]):
                    print(f"Sonderfall: Ziehe 'fettsäuren' mit 'gesättigte' nach links.")
                    
                    for box in target_boxes[:]: # Kopie der Liste um alle Indizes richtig zu durchlaufen
                        if box[4].strip() == "fettsäuren":
                            words_per_center[source_cid].append(box)
                            words_per_center[target_cid].remove(box)  # Entferne die Box aus der Zielspalte
                            continue
                continue

        row_text = {}
        row_boxes = {}
        for cid, boxes in words_per_center.items():
            if boxes:
                used_centers.add(cid)  # Tracke genutzte Cluster-Mittelpunkte
                row_boxes[cid] = boxes  # Speichere die Boxen für diese Spalte
                tokens = [box[4] for box in boxes]
                row_text[cid] = re.sub(r'\s+', " ", " ".join(tokens).strip())  # Füge die Tokens zusammen und entferne überflüssige Leerzeichen
            else:
                row_boxes[cid] = []  # Falls keine Boxen vorhanden sind, setze auf leer
                row_text[cid] = ""  # Setze den Text für diese Spalte auf leer

        rows_raw_text.append(row_text)  # Füge die Zeile zum Roh-Output hinzu
        rows_raw_boxes.append(row_boxes)  # Füge die Boxen der Zeile zum Roh-Output hinzu   

    # Nur die Spalten behalten, die IRGENDWO benutzt wurden (Reihenfolge beibehalten)
    non_empty_indices = [i for i in range(len(centers)) if i in used_centers]

    if not non_empty_indices:
        centers_filtered = []
        rows_output = [{"label": "", "values": []} for _ in rows_raw_text]  # Leere Ausgabe, wenn keine Spalten übrig sind

    else:
        centers_filtered = [centers[i] for i in non_empty_indices]
        index_map = {old: new for new, old in enumerate(non_empty_indices)}  # Mappe alte Indizes auf neue

        rows_compact_text = []
        for row in rows_raw_text:
            rows_compact_text.append({index_map[old]: row.get(old, "") for old in non_empty_indices})

            # print("Row compact text:", rows_compact_text)

        rows_compact_boxes = []
        for row in rows_raw_boxes:
            rows_compact_boxes.append({index_map[old]: row.get(old, []) for old in non_empty_indices})
            

    # Finde die Label-Spalte
    label_cluster = _detect_label_cluster(rows_compact_boxes, centers_filtered)

    # Merge "davon" Zeilen, falls Label-Spalte gefunden wurde
    rows_compact_text, rows_compact_boxes = _merge_davon_rows(rows_compact_text, rows_compact_boxes, label_cluster)

    # Merge zweizeilige Energie-Zeilen
    rows_compact_text, rows_compact_boxes = _merge_multiline_energy_rows(rows_compact_text, rows_compact_boxes, label_cluster, centers_filtered)

    # Ermittle Titel, Spaltennamen und Fußnote
    title, column_names = _derive_title_and_column_names(header_rows, centers_filtered, label_cluster)
    footer = " ".join(box[4] for row in footer_rows for box in row if box[4] and box[4].strip())  # Füge den Footer-Text zusammen

    # Erstelle die finale Ausgabe
    rows_output = []
    for row in rows_compact_text:
        label = ""
        if label_cluster is not None and label_cluster in row:
            label = re.sub(r"\s+", " ", row[label_cluster].strip())

        values = []
        for i in range(len(centers_filtered)):
            if i == label_cluster:
                continue
            values.append(re.sub(r"\s+", " ", row.get(i, "").strip()))  # Füge die Werte der Spalten hinzu, außer der Label-Spalte

        rows_output.append({"label": label, "values": values})

    result_json = {
                "title": title,
                "columns": column_names,
                "rows": rows_output,
                "footer": footer
            }
    
    #print("\nErgebnis (JSON-Format):")
    #print(json.dumps(result_json, ensure_ascii=False, indent=4))
    
    return result_json

# ==============================================================================
# HELFERFUNKTIONEN 
# ==============================================================================

def _normalize_ingredients(raw_text: str) -> str:

    text = unicodedata.normalize("NFKC", raw_text)  # Unicode Normalisierung und Kleinschreibung
    text = text.replace("\u00AD", "")
    text = text.translate(str.maketrans({
        "\u2010": "-", "\u2011": "-", "\u2012": "-", "\u2013": "-", "\u2014": "-", "\u2212": "-"
    }))

    text = _join_hyphenated_lines(text)
    
    text = text.replace("\n", " ")

    sub_rules = {
        r'\s+': ' ',
        r'[.;:,„“‚‘`´❝❞°><]': ' ',
        r'[([{]': "(",
        r'[}\])]': ")",
        r'\b[qa]\b': 'g',   # Alleinstehendes q oder a → g
    }

    for pattern, replace in sub_rules.items():
        text = re.sub(pattern, replace, text)

    text = re.sub(r'\s+', ' ', text).strip()

    for regex, replacement in KEY_CORRECTIONS:
        text = regex.sub(replacement, text)  # Wende die Regex-Korrekturen an

    text = _cut_to_ingredients(text)

    number_pattern = re.compile(
        r'^\(?\s*<?\s*'             # optional: Klammer, <
        
        # Optionaler Block für beschreibende Präfixe
        r'(?:ca\.?|approx\.?|~)?\s*' # Erkennt "ca.", "ca", "approx.", "approx", "~"
        
        r'('
            r'\d+(?:[.,/]\d+)*'     # Zahl mit optionalen Dezimaltrennzeichen
            r'(?:\s*[a-zA-Z%]+)?'    # optional Einheit
        r'|'
            r'[a-zA-Z]+/\d+(?:[.,/]\d+)*'
        r')'
        r'(?:\s*/\s*'
            r'\d+(?:[.,/]\d+)*(?:\s*[a-zA-Z%]+)?)*'
        r'\s*\)?$'
    )

    filtered_words = []
    # for word in text.split():
    #     # Korrigiere numerische Fehler

    #     word_without_parenthesis = word
    #     if word_without_parenthesis.startswith("("):
    #         word_without_parenthesis = word_without_parenthesis[1:]  # Entferne die öffnende Klammer
    #     if word_without_parenthesis.endswith(")"):
    #         word_without_parenthesis = word_without_parenthesis[:-1]  # Entferne die schließende Klammer

    #     if (number_pattern.fullmatch(word_without_parenthesis)):
    #         word = re.sub(r'\s+', ' ', word).strip().lower()
    #         filtered_words.append(word)
    #     elif (re.fullmatch(r"[a-zA-ZÄÖÜäöüß]+(?:-[a-zA-ZÄÖÜäöüß]+)*", word_without_parenthesis) and len(word_without_parenthesis) >= 2):
    #         filtered_words.append(word)

    #     elif word_without_parenthesis in GOLDEN_KEYS:
    #         filtered_words.append(word)

    #     else:
    #         print(f"Unpassendes Wort: {word} wird ignoriert.")

    for word in text.split():
        if number_pattern.fullmatch(word):
            filtered_words.append(word)
        elif len(word) >= 2:
            filtered_words.append(word)
        elif word in GOLDEN_KEYS:
            filtered_words.append(word)

    # Zahlen mit Einheiten verschmelzen
    filtered_words = _merge_number_with_unit(filtered_words)

    text = " ".join(filtered_words)  # Füge die gefilterten Wörter wieder zusammen
            
    text = re.sub(r"\s+", ' ', text).strip().lower()

    text = _add_lost_commata(text)  # Füge verlorene Kommas hinzu

    return text

def _merge_number_with_unit(filtered_words: list) -> list:
    merged_words = []
    skip_next = False
    for i, w in enumerate(filtered_words):
        if skip_next:
            skip_next = False
            continue

        # prüfe ob aktuelles Wort eine Zahl ist (z.B. "100", "3.5", "1/2")
        if re.fullmatch(r"\d+(?:[.,/]\d+)?", w):
            if i+1 < len(filtered_words) and re.fullmatch(r"(g|kg|mg|µg|ml|l|cl|dl|%|kcal|kalorien)", filtered_words[i+1]):
                merged_words.append(w + filtered_words[i+1])  # 100 + g → 100g
                skip_next = True
            else:
                merged_words.append(w)
        else:
            merged_words.append(w)

    return merged_words

def _normalize_nutrition(word: str) -> str:

    word = unicodedata.normalize("NFKC", word).lower()

    # print(f"Vor der Normalisierung durch sub_rules: {word}")

    word = re.sub(r'[/:;]', ' ', word)

    # print(f" Nach erster Normalisierung durch sub_rules: {word}")
    sub_rules = {
        # 1. Sonderzeichen-Bereinigung (kombiniert)
        r'[\"\'„"‚\'`´❝❞°=()]': '',  # Alle Anführungszeichen und Klammern entfernen
        
        # 2. Dezimaltrennzeichen normalisieren
        r',': '.',  # Kommas durch Punkte ersetzen
        
        # 3. OCR-Korrekturen für Ziffern (kombiniert)
        r'<[olsB]': lambda m: '<' + {'o': '0', 'l': '1', 's': '5', 'B': '8'}[m.group()[-1]],
        r'\b[olsB],': lambda m: {'o': '0', 'l': '1', 's': '5', 'B': '8'}[m.group()[0]] + '.',
        
        # 4. Einheiten-Korrekturen
        r'k[731!l\]Ww\})]': 'kj',  # kj Korrekturen
        r'k[ceaiou]a[t\]1Il]': 'kcal',  # kcal Korrekturen
        r'([0-9])[ol](?=\s|$)': r'\1g',  # "10o" oder "10l" -> "10g" nur am Wortende
        
        # 5. Trennzeichen normalisieren
        r'(\d)\.(?!\d)': r'\1 ',  # Punkte nach Zahlen (keine Dezimalstelle) -> Leerzeichen
        r'(?<=[A-Za-zÄÖÜäöüß])\.(?=[A-Za-zÄÖÜäöüß])': ' ',  # Punkte zwischen Buchstaben
        
        # 6. Mehrfache Zeichen bereinigen
        r'\.{2,}': '.',  # Mehrere Punkte -> ein Punkt
        r'\s{2,}': ' ',  # Mehrere Leerzeichen -> ein Leerzeichen
        
        # 7. Bindestrich-Bereinigung
        r'(^|\s)-(?=\s*[a-zA-Z])': r'\1',  # Bindestrich vor Buchstaben entfernen
    }

    for pattern, replace in sub_rules.items():
        word = re.sub(pattern, replace, word)

    # Zahl + Einheit im selben Token: Leerzeichen entfernen (z.B. "< 0.5 g" -> "<0.5g")
    word = re.sub(
        r'''(?ix)
            (?<!\w)                 # kein Buchstabe/Ziffer davor
            (<?) \s*                # optional "<"
            (\d+(?:[.,]\d+)*) \s*   # Zahl
            (kj|kcal|g|mg|µg|ug|ml|l|%)  # Einheit
            (?!\w)                  # kein Buchstabe/Ziffer danach
        ''',
        r'\1\2\3',
        word
    )

    stripped = word.strip(" .,;") # entferne Leerzeichen, Punkte, Kommas und Semikolons am Anfang und Ende
    # print(f"Nach der Normalisierung nach sub_rules: {stripped}")
    words = stripped.split() # teile den String in einzelne Wörter
    corrected_words = []

    for word in words:
        for regex, canonical in KEY_CORRECTIONS:
            if regex.fullmatch(word):
                word = canonical
                break
        else:  # fuzzy matching falls kein Regex passt
            fuzzy_corrected = _fuzzy_correct(word, GOLDEN_KEYS, threshold=80)
            if fuzzy_corrected != word:
                word = fuzzy_corrected
                # print(f"Fuzzy-Korrektur: {word}")
        corrected_words.append(word)
        # print(f"Nach der Normalisierung nach KEY_CORRECTIONS: {word}")

    # Nach der Korrektur (Regex + Fuzzy)
    valid_words = set(GOLDEN_KEYS)

    # Regex für Zahlenangaben
    number_pattern = re.compile(
        r'^\s*'                      # Optionale Leerzeichen am Anfang
        r'[\(\<]?\s*'               # Optional: öffnende Klammer oder <
        r'(?:ca\.?|approx\.?|etwa|~|≈|±)?\s*'  # Erweiterte Präfixe
        r'(?:'
            # Hauptzahl-Gruppen:
            
            # 1. Normale Zahlen mit Einheiten
            r'(?:'
                r'\d+(?:[.,]\d+)?'          # Hauptzahl (123 oder 12.3 oder 12,3)
                r'(?:\s*[-–]\s*\d+(?:[.,]\d+)?)?'  # Optional: Bereich (12-15)
                r'(?:\s*[a-zA-ZäöüÄÖÜß%°]+)?'      # Optional: Einheit
            r')'
            
            r'|'
            
            # 2. Brüche (1/2, 3/4, etc.)
            r'(?:'
                r'\d+\s*/\s*\d+'
                r'(?:\s*[a-zA-ZäöüÄÖÜß%°]+)?'
            r')'
            
            r'|'
            
            # 3. Verhältnisse
            # Mischungen (Protein/100g, 5ml/kg, etc.)
            r'(?:'
                r'[a-zA-ZäöüÄÖÜß]+\s*/\s*\d+(?:[.,]\d+)?'
                r'(?:\s*[a-zA-ZäöüÄÖÜß%°]+)?'
            r')'
            
            r'|'
            
            # 4. Komplexe Einheiten (mg/100g, kcal/100ml, etc.)
            r'(?:'
                r'\d+(?:[.,]\d+)?\s*'
                r'[a-zA-ZäöüÄÖÜß%°]+\s*/\s*'
                r'\d+(?:[.,]\d+)?\s*'
                r'[a-zA-ZäöüÄÖÜß%°]+'
            r')'
        r')'
        
        # Optionale weitere Werte (für Listen wie "12g, 15g, 20g")
        r'(?:\s*[,;]\s*'
            r'(?:'
                r'\d+(?:[.,]\d+)?'
                r'(?:\s*[-–]\s*\d+(?:[.,]\d+)?)?'
                r'(?:\s*[a-zA-ZäöüÄÖÜß%°]+)?'
            r')'
        r')*'
        
        r'\s*[\)\>]?\s*$'           # Optional: schließende Klammer oder >
    )

    filtered_words = []
    for word in corrected_words:

        corrected_number = _correct_numeric_errors(word)  # Korrigiere eindeutige numerische Fehler

        parts = corrected_number.split()  # Teile den korrigierten String in Teile

        for part in parts:

            # Prüfe, ob der Teil ein gültiges Wort oder eine Zahl ist
            if part in valid_words or number_pattern.match(part):
                filtered_words.append(part)
            elif len(part) >= 3 and part.isalpha():
                filtered_words.append(part)
            else:
                print(f"Ausreißer entfernt: {word}")


    return " ".join(filtered_words)  # Füge die Wörter wieder zu einem String zusammen

def _formate_raw_text(ocr_raw_string):

    # Hier können Sie die Formatierung des Rohtextes implementieren
    ocr_raw_string = ocr_raw_string.replace("\n", " ").replace("\r", "")
    return ocr_raw_string.strip()

def _process_cv2_picture(img_bgr):
    """
    Vorverarbeitung des Bildes für die Texterkennung
    Erwartet: BGR-Bild
    Liefert: RGB-Bild für Tesseract
    """

    # 1. Upscaling

    height, width = img_bgr.shape[:2]
    min_dim = 1600
    scale = max(1.0, min_dim / min(height, width)) # Prüfe ob die kleinere Seite kleiner als min_dim ist
    if scale > 1.0:
        new_size = (int(round(width * scale)), int(round(height * scale)))
        img_bgr = cv2.resize(img_bgr, new_size, interpolation=cv2.INTER_CUBIC)

    # 2. Deskewing (Bild begradigen)
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    inverted = cv2.bitwise_not(gray) # canny needs white lines on black background to detect edges
    edges = cv2.Canny(inverted, 50, 150, apertureSize=3) # detect edges in the image
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, 100, minLineLength=100, maxLineGap=10) # # detect lines in the image
    
    angles = []
    if lines is not None:
        
        for line in lines:
            x1, y1, x2, y2 = line[0]
            dx, dy = x2 - x1, y2 - y1
            if dx == 0:
                continue
            
            # Winkelberechnung direkt in Grad
            angle = np.degrees(np.arctan2(dy, dx))
            
            # Auf Bereich [-45, 45] falten
            if angle < -45:
                angle += 90
            elif angle > 45:
                angle -= 90

            if -15 < angle < 15:
                angles.append(angle)

    if angles:
        median_angle = np.median(angles)
        (h, w) = img_bgr.shape[:2]
        center = (w // 2, h // 2)
            
        M = cv2.getRotationMatrix2D(center, median_angle, 1.0)

        img_bgr = cv2.warpAffine(img_bgr, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE) # drehe Bild, BorderReplicate füllt Ecken mit Randpixeln

    
    # # 3. Graustufen und Beleuchtungskorrektur
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

    kernel = max(15, (min(gray.shape) // 30) | 1)  # Kernelgröße anpassen, mindestens 15 und ungerade (um Mittelpunkt zu haben), um Hintergrund zu filtern
    background = cv2.medianBlur(gray, kernel) # schätzen von weichem Hintergrund (großer, ungerader Kernel) -> Details verschwinden, Beleuchtung bleibt übrig
    illum = cv2.divide(gray, background, scale=255)# Beleuchtung wird überall vereinheitlicht (Illumination)
    illum = np.clip(illum, 0, 255).astype(np.uint8)
    illum = cv2.addWeighted(gray, 0.6, illum, 0.4, 0) # Divisionseffekt abmildern, um dickere Striche zu behalten -> Mische Teile vom Originalbild mit der beleuchteten Version

    # 4. Highlight und Glare Reduktion -> Überbelichtungen reduzieren
    mask = (illum >= 240).astype(np.uint8) * 255 # Maske für extrem helle Bereiche (nahe 255), die korrigiert werden sollen (werden markiert)
    if cv2.countNonZero(mask) > 0: # Wenn es überbelichtete Bereiche gibt
        # Wende die Maske an, um die überbelichteten Bereiche zu reduzieren
        blurred = cv2.GaussianBlur(illum, (0, 0), sigmaX=3, sigmaY=3) # Berechne Kernelgröße aus der Standardabweichung
        mask_soft = cv2.GaussianBlur(mask, (0, 0), sigmaX=3, sigmaY=3).astype(np.float32) / 255.0 # Weiche Maske für die Überbelichtungsbereiche (keine harte) zwischen 0 (kein Highlight) und 1 (Highlight)
        glare = (illum.astype(np.float32) * (1 - mask_soft) + blurred.astype(np.float32) * mask_soft) # wenn maske = 1 -> nimm das geglättete Bild, sonst nimm das originale
        glare = np.clip(glare, 0, 255).astype(np.uint8)  # Stelle sicher, dass die Werte im gültigen Bereich liegen
    else:
        glare = illum

    # 4. CLAHE (Erhöhung von lokalen Kontrasten) -> Teilt Bild in Grids und berechnet für Grid die Kontraste
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(glare)

    # 5. Denoising (non local means) -> suche nach ähnlichen Pixelmustern und reduziere Rauschen (Elemente, die nicht zu den Mustern gehören)
    denoised = cv2.fastNlMeansDenoising(clahe, None, h=10, templateWindowSize=10, searchWindowSize=21)

    # 6. Sanftes Unsharp Masking (Unsharp Masking) -> Schärfe erhöhen, ohne Rauschen zu verstärken
    blur = cv2.GaussianBlur(denoised, (0, 0), sigmaX=0.8, sigmaY=0.8) # leichte weichzeichnung des denoised bild (nur kleine glättung mit 0.8)
    sharp = cv2.addWeighted(denoised, 1.5, blur, -0.6, 0) # man subtrahiert die weichzeichnung und erhält dadurch Kanten
    sharp = np.clip(sharp, 0, 255).astype(np.uint8)

    # 7. Binarisierung (Otsu)
    _, binary = cv2.threshold(sharp, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    black_ratio = 1.0 - (binary.mean() / 255.0)
    if black_ratio < 0.05 or black_ratio > 0.95:
        binary = cv2.adaptiveThreshold(
            sharp, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY,
            blockSize=31, C=10
        )

    num_white_pixels = cv2.countNonZero(binary)
    height, width = binary.shape
    total_pixels = height * width

    # wenn weniger als die Hälfte aller Pixel weiß sind, dann invertiere das Bild (da Hintergrund dann schwarz ist)
    if num_white_pixels < (total_pixels / 2):
        binary = cv2.bitwise_not(binary)

    # Fülle Löcher in der Binarisierung
    closing_kernel = np.ones((3, 3), np.uint8)
    opening_kernel = np.ones((6, 6), np.uint8)
    inv = cv2.bitwise_not(binary) # weil Morphologie auf weiße Bereiche (Vordergrund) wirkt
    inv = cv2.morphologyEx(inv, cv2.MORPH_CLOSE, closing_kernel, iterations=1)
    inv = cv2.morphologyEx(inv, cv2.MORPH_OPEN, opening_kernel, iterations=1)
    binary = cv2.bitwise_not(inv)

    output_rgb = cv2.cvtColor(binary, cv2.COLOR_GRAY2RGB)
    return output_rgb  # Rückgabe des vorverarbeiteten Bildes

def _fuzzy_correct(word, key_list, threshold=80):
    match, score, _ = process.extractOne(word, key_list, scorer=fuzz.ratio)
    if score >= threshold:
        return match  # Korrigiere zum gefundenen Key
    return word      # Lass Wort unverändert

def _correct_numeric_errors(word: str) -> str:
        """
        Korrigiert eindeutige OCR-Fehler in numerischen Werten.
        """

        # Fall 1 - Alleinstehende Null gefolgt von einer 9. -> Wahrscheinlich "0g".
        # Passt NUR auf "09".
        if re.fullmatch(r'09', word):
            corrected = "0g"
            print(f"Sichere numerische Korrektur (g-als-9-Fehler): '{word}' -> '{corrected}'")
            return corrected

        # Fall 2 - Führende Null mit expliziter 'g'-Einheit. -> Wahrscheinlich "0,Xg".
        # Passt auf "02g", "05g", aber NICHT auf das alleinstehende "02".
        elif re.fullmatch(r'0\d+g', word): 
            numeric_part = word.removesuffix('g')
            corrected = f"{numeric_part[0]}.{numeric_part[1:]}g"
            print(f"Sichere numerische Korrektur (fehlendes Komma): '{word}' -> '{corrected}'")
            return corrected
        
        return word

def _split_label_values(line):
        """
        Teilt eine Zeile in einen Label-Teil (Text) und einen Werte-Teil (Zahlen/Einheiten).
        """
        tokens = line.split()
        split_index = -1
        for i, token in enumerate(tokens):
            # Prüfe, ob der Token mit einer Ziffer beginnt.
            # Es erfasst "435", "2.0g", "0.8" etc.
            if re.match(r'^\d', token):
                split_index = i
                break  # Wir haben den ersten Wert gefunden, die Suche kann stoppen.

        if split_index != -1:
            # Alles vor dem Split-Punkt ist das Label
            key = ' '.join(tokens[:split_index])
            # Alles ab dem Split-Punkt ist der Wert
            value = ' '.join(tokens[split_index:])
            return key, value
        return line, ''

def _combine_splitted_rows(lines: list) -> list:
    merged_lines = []
    i = 0
    # Keywords, die oft am Anfang der zweiten Zeile eines getrennten Keys stehen
    CONTINUATION_KEYWORDS = {'fettsäuren', 'gesättigte', 'zucker'}
    
    while i < len(lines):
        current_line = lines[i]
        # Prüfe, ob die Zeile nur Text enthält (wahrscheinlich ein halber Key)
        if not any(char.isdigit() for char in current_line):
            # Prüfe, ob die nächste Zeile existiert und mit einem Continuation-Keyword beginnt
            if i + 1 < len(lines) and lines[i+1].split() and lines[i+1].split()[0] in CONTINUATION_KEYWORDS:
                # Füge die Zeilen zusammen
                merged_lines.append(current_line.strip() + ' ' + lines[i+1].strip())
                i += 2  # Überspringe die nächste Zeile, da sie verarbeitet wurde
                continue
        
        merged_lines.append(current_line)
        i += 1
    # print("Nach Merge:", merged_lines)
    return merged_lines

def _compute_clusters_by_DBSCAN(processed_rows):
    """
    Findet Spaltenzentren robust mit DBSCAN-Clustering.
    Dieser Ansatz ist adaptiv und benötigt keinen manuellen Faktor.
    """
    all_boxes = [box for row in processed_rows for box in row]
    if len(all_boxes) < 2:
        return []

    # 1. Sammle alle X-Mittelpunkte und Wort-Breiten
    x_centers = np.array([(box[0] + box[2]) / 2 for box in all_boxes]).reshape(-1, 1)
    word_widths = [box[2] - box[0] for box in all_boxes if (box[2] - box[0]) > 0]

    if not word_widths:
        return [np.mean(x_centers)]

    # Schätze den adaptiven Nachbarschaftsradius (eps)
    # 'eps' ist der maximale Abstand zwischen Wörtern in derselben Spalte.
    median_width = np.median(word_widths)
    eps = median_width  # Erlaube einen Abstand von der medianen Wortbreite

    # Führe DBSCAN aus
    # min_samples: Wie viele Wörter müssen minimal eine Spalte bilden (Minimalwert 2, um keine Spalten zu verlieren)
    db = DBSCAN(eps=eps, min_samples=2).fit(x_centers)
    
    # Die Labels_ enthalten die Cluster-ID für jedes Wort
    labels = db.labels_
    unique_labels = set(labels)
    
    # Berechne die Zentren der gefundenen Cluster
    centers = []
    for k in unique_labels:
        if k == -1:
            continue
        
        cluster_points = x_centers[labels == k]
        # Berechne den Mittelpunkt dieses Clusters
        center = np.mean(cluster_points)
        centers.append(center)
        
    print(f"DBSCAN: Mediane Wortbreite={median_width:.2f}, eps={eps:.2f}, Gefundene Spalten={len(centers)}")
    
    return sorted(centers)

def _merge_clusters_semantically(centers, processed_rows):
    """
    
    """
    if not centers:
        return []
    
    SEMANTIC_PAIRS = {
        "gesättigte": "fettsäuren",
        "davon": "zucker",
        "kj": "kcal"
    }

    merged_centers = sorted(centers) 

    for row in processed_rows:
        for i, box in enumerate(row):
            word1 = box[4]  # Text des Wortes
            if word1 in SEMANTIC_PAIRS and i < len(row) - 1:
                box2 = row[i + 1]  # Nächstes Wort
                word2 = box2[4]  # Nächstes Wort

                if word2 == SEMANTIC_PAIRS[word1]:
                    # Finde den Cluster des ersten Wortes
                    cid1 = int(np.argmin([abs((box[0] + box[2]) / 2 - center) for center in merged_centers]))
                    # Finde den Cluster des zweiten Wortes
                    cid2 = int(np.argmin([abs((box2[0] + box2[2]) / 2 - center) for center in merged_centers]))

                    if cid1 != cid2:
                        center1 = merged_centers[cid1]
                        center2 = merged_centers[cid2]

                        print(f"Semantisches Merging: Führe Cluster um {center1:.0f} und {center2:.0f} zusammen wegen '{word1} {word2}'.")

                        new_center = (center1 + center2) / 2
                        merged_centers[cid1] = new_center

                        temp_centers = [center for i, center in enumerate(merged_centers) if i != cid2 and i != cid1]
                        temp_centers.append(new_center)
                        merged_centers = sorted(temp_centers)  # Sortiere die neuen Zentren

                        return _merge_clusters_semantically(merged_centers, processed_rows)  # Rekursiv aufrufen, um weitere Zusammenführungen zu prüfen
                    
    return merged_centers  # Rückgabe der finalen Zentren nach semantischem Merging

def _compute_clusters_by_x(x_values):
    """
    x_values sind die mittleren x-Werte aller erkannten Wörter
    Rückgabe sind die Mittelpunkte aller Cluster und die x-Werte je Cluster als verschachtelte Liste
    """

    if not x_values:
        return []
    
    xs_sorted = sorted(x_values)
    if len(xs_sorted) == 1:
        return(xs_sorted[0], xs_sorted[0])
    
    gaps = []

    for i in range(len(xs_sorted) - 1):
        gap = xs_sorted[i+1] - xs_sorted[i]
        gaps.append(gap)

    median = float(np.median(gaps))
    mad = float(np.median([abs(gap - median) for gap in gaps]))  # Median der absoluten Abweichungen vom Median
    print("mad:", mad)

    # Definiere Trennschwelle ab wann neue Spalte
    T = median + 21 * (mad if mad > 0 else median)

    clusters = [[xs_sorted[0]]] # Der erste Wert ist immer der Beginn einer Spalte

    for i in range(1, len(xs_sorted)):
        gap = xs_sorted[i] - xs_sorted[i-1]
        if gap > T:
            clusters.append([xs_sorted[i]])
        else:
            clusters[-1].append(xs_sorted[i])

    centers = [float(np.mean(cluster)) for cluster in clusters]

    return centers

def _compute_clusters_by_agglomerative_clustering(processed_rows, all_boxes, min_coverage_ratio, distance_threshold=25.0, min_cluster_size=2):
    """
    Spaltendetection mit Agglomerative Clustering
    """

    all_boxes = [box for row in processed_rows for box in row]
    if len(all_boxes) < 2:
        return []
    
    # 1. Extrahiere alle mittleren x-Werte
    x_coords = np.array([(box[0] + box[2]) / 2 for box in all_boxes])

    # 2. 2D-Array für Clustering erstellen
    coordinates = np.column_stack([x_coords, np.zeros(len(x_coords))])
    
    # 3. Wende Agglomerative Clustering an
    clustering = AgglomerativeClustering(
        n_clusters=None,                    # Lass Algorithmus Anzahl bestimmen
        distance_threshold= 1.7 * distance_threshold,  # Stopp-Kriterium für Clustering
        metric='manhattan',               # Manhattan-Distanz (gut für X-Koordinaten)
        linkage='complete'                  # Complete linkage (max distance in cluster)
    )

    cluster_labels = clustering.fit_predict(coordinates)

    unique_labels = np.unique(cluster_labels)
    column_centers = []
    for label in unique_labels:
        # Finde alle Punkte in diesem Cluster
        cluster_mask = cluster_labels == label
        cluster_points = x_coords[cluster_mask]
        
        # Prüfe Mindestgröße
        if len(cluster_points) >= min_cluster_size:
            # Berechne Spalten-Zentrum als Durchschnitt
            column_center = np.mean(cluster_points)
            column_centers.append(column_center)
            
            print(f"  Cluster {label}: {len(cluster_points)} Punkte, Zentrum bei X={column_center:.1f}")
        else:
            print(f"  Cluster {label}: {len(cluster_points)} Punkte - zu klein, ignoriert")

    # 5. Validiere Spalten-Coverage (optional)
    validated_centers = []
    for center in column_centers:
        coverage = _calculate_column_coverage(center, processed_rows, distance_threshold/2)
        if coverage >= min_coverage_ratio:
            validated_centers.append(center)
            print(f"  ✓ Spalte X={center:.1f} akzeptiert (Coverage: {coverage:.2f})")
        else:
            print(f"  ✗ Spalte X={center:.1f} abgelehnt (Coverage: {coverage:.2f})")
    
    return sorted(validated_centers)

def _calculate_column_coverage(center_x, processed_rows, tolerance):
    """Berechnet Coverage einer Spalte (wie viele Zeilen haben ein Element in dieser Spalte)."""
    
    rows_with_element = 0
    
    for row in processed_rows:
        has_element = False
        for box in row:
            box_center = (box[0] + box[2]) / 2
            if abs(box_center - center_x) <= tolerance:
                has_element = True
                break
        
        if has_element:
            rows_with_element += 1
    
    return rows_with_element / len(processed_rows) if processed_rows else 0

def _compute_columns_agglomerative_advanced(processed_rows, all_boxes, min_row_coverage_ratio):
    """
    Erweiterte Version mit automatischer Parameter-Schätzung.
    """
    
    all_boxes = [box for row in processed_rows for box in row]

    # 1. Schätze optimalen distance_threshold automatisch
    word_widths = [box[2] - box[0] for box in all_boxes if (box[2] - box[0]) > 0]
    
    if not word_widths:
        return []
    
    # Heuristik: distance_threshold = mittlere Wortbreite
    # Grund: Wörter in derselben Spalte sollten näher als eine Wortbreite sein
    median_word_width = np.median(word_widths)
    distance_threshold = median_word_width
    
    # 2. Schätze min_cluster_size basierend auf Anzahl Zeilen
    min_cluster_size = max(2, len(processed_rows) // 1.4)  # Mindestens 70% der Zeilen

    print(f"Auto-Parameter: distance_threshold={distance_threshold:.2f}, min_cluster_size={min_cluster_size}")
    
    return _compute_clusters_by_agglomerative_clustering(
        processed_rows, 
        all_boxes,
        min_row_coverage_ratio, 
        distance_threshold, 
        min_cluster_size
    )

def _process_calculated_clusters(processed_rows, min_row_coverage_ratio):
    """
    Funktion nimmt von allen Wörtern die x-Werte, führt basierend darauf ein Clustering durch und entfernt bzw. mergt zu schwache Cluster, die am Ende die vorhandenen Spalten darstellen
    """

    x_values = []
    all_boxes = []

    for i, row in enumerate(processed_rows):
        for box in row:
            x_values.append((box[0] + box[2]) / 2) # Füge den mittleren x-Wert des Wortes hinzu
            all_boxes.append((i, box))  # Füge Zeilennummer + Box hinzu

    # centers = _compute_clusters_by_x(x_values)
    # centers = _compute_clusters_by_DBSCAN(processed_rows, all_boxes)  # Alternativ: DBSCAN verwenden
    #centers = _merge_clusters_semantically(raw_centers, processed_rows)  # Semantisches Merging der Cluster
    centers = _compute_columns_agglomerative_advanced(processed_rows, all_boxes, min_row_coverage_ratio=0.2)

    if len(centers) < 2:
        print("Weniger als 2 Spalten gefunden, versuche mit niedrigeren Parametern...")
        
        # Fallback mit niedrigeren Anforderungen
        centers = _compute_clusters_by_agglomerative_clustering(
            processed_rows, 
            all_boxes,
            min_coverage_ratio=0.1,  # Niedrigere Coverage-Anforderung
            distance_threshold=50.0,     # Größerer Threshold
            min_cluster_size=1           # Kleinere Cluster erlauben
        )

    return centers

    # if not centers:
    #     return []

    # box_to_cluster = []
    # for row, box in all_boxes:
    #     x_center = (box[0] + box[2]) / 2
    #     center_id = int(np.argmin([abs(x_center - center) for center in centers])) # np.argmin gibt den Index des kleinsten Wertes zurück
    #     box_to_cluster.append((center_id, row))


    # # Prüfe, ob einem Cluster eine Mindestanzahl an Zeilen zugeordnet wurde, um Abdeckung zu prüfen
    # n_rows = len(processed_rows)
    # min_rows = max(2, int(np.ceil(min_row_coverage_ratio * n_rows)))  # Mindestens 2 Zeilen oder ein Prozentsatz der Gesamtzeilen

    # coverage = [] # Gibt die Menge an vorhandenen verschiedenen Zeilen in einem Cluster an 

    # for center_id in range(len(centers)):
    #     different_rows = {r for (c, r) in box_to_cluster if center_id == c} # Füge ZeilenID hinzu, wenn sie in dem Cluster vorkommt
    #     coverage.append(len(different_rows))

    # # Merge schwache Cluster mit Nachbarn
    # keep = [True] * len(centers)
    # for center_id, coverage in enumerate(coverage):
    #     if coverage < min_rows:
    #         # zu wenige Zeilen -> markiere, dass gemergt werden muss
    #         keep[center_id] = False

    # if all(keep) == True:
    #     return centers
    
    # kept_centers = [c for i, c in enumerate(centers) if keep]

    # if kept_centers:
    #     return kept_centers # Merge noch einbauen?
    
    # return [float(np.mean(x_values))] # falls alle Cluster entfernt werden, nimm die mittlere aller x-Werte als Spalte

def _is_numeric(word: str) -> bool:
    """ Überprüft, ob ein Wort eine Zahl oder eine Zahl mit Einheit ist. """

    return bool(re.search(r"\d", word))

def _detect_label_cluster(processed_rows, centers):
    """
    Funktion nimmt die verarbeiteten Zeilen und die Cluster-Mittelpunkte und versucht, die Spalte mit den Labels (Energie etc.), welche die geringste Anzahl an Zahlen enthält, zu finden.
    """

    if not centers:
        return None
    
    counts = []

    for cid, center in enumerate(centers):
        tokens = []

        for row in processed_rows:
            for box in row:
                boxes = row.get(cid, [])
                for box in boxes:
                    # box = (x1, y1, x2, y2, text)
                    tokens.append(box[4])

        if not tokens:
            number_ratio = 1.0

        else:
            number_ratio = sum(1 for token in tokens if _is_numeric(token)) / len(tokens)  # Anteil der Zahlen in den Tokens

        counts.append((cid, number_ratio, center))  # Speichere die Cluster-ID, den Anteil der Zahlen und den Mittelpunkt


    # Finde den Cluster mit dem niedrigsten Anteil an Zahlen
    cid, number_ratio, _ = sorted(counts, key=lambda x: (x[1], x[2]))[0]  # Cluster mit dem niedrigsten Anteil an Zahlen - bei Gleichstand das Cluster weiter links

    if number_ratio > 0.6:
        cid = int(np.argmin(centers))  # Wenn alle Cluster zu viele Zahlen haben, nimm das Cluster ganz links

    return cid

def _find_body_start_index(processed_rows):
    """
    Findet den Index der ersten Zeile, die mit einem Label beginnt.
    """

    START_KEYWORDS = {"energie", "brennwert", "fett", "gesättigte", "fettsäuren", "kohlenhydrate", "zucker", "eiweiß", "salz"}
    for i, row in enumerate(processed_rows):
        words = {box[4] for box in row}  # Extrahiere die Wörter aus der Box
        if words & START_KEYWORDS:
            return i
    return len(processed_rows)  # Falls kein Start-Keyword gefunden, gib die Länge der Liste zurück (keine Body-Zeilen)

def _find_footer_start_index(processed_rows):
    """
    Findet den Index der ersten Zeile, die zum Footer gehört.
    """

    END_KEYWORDS = {"salz"}

    for i, row in enumerate(processed_rows):
        words = {box[4] for box in row}  # Extrahiere die Wörter aus der Box
        if words & END_KEYWORDS:
            return i + 1  # Rückgabe des Index der ersten footer-Zeile

    return len(processed_rows)  # Falls kein Footer-Keyword gefunden, gib die Länge der Liste zurück (keine Footer-Zeilen)

def _is_number_or_unit(tok: str) -> bool:
    t = tok.lower()
    if re.fullmatch(r'[<>]?\d+(?:[.,]\d+)?', t):
        return True
    if any(u in t for u in ("g", "mg", "ml", "%", "kj", "kcal")):
        return True
    return False

def _row_metrics(row, centers):
    # Breiten/Koordinaten
    x1s  = [b[0] for b in row]
    x2s = [b[2] for b in row]
    mids = [0.5*(b[0]+b[2]) for b in row]
    widths = [b[2]-b[0] for b in row]
    row_x1, row_x2 = (min(x1s), max(x2s)) if row else (0, 0)
    row_width = max(1, row_x2 - row_x1)

    # Füllfaktor
    coverage = (sum(widths) / row_width) if row else 0

    # Alignment an Centers
    if row and centers:
        tol = np.median(widths) * 0.2  # Toleranz für die Zentren
        aligned = 0
        hit_cids = set()
        for mid in mids:
            diffs = [abs(mid - center) for center in centers]
            cid = int(np.argmin(diffs))  # Finde den nächsten Cluster
            if diffs[cid] < tol:
                aligned += 1
                hit_cids.add(cid)
        aligned_words_ratio = aligned / len(row) if row else 0 # Anteil der Wörter, die zu einem Cluster passen
    else:
        aligned_words_ratio = 0
        hit_cids = set()
        tol = 0

    # Content-Hinweise
    texts = [b[4] for b in row]
    has_header_keywords = any(key in texts for key in (" je ", " pro ", " per ", "je", "pro"))
    numeric_ratio = (sum(1 for t in texts if _is_number_or_unit(t)) / len(texts)) if texts else 0.0

    # Vertikales Maß
    y_top = min((b[1] for b in row), default=0)
    y_bot = max((b[3] for b in row), default=0)

    return {
        "row_w": row_width,
        "coverage": coverage,
        "aligned_words_ratio": aligned_words_ratio,
        "hit_cols": len(hit_cids),
        "has_header_keywords": has_header_keywords,
        "numeric_ratio": numeric_ratio,
        "y_top": y_top,
        "y_bot": y_bot,
        "texts": texts,
    }

def _derive_title_and_column_names(header_rows, centers, label_cluster):
    """
    Leitet die Spaltennamen aus den Header-Zeilen ab, basierend auf den Cluster-Mittelpunkten und ermittelt einen eventuellen Titel.
    """
    if not centers or not header_rows:
        return "", [""] * len(centers)  # Falls keine Zentren oder Header-Zeilen vorhanden sind, gib leere Spaltennamen zurück
    
    # Body-Breite aus Centers
    body_width = max(centers) - min(centers) if len(centers) > 1 else 200  # Default-Breite falls nur ein Cluster
    print("Body-Width:", body_width)

    line_heights = [b[3] - b[1] for row in header_rows for b in row]  # Höhe der Boxen in den Header-Zeilen
    median_height = np.median(line_heights) if line_heights else 20  # Median der Zeilenhöhe, Standardwert 20

    # Metriken der ersten Header-Zeile
    first_row = header_rows[0]

    metrics1 = _row_metrics(first_row, centers)

    # Prüfe Abstand zunächster Zeile

    big_vertical_gap = False
    if len(header_rows) > 1:
        metrics2 = _row_metrics(header_rows[1], centers)
        vertical_gap = abs(metrics2["y_top"] - metrics1["y_bot"])
        big_vertical_gap = vertical_gap > 1.2 * median_height  # Großer vertikaler Abstand, wenn größer als 1.2 * Medianhöhe der Zeilen

    else:
        metrics2 = None  # Keine zweite Zeile vorhanden, also kein vertikaler Abstand


    # Entscheidung, ob es sich um einen Titel handelt
    high_coverage = metrics1["coverage"] >= 0.6  # Füllfaktor größer als 0.6
    # print(f"Row Metrics: {metrics1}")
    poor_alignment = metrics1["aligned_words_ratio"] <= 0.5  # Weniger als 50% der Wörter sind zu Clustern ausgerichtet
    wide_enough = metrics1["row_w"] > body_width * 0.7  # Breite der Zeile größer als 70% der Body-Breite
    content_looks_like_header = metrics1["has_header_keywords"] and metrics1["numeric_ratio"] > 0.4  # Enthält Header-Keywords und mehr als 40% numerische Werte

    is_title = (
        high_coverage and
        poor_alignment and
        (wide_enough or big_vertical_gap) and not # Breite oder großer vertikaler Abstand
        content_looks_like_header
    )

    if is_title:
        title = re.sub(r"\s+", " ", " ".join(t for _,_,_,_,t in sorted(first_row, key=lambda b: b[0]))).strip()
        column_rows = header_rows[1:]

    else:
        title= ""
        column_rows = header_rows  # Nutze alle Header-Zeilen als Spaltennamen

    # --- Spaltennamen nach Centers aggregieren --------------------------------
    words_by_center = {i: [] for i in range(len(centers))}
    for row in column_rows:
        for b in row:
            cx = 0.5 * (b[0] + b[2])
            diffs = [abs(cx - c) for c in centers]
            cid = int(np.argmin(diffs))
            words_by_center[cid].append(b[4])

    final_columns = []
    for i in range(len(centers)):
        tokens = words_by_center[i]
        name = re.sub(r"\s+", " ", " ".join(tokens)).strip() if tokens else ""
        final_columns.append(name)

    return title, final_columns

def _is_empty_value_row(row, label_idx):
    """
    Prüft ob alle Spalten außer Label leer sind.
    """

    for cid, text in row.items():
        if cid == label_idx:
            continue
        if text.strip():
            return False  # Wenn ein Text in einer anderen Spalte vorhanden ist, ist die Zeile nicht leer

    return True

def _merge_davon_rows(rows_text, rows_boxes, label_cluster):

    if label_cluster is None:
        return rows_text, rows_boxes

    output_text = []
    output_boxes = []
    i = 0

    while i < len(rows_text):

        current_row_text = rows_text[i]
        current_row_boxes = rows_boxes[i]

        label_row = current_row_text.get(label_cluster, "").strip()

        if label_row.startswith("davon") and label_row != "davon zucker" and _is_empty_value_row(current_row_text, label_cluster) and (i + 1 < len(rows_text)):
            next_row_text = rows_text[i + 1]
            print(next_row_text)
            next_row_boxes = rows_boxes[i + 1]
            print(next_row_boxes)

            # Nächste Zeile darf nicht leer sein
            next_row_has_values = any(
                cid != label_cluster and next_row_text.get(cid, "").strip()
                for cid in next_row_text.keys()
            )

            label_next_row = next_row_text.get(label_cluster, "").strip()

            if next_row_has_values and label_next_row:
                # Füge den Text der nächsten Zeile zum aktuellen Label hinzu
                merged_label_text = re.sub(r"\s+", " ", f"{label_row} {label_next_row}".strip())

                # label-Boxen: erst die "davon"-Boxen, dann die Folgezeile
                merged_label_boxes = (current_row_boxes.get(label_cluster, []) or []) + (next_row_boxes.get(label_cluster, []) or [])

                # baue neue Zeile j (ersetzt Zeile i+1)
                new_row_t = dict(next_row_text)
                new_row_b = dict(next_row_boxes)
                new_row_t[label_cluster] = merged_label_text
                new_row_b[label_cluster] = merged_label_boxes

                # Zeile i wird NICHT ausgegeben, nur die gemergte j
                output_text.append(new_row_t)
                output_boxes.append(new_row_b)

                i += 2
                continue  # nächstes Paar prüfen

        # Default: Zeile unverändert übernehmen
        output_text.append(current_row_text)
        output_boxes.append(current_row_boxes)
        i += 1

    return output_text, output_boxes

def _merge_multiline_energy_rows(rows_text, rows_boxes, label_cluster, centers):
    """
    Mergt mehrere Zeilen, die zu einer Energiezeile gehören, zu einer einzigen Zeile.
    """
    if label_cluster is None or not rows_text:
        return rows_text, rows_boxes
    
    ENERGY_KEYS = {"energie", "brennwert"}
    ENERGY_VALUE_PATTERN = re.compile(r"(kj|kcal)", re.IGNORECASE)

    output_text = []
    output_boxes = []
    i = 0

    while i < len(rows_text):
        current_row_text = rows_text[i]
        current_row_boxes = rows_boxes[i]

        if any(word in ENERGY_KEYS for word in current_row_text.get(label_cluster, "").split()):
            
            next_row = i + 1

            while next_row < len(rows_text): # Es könnten mehrere Zeilen zu einer Energiezeile gehören
                next_row_text = rows_text[next_row] if next_row < len(rows_text) else None
                next_row_boxes = rows_boxes[next_row] if next_row < len(rows_boxes) else None
                next_row_label = next_row_text.get(label_cluster, "").strip() if next_row_text else ""

                has_energy_values = any(
                    ENERGY_VALUE_PATTERN.search(val or "")
                    for cid, val in next_row_text.items() 
                    if cid != label_cluster
                )

                if next_row_label == "" and has_energy_values:
                    # Werte spaltenweise anhängen
                    for cid, _ in enumerate(centers):
                        if cid == label_cluster:
                            continue
                        text_current = current_row_text.get(cid, "").strip()
                        text_next = next_row_text.get(cid, "").strip()

                        if text_next:
                            current_row_text[cid] = f"{text_current} {text_next}".strip() 
                            current_row_boxes[cid] = (current_row_boxes.get(cid, []) or []) + (next_row_boxes.get(cid, []) or [])

                            print(f"Zeile {i} und {next_row} wegen Energiezusammengehörigkeit gemerged: {current_row_text[cid]}")

                    next_row += 1  # Nächste Zeile prüfen
                    continue
                break  # Wenn die nächste Zeile nicht leer ist oder kein Energiewert enthält, beende die Schleife

            output_text.append(current_row_text)
            output_boxes.append(current_row_boxes)
            i = next_row
            continue

        # Default: Zeile unverändert übernehmen
        output_text.append(current_row_text)
        output_boxes.append(current_row_boxes)
        i += 1

    return output_text, output_boxes

def _join_hyphenated_lines(text: str) -> str:
    text = unicodedata.normalize("NFKC", text)  # Normalisiere den Text
    text = text.replace("\u00AD", "")  # Soft Hyphen entfernen
    text = text.translate(str.maketrans({
        "\u2010": "-",  # Hyphen
        "\u2011": "-",  # Non-breaking Hyphen
        "\u2012": "-",  # Figure Dash
        "\u2013": "-",  # En Dash
        "\u2014": "-",  # Em Dash
        "\u2212": "-",  # Minus
    }))

    lines = text.splitlines()
    output = []
    i = 0

    while i < len(lines):
        line = lines[i].rstrip()
        j = i
        print(f"Verarbeite Zeile {i}: {line}")

        while line.endswith('-'):
            print("Zeile endet mit -")
            # nächste nicht leere Zeile finden
            k = j + 1

            while k < len(lines) and lines[k].strip() == "":
                k += 1
            if k >= len(lines):
                break

            next_line = lines[k].lstrip()
            
            next_line_first_word = next_line.split()[0] if next_line else ""

            # Fall A: Echtes Bindestrichwort beibehalten
            if next_line_first_word[:1].isupper() or next_line_first_word[:1].isdigit():  # Prüfe, ob das erste Zeichen des nächsten Wortes ein Großbuchstabe ist
                # Wenn ja, behalte den Bindestrich
                line = line[:-1] + "-" + next_line
            
            # Sonderzeichen am Anfang
            elif next_line_first_word and next_line_first_word[0] in ".;:,„“‚‘`´❝❞°":
                cleaned = next_line_first_word
                while cleaned and cleaned[0] in ".;:,„“‚‘`´❝❞°":
                    cleaned = cleaned[1:]

                # Prüfe, ob noch ein Buchstabe oder eine Zahl übrig ist
                if cleaned and (cleaned[0].isalpha() or cleaned[0].isdigit()):
                    # Füge die bereinigte Zeile an
                    line = line[:-1] + cleaned + next_line[len(next_line_first_word):]

            # Fall B: Silbentrennung - Zusammenziehen ohne Bindestrich

            else:
                line = line[:-1] + next_line

            j = k

        output.append(line)
        i = j + 1

    return "\n".join(output)

def _cut_to_ingredients(text:str) -> str:
    """
    Schneidet alles vor "Zutaten" weg.
    Für das Ende:
    - Prüft, ob eine der Marken ("enthalten", "glutenfrei", "laktosefrei", "qs", "ware")
        in der Zutatenliste vorkommt. Wenn ja, nimm den *letzten* Treffer (größter Index)
        als Ende (inklusive des gefundenen Wortes).
    - Wenn keine Marke gefunden wurde, prüfe die ABORT_PHRASES und nimm die *erste* gefundene
        Position (kleinster Index) als Ende.
    - Wenn gar nichts gefunden wurde, gib den Text ab "Zutaten" unverändert zurück.
    """

    zutaten_search = re.search(r"\bzutaten\b\s*[:\-–—]?\s*", text, re.I) # Ignoriere Groß-/Kleinschreibung
    if zutaten_search:
        text = text[zutaten_search.start():]  # Alles ab "Zutaten" behalten

    # Endmarken definieren und letzten Treffer finden
    end_markers = ["enthalten", "glutenfrei", "laktosefrei", "qs", "ware"]
    last_end_idx = None

    for w in end_markers:
        # Alle Vorkommen des Wortes (case-insensitiv, wortgrenzen) finden (finditer)
        for m in re.finditer(rf"\b{re.escape(w)}\b", text, re.I):
            end_pos = m.end()  # bis einschließlich des Wortes schneiden
            if last_end_idx is None or end_pos > last_end_idx:
                last_end_idx = end_pos

    if last_end_idx is not None:
        return text[:last_end_idx].strip()

    # Keine Endmarke gefunden -> Abbruchphrasen prüfen (erste gefundene Stelle)
    abort_cutoffs = []
    for phrase in ABORT_PHRASES:
        m = re.search(phrase, text, re.I)
        if m:
            abort_cutoffs.append(m.start())

    if abort_cutoffs:
        cutoff = min(abort_cutoffs)
        return text[:cutoff].strip()

    # 4) Fallback: nichts zu schneiden
    return text.strip()

def _add_lost_commata(text: str) -> str:
    """
    Fügt verlorene Kommas in den Text wieder hinzu.
    """
    text = re.sub(r'(\d+)\s+(\d+)([g%])', r'\1.\2\3', text)
    return text

def _merge_num_unit_in_row(row):
    """
    row: List[box]; box = (x1,y1,x2,y2,text). Merged Tokens behalten die erste Box, text wird zusammengezogen.
    """

    unit_set = {"g","mg","µg","ug","ml","l","kj","kcal","%"}
    numish = re.compile(r'^(?:<\s*)?(?:ca\.?|approx\.?|~\s*)?\d+(?:[.,]\d+)*(?:/\d+(?:[.,]\d+)*)?$')

    merged = []
    i = 0
    while i < len(row):
        current_word = row[i]

        if i + 1 < len(row):
            next_word = row[i + 1]

            current_core = current_word[4].strip().strip("()[]{}").lower()
            next_core = next_word[4].strip().strip("()[]{}").lower()

            if numish.match(current_core) and next_core in unit_set:
                new_text = re.sub(r'\s+$', '', current_word[4]) + re.sub(r'^\s+', '', next_word[4])
                merged.append((current_word[0], current_word[1], current_word[2], current_word[3], new_text))
                i += 2
                continue
        
        merged.append(current_word)
        i += 1
    return merged

def _visualize_ocr_boxes(image: np.ndarray, ocr_data: Dict, output_path: Path):
    """
    Zeichnet die von Tesseract erkannten Bounding Boxes auf ein Bild und speichert es.

    Args:
        image: Das Bild, auf das gezeichnet werden soll (typischerweise das vorverarbeitete Bild).
        ocr_data: Das Dictionary-Ergebnis von pytesseract.image_to_data.
        output_path: Der vollständige Pfad zum Speichern des Ergebnisbildes.
    """
    # Kopie des Bildes erstellen, um das Original nicht zu verändern
    img_with_boxes = image.copy()

    # Wenn das Bild in Graustufen ist, konvertiere es zu BGR, um farbige Boxen zu zeichnen
    if len(img_with_boxes.shape) == 2:
        img_with_boxes = cv2.cvtColor(img_with_boxes, cv2.COLOR_GRAY2BGR)

    num_boxes = len(ocr_data['text'])
    for i in range(num_boxes):
        # Nur Boxen mit einer gewissen Konfidenz und Inhalt berücksichtigen
        text = ocr_data['text'][i].strip()

        if text:  # Schwellenwert von 60, kann angepasst werden
            x, y, w, h = ocr_data['left'][i], ocr_data['top'][i], ocr_data['width'][i], ocr_data['height'][i]
            
            # Zeichne ein grünes Rechteck um jedes erkannte Wort
            cv2.rectangle(img_with_boxes, (x, y), (x + w, y + h), (0, 255, 0), 3)
            
    # Das Bild mit den gezeichneten Boxen speichern
    cv2.imwrite(str(output_path), img_with_boxes)
    print(f"OCR-Visualisierung gespeichert unter: {output_path}")