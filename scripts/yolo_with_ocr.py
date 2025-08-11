# Einbindung der OpenAI API 

from dotenv import load_dotenv
import os
import tkinter as tk # TK Interface 
from tkinter import filedialog, messagebox 
from PIL import Image, ImageTk
import numpy as np
import cv2
import matplotlib.pyplot as plt
from ultralytics import YOLO 
import pytesseract
from pytesseract import Output
import re
import unicodedata
from rapidfuzz import process, fuzz

# Ursprüngliche GOLDEN_KEYS
BASE_GOLDEN_KEYS = [
    "energie", "brennwert", "durchschnittliche", "durchschnittlich", "nährwerte", "pro", "fett", "gesättigte", "fettsäuren", "nährwertinformationen",
    "kohlenhydrate", "davon", "-davon", "- davon", "zucker", "eiweiß", "salz", "g", "kcal", "kj",
    "%", "mg", "ml", "cal", "kj/kcal", "-zucker", "davon:", "-davon:", "- davon:", r"%rm", r"%rm*", "rm", "-gesättigte", "- gesättigte", "je"
]

# Ergänze alle Varianten mit Klammern
GOLDEN_KEYS = BASE_GOLDEN_KEYS + [f"({key})" for key in BASE_GOLDEN_KEYS] + [f"{key})" for key in BASE_GOLDEN_KEYS] + [f"({key}" for key in BASE_GOLDEN_KEYS]

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
    (re.compile(r'f[e3][t7]{1,3}', re.I), "fett"),

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
    (re.compile(r'e[i1l][wvv][e3][i1l][sß]', re.I | re.U), "eiweiß"),

    # salz (sa1z, sa|z, sa!z)
    (re.compile(r's[aä@][l1|!][z2]', re.I), "salz"),

    # kJ (kJ, kj, k j, k.j., k-j.)
    (re.compile(r'^(k|kj|k j|k\.j\.|k-?j\.?)$', re.I), "kj"),

    # kcal (kcal, kcal., kcal-, k cal, k.cal, k-cal)
    (re.compile(r'^(kcal|k cal|k\.cal|k-?cal)$', re.I), "kcal"),

    # nährwertinformationen (sehr langes Wort, fehleranfällig in der Mitte)
    (re.compile(r'n[äa@]hr?w[e3]rt.{5,15}[t7]ion[e3]n', re.I | re.U), "nährwertinformationen")
]
# -----------------------------------------------------------


class ImageAnalyzer:

    

    def __init__(self, root):

        # Class-Mapper
        self.className_to_id = {
        "ingredients": 0,
        "nutrition": 1
        }

        self.root = root
        self.root.title("OCR Verpackungsanalyse")

        self.image_paths = None # Abspeicherung der Bildpfade (mehrere Bilder pro Produkt)

        self.uploadButton = tk.Button(root, text="Bilder auswählen", command=self.upload_images)
        self.uploadButton.pack(pady=10) # 10 Pixel Abstand nach unten

        self.analyzeButton = tk.Button(root, text="Bild analysieren", command=self.analyze_image)
        self.analyzeButton.pack(pady=10)

        # Class selection dropdown
        self.selectedClass = tk.StringVar(value="ingredients")

        tk.Label(root, text="Wähle eine Klasse:").pack(pady=5)
        self.class_dropdown = tk.OptionMenu(
            root, self.selectedClass, *self.className_to_id.keys()
        )
        self.class_dropdown.pack(pady=5)

        # Frame (Container) für Bilder nebeneinander
        self.main_frame = tk.Frame(root)
        self.main_frame.pack(pady=10)

        # Linker Bereich für Thumbnails
        self.left_frame = tk.Frame(self.main_frame)
        self.left_frame.pack(side="left", padx=10)

        # Mittlerer Bereich für Bild mit Bounding Boxen
        self.yolo_frame = tk.Frame(self.main_frame)
        self.yolo_frame.pack(side="left", padx=10)

        # Rechter Bereich für gecropptes und preprocessed Bild
        self.cropped_frame = tk.Frame(self.main_frame)
        self.cropped_frame.pack(side="right", padx=10)

        #Bild mit BBoxen
        self.image_with_bboxes = tk.Label(self.yolo_frame)
        self.image_with_bboxes.pack()

        #Cropped Bild
        self.croppedImg = tk.Label(self.cropped_frame)
        self.croppedImg.pack()

        # Bild mit Textboxen
        self.textboxes = tk.Label(self.cropped_frame)
        self.textboxes.pack()

        #Erkannter Text
        self.text_output = tk.Label(root, text="")
        self.text_output.pack(pady=10)

        # Liste um Thumbnails dynamisch zu speichern
        self.thumbnail_labels = []


    def upload_images(self):
        path = filedialog.askopenfilenames(
            filetypes=[("Image files", "*.png;*.jpg;*.jpeg;*.tif;*.bmp")]
        )
        if not path:
            return

        # Konvertiere die Pfade in eine Liste, falls mehrere Bilder ausgewählt wurden
        path_list = list(path)

        self.image_paths = path_list  # Speichere die Bildpfade in der Instanzvariable

        # delete old thumbnails
        for lbl in self.thumbnail_labels:
            lbl.destroy()
        self.thumbnail_labels.clear()

        # create new labels + thumbnails
        
        for i, img_path in enumerate(path_list):
            img = Image.open(img_path)
            
            lbl = tk.Label(self.left_frame)
            lbl.grid(row=i // 3, column=i % 3, padx=5, pady=5)  # 3 Bilder pro Zeile    

            self.create_picture_preview(img, lbl, size=(200, 200))
            self.thumbnail_labels.append(lbl)



    def create_picture_preview(self, img, image_label, size):

        if isinstance(img, np.ndarray):
            img_pil = Image.fromarray(img)
        
        elif isinstance(img, Image.Image):
            img_pil = img
        
        else:
            raise ValueError("Ungültiges Bildformat")

        img_pil.thumbnail(size)
        photo = ImageTk.PhotoImage(img_pil)
        image_label.config(image=photo)
        image_label.image = photo  # Referenz speichern

    
    def analyze_image(self):
        print("Die Bilder die übergeben worden sind: ", self.image_paths)
        
        if not (self.image_paths):
            messagebox.showwarning("Keine Bilder", "Bitte zuerst ein Bild auswählen.")
            return
        
        target_class = self.selectedClass.get()
        
        self.main(self.image_paths, target_class)

        
    def process_cv2_picture(self, img):

        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        inverted = cv2.bitwise_not(gray) # canny needs white lines on black background to detect edges
        edges = cv2.Canny(inverted, 50, 150, apertureSize=3) # detect edges in the image
        lines = cv2.HoughLinesP(edges, 1, np.pi / 180, 100, minLineLength=100, maxLineGap=10) # # detect lines in the image
        
        if lines is None:
            print("Keine Linien zur Bestimmung des Winkels gefunden. Bearbeite Originalbild weiter.")
            deskewed_image = img
        else:
            angles = []
            for line in lines:
                x1, y1, x2, y2 = line[0]
                dx, dy = x2 - x1, y2 - y1
                if dx == 0:
                    continue
                
                # Winkelberechnung direkt in Grad
                angle = np.degrees(np.arctan2(dy, dx))
                
                # Auf Bereich [-45, 45] falten (Ihre Logik ist hier gut)
                if angle < -45:
                    angle += 90
                elif angle > 45:
                    angle -= 90

                if -15 < angle < 15:
                    angles.append(angle)

            if not angles:
                print("Keine gültigen Winkel gefunden. Bearbeite Originalbild weiter.")
                deskewed_image = img
            else:
                median_angle = np.median(angles)
                print(f"Erkannter Neigungswinkel: {median_angle:.2f} Grad")

                (h, w) = img.shape[:2]
                center = (w // 2, h // 2)
                
                M = cv2.getRotationMatrix2D(center, median_angle, 1.0)
                
                deskewed_image = cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)

        # Der Rest des Codes ist korrekt
        gray_deskewed = cv2.cvtColor(deskewed_image, cv2.COLOR_BGR2GRAY)
        _, binary = cv2.threshold(gray_deskewed, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        #processed = cv2.bitwise_not(gray_deskewed)

        return binary  # Rückgabe des vorverarbeiteten Bildes
        
    def draw_bboxes(self, img, boxes):
        
        for box in boxes:
            # Coordinates
            x1, y1, x2, y2, _ = box
            # left is the distance from the upper-left corner of the bounding box, to the left border of the image
            # # top is the distance from the upper-left corner of the bounding box to the top border of the image.
            top_left = (x1, y1)
            bottom_right = (x2, y2) # hier y + h, WEIL DIE Y-ACHSE NACH UNTEN GERICHTET IST (also wird y + h ein kleinerer Wert)

            # Box params
            green = (0, 255, 0)
            thickness = 1  # The function-version uses thinner lines

            cv2.rectangle(img, top_left, bottom_right, green, thickness)

        self.create_picture_preview(img, self.textboxes, size=(200, 200))

    def fuzzy_correct(self, word, key_list, threshold=80):
        match, score, _ = process.extractOne(word, key_list, scorer=fuzz.ratio)
        if score >= threshold:
            return match  # Korrigiere zum gefundenen Key
        return word      # Lass Wort unverändert

    # Korrektur kontextunabhängiger numerischer Fehler
    def correct_numeric_errors(self, word: str) -> str:
        """
        Korrigiert eindeutige OCR-Fehler in numerischen Werten.
        """

        # Fall 1 - Alleinstehende Null gefolgt von einer 9. -> Wahrscheinlich "0 g".
        # Passt NUR auf "09".
        if re.fullmatch(r'09', word):
            corrected = "0 g"
            print(f"Sichere numerische Korrektur (g-als-9-Fehler): '{word}' -> '{corrected}'")
            return corrected

        # Fall 2 - Führende Null mit expliziter 'g'-Einheit. -> Wahrscheinlich "0,X g".
        # Passt auf "02g", "05g", aber NICHT auf das alleinstehende "02".
        elif re.fullmatch(r'0\d+g', word): 
            numeric_part = word.removesuffix('g')
            corrected = f"{numeric_part[0]},{numeric_part[1:]} g"
            print(f"Sichere numerische Korrektur (fehlendes Komma): '{word}' -> '{corrected}'")
            return corrected
        
        return word
    
    ####################################
    # Funktionen zur Konvertierung des Ergebnisses in ein strukturiertes Format
    ####################################

    def split_label_values(self, line):
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
    
    def combine_splitted_rows(self, lines: list) -> list:
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
        print("Nach Merge:", merged_lines)
        return merged_lines
    

    def parse_ocr(self, textblock: list) -> dict:
        """
        Nimmt einen Textblock und extrahiert Schlüssel-Wert-Paare.
        Beispiel: "Energie 100 kcal" -> {"Energie": "100 kcal"}

        """

        print("Übergebener Textblock:", textblock)

        START_KEYWORDS = {"energie", "brennwert", "davon", "fett", "gesättigte", "fettsäuren", "kohlenhydrate", "zucker"}
        END_KEYWORD = "salz"

        result = {}
        header_lines = []
        body_lines = []
        footer_lines = []

        # Zustände initialisieren

        current_state = "HEADER"  # Start im Header-Zustand

        for line in textblock:
            words = line.split()

            # Prüfe, ob das Wort ein Start-Keyword ist
            if current_state == "HEADER" and any(word.lower() in START_KEYWORDS for word in words):
                current_state = "BODY"

            if current_state == "HEADER":
                header_lines.append(line)
            elif current_state == "BODY":
                body_lines.append(line)
            elif current_state == "FOOTER":
                footer_lines.append(line)

            if current_state == "BODY" and any(word.lower() in END_KEYWORD for word in words):
                # Wenn das End-Keyword erreicht ist, wechsle in den Footer-Zustand
                current_state = "FOOTER"

        # Verarbeite Header-Zeilen
        result["title"] = " ".join(header_lines).strip() if header_lines else ""

        restructered_bodyLines = self.combine_splitted_rows(body_lines)

        # Verarbeite Body-Zeilen
        for line in restructered_bodyLines:


            key, value = self.split_label_values(line)
            key = key.strip()
            value = value.strip()

            if key or value:
                # Füge das Schlüssel-Wert-Paar zum Ergebnis hinzu
                if key in result:
                    # Falls der Key schon existiert, hänge den Wert an
                    result[key] += f" {value}"
                else:
                    result[key] = value

        # Verarbeite Footer-Zeilen
        result["footnote"] = " ".join(footer_lines).strip() if footer_lines else ""
                
        return result

    # Hauptfunktion zur Normalisierung des Textes
    # Diese Funktion nimmt einen Textstring und wendet verschiedene Regeln an, um ihn zu bereinigen und zu normalisieren.
    # Sie verwendet sowohl reguläre Ausdrücke als auch Fuzzy-Matching, um sicherzustellen, dass die wichtigsten Schlüsselwörter korrekt erkannt werden.
    def normalize(self, line: str) -> str:

        line = unicodedata.normalize("NFKC", line).lower()

        sub_rules = {

            # Entferne Punkte, die KEINE Dezimaltrennzeichen sind, und ersetze sie durch ein Leerzeichen.
            # "1189.k7" -> "1189 k7". "9.4" bleibt unberührt.
            r'(\d)\.(?!\d)': r'\1 ',

            r'[\"\'„“‚‘`´❝❞°]': '',  # Entferne Anführungszeichen und weitere Sonderzeichen

            r',': '.',  # Ersetze Kommas durch Punkte (für Dezimalzahlen)

            r'<o': '<0',  # Speziell für deinen Fall <o,5g -> <0,5g
            r'<l': '<1',  # Analog für den Fall <l,5g -> <1,5g
            r'<s': '<5',  # Analog für den Fall <s,5g -> <5,5g
            r'<B': '<8',  # Analog für den Fall <B,5g -> <8,5g
            r'\bo,': '0.',  # Für Fälle wie o,5g -> 0,5g (o nur am Wortanfang)
            r'\bl,': '1.',  # Für Fälle wie l,5g -> 1,5g (l nur am Wortanfang)
            r'\bs,': '5.',  # Für Fälle wie s,5g -> 5,5g (s nur am Wortanfang)
            r'\bB,': '8.',  # Für Fälle wie B,5g -> 8,5g (B nur am Wortanfang)
            # "k7" -> "kj", "k1" -> "kj"
            r'k[71!l\]Ww)]': 'kj',  # kj Korrekturen
            r'kca[t\]1I]': 'kcal',  # kcal Korrekturen

            # Betrachte Schrägstrich als Trennzeichen
            r'/': ' ',

            r'\.\.': '.',           # Ersetzt ".." mit "."



            r'([0-9])o': r'\1g', # 10 o -> 10 g
            r'([0-9])l': r'\1g',
            r'([;])': '.', # Ersetze Semikolons mit .
            r'\s{2,}': ' ', # Entferne überflüssige Leerzeichen
            r'\s*,\s*': '.' # Entferne Leerzeichen vor und nach Kommas

        }

        print(f"Vor der Normalisierung nach sub_rules: {line}")

        for pattern, replace in sub_rules.items():
            line = re.sub(pattern, replace, line)

        stripped = line.strip(" .,;") # entferne Leerzeichen, Punkte, Kommas und Semikolons am Anfang und Ende
        print(f"Nach der Normalisierung nach sub_rules: {stripped}")
        words = stripped.split() # teile den String in einzelne Wörter
        corrected_words = []

        for word in words:
            for regex, canonical in KEY_CORRECTIONS:
                if regex.fullmatch(word):
                    word = canonical
                    break
            else:  # fuzzy matching falls kein Regex passt
                fuzzy_corrected = self.fuzzy_correct(word, GOLDEN_KEYS, threshold=80)
                if fuzzy_corrected != word:
                    word = fuzzy_corrected
                    print(f"Fuzzy-Korrektur: {word}")
            corrected_words.append(word)
            print(f"Nach der Normalisierung nach KEY_CORRECTIONS: {word}")

        # Nach der Korrektur (Regex + Fuzzy)
        valid_words = set(GOLDEN_KEYS)

        # Regex für Zahlenangaben
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
        for word in corrected_words:

            corrected_number = self.correct_numeric_errors(word)  # Korrigiere eindeutige numerische Fehler

            parts = corrected_number.split()  # Teile den korrigierten String in Teile


            for part in parts:

                part_without_parentheses = part

                # Prüfe, ob das Wort von Klammern umschlossen ist
                if part.startswith('(') and part.endswith(')'):
                    # Entferne die Klammern nur für den Check
                    part_without_parentheses = part[1:-1]

                # Prüfe, ob der Teil ein gültiges Wort oder eine Zahl ist
                if part in valid_words or number_pattern.match(part):
                    filtered_words.append(part)
                elif len(part_without_parentheses) >= 4 and part_without_parentheses.isalpha():
                    # z.B. Filter nach Minimal-Länge und Buchstaben
                    filtered_words.append(part)
                else:
                    print(f"Ausreißer entfernt: {word}")


        return " ".join(filtered_words)  # Füge die Wörter wieder zu einem String zusammen
        
    
    def main(self, image_paths, target_class):

        target_class_id = self.className_to_id[target_class]

        model = YOLO("yolo_results/model_yolov8s/weights/best.pt")

        print(model.names)

        images = [cv2.imread(path) for path in image_paths]

        results = model(images)

        best_box = None
        best_conf = -1
        best_image_path = None

        for img_path, result in zip(image_paths, results):
            
            for box in result.boxes:
                cls_id = int(box.cls[0])
                conf = box.conf[0]

                if cls_id == target_class_id and conf > best_conf and conf > 0.5:
                    best_conf = conf
                    best_box = box
                    best_image_path = img_path


        if best_box is None:
            messagebox.showinfo("Ergebnis", f"Keine Objekte der Klasse '{target_class}' erkannt.")
            return
        
        x1, y1, x2, y2 = best_box.xyxy[0].tolist()  # Koordinaten der Bounding Box

        best_image = cv2.imread(best_image_path)

        cv2.rectangle(best_image, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
        label = f"{model.names[target_class_id]}, {best_conf:.2f} %"
        cv2.putText(best_image, label, (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 0, 0), 6)
        # Bild mit Bounding Boxes anzeigen
        self.create_picture_preview(cv2.cvtColor(best_image, cv2.COLOR_BGR2RGB), self.image_with_bboxes, size=(640, 480))

        cropped = best_image[int(y1):int(y2), int(x1):int(x2)]  # Bild zuschneiden
        
        if cropped.size == 0:
            messagebox.showwarning("Warnung", "Cropped Bounding Box war leer – übersprungen.")
            return
        
        preprocessed = self.process_cv2_picture(cropped)  # Bild vorverarbeiten

        # Zeige das vorverarbeitete Bild an
        self.create_picture_preview(cv2.cvtColor(preprocessed, cv2.COLOR_BGR2RGB), self.croppedImg, size=(640, 480))
        config = r'--oem 3 --psm 6'  # OCR Engine Mode + Page Segmentation Mode  ----- oem 3: Verwende die beste verfügbare OCR-Engine (LSTM oder Legacy) -----  psm 1: Erwartet einen einheitlichen Textblock (z. B. Absatz oder mehrere Zeilen).

        text = pytesseract.image_to_string(preprocessed, lang='deu', config=config, output_type='string')
        print(f"Output ohne postprocessing: " + text)
        data = pytesseract.image_to_data(preprocessed, lang='deu', config=config, output_type=Output.DICT)

        boxes = []

        for i, txt in enumerate(data["text"]):
            txt = txt.strip()
            if not txt:
                continue
            x1, y1 = data["left"][i],  data["top"][i]
            w , h  = data["width"][i], data["height"][i]

            print(txt)
            
            normalized_txt = self.normalize(txt)

            print(f"Normalized Text: {normalized_txt}")
            boxes.append((x1, y1, x1 + w, y1 + h, normalized_txt))

        self.draw_bboxes(preprocessed, boxes)

        # mittelpunkte_bboxes = [[(x1 + 0.5*(abs(x2-x1))), (y1 + 0.5*(abs(y2-y1)))] for x1,y1,x2,y2 in boxes] # Liste der Mittelpunkte der Boxen

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

        final_processed_rows = []

        energy_keywords = {"energie", "brennwert"}

        for row in rows: # row ist ein List[Box]
            row.sort(key=lambda b: b[0])  # Sortiere Boxen in der Zeile nach x1 (linke Kante)
            words = [box[4] for box in row]


            # Prüfe VORHER, ob es sich um eine Energie-Zeile handelt.
            is_energy_line = any(key in words for key in energy_keywords)
            is_salt_line = "salz" in words
            # Prüfe zuerst die aktuelle Zeile
            is_nutrition_line = any(w in {"nährwerte", "nährwertinformationen"} for w in words)
            
            # Wenn nicht in der aktuellen Zeile gefunden, prüfe die vorherige
            if not is_nutrition_line:
                try:
                    current_row_index = rows.index(row)
                    if current_row_index > 0:
                        previous_row_boxes = rows[current_row_index - 1]
                        # Extrahiere die Wörter aus den Boxen der vorherigen Zeile
                        words_in_previous_row = [box[4] for box in previous_row_boxes]
                        
                        if any(w in {"nährwerte", "nährwertinformationen"} for w in words_in_previous_row):
                            is_nutrition_line = True
                except ValueError:
                    pass

            corrected_boxes = []
            for i, box in enumerate(row):
                word = box[4]

                # Heuristik 1: "739" -> "7,3 g" (Fehler: Komma und Einheit fehlen)
                if re.fullmatch(r'[1-9]\d9', word) and is_energy_line == False and is_nutrition_line == False:
                    corrected_word = f"{word[0]}.{word[1]} g"
                    print(f"Kontext-Korrektur (Heuristik 1): '{word}' -> '{corrected_word}'")
                    new_box = (box[0], box[1], box[2], box[3], corrected_word)  # Ersetze das Wort in der Box
                    corrected_boxes.append(new_box)
                    continue

                # HEURISTIK 2: "289g" -> "28 g" (Fehler: Leerzeichen als '9' gelesen)
                elif re.fullmatch(r'\d{2}9g', word) and is_energy_line == False:
                    corrected_word = f"{word[:2]} g"
                    print(f"Kontext-Korrektur (Heuristik 2): '{word}' -> '{corrected_word}'")
                    new_box = (box[0], box[1], box[2], box[3], corrected_word)  # Ersetze das Wort in der Box
                    corrected_boxes.append(new_box)
                    continue
                
                # HEURISTIK 2a: "2360" -> "23,6 g"
                elif re.fullmatch(r'\d{3}(g|0)', word) and is_energy_line == False and is_salt_line == False and is_nutrition_line == False :
                    corrected_word = f"{word[:2]}.{word[2]} g"
                    print(f"Kontext-Korrektur (Heuristik 2a): '{word}' -> '{corrected_word}'")
                    new_box = (box[0], box[1], box[2], box[3], corrected_word)  # Ersetze das Wort in der Box
                    corrected_boxes.append(new_box)
                    continue

                # HEURISTIK 2b: "16,29" -> "16,2 g"
                # Das Muster (Zahl, eine Ziffer, dann eine 9) ist extrem wahrscheinlich ein
                # OCR-Fehler für "...g".
                elif re.fullmatch(r'\d+[.,]\d9', word) and is_salt_line == False:
                    corrected_word = f"{word[:-1]} g"
                    print(f"Kontext-Korrektur (Heuristik 2b): '{word}' -> '{corrected_word}'")
                    new_box = (box[0], box[1], box[2], box[3], corrected_word)  # Ersetze das Wort in der Box
                    corrected_boxes.append(new_box)
                    continue

                # Heuristik 3: Salz "200" -> "2,00" (kontextbasiert)
                # ========================================================
                # Prüfe, ob das VORHERIGE Wort "salz" war.
                if i > 0 and words[i-1] == "salz" :

                    if re.fullmatch(r'\d{3}9', word):

                        corrected_word = f"{word[0]}.{word[1:3]} g"
                        print(f"Kontext-Korrektur (Salz): '{word}' -> '{corrected_word}'")
                        new_box = (box[0], box[1], box[2], box[3], corrected_word)  # Ersetze das Wort in der Box
                        corrected_boxes.append(new_box)
                        continue

                    elif re.fullmatch(r'\d{3}', word):
                        # Wenn es eine 3-stellige Zahl ist, die KEIN "9" am Ende hat,
                        # dann ist es wahrscheinlich ein Fehler für "2,00 g".
                        corrected_word = f"{word[0]}.{word[1:]} g"
                        print(f"Kontext-Korrektur (Salz): '{word}' -> '{corrected_word}'")
                        new_box = (box[0], box[1], box[2], box[3], corrected_word)  # Ersetze das Wort in der Box
                        corrected_boxes.append(new_box)
                        continue

                    elif re.fullmatch(r'\d[.,]\d{2}9', word) or re.fullmatch(r'\d[.,]\d{2}9g', word):
                        # Wenn es eine Zahl mit 2 Nachkommastellen und einer 9 am Ende ist,
                        # dann ist es wahrscheinlich ein Fehler für "2,00 g".
                        corrected_word = f"{word[0]}.{word[2:4]} g"
                        print(f"Kontext-Korrektur (Salz): '{word}' -> '{corrected_word}'")
                        new_box = (box[0], box[1], box[2], box[3], corrected_word)  # Ersetze das Wort in der Box
                        corrected_boxes.append(new_box)
                        continue

                # Wenn keine kontextsensitive Regel gegriffen hat, behalte das Wort bei.
                corrected_boxes.append(box)

            final_processed_rows.append(corrected_boxes)

        # Gib das endgültig prozessierte Ergebnis aus
        print(f"\nErgebnis nach finalem Line - Postprocessing:")
        for i, row in enumerate(final_processed_rows, 1):
            # Extrahiere nur den Text für die Anzeige
            line_text = " ".join(box[4] for box in row)
            print(f"{i:02d}: {line_text}")

        # show text in GUI
        self.text_output.config(text=text.strip() or "Kein Text erkannt.")

        ### BEGINN SPALTENLOGIK ###

        


        


        ### AUFRUF DER JSON-PARSE FUNKTION ###
        # ocr_dict = self.parse_ocr(result)
        # print(ocr_dict)

# Teste den Code
if __name__ == "__main__":
    root = tk.Tk()
    app = ImageAnalyzer(root)
    root.mainloop()