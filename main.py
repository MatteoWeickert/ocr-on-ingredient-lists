import cv2
import pytesseract
from pytesseract import Output
import tkinter as tk # TK Interface 
from tkinter import filedialog, messagebox 
from PIL import Image, ImageTk
import numpy as np
from tkinter import ttk
import re
import math
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_distances
from sklearn.cluster import DBSCAN

class ImageAnalyzer:

    def __init__(self, root):
        self.root = root
        self.root.title("OCR Verpackungsanalyse")

        self.image_path = None # Abspeicherung des Bildpfades

        self.uploadButton = tk.Button(root, text="Bild auswählen", command=self.upload_image)
        self.uploadButton.pack(pady=10) # 10 Pixel Abstand nach unten

        self.analyzeButton = tk.Button(root, text="Bild analysieren", command=self.analyze_image)
        self.analyzeButton.pack(pady=10)

        # Frame (Container) für Bilder nebeneinander
        image_frame = tk.Frame(root)
        image_frame.pack(pady=10)

        #Uploadbild
        self.image_label = tk.Label(image_frame)
        self.image_label.pack(side="left", pady=10)

        # Processed Image
        self.processed_image = tk.Label(image_frame)
        self.processed_image.pack(side="left", pady=10)

        #Bild mit BBoxen
        self.image_with_bboxes = tk.Label(image_frame)
        self.image_with_bboxes.pack(side="left", pady=10)

        #Erkannter Text
        self.text_output = tk.Label(root, text="")
        self.text_output.pack(pady=10)


    def analyze_image(self):
        print("Der Bildpfad der übergeben wird ist: ", self.image_path)
        
        if not (self.image_path):
            messagebox.showwarning("Kein Bild", "Bitte zuerst ein Bild auswählen.")
            return
        
        self.image_to_text(self.image_path)

        self.draw_bboxes(self.image_path)


    
    def upload_image(self):
        path = filedialog.askopenfilename(
            filetypes=[("Image files", "*.png;*.jpg;*.jpeg;*.tif;*.bmp")]
        )
        if not path:
            return
        
        self.image_path = path
        print("Bild geladen:", self.image_path)

        img = Image.open(path)
        self.create_picture_preview(img, self.image_label)


    def create_picture_preview(self, img, image_label):
        #Bildvorschau
        if isinstance(img, np.ndarray):
            img_pil = Image.fromarray(img)
        
        elif isinstance(img, Image.Image):
            img_pil = img
        
        else:
            raise ValueError("Ungültiges Bildformat")

        img_pil.thumbnail((300,300))
        photo = ImageTk.PhotoImage(img_pil)
        image_label.config(image=photo)
        image_label.image = photo  # wichtig: Referenz speichern!


    def process_cv2_picture(self, img):

        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY) # RGB zu GRAY, weil PIL das Bild in RGB speichert
        #blurred = cv2.GaussianBlur(gray, (3, 3), 0) # Bild wird unscharf gemacht, um Rauschen zu entfernen
        inverted = cv2.bitwise_not(gray) # Bildinvertierung: helle Pixel werden dunkel und umgekehrt -> Schrift hell, Background dunkel
        _, binary = cv2.threshold(inverted, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU) # OTSU berechnet optimalen Schwellenwert für das gesamte Bild -> alle darunter werden schwarz, darüber (also der Text) weiß
        #binary = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 51, 2)
        processed = cv2.bitwise_not(binary) # Umkehrung Binärbild: Text wird schwarz, Hintergrund weiß

        return processed
        

    def image_to_text(self, image_path):
        """
        A function to read text from an image. 
        """
        img_pil = Image.open(image_path).convert("RGB")
        if img_pil is None:
            self.text_output.config(text="Fehler: Bild konnte nicht geladen werden.")
            return
        
        # Konvertiere PIL → OpenCV
        img = np.array(img_pil)

        if img is None:
            self.text_output.config(text="Fehler: Bild konnte nicht zu numpy array konvertiert werden.")
            print("Fehler: Bild konnte nicht zu numpy array konvertiert werden.")
            return
        
        processed = self.process_cv2_picture(img)

        cuttedImage = self.findIngredients(processed)

        self.create_picture_preview(cuttedImage, self.processed_image)
        
        config = r'--oem 3 --psm 6'  # OCR Engine Mode + Page Segmentation Mode  ----- oem 3: Verwende die beste verfügbare OCR-Engine (LSTM oder Legacy) -----  psm 6: Erwartet einen einheitlichen Textblock (z. B. Absatz oder mehrere Zeilen).
        text = pytesseract.image_to_string(cuttedImage, lang='deu', config=config)
        print(text)

        self.text_output.config(text=text.strip()) # in GUI anzeigen lassen

        return text.strip() 
    
    def findIngredientsClusterAnalysis(self, preProcessedImg):

        data = pytesseract.image_to_data(preProcessedImg, output_type=Output.DICT, lang="deu", config='--oem 3 --psm 6')

        boxes = [] # Liste aller Boxen

        for i in range(len(data["text"])):
            if data["text"][i].strip() == "":
                continue

            # Koordinaten der Boxen
            x1 = data["left"][i]
            y1 = data["top"][i]
            x2 = data["left"][i] + data["width"][i]
            y2 = data["top"][i] + data["height"][i]

            boxes.append((x1, y1, x2, y2))

        mittelpunkte_bboxes = [[(x1 + 0.5*(abs(x2-x1))), (y1 + 0.5*(abs(y2-y1)))] for x1,y1,x2,y2 in boxes] # Liste der Mittelpunkte der Boxen

        coords = np.array(mittelpunkte_bboxes) # Konvertiere die Liste in ein NumPy-Array
        db = DBSCAN(eps=75, min_samples=5).fit(coords) # eps: maximaler Abstand zwischen zwei Punkten, um sie als Nachbarn zu betrachten; min_samples: minimale Anzahl von Punkten, um einen Cluster zu bilden
        labels = db.labels_ # jede Box hat ein Label, das angibt, zu welchem Cluster sie gehört (oder -1, wenn sie kein Cluster hat)

        clusters = {} # Dictionary, das die Cluster speichert
        for box, label in zip(boxes, labels): # Jeder Box wird ihr Label zugewiesen
            if label == -1:
                continue # -1 bedeutet, dass die Box kein Cluster hat

            if label not in clusters:
                clusters[label] = []  # Liste initialisieren, wenn Label noch nicht vorhanden

            clusters[label].append(box)

            # 2. Umschließende Bounding Box für jeden Cluster berechnen

        cluster_bboxes = {}
        for label, cluster_boxes in clusters.items():
            x1_vals = [box[0] for box in cluster_boxes]
            y1_vals = [box[1] for box in cluster_boxes]
            x2_vals = [box[2] for box in cluster_boxes]
            y2_vals = [box[3] for box in cluster_boxes]

            x_min = min(x1_vals)
            y_min = min(y1_vals)
            x_max = max(x2_vals)
            y_max = max(y2_vals)

            if label not in cluster_bboxes:
                cluster_bboxes[label] = []  # Liste initialisieren, wenn Label noch nicht vorhanden

            cluster_bboxes[label] = (x_min, y_min, x_max, y_max)

        # Finde BBox mit dem Label "Zutaten"
        zutaten_label = None
        pattern = r"z[uva2]t[aiuä]t[aeä]n[:\s]?" # die []- Ausdrücke können eintreten und [:\s] bedeutet ein optionaler Doppelpunkt    

        for label, min_max_box in cluster_bboxes.items():
            x1, y1, x2, y2 = min_max_box
            text = pytesseract.image_to_string(preProcessedImg[y1:y2, x1:x2], lang='deu', config='--oem 3 --psm 6')
            print(f"[Cluster {label}] erkannter Text:\n{text.strip()}\n")  # Debug-Ausgabe
            if re.search(pattern, text.lower()):
                zutaten_label = label
                break
            if zutaten_label is not None:
                break

        if zutaten_label is None:
            print("Zutatenliste nicht gefunden")
            raise ValueError("Zutatenliste nicht gefunden") 

        x_min, y_min, x_max, y_max = cluster_bboxes[zutaten_label]  # Hole die Bounding-Box für die Zutatenliste, wenn sie gefunden wurde

        return preProcessedImg[y_min:y_max, x_min:x_max] if zutaten_label is not None else None # Rückgabe der Bounding-Box für die Zutatenliste oder None, wenn keine gefunden wurde


    
    def findIngredients(self, preProcessedImg):
        data = pytesseract.image_to_data(preProcessedImg, output_type=Output.DICT, lang="deu", config='--oem 3 --psm 6')
        
        print(data)

        zutaten_index = None

        # Regex-Erkennung von "Zutaten"
        pattern = r"z[uva2]t[aiu]t[ae]n[:\s]" # die []- Ausdrücke können eintreten und [:\s] bedeutet ein optionaler Doppelpunkt

        for i, word in enumerate(data["text"]):
            if re.fullmatch(pattern, word.lower()):
                zutaten_index = i 
                break

        if not zutaten_index:
            print("Zutaten nicht erkannt")
            raise ValueError("Zutatenliste nicht gefunden")

        boxes = []
        boxes.append((data["left"][zutaten_index], data["top"][zutaten_index], data["left"][zutaten_index] + data["width"][zutaten_index], data["top"][zutaten_index] + data["height"][zutaten_index]))
        n_boxes = len(data["text"])

        zutaten_y = data["top"][zutaten_index] # y-Koordinate der Zutatenbox
        zutaten_x = data["left"][zutaten_index] # x-Koordinate der Zutatenbox

        ingredient_indizes = [zutaten_index] # Liste für die Indizes der Zutaten

        # Sammmle alle Wörter, die in der Nähe der Zutatenbox sind 
        for i in range(zutaten_index + 1, n_boxes):

            if data["text"][i].strip() == "":
                continue

            if data["left"][i] >= zutaten_x - 10 and data["top"][i] >= zutaten_y - 30: # Wenn die Box rechts von der Zutatenbox ist, dann ist sie relevant
                ingredient_indizes.append(i) # Füge den Index der Zutatenbox zur Liste hinzu

        # VERTIKALER CUTOFF
        # ---------------------------------
        # Sortiere die Indizes nach der erst nach y-Koordinate, dann nach x-Koordinate
        ingredient_indizes.sort(key=lambda i: (data["top"][i], data["left"][i])) # Damit sind die der Zutaten-Box am nächsten liegenden Wörter zuerst in der Liste

        # Speichere die y-Werte der Zutaten, um den cutoff point zu bestimmen, ab wann eine Box nicht mehr zur Zutatenliste gehört
        y_vals = [data["top"][i] for i in ingredient_indizes]

        # Berechne Differenzen zwischen benachbarten y-Werten
        y_diffs = [y2-y1 for y1,y2 in zip(y_vals[:-1], y_vals[1:])] # zip(y_vals[:-1]) gibt alle y-Werte bis auf den letzten zurück, zip(y_vals[1:]) gibt alle y-Werte ab dem zweiten zurück

        # Berechne den Mittelwert der Differenzen
        mean_y_diff = sum(y_diffs) / len(y_diffs) if y_diffs else 0

        # Berechne den cutoff point, ab wann eine Box nicht mehr zur Zutatenliste gehört (Abstand zu groß)
        cutoff_unten = len(ingredient_indizes)
        for idx, diff in enumerate(y_diffs):
            if diff > mean_y_diff * 4:
                cutoff_unten = idx + 1
                break

        # ZEILENWEISE GRUPPIERUNG (für rechten Cutoff)
        # ----------------------------------------------
        # Gruppiere Boxen in Zeilen (basierend auf y-Koordinate mit Toleranz)
        y_tolerance = 5  # Toleranz für die y-Koordinate
        zeilen = []
        aktuelle_zeile = []
        for i in sorted(ingredient_indizes, key=lambda j: (data["top"][j], data["left"][j])):
            if not aktuelle_zeile:
                aktuelle_zeile.append(i)
            else:
                if abs(data["top"][i] - data["top"][aktuelle_zeile[-1]]) <= y_tolerance:
                    aktuelle_zeile.append(i)
                else:
                    zeilen.append(aktuelle_zeile)
                    aktuelle_zeile = [i]
        if aktuelle_zeile:
            zeilen.append(aktuelle_zeile)

        # RECHTER CUTOFF (maximale x-Position pro Zeile)
        # ------------------------------------------------
        # Berechne den Median der maximalen x-Positionen pro Zeile
        max_x_pro_zeile = [max(data["left"][i] + data["width"][i] for i in zeile) for zeile in zeilen]
        rechts_cutoff_x = sorted(max_x_pro_zeile)[len(max_x_pro_zeile) // 2]  # Median (wir greifen auf das mittlere Element der sortierten List zu)

        # KOMBINIERTE FILTERUNG
        # -------------------------
        relevant_indices = []
        for i in ingredient_indizes[:cutoff_unten]:
            # Box muss links vom rechten Cutoff liegen
            if (data["left"][i] + data["width"][i]) <= rechts_cutoff_x + 10:  # Kleine Toleranz
                relevant_indices.append(i)

        # Berechne die Bounding-Box für alle relevanten Boxen
        # Entpacke x1, y1, x2, y2 aus allen Boxen
        x1_vals = [data["left"][i] for i in relevant_indices]
        y1_vals = [data["top"][i] for i in relevant_indices]
        x2_vals = [data["left"][i] + data["width"][i] for i in relevant_indices]
        y2_vals = [data["top"][i] + data["height"][i] for i in relevant_indices]
        
        # Ermittle den kleinsten/größten Wert
        x_min = min(x1_vals)
        y_min = min(y1_vals)
        x_max = max(x2_vals)
        y_max = max(y2_vals)

        # Speicherung der Gesamtbbox auf dem Bild
        roi = preProcessedImg[y_min:y_max, x_min:x_max]

        # Rückgabe als Gesamt-Bounding-Box
        return roi      



    def draw_bboxes(self, input_path):
        img_pil = Image.open(input_path).convert("RGB")

        if img_pil is None:
            print("Fehler: Bild konnte nicht geladen werden.")
            return

        # Konvertiere PIL → OpenCV (NumPy-Array im BGR-Format)
        img = np.array(img_pil)


        if img is None:
            print("Fehler: Bild konnte nicht zu numpy array konvertiert werden.")
            return

        processed = self.process_cv2_picture(img)

        cutted_image = self.findIngredients(processed)

        # Extract data
        config = r'--oem 3 --psm 6'
        data = pytesseract.image_to_data(cutted_image, output_type=Output.DICT, lang='deu', config=config) # gibt Background-Daten des Bildes zurück 
        print(data)
        n_boxes = len(data["text"])

        for i in range(n_boxes):
            if data["conf"][i] == -1:
                continue
            # Coordinates
            x, y = data["left"][i], data["top"][i]
            # left is the distance from the upper-left corner of the bounding box, to the left border of the image
            # # top is the distance from the upper-left corner of the bounding box to the top border of the image.

            w, h = data["width"][i], data["height"][i]

            # Corners
            top_left = (x, y)
            bottom_right = (x + w, y + h) # hier y + h, WEIL DIE Y-ACHSE NACH UNTEN GERICHTET IST (also wird y + h ein kleinerer Wert)

            # Box params
            green = (0, 255, 0)
            thickness = 1  # The function-version uses thinner lines

            cv2.rectangle(cutted_image, top_left, bottom_right, green, thickness)

        self.create_picture_preview(cutted_image, self.image_with_bboxes)



# Start der Anwendung
if __name__ == "__main__":
    root = tk.Tk()
    app = ImageAnalyzer(root)
    root.mainloop()