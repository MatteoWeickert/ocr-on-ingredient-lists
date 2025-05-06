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

    
    def findIngredients(self, preProcessedImg):
        data = pytesseract.image_to_data(preProcessedImg, output_type=Output.DICT, lang="deu", config='--oem 3 --psm 6')
        
        print(data)

        # 1. Robuste "Zutaten"-Erkennung mit erweiterten Patterns
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

        for i in range(n_boxes):

            current_box = (data["left"][i], data["top"][i], data["left"][i] + data["width"][i], data["top"][i] + data["height"][i])

            is_close = self.is_near_existing_boxes_and_not_left(current_box, boxes)

            if abs(data["height"][zutaten_index] - data["height"][i]) < 3 and is_close:
                x, y, h, w = data["left"][i], data["top"][i], data["height"][i], data["width"][i]
                boxes.append((x, y, x + w, y + h)) # Füge BBox des Wortes zur BBox Liste hinzu
                print(boxes)
        

        # Entpacke x1, y1, x2, y2 aus allen Boxen
        x1_vals = [box[0] for box in boxes]
        y1_vals = [box[1] for box in boxes]
        x2_vals = [box[2] for box in boxes]
        y2_vals = [box[3] for box in boxes]
        
        # Ermittle den kleinsten/größten Wert
        x_min = min(x1_vals)
        y_min = min(y1_vals)
        x_max = max(x2_vals)
        y_max = max(y2_vals)

        # Speicherung der Gesamtbbox auf dem Bild
        roi = preProcessedImg[y_min:y_max, x_min:x_max]

        # Rückgabe als Gesamt-Bounding-Box
        return roi

                
    def is_near_existing_boxes_and_not_left(self, current_box, existing_boxes):
        new_x, new_y, new_x2, new_y2 = current_box

        zutaten_x, zutaten_y, zutaten_x2, zutaten_y2 = existing_boxes[0] # Die erste Box ist die Zutatenbox

        for box in existing_boxes:
            x, y, x2, y2 = box

            if (abs(y2 - new_y) < 15 or abs(x2 - new_x) < 15) and new_x > zutaten_x - 5: # Prüfe ob die neue Box in der Nähe einer schon hinzugefügten Box ist und nicht links von der Zutatenbox (mit kleinem Toleranzabstand)
                return True
        
        return False       



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