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

        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY) # RGB zu GRAY, weil PIL das Bild in RGB speichert
        #blurred = cv2.GaussianBlur(gray, (3, 3), 0) # Bild wird unscharf gemacht, um Rauschen zu entfernen
        inverted = cv2.bitwise_not(gray) # Bildinvertierung: helle Pixel werden dunkel und umgekehrt -> Schrift hell, Background dunkel
        _, binary = cv2.threshold(inverted, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU) # OTSU berechnet optimalen Schwellenwert für das gesamte Bild -> alle darunter werden schwarz, darüber (also der Text) weiß
        #binary = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 51, 2)
        processed = cv2.bitwise_not(binary) # Umkehrung Binärbild: Text wird schwarz, Hintergrund weiß

        return processed

    
    def main(self, image_paths, target_class):

        target_class_id = self.className_to_id[target_class]

        model = YOLO("yolo_results/model/weights/best.pt")

        print(model)
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
        # OCR auf dem vorverarbeiteten Bild ausführen
        config = r'--oem 3 --psm 6'  # OCR Engine Mode + Page Segmentation Mode  ----- oem 3: Verwende die beste verfügbare OCR-Engine (LSTM oder Legacy) -----  psm 6: Erwartet einen einheitlichen Textblock (z. B. Absatz oder mehrere Zeilen).
        text = pytesseract.image_to_string(preprocessed, lang='deu', config=config)
        print(f"Erkannter Text: {text}")

        # show text in GUI
        self.text_output.config(text=text.strip() or "Kein Text erkannt.")

# Teste den Code
if __name__ == "__main__":
    root = tk.Tk()
    app = ImageAnalyzer(root)
    root.mainloop()