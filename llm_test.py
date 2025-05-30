# Einbindung der OpenAI API 

from openai import OpenAI
from dotenv import load_dotenv
import os
import tkinter as tk # TK Interface 
from tkinter import filedialog, messagebox 
from PIL import Image, ImageTk
import numpy as np
import base64
import cv2
from io import BytesIO
from collections import defaultdict


class ImageAnalyzer:

    def __init__(self, root):
        self.root = root
        self.root.title("OCR Verpackungsanalyse")

        self.image_paths = None # Abspeicherung der Bildpfade (mehrere Bilder pro Produkt)

        self.uploadButton = tk.Button(root, text="Bilder auswählen", command=self.upload_image)
        self.uploadButton.pack(pady=10) # 10 Pixel Abstand nach unten

        self.uploadFolderButton = tk.Button(root, text="Ordner auswählen", command=self.upload_folder)
        self.uploadFolderButton.pack(pady=10) # 10 Pixel Abstand nach unten

        self.analyzeButton = tk.Button(root, text="Bild analysieren", command=self.analyze_image)
        self.analyzeButton.pack(pady=10)

        # Frame (Container) für Bilder nebeneinander
        self.image_frame = tk.Frame(root)
        self.image_frame.pack(pady=10)

        # Liste für Labels und Bilder
        self.image_labels= []
        self.image_photos = []

        #Erkannter Text
        self.text_output = tk.Label(root, text="")
        self.text_output.pack(pady=10)


    def upload_image(self):
        paths = filedialog.askopenfilenames(
            filetypes=[("Image files", "*.png;*.jpg;*.jpeg;*.tif;*.bmp")]
        )
        if not paths:
            return

        self.image_paths = paths
        print("Bilder geladen:", self.image_paths)

        for label in self.image_labels:
            label.destroy()

        self.image_labels.clear()
        self.image_photos.clear()

        for i, path in enumerate(paths):
            img = Image.open(path)
            img.thumbnail((300, 300))
            image_label = tk.Label(self.image_frame)
            image_label.grid(row=0, column=i, padx=10, pady=10)
            self.create_picture_preview(img, image_label)
            self.image_labels.append(image_label)

    def upload_folder(self):
        folder_path = filedialog.askdirectory()
        if not folder_path:
            return

        # Suche nach Bilddateien im ausgewählten Ordner
        valid_extensions = (".png", ".jpg", ".jpeg", ".tif", ".bmp")
        paths = [
            os.path.join(folder_path, f) # Kombiniert Ordnerpfad mit Bilddatei
            for f in os.listdir(folder_path) # Listet alle Elemente des Ordners auf
            if f.lower().endswith(valid_extensions) # Prüft, ob Element ein Bild
        ]

        if not paths:
            return

        self.image_paths = paths
        print("Bilder geladen:", self.image_paths)

        for label in self.image_labels:
            label.destroy()

        self.image_labels.clear()
        self.image_photos.clear()

        for i, path in enumerate(paths):
            img = Image.open(path)
            img.thumbnail((300, 300))
            image_label = tk.Label(self.image_frame)
            image_label.grid(row=0, column=i, padx=10, pady=10)
            self.create_picture_preview(img, image_label)
            self.image_labels.append(image_label)

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
        self.image_photos.append(photo)

    
    def analyze_image(self):
        print("Die Bilder die übergeben worden sind: ", self.image_paths)
        
        if not (self.image_paths):
            messagebox.showwarning("Keine Bilder", "Bitte zuerst ein Bild auswählen.")
            return
        
        self.main(self.image_paths)

        
    def process_cv2_picture(self, img):

        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY) # RGB zu GRAY, weil PIL das Bild in RGB speichert
        #blurred = cv2.GaussianBlur(gray, (3, 3), 0) # Bild wird unscharf gemacht, um Rauschen zu entfernen
        inverted = cv2.bitwise_not(gray) # Bildinvertierung: helle Pixel werden dunkel und umgekehrt -> Schrift hell, Background dunkel
        _, binary = cv2.threshold(inverted, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU) # OTSU berechnet optimalen Schwellenwert für das gesamte Bild -> alle darunter werden schwarz, darüber (also der Text) weiß
        #binary = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 51, 2)
        processed = cv2.bitwise_not(binary) # Umkehrung Binärbild: Text wird schwarz, Hintergrund weiß

        return processed


    def configure(self):
        # Laden des .env-Files mit dem API-Keys
        load_dotenv()


    # Funktion zur Konvertierung des Bildes in ein Byte-Array
    def encode_image(self, img):
        # img kann ein numpy array oder ein PIL Image sein
        if isinstance(img, np.ndarray):
            img = Image.fromarray(img) # Umwandlung von numpy array in PIL Image
        elif not isinstance(img, Image.Image):
            raise ValueError("Ungültiges Bildformat")
        
        # Bild → BytesIO (Speicherpuffer) → Bytes → Base64-Text → Data-URL        

        buffered = BytesIO() # BytesIO-Objekt zum Speichern des Bildes im Arbeitsspeicher
        img.save(buffered, format="PNG") # Bild wird in PNG Format gespeichert
        img_bytes= buffered.getvalue() # Bild wird in Bytes umgewandelt
        base64_str = base64.b64encode(img_bytes).decode('utf-8') # Bilddaten werden in einen Base64-String umgewandelt, sodass sie als Text gespeichert werden können
        return f"data:image/png;base64,{base64_str}" # Erzeugug der Data-URL für das Bild

    
    def main(self, image_paths):

        encoded_images = []

        ############## Bildverarbeitung & Encoding ################
        for image_path in image_paths:
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
            
            processed_img = self.process_cv2_picture(img)
            
            encoded_img = self.encode_image(processed_img)
            encoded_images.append((encoded_img, image_path))


        self.configure()
        client = OpenAI(api_key=os.getenv('api_key')) # Zugriff auf den API-Key

        content = [
            {
            "type": "input_text",
            "text": "Die Bilder zeigen die Verpackung eines Produkts aus verschiedenen Perspektiven. Bitte extrahiere die Zutatenliste und die Nährwerttabelle des Produkts. Wenn du keine Zutatenliste oder Nährwerttabelle findest, gib bitte an, dass diese nicht vorhanden sind."
            }
        ]


        # Gruppiere die Bilder nach Produkt-ID (alles vor dem Unterstrich)
        product_images = defaultdict(list)
        for encoded_img, image_path in encoded_images:
            basename = os.path.basename(image_path) # Extrahiert den Dateinamen
            product_id = basename.split('_')[0] # Speichert Produktnummer
            product_images[product_id].append((encoded_img, image_path))

        # Vorbereitung API Abfrage
        for product_id, images in product_images.items():
            # Baue den Content für dieses Produkt
            product_content = [
            {
                "type": "input_text",
                "text": "Die Bilder zeigen die Verpackung eines Produkts aus verschiedenen Perspektiven. Bitte extrahiere die Zutatenliste und die Nährwerttabelle des Produkts. Wenn du keine Zutatenliste oder Nährwerttabelle findest, gib bitte an, dass diese nicht vorhanden sind."
            }
            ]

            for encoded_img, image_path in images:
                product_content.append({
                    "type": "input_image",
                    "image_url": encoded_img
                })

            # Erstellen der Anfrage an die API für dieses Produkt
            response = client.responses.create(
            model="gpt-4.1-mini",
            input=[
                {
                "role": "user",
                "content": product_content
                }
            ]
            )

            print(f"Produkt {product_id}:")
            print(response.output_text)

        # # Erstellen der Anfrage an die API
        # response = client.responses.create(
        #     model="gpt-4.1-mini",
        #     input=[
        #         {
        #             "role": "user",
        #             "content": content
        #         } 
        #     ]
        # )

        # print(response.output_text)

# Teste den Code
if __name__ == "__main__":
    root = tk.Tk()
    app = ImageAnalyzer(root)
    root.mainloop()