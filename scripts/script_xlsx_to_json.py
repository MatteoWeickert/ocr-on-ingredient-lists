import json
import pandas as pd

xlsx_file = "C:/Users/maweo/OneDrive - Universität Münster/Dokumente/Studium/Semester 6/Bachelorarbeit/ocr_test/GT_Nutrition.xlsx"
export_file = "C:/Users/maweo/OneDrive - Universität Münster/Dokumente/Studium/Semester 6/Bachelorarbeit/ocr_test/GT_Nutrition.json"
spaltennamen = ["ProduktNR", "Text"]

df = pd.read_excel(xlsx_file, header=0, names=spaltennamen)

def parse_json_string(text_str):
            try:
                # json.loads() wandelt einen String in ein Python-Objekt (dict/list) um
                return json.loads(text_str)
            except (json.JSONDecodeError, TypeError):
                # Falls der Inhalt der Zelle kein gültiger JSON-String ist 
                # (oder leer/kein String ist), behalten wir den Originalwert bei.
                return text_str
            
df["Text"] = df["Text"].apply(parse_json_string)
daten_als_liste = df.to_dict(orient='records')


with open(export_file, 'w', encoding='utf-8') as json_datei:
    # indent=4 sorgt für eine schöne, lesbare Formatierung der JSON-Datei
    # ensure_ascii=False stellt sicher, dass Umlaute und Sonderzeichen korrekt dargestellt werden
    json.dump(daten_als_liste, json_datei, ensure_ascii=False, indent=4)

print(f"Erfolgreich! Die Daten wurden von '{xlsx_file}' nach '{export_file}' konvertiert.")
