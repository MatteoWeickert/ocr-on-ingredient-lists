from ultralytics import YOLO

# Lade das YOLOv8m-Modell
model = YOLO("yolov8m.pt")

# Trainiere es mit deinen Daten
model.train(
    data="data.yaml",
    epochs=50,
    imgsz=640,
    batch=16,
    device=0
)