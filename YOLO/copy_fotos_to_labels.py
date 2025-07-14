# this script copies all the images that are associated to labels into the images/train directory

import os
import shutil 

images_dir = "./Fotos-Labbase/Fotos-Labbase" # source directory containing images
labels_dir = "YOLO/labels/train" # directory containing label files
train_images_dir = "YOLO/images/train" # destination directory for training images

os.makedirs(train_images_dir, exist_ok=True)  # create destination directory if it doesn't exist

for label_file in os.listdir(labels_dir):
    base_name = os.path.splitext(label_file)[0]  # remove file extension
    image_file = f"{base_name}.jpg"  # images are in .jpg format
    source_image_path = os.path.join(images_dir, image_file)
    destination_image_path = os.path.join(train_images_dir, image_file)

    if os.path.exists(source_image_path):
        shutil.copy2(source_image_path, destination_image_path)
        print(f"Kopiert: {image_file}")
    else:
        print(f"❌ Bild nicht gefunden: {image_file}")
