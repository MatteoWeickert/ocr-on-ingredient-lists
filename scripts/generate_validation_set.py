# this script select randomly 10% of the training images and their corresponding labels for validation and copies them to a validation directory

import os
import random
import shutil

VAL_SPLIT = 0.1 # percentage of images to select for validation

# directories
train_images_dir = "YOLO/images/train"  # source directory containing training images
train_labels_dir = "YOLO/labels/train"  # directory containing training labels
val_images_dir = "YOLO/images/val"  # destination directory for validation images
val_labels_dir = "YOLO/labels/val"  # destination directory for validation labels

# create directories 
os.makedirs(val_images_dir, exist_ok=True)
os.makedirs(val_labels_dir, exist_ok=True)

# extract images
image_files = [f for f in os.listdir(train_images_dir) if f.endswith('.jpg')]

#random selection of images for validation
val_count = int(len(image_files) * VAL_SPLIT)
val_files = random.sample(image_files, val_count)

# move selected images and their corresponding labels to validation directories
for image_file in val_files:

    label_file = os.path.splitext(image_file)[0] + ".txt"  # corresponding label file

    # directories for source and destination
    src_image_path = os.path.join(train_images_dir, image_file)
    src_label_path = os.path.join(train_labels_dir, label_file)
    dest_image_path = os.path.join(val_images_dir, image_file)
    dest_label_path = os.path.join(val_labels_dir, label_file)

    # copy image and label files if they exist
    if os.path.exists(src_image_path):
        shutil.move(src_image_path, dest_image_path)
        print(f"Copied image: {image_file}")

    if os.path.exists(src_label_path):
        shutil.move(src_label_path, dest_label_path)
        print(f"Copied label: {label_file}")