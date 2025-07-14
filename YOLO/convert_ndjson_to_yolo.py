# this script converts an NDJSON file with training data into a YOLO format 

import os
import json

# input NDJSON file
input_ndjson = "YOLO/dataset/train/train_data.ndjson"

# set output directory
output_dir = "YOLO/labels/train"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# class id mapping
class_mapping = {
    "ingredients": 0,
    "nutrition": 1
}

# function to calculate relative coordinates (between 0 and 1)

def convert_bbox(bbox, img_width, img_height):
    x_center = (bbox["left"] + bbox["width"] / 2) / img_width
    y_center = (bbox["top"] + bbox["height"] / 2) / img_height
    width = bbox["width"] / img_width
    height = bbox["height"] / img_height
    return x_center, y_center, width, height

# process the NDJSON file

with open(input_ndjson, "r") as input_file:
    for line in input_file:
        data = json.loads(line)
        external_id = data.get("data_row", {}).get("external_id", "")
        img_width = data.get("media_attributes", {}).get("width", "")
        img_height = data.get("media_attributes", {}).get("height", "")

        labels = []

        for project in data.get("projects", {}).values(): #iterate through all values of projects
            for label in project.get("labels", []):
                for object in label.get("annotations", {}).get("objects", []):
                    category = object.get("value", "")
                    bbox = object.get("bounding_box", {})

                    if category in class_mapping and bbox:
                        class_id = class_mapping[category]
                        x_center, y_center, width, height = convert_bbox(bbox, img_width, img_height)
                        labels.append(f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}") # 6 decimal places for precision

        if labels:
            label_file_name = os.path.splitext(external_id)[0] + ".txt"  # remove file extension
            label_file_path = os.path.join(output_dir, label_file_name)
            with open(label_file_path, "w") as label_file:
                label_file.write("\n".join(labels) + "\n")