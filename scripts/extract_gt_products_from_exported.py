# this script extracts the data that has GT text annotations from the exported Labelbox dataset

import json

# product numbers with GT text
with open("YOLO/dataset/eval/eval_product_nr.json", "r") as f:
    eval_product_ids = set(json.load(f)) # Umwandlung in Set, um Duplikate zu entfernen und schnellere Suche zu ermöglichen

# exported labels
exported_ndjson = "YOLO/export.ndjson"

# Zieldateien
eval_output_file = "YOLO/dataset/eval/eval_data.ndjson"
train_output_file = "YOLO/dataset/train/train_data.ndjson"

# lists to store extracted data
eval_data = []
train_data = []

with open(exported_ndjson, "r") as input_file:
    for line in input_file:
        data = json.loads(line)
        external_id = data.get("data_row", {}).get("external_id", "")
        product_id = external_id.split("_")[0] # convert from "product_1234.jpg" to "product"

        if product_id in eval_product_ids:
            eval_data.append(line)
        else:
            train_data.append(line)

with open(eval_output_file, "w") as eval_file:
    eval_file.writelines(eval_data)

with open(train_output_file, "w") as train_file:
    train_file.writelines(train_data)