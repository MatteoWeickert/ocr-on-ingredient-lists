from eval_llm import run_llm
from eval_traditional import run_traditional
from eval_helpers import _validate_config

if __name__ == "__main__":
    cfg = {
        "img_dir": "C:\\Users\\maweo\\OneDrive - Universit\u00e4t M\u00fcnster\\Dokumente\\Studium\\Semester 6\\Bachelorarbeit\\ocr_test\\eval\\ingredients\\test_img",
        "gt_json": "C:\\Users\\maweo\\OneDrive - Universit\u00e4t M\u00fcnster\\Dokumente\\Studium\\Semester 6\\Bachelorarbeit\\ocr_test\\eval\\ingredients\\GT_Ingredients_bbox.json",
        "model": "C:\\Users\\maweo\\OneDrive - Universit\u00e4t M\u00fcnster\\Dokumente\\Studium\\Semester 6\\Bachelorarbeit\\ocr_test\\yolo_results\\model_yolov8s\\weights\\best.pt",
        "class_filter": "ingredients",
        "prefered_analysis_method": "traditional", # "llm" or "traditional" or "both"
        "out_dir": "C:\\Users\\maweo\\OneDrive - Universit\u00e4t M\u00fcnster\\Dokumente\\Studium\\Semester 6\\Bachelorarbeit\\ocr_test\\eval\\eval_outputs",
        "traditional_script": "C:\\Users\\maweo\\OneDrive - Universit\u00e4t M\u00fcnster\\Dokumente\\Studium\\Semester 6\\Bachelorarbeit\\ocr_test\\eval\\traditional_pipeline.py",
        "llm_script": "C:\\Users\\maweo\\OneDrive - Universit\u00e4t M\u00fcnster\\Dokumente\\Studium\\Semester 6\\Bachelorarbeit\\ocr_test\\eval\\llm_pipeline.py"
    }
    _validate_config(cfg)

    if cfg["prefered_analysis_method"] == "llm":
        run_llm(cfg)
    elif cfg["prefered_analysis_method"] == "traditional":
        run_traditional(cfg)
    elif cfg["prefered_analysis_method"] == "both":
        run_llm(cfg)
        run_traditional(cfg)
