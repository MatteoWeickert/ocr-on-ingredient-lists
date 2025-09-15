from eval_llm import run_llm
from eval_traditional import run_traditional
from eval_helpers import _validate_config

import os, json

if __name__ == "__main__":
    cfg_path = os.environ.get("EVAL_CONFIG", os.path.join(os.path.dirname(__file__), "config_local.json"))
    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg = json.load(f)
    _validate_config(cfg)

    if cfg["prefered_analysis_method"] == "llm":
        run_llm(cfg)
    elif cfg["prefered_analysis_method"] == "traditional":
        run_traditional(cfg)
    elif cfg["prefered_analysis_method"] == "both":
        run_llm(cfg)
        run_traditional(cfg)
