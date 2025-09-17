import argparse
import json
import os
from eval_llm import run_llm
from eval_traditional import run_traditional
from eval_helpers import _validate_config

def main():
    parser = argparse.ArgumentParser(description="OCR Evaluation Pipeline")
    parser.add_argument("--config", help="Path to the config file.", required=True)
    parser.add_argument("--oem", help="OCR Engine Mode. Default: 1", default=1)
    parser.add_argument("--psm", type=int, default=6, help="Tesseract Page Segmentation Mode (PSM). Default: 6")
    parser.add_argument("--temperature", type=float, default=0.2, help="LLM Temperature setting. Default: 0.2")
    parser.add_argument("--llm_model", type=str, default="gpt-4o", help="LLM model to use. Default: gpt-4o")
    parser.add_argument("--distance_threshold_factor", type=float, default=1.7, help="Factor for distance threshold in agglomerative clustering. Default: 1.7")

    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = json.load(f)

    _validate_config(cfg)

    if cfg["prefered_analysis_method"] == "llm":
        run_llm(cfg, temperature=args.temperature, llm_model=args.llm_model)
    elif cfg["prefered_analysis_method"] == "traditional":
        run_traditional(cfg, oem=args.oem, psm=args.psm, distance_threshold_factor=args.distance_threshold_factor)
    elif cfg["prefered_analysis_method"] == "both":
        run_llm(cfg, temperature=args.temperature, llm_model=args.llm_model)
        run_traditional(cfg, oem=args.oem, psm=args.psm, distance_threshold_factor=args.distance_threshold_factor)

if __name__ == "__main__":
    main()