#!/bin/bash
#SBATCH --job-name=ocr_eval
#SBATCH --time=01:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --output=/scratch/tmp/mweicker/ocr-on-ingredient-lists/logs/%x_%j.out
#SBATCH --error=/scratch/tmp/mweicker/ocr-on-ingredient-lists/logs/%x_%j.err

# ======================================================================
# Initialisierung von Conda für die Batch-Umgebung
source /home/m/mweicker/miniforge3/etc/profile.d/conda.sh
# ======================================================================

set -euo pipefail

# 1) Ins Repo & venv aktivieren
cd /scratch/tmp/mweicker/ocr-on-ingredient-lists
conda activate ocr_env

# 2) Argumente: Config-Pfad und Run-Name
CFG="${1:-eval/configs/ingredients_trad_hpc.json}"
RUN="${2:-run_$(date +%F_%H-%M)}"

# 3) (optional) commit/branch ins Log
echo "Git branch: $(git rev-parse --abbrev-ref HEAD)"
echo "Git commit: $(git rev-parse --short HEAD)"
echo "Config: $CFG"
echo "Run: $RUN"

# 4) Config per Env an main.py übergeben
export EVAL_CONFIG="$CFG"

# 5) Start
echo "Starting Python application inside Conda environment..."
which python # Zur Kontrolle, sollte jetzt auf den Conda-Python-Pfad zeigen
python eval/main.py | tee "runs/${RUN}.log"
echo "Python application finished."