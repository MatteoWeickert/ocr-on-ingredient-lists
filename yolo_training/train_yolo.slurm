#!/bin/bash
#SBATCH --job-name=yolo8m-train
#SBATCH --output=output_%j.log
#SBATCH --error=error_%j.log
#SBATCH --time=04:00:00
#SBATCH --partition=o0cvmls
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --mail-user=mweicker@uni-muenster.de
#SBATCH --mail-type=END,FAIL

module purge
module load palma/2021a Miniconda3/4.9.2

source ~/envs/yolo8/etc/profile.d/conda.sh
conda activate ~/envs/yolo8

pip install ultralytics

cd /scratch/tmp/mweicker/yolo_training
python train.py

echo "Training completed. Check the output and error logs for details. $(date)"
nvidia-smi
