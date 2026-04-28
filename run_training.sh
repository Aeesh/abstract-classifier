#!/bin/bash
#SBATCH --job-name=abstract-classifier
#SBATCH --partition=GPU-shared
#SBATCH --gres=gpu:v100-32:1
#SBATCH --time=03:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=5
#SBATCH --mem=32G
#SBATCH --output=logs/train_%j.out
#SBATCH --error=logs/train_%j.err

mkdir -p logs

module load anaconda3
conda activate arxiv-ml

cd $SLURM_SUBMIT_DIR

# Log in to WandB non-interactively using your API key
export WANDB_API_KEY= WANDB_API_KEY


python train.py