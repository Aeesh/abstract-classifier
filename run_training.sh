#!/bin/bash
#SBATCH --job-name=abstract-classifier
#SBATCH --partition=GPU-shared
#SBATCH --gres=gpu:1
#SBATCH --time=08:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=5
#SBATCH --mem=20G
#SBATCH --output=logs/train_%j.out
#SBATCH --error=logs/train_%j.err


module load anaconda3
conda activate arxiv-ml

cd $SLURM_SUBMIT_DIR

# safe temp + cache
export TMPDIR=/local/$USER/tmp
mkdir -p $TMPDIR

export WANDB_API_KEY= WANDB_API_KEY
export TRANSFORMERS_CACHE=$HF_HOME
mkdir -p $HF_HOME

# Log in to WandB non-interactively using your API key
export WANDB_API_KEY= WANDB_API_KEY


# run script passed from sbatch
python -u $1

# # Submit
# sbatch run_training.sh evaluate.py

# # Check status
# squeue -u $USER

# # Watch output live
# tail -f logs/train_JOBID.out