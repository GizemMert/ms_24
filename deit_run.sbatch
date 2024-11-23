#!/bin/bash

#SBATCH --output=o_cap.job
#SBATCH --error=e_cap.job
#SBATCH --partition=gpu_p
#SBATCH --qos=gpu_normal
#SBATCH --gres=gpu:1

#SBATCH --time=10:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=60G
#SBATCH --nice=10000

# Load the conda environment
source /home/aih/gizem.mert/tools/apps/mamba/etc/profile.d/conda.sh
conda activate env

# Run the Python script
python main_deit.py