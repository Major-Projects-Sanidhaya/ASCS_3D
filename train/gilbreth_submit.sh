#!/bin/bash
#SBATCH --job-name=ascs3d_phase1
#SBATCH --account=pfw-cs
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --gpus-per-node=1
#SBATCH --time=04:00:00
#SBATCH --output=logs/train_%j.out
#SBATCH --error=logs/train_%j.err

module load anaconda
conda activate ascs3d

cd $SLURM_SUBMIT_DIR
mkdir -p logs checkpoints

# Phase 1: health/collision reward, 32 parallel envs
python train/train_phase1.py \
    --phase 1 \
    --n-envs 32 \
    --steps 500000 \
    --scenario default

# Phase 2: add coverage reward, continue from phase 1 checkpoint
python train/train_phase1.py \
    --phase 2 \
    --n-envs 32 \
    --steps 500000 \
    --load checkpoints/final_phase1_default \
    --scenario default

# Phase 3: add tasking reward
python train/train_phase1.py \
    --phase 3 \
    --n-envs 32 \
    --steps 500000 \
    --load checkpoints/final_phase2_default \
    --scenario default

# Phase 4: multi-zone generalisation
python train/train_phase1.py \
    --phase 4 \
    --n-envs 32 \
    --steps 500000 \
    --load checkpoints/final_phase3_default \
    --scenario forest_fire
