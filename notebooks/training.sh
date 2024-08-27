#!/bin/bash
#SBATCH --job-name=training-Res50-1.9

#SBATCH --nodes=1 
#SBATCH --cpus-per-task=1
#SBATCH --time=24:00:00
#SBATCH --account=ajitj0
#SBATCH --partition=spgpu
#SBATCH --gres=gpu:1
#SBATCH --gpu_cmode=exclusive
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-task=1
#SBATCH --mail-type=NONE
#SBATCH  --mem=50GB

export NCCL_IB_DISABLE=1
export NCCL_P2P_DISABLE=1

cd "$SLURM_SUBMIT_DIR" 

# Activate conda environment
eval "$(conda shell.bash hook)"
conda activate detectron2env

module load cuda/11.8.0
module load cudnn/11.8-v8.7.0
module load gcc/11.2.0

srun python /home/anishjv/cell_seg_classify/notebooks/training_notebooks/detectron2/tools/lazyconfig_train_net.py --config-file r50_1.9.yaml