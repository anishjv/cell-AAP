#!/bin/bash

# ENVIRONMENT SETUP
# The user must ensure a conda environment exists that has all necessary dependencies (Detectron2, PyTorch, etx.) installed
# Activate conda environment in which detectron2 is installed
# This assumes the user has initialized conda in their shell (e.g., with 'conda init')
eval "$(conda shell.bash hook)"
conda activate biop

# DEPENDENCY LOADING
# If using an HPC system, the user may need to load the following dependencies: 
# module load cuda/11.8.0
# module load cudnn/11.8-v8.7.0
# module load gcc/11.2.0


python3 training.py

