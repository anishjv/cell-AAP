#!/bin/bash

# ENVIRONMENT SETUP
# The user must ensure a conda environment exists that has all necessary dependencies (Detectron2, PyTorch, etx.) installed
# Activate conda environment in which detectron2 is installed
# This assumes the user has initialized conda in their shell (e.g., with 'conda init')
eval "$(conda shell.bash hook)"
conda activate detectron2env

# DEPENDENCY LOADING
# If using an HPC system, the user may need to load the following dependencies: 
# module load cuda/11.8.0
# module load cudnn/11.8-v8.7.0
# module load gcc/11.2.0


python3 - <<'EOF'
import sys
import os
from detectron2.data.datasets import register_coco_instances

# --- Configuration ---
# Set the base directory relative to where the script is executed.
# NOTE: The user must modify this path to match their setup.

BASE_DIR = os.path.expanduser("example_dataset")
CONFIG_FILE = "config.yaml" # Assume config is in the same directory as the script
TRAIN_SCRIPT = "/path/to/your/detectron/copy/detectron2/tools/lazyconfig_train_net.py"
NUM_GPUS = "1" # Set the number of GPUs to use (must be installed and available)

# Register datasets
for d in ["train", "test"]:
    dataset_name = f"example_dataset_{d}"
    json_path = os.path.join(BASE_DIR, f"{d}/example_dataset_{d}.json")
    image_dir = os.path.join(BASE_DIR, f"{d}/images/")

    if not os.path.exists(json_path) or not os.path.exists(image_dir):
        print(f"Error: Required paths for {dataset_name} do not exist.")
        print(f"JSON: {json_path}")
        print(f"Images: {image_dir}")
        sys.exit(1)

    register_coco_instances(
        dataset_name, 
        {},
        json_path, 
        image_dir
    )
print("Successfully registered datasets")

# ---- Specify command-line arguments for training ----
# These arguments are passed to the training script.
sys.argv = [
    "lazyconfig_train_net.py",
    "--config-file",
    CONFIG_FILE,
    "--num-gpus",
    NUM_GPUS,
]

# ---- Run training script in the same process ----
if not os.path.exists(TRAIN_SCRIPT):
    print(f"Error: Training script not found at {TRAIN_SCRIPT}")
    sys.exit(1)

with open(TRAIN_SCRIPT) as f:
    code = compile(f.read(), TRAIN_SCRIPT, "exec")
    exec(code)
EOF

