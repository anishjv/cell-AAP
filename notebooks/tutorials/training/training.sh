#!/bin/bash
#SBATCH --job-name=training
#SBATCH --nodes=1 
#SBATCH --cpus-per-task=1
#SBATCH --time=48:00:00
#SBATCH --account=ajitj0
#SBATCH --partition=spgpu
#SBATCH --gres=gpu:1
#SBATCH --gpu_cmode=exclusive
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-task=1
#SBATCH --mail-type=NONE
#SBATCH --mem=50GB

export NCCL_IB_DISABLE=1
export NCCL_P2P_DISABLE=1

cd "$SLURM_SUBMIT_DIR" 

# Activate conda environment in which detectron2 is installed
eval "$(conda shell.bash hook)"
conda activate detectron2env

module load cuda/11.8.0
module load cudnn/11.8-v8.7.0
module load gcc/11.2.0

# Run registration + training in one Python process
srun python3 - <<'EOF'
import sys
from detectron2.data.datasets import register_coco_instances

#Register datasets
for d in ["train", "test"]: #I have two datasets to register: training and testing
    register_coco_instances(
        f"HeLa_{d}_0.2", #this specifies the name that will be registered 
        {},
        f"/home/anishjv/cell_seg_classify/datasets/HeLa_0.2/{d}/hela_0.2_{d}.json", #this specified where the COCO-format json file is located
        f"/home/anishjv/cell_seg_classify/datasets/HeLa_0.2/{d}/images/" #this specifies where the images are located
    )
print("Successfully registered datasets")

# ---- Specify command-line arguments for training ----
sys.argv = [
    "lazyconfig_train_net.py",
    "--config-file",
    "config.yaml", #here you must specify the path to your config file
    "--num-gpus",
    "1",
]

# ---- Run training script in the same process ----
train_script = "/path/to/your/detectron/copy/detectron2/tools/lazyconfig_train_net.py"
with open(train_script) as f:
    code = compile(f.read(), train_script, "exec")
    exec(code)
EOF

