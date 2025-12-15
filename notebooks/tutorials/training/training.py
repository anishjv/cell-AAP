import sys
import os
from detectron2.data.datasets import register_coco_instances

# Configuration 
# Set the base directory relative to where the script is executed.
# NOTE: The user must modify this path to match their setup.
BASE_DIR = os.path.expanduser("example_dataset")
CONFIG_FILE = "config.yaml"
TRAIN_SCRIPT = "/path/to/local/detectron2/tools/lazyconfig_train_net.py"
NUM_GPUS = "1"

# Register datasets
for d in ["train", "test"]:
    dataset_name = f"example_dataset_{d}"
    # Use os.path.join for robust path building
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

# Specify command-line arguments for training 
sys.argv = [
    "lazyconfig_train_net.py",
    "--config-file",
    CONFIG_FILE,
    "--num-gpus",
    NUM_GPUS,
]

# Run training script in the same process 
if not os.path.exists(TRAIN_SCRIPT):
    print(f"Error: Training script not found at {TRAIN_SCRIPT}")
    sys.exit(1)

# This is the core logic that the spawned processes need to re-run:
with open(TRAIN_SCRIPT) as f:
    code = compile(f.read(), TRAIN_SCRIPT, "exec")
    exec(code)