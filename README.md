# Cellular Annotation & Perception Pipeline

![](https://github.com/anishjv/cell-AAP/blob/main/images/fig_1.png?raw=true)

## Description
Cell-APP automates the generation of cell masks (and classifications too!), enabling users to create 
custom instance segmentation training datasets in transmitted-light microscopy. 

To learn more, read our preprint: https://www.biorxiv.org/content/10.1101/2025.01.23.634498v2.

For questions regarding installation or usage, contact: anishjv@umich.edu

## Usage 
1. Users who wish to segment HeLa, U2OS, HT1080, or RPE-1 cell lines may try our pre-trained model. These models can be used through our GUI (see **Installation**) and their weights can be downloaded at: https://zenodo.org/communities/cellapp/records?q=&l=list&p=1&s=10. To learn about using pre-trained models through the GUI, see this video: 



2. Users who wish to segment their own cell lines may: (a) try our "general" model (GUI/weight download) or (b) 
train a custom model by creating an instance segmentation dataset via our *Dataset Generation GUI* (see **Installation**). To learn about creating custom datasets through the GUI, see this video: 

## Installation

`cell-AAP` requires Python **3.11–3.12**. We recommend installing into a clean virtual environment (via `conda` or `venv`) to avoid dependency conflicts.

### 1. Create and activate an environment

With **conda**:
```bash
conda create -n cellapp -c conda-forge python=3.11
conda activate cellapp
```

Or with **venv**:
```bash 
python -m venv cellapp
source cellapp/bin/activate  # Linux/Mac
cellapp\Scripts\activate     # Windows PowerShell
```

### 2. Install Pytorch:
```bash
conda install -c pytorch -c conda-forge pytorch torchvision #Mac
pip install torch torchvision #Linux/Windows
```

### 3. Install Cell-APP:
```bash
pip install cell-AAP
```
### 4. Finally, detectron2 must be built from source atop Cell-APP:
```bash
    
#Mac
git clone https://github.com/facebookresearch/detectron2.git
CC=clang CXX=clang++ ARCHFLAGS="-arch arm64" python -m pip install -e detectron2 --no-build-isolation

#Linux/Windows
git clone https://github.com/facebookresearch/detectron2.git
python -m pip install -e detectron2 --no-build-isolation
```

## Napari Plugin Usage

1. To open napari simply type "napari" into the command line, ensure that you are working the correct environment
2. To instantiate the plugin, navigate to the "Plugins" menu and hover over "cell-AAP"
3. You should see three plugin options; two relate to *Usage 1*; one relates to *Usage 2*. 











