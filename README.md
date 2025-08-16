# Cellular Annotation & Perception Pipeline

![](https://github.com/anishjv/cell-AAP/blob/main/images/fig_1.png?raw=true)

## Description
Cell-APP automates the generation of cell masks (and classifications too!), enabling users to create 
custom instance segmentation training dataset in transmitted-light microscopy. To learn more, read our preprint: https://www.biorxiv.org/content/10.1101/2025.01.23.634498v2. 

## Usage 
1. Users who wish to segment HeLa, U2OS, HT1080, or RPE-1 cell lines may try our pre-trained model. These models can be used through our GUI (see **Installation**) and their weights can be downloaded at: https://zenodo.org/communities/cellapp/records?q=&l=list&p=1&s=10. To learn about using pre-trained models through the GUI, see this video: 



2. Users who wish to segment their own cell lines may: (a) try our "general" model (GUI/weight download) or (b) 
train a custom model by creating an instance segmentation dataset via our *Dataset Generation GUI* (see **Installation**). To learn about creating custom datasets through the GUI, see this video: 

## Installation 
We highly recommend installing cell-APP in a clean conda environment. To do so, you must have [miniconda](https://docs.anaconda.com/free/miniconda/#quick-command-line-install) or [anaconda](https://docs.anaconda.com/free/anaconda/) installed.

Once a conda distribution has been installed:

1. Create and activate a clean environment 

        conda create -n cell-aap-env python=3.11.0
        conda activate cell-app-env

2. Within this environment, install pip

        conda install pip

3. Then install the package from PyPi (the package bears the name "cell-AAP;" a historical quirk)

        pip install cell-AAP --upgrade

4. Finally detectron2 must be built from source, atop cell-AAP
    
        #For MacOS
        CC=clang CXX=clang++ ARCHFLAGS="-arch arm64" python -m pip install 'git+https://github.com/facebookresearch/detectron2.git'

        #For other operating systems 
        python -m pip install 'git+https://github.com/facebookresearch/detectron2.git'



## Napari Plugin Usage

1. To open napari simply type "napari" into the command line, ensure that you are working the correct environment
2. To instantiate the plugin, navigate to the "Plugins" menu and hover over "cell-AAP"
3. You should see three plugin options; two relate to *Usage 1*; one relate to *Usage 2*. 











