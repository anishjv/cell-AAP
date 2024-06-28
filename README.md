# cellular-Automated Annotation Pipeline
Utilities for the semi-automated generation of instance segmentation annotations to be used for neural network training. Utilities are built ontop of [UMAP](https://github.com/lmcinnes/umap), [HDBSCAN](https://arxiv.org/abs/1911.02282) and a finetuned encoder version of FAIR's [Segment Anything Model](https://github.com/facebookresearch/segment-anything/tree/main?tab=readme-ov-file) developed by Computational Cell Analytics for the project [micro-sam](https://github.com/computational-cell-analytics/micro-sam/tree/master/micro_sam/sam_annotator). In addition to providing utilies for annotation building, we train a network, FAIR's [detectron2](https://github.com/facebookresearch/detectron2) to 
1. Demonstrate the efficacy of our utilities. 
2. Be used for microscopy annotation of supported cell lines 

Supported cell lines currently include:
1. HeLa

In development cell lines currently include:
1. U2OS
2. HT1080
3. Yeast

We've developed a napari application for the usage of this pre-trained network and propose a transfer learning schematic for the handling of new cell lines. 



# Installation 
We highly recommend installing cell-AAP in a clean conda environment. To do so you must have [miniconda](https://docs.anaconda.com/free/miniconda/#quick-command-line-install) or [anaconda](https://docs.anaconda.com/free/anaconda/) installed.

If a conda distribution has been installed:

1. Create and activate a clean environment 

        conda create -n cell-aap-env
        conda activate cell-app-env

2. Within this enviroment install pip

        conda install pip

3. Then install cell-AAP from PyPi

        pip install cell-AAP --upgrade

4. Finally detectron2 must be built from source, atop cell-AAP
    
        #For MacOS
        CC=clang CXX=clang++ ARCHFLAGS="-arch arm64" python -m pip install 'git+https://github.com/facebookresearch/detectron2.git'

        #For other operating systems 
        python -m pip install 'git+https://github.com/facebookresearch/detectron2.git'



# Napari Plugin Usage

1. To open napari simply type "napari" into the command line, ensure that you are working the correct environment
2. To instantiate the plugin navigate to the "Plugins" menu and select "cell-AAP"
3. You should now see the Plugin, where you can select an image, display it, and run inference on it. 


# Configs Best Practices

If running inference on large volumes of data, i.e. timeseries data >= 300 MB in size, we recommed to procceed in the following manner. 

1. Assemble a small, < 100 MB, substack of your data using python or a program like [ImageJ](https://imagej.net/ij/download.html)
2. Use this substack to find the optimal parameters for your data, (Number of Cells, Confidence)
3. Run Inference over the volume using the discovered optimal parameters

Note: Finding the optimal set of parameters requires some trial and error, to assist we've created a table. 

| Classifications $\Downarrow$ Detections $\Rightarrow$ | **Too few**                            | **Too many**                             |
|----------------------------|----------------------------------------|------------------------------------------|
| **Dropping M-phase**       | Confidence $\Downarrow$ <br> Number of Cells $\Uparrow$ | Confidence $\Downarrow$ <br> Number of cells $\Downarrow$ |
| **Missclasifying M-phase** | Confidence $\Uparrow$ <br> Number of Cells $\Uparrow$   | Confidence $\Uparrow$ <br> Number of Cells $\Downarrow$   |


# Intepreting Results 

Once inference is complete the following colors indicate class prediction
- Red: Non-mitotic
- Blue: Mitotic
- Purple: Interclass double prediction

Note: Interclass double predictions are often early prophase cells that the network is not "confident" in, to mitigate such predictions increase the minimum confidence threshold. This will typically result in most double predictions regressing to the Non-mitotic class. 










