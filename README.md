# cellular-Automated Annotation Pipeline
Utilities for the semi-automated generation of instance segmentation annotations to be used for neural network training. Utilities are built ontop of [UMAP](https://github.com/lmcinnes/umap), [HDBSCAN](https://arxiv.org/abs/1911.02282) and a finetuned encoder version of FAIR's [Segment Anything Model](https://github.com/facebookresearch/segment-anything/tree/main?tab=readme-ov-file) developed by Computational Cell Analytics for the project [micro-sam](https://github.com/computational-cell-analytics/micro-sam/tree/master/micro_sam/sam_annotator). In addition to providing utilies for annotation building, we train a network, FAIR's [detectron2](https://github.com/facebookresearch/detectron2) to 
1. Demonstrate the efficacy of our utilities. 
2. Be used for microscopy annotation of supported cell lines 

Supported cell lines currently include:
1. HeLa

In development cell lines currently include:
1. U2OS
2. HT1080

We've developed a napari application for the usage of this pre-trained network and propose a transfer learning schematic for the handling of new cell lines. 




