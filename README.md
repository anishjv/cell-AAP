# cellular Segmentation Annotation Pipeline
Utilities for the semi-auotomated generation of instance segmentation annotations to be used for neural network training. Utilities are built ontop of MAIR's [Segment Anything Model](https://github.com/facebookresearch/segment-anything/tree/main?tab=readme-ov-file), [UMAP](https://github.com/lmcinnes/umap) and [HDBSCAN](https://arxiv.org/abs/1911.02282). In addition to providing utilies for annotation building, we train a network, MAIR's [detectron2](https://github.com/facebookresearch/detectron2) to 
1. Demonstrate the efficacy of our utilities. 
2. Be used for microscopy annotation of supported cell lines 

Supported cell lines currently include:
1. HeLa

We've developed a napari application for the usage of this pre-trained network and propose a transfer learning schematic for the handling of new cell lines. 




