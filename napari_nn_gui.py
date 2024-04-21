from magicgui import magicgui
import napari
import cv2, torch
import numpy as np
import os
from napari.layers import Image, Labels
from napari.types import LabelsData
from detectron2.utils.logger import setup_logger
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog
from data_module import data_utils as dt
from skimage.measure import label

setup_logger()
TORCH_VERSION = ".".join(torch.__version__.split(".")[:2])
CUDA_VERSION = torch.__version__.split("+")[-1]


cfg = get_cfg()
cfg.merge_from_file("20240419_171905/config.yaml")
cfg.MODEL.WEIGHTS = os.path.join("20240419_171905", "model_final.pth")
cfg.MODEL.DEVICE = 'cpu' 
predictor = DefaultPredictor(cfg)

@magicgui(call_button='inference', layout = 'vertical')
def inference(layer_image: Image, predictor = predictor, confidence_threshold = 0.1) -> LabelsData:

    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = confidence_threshold
    mask_array = np.empty((layer_image.data.shape[0], int(layer_image.data.shape[1]), int(layer_image.data.shape[2])), dtype='int32')
    for frame in range(layer_image.data.shape[0]):
        img = dt.bw_to_rgb(layer_image.data[frame]).astype('float32')
        output = predictor(img)
        segmentations = np.asarray(output['instances'].pred_masks.to('cpu'))
        segmentations = label(np.logical_or.reduce(segmentations, axis =0))
        mask_array[frame] = segmentations

    return mask_array

viewer = napari.Viewer()
viewer.window.add_dock_widget(inference, area = 'right')
napari.run()






