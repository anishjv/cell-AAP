from magicgui import magicgui
import napari
import torch
import numpy as np
import os
import pathlib
import tifffile as tiff
import datetime
from napari.layers import Image
from napari.types import LabelsData
from detectron2.utils.logger import setup_logger
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from data_module import data_utils as dt
from skimage.measure import label

setup_logger()
TORCH_VERSION = ".".join(torch.__version__.split(".")[:2])
CUDA_VERSION = torch.__version__.split("+")[-1]
model = '20240425_134343'

cfg = get_cfg()
cfg.merge_from_file(model + "/config.yaml")
cfg.MODEL.WEIGHTS = os.path.join(model, "model_final.pth")
cfg.MODEL.DEVICE = 'cpu' 
cfg.TEST.DETECTIONS_PER_IMAGE = 100000
predictor = DefaultPredictor(cfg)

@magicgui(call_button='inference', layout = 'vertical')
def inference(save_mask: bool,
              layer_image: Image, 
              predictor = predictor, 
              confidence_threshold = 0.1,
              filepath = pathlib.Path("default/path")) -> LabelsData:
    
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = confidence_threshold

    if len(layer_image.data.shape) == 3:
        mask_array = np.empty(
                            (layer_image.data.shape[0], int(layer_image.data.shape[1]), int(layer_image.data.shape[2])), 
                            dtype='int32'
                            )
        for frame in range(layer_image.data.shape[0]):
            img = dt.bw_to_rgb(layer_image.data[frame]).astype('float32')
            output = predictor(img)
            segmentations = np.asarray(output['instances'].pred_masks.to('cpu'))
            segmentations = label(np.logical_or.reduce(segmentations, axis =0))
            mask_array[frame] = segmentations

    elif len(layer_image.data.shape) == 2:
        img = dt.bw_to_rgb(layer_image.data.astype('float32'))
        output = predictor(img)
        segmentations = np.asarray(output['instances'].pred_masks.to('cpu'))
        segmentations = label(np.logical_or.reduce(segmentations, axis = 0))
        mask_array = segmentations

    if save_mask == True:
        if filepath == pathlib.Path('default/path'):
            filepath = os.getcwd()
        now = datetime.datetime.now()
        tiff.imwrite(
                    os.path.join(filepath, f'{now.strftime("%Y%m%d_%H%M%S")}_{model}.tiff'), 
                    mask_array.astype('uint8'), 
                    compression = "jpeg",
                    compressionargs={"level": 20}
                    )


    return mask_array

viewer = napari.Viewer()
viewer.window.add_dock_widget(inference, area = 'right')
napari.run()






