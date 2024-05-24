from __future__ import annotations
import logging

import napari
import cell_AAP.napari.ui as ui # type: ignore

import numpy as np 
import sys
import cv2
import tifffile as tiff
import re
import os
import torch
import skimage.measure

from detectron2.utils.logger import setup_logger
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.data import MetadataCatalog
from qtpy import QtWidgets

sys.path.append("/Users/whoisv/cell-AAP/cell_AAP/")
from data_module import annotation_utils as au # type: ignore

setup_logger()
TORCH_VERSION = ".".join(torch.__version__.split(".")[:2])
CUDA_VERSION = torch.__version__.split("+")[-1]
model = "/Users/whoisv/cell-AAP/models/20240520_140429_1.8"
model_version = "final"
cellseg_metadata = MetadataCatalog.get("cellseg_train_1.8")


__all__ = [
    "create_cellAAP_widget",
]

# get the logger instance
logger = logging.getLogger(__name__)

# if we don't have any handlers, set one up
if not logger.handlers:
    # configure stream handler
    log_fmt = logging.Formatter(
        "[%(levelname)s][%(asctime)s] %(message)s",
        datefmt="%Y/%m/%d %I:%M:%S %p",
    )
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(log_fmt)

    logger.addHandler(console_handler)
    logger.setLevel(logging.DEBUG)



def create_cellAAP_widget() -> ui.cellAAPWidget:
    "Creates instance of ui.cellAAPWidget and sets callbacks"

    cellaap_widget = ui.cellAAPWidget(
        napari_viewer = napari.current_viewer(),
        cfg = get_cfg()
    )
    
    cellaap_widget.cfg.merge_from_file(model + "/config.yaml")
    cellaap_widget.cfg.MODEL.WEIGHTS = os.path.join(model, f"model_{model_version}.pth")
    cellaap_widget.cfg.MODEL.DEVICE = "cpu"


    cellaap_widget.inference_button.clicked.connect(
        lambda: run_inference(cellaap_widget)                               
    )

    cellaap_widget.display_button.clicked.connect(
        lambda: display(cellaap_widget)
    )

    cellaap_widget.image_selector.clicked.connect(
        lambda: grab_file(cellaap_widget)
    )

    cellaap_widget.path_selector.clicked.connect(
        lambda : grab_directory(cellaap_widget)
    )

    cellaap_widget.set_configs.clicked.connect(
        lambda : configure(cellaap_widget)
    )

    return cellaap_widget



def run_inference(cellaap_widget : ui.cellAAPWidget):
    """
    Runs inference on image returned by self._image_select(), saves inference result if save selector has been checked
    ----------------------------------------------------------------------------------------------------------------
    Inputs:
        cellapp_widget: instance of ui.cellAAPWidget()
    """
    prog_count = 0
    mask_array = []
    points = ()
    print('Inference')
    try:
        name, im_array = image_select(cellaap_widget)
    except AttributeError:
        napari.utils.notifications.show_error(
            'No Image has been selected'
        )
        return 
    
    cellaap_widget.predictor = DefaultPredictor(cellaap_widget.cfg)
    if len(im_array.shape) == 3:
        for frame in range(im_array.shape[0]):
            prog_count += 1
            cellaap_widget.progress_bar.setMaximum(im_array.shape[0])
            cellaap_widget.progress_bar.setValue(prog_count)
            img = au.bw_to_rgb(im_array[frame].astype("float32"))
            segmentations, mitotic_centroids = inference(cellaap_widget, img, frame)
            mask_array.append(segmentations.astype('uint8'))
            points += (mitotic_centroids, )

    elif len(im_array.shape) == 2:
        prog_count += 1
        cellaap_widget.progress_bar.setValue(prog_count)
        img = au.bw_to_rgb(im_array.astype("float32"))
        segmentations, mitotic_centroids = inference(cellaap_widget, img)
        mask_array.append(segmentations.astype('uint8'))
        points += (mitotic_centroids, )

    name = name.replace(".", "/").split("/")[-2]

    if cellaap_widget.save_selector.isChecked() == True:
        try:
            filepath = cellaap_widget.dir_grabber
        except AttributeError:
            napari.utils.notifications.show_error(
            'No Directory has been selected - will save output to current working directory'
        )
            filepath = os.getcwd() 
            pass

        tiff.imwrite(
            os.path.join(filepath, f"{model}_{model_version}_{name}_.tif"),
            np.array(mask_array).astype("uint16"),
        )

        
    points_array = np.vstack(points)
    cellaap_widget.progress_bar.reset()
    cellaap_widget.viewer.add_labels(np.array(mask_array), name = name + 'inf', opacity = 0.3)
    cellaap_widget.viewer.add_points(points_array, ndim = points_array.shape[1], name = name + 'mitotic_centroids', size = 15)


def inference(cellaap_widget : ui.cellAAPWidget, img : np.ndarray, frame_num: int = None) -> tuple[np.ndarray, list[np.ndarray[tuple]]]: 
    '''
    Runs the actual inference -> Detectron2 -> masks
    ------------------------------------------------
    INPUTS:
        cellaap_widget: instance of ui.cellAAPWidget()
    '''
    if img.shape[0] != 2048:
        half_diff = (2048 - img.shape) / 2
        img = cv2.copyMakeBorder(
                img, 
                half_diff, 
                half_diff, 
                half_diff, 
                half_diff, 
                cv2.BORDER_CONSTANT, 
                value = img.mean()
                )

    output = cellaap_widget.predictor(img)
    segmentations = output["instances"].pred_masks.to("cpu")
    labels = output['instances'].pred_classes.to('cpu')

    seg_labeled = np.zeros_like(segmentations[0], int)
    for i, mask in enumerate(segmentations):
        if labels[i] == 0:
            seg_labeled[mask] = 1
        else:
            seg_labeled[mask] = 100

    mitotic_centroids = []
    for i, class_label in enumerate(labels):
        if class_label == 1:
            labeled_mask = skimage.measure.label(segmentations[i])
            centroid = skimage.measure.centroid(labeled_mask)
            if frame_num != None:
                centroid = np.array([frame_num, centroid[0], centroid[1]])

            mitotic_centroids.append(centroid)
            
        else:
            pass

    # Quicker but does not work well if model predicts overlapping masks
    #segmentations = np.logical_xor.reduce(segmentations, axis = 0)
    #seg_labeled = label(segmentations)

    return seg_labeled ,mitotic_centroids

def configure(cellaap_widget : ui.cellAAPWidget):
    '''
    Configures some tunable parameters for Detectron2
    ------------------------------------------------
    INPUTS:
        cellaap_widget: instance of ui.cellAAPWidget()
    '''
    if cellaap_widget.confluency_est.value():
        cellaap_widget.cfg.TEST.DETECTIONS_PER_IMAGE = cellaap_widget.confluency_est.value()
    if cellaap_widget.thresholder.value():
        cellaap_widget.cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = cellaap_widget.thresholder.value()
    napari.utils.notifications.show_info(
        f"Configurations successfully saved"
    )


def image_select(cellaap_widget: ui.cellAAPWidget):
    """
    Returns the path selected in the image selector box and the array corresponding the to path
    -------------------------------------------------------------------------------------------
    """
    if (
        re.search(
            r"^.+\.(?:(?:[tT][iI][fF][fF]?)|(?:[tT][iI][fF]))$",
            str(cellaap_widget.file_grabber),
        )
        == None
    ):
        layer_data = cv2.imread(
                                str(cellaap_widget.file_grabber), 
                                cv2.IMREAD_GRAYSCALE
                                )
    else:
        layer_data = tiff.imread(
                                str(cellaap_widget.file_grabber)
                                )

    return str(cellaap_widget.file_grabber), layer_data

def display(cellaap_widget: ui.cellAAPWidget):
    """
    Displays file in Napari gui if file has been selected, also returns the 'name' of the image file
    ------------------------------------------------------------------------------------------------
    INPUTS:
        cellaap_widget: instance of ui.cellAAPWidget()
    """
    try:
        name, layer_data = image_select(cellaap_widget)
    except AttributeError:
        napari.utils.notifications.show_error(
            'No Image has been selected'
        )
        return 
    
    name = name.replace(".", "/").split("/")[-2]
    cellaap_widget.viewer.add_image(layer_data, name=name)

    

def grab_file(cellaap_widget):
    '''
    Initiates a QtWidget.QFileDialog instance and grabs a file
    -----------------------------------------------------------
    INPUTS:
        cellaap_widget: instance of ui.cellAAPWidget()
    '''

    file_filter = 'TIFF (*.tiff, *.tif);; Other (*.jpg, *.png)'
    file_grabber = QtWidgets.QFileDialog.getOpenFileName(
        parent = cellaap_widget,
        caption = 'Select a file',
        directory = os.getcwd(),
        filter = file_filter,
    )

    cellaap_widget.file_grabber = file_grabber[0]
    napari.utils.notifications.show_info(
        f"File: {file_grabber[0]} is queued for inference/display"
    )


def grab_directory(cellaap_widget):
    '''
    Initiates a QtWidget.QFileDialog instance and grabs a directory
    -----------------------------------------------------------
    INPUTS:
        cellaap_widget: instance of ui.cellAAPWidget()I
    '''

    dir_grabber = QtWidgets.QFileDialog.getExistingDirectory(
        parent = cellaap_widget,
        caption = 'Select a directory to save inference result'
    )

    cellaap_widget.dir_grabber = dir_grabber
    napari.utils.notifications.show_info(
        f"Directory: {dir_grabber} has been selected"
    )
