from __future__ import annotations
import logging

import napari
import cell_AAP.napari.ui as ui  # type:ignore
import cell_AAP.annotation.annotation_utils as au  # type:ignore

import numpy as np
import cv2
import tifffile as tiff
import re
import os
import torch
import skimage.measure
import pooch

from detectron2.utils.logger import setup_logger
from detectron2.engine import DefaultPredictor
from detectron2.engine.defaults import create_ddp_model
from detectron2.config import get_cfg
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import LazyConfig, instantiate
from qtpy import QtWidgets
import timm
from typing import Optional

setup_logger()

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
        napari_viewer=napari.current_viewer(), cfg=None
    )

    cellaap_widget.inference_button.clicked.connect(
        lambda: run_inference(cellaap_widget)
    )

    cellaap_widget.display_button.clicked.connect(lambda: display(cellaap_widget))

    cellaap_widget.image_selector.clicked.connect(lambda: grab_file(cellaap_widget))

    cellaap_widget.path_selector.clicked.connect(lambda: grab_directory(cellaap_widget))

    cellaap_widget.save_selector.clicked.connect(lambda: save(cellaap_widget))

    cellaap_widget.set_configs.clicked.connect(lambda: configure(cellaap_widget))


    return cellaap_widget


def run_inference(cellaap_widget: ui.cellAAPWidget):
    """
    Runs inference on image returned by self._image_select(), saves inference result if save selector has been checked
    ----------------------------------------------------------------------------------------------------------------
    Inputs:
        cellapp_widget: instance of ui.cellAAPWidget()
    """
    prog_count = 0
    mask_array = []
    points = ()
    try:
        name, im_array = image_select(cellaap_widget)
    except AttributeError:
        napari.utils.notifications.show_error("No Image has been selected")
        return

    try:
        assert cellaap_widget.configured == True
    except AssertionError:
        napari.utils.notifications.show_error(
            "You must configure the model before running inference"
        )
        return

    if len(im_array.shape) == 3:
        for frame in range(im_array.shape[0]):
            prog_count += 1
            cellaap_widget.progress_bar.setMaximum(im_array.shape[0])
            cellaap_widget.progress_bar.setValue(prog_count)
            img = au.bw_to_rgb(im_array[frame].astype("float32"))
            segmentations, mitotic_centroids = inference(cellaap_widget, img, frame)
            mask_array.append(segmentations.astype("uint8"))
            if len(mitotic_centroids) != 0:
                points += (mitotic_centroids,)

    elif len(im_array.shape) == 2:
        prog_count += 1
        cellaap_widget.progress_bar.setValue(prog_count)
        img = au.bw_to_rgb(im_array.astype("float32"))
        segmentations, mitotic_centroids = inference(cellaap_widget, img)
        mask_array.append(segmentations.astype("uint8"))
        if len(mitotic_centroids) != 0:
            points += (mitotic_centroids,)

    cellaap_widget.name = name.replace(".", "/").split("/")[-2]
    cellaap_widget.mask_array = np.array(mask_array)
    model_name = cellaap_widget.model_selector.currentText()

    cellaap_widget.progress_bar.reset()
    cellaap_widget.viewer.add_image(
        im_array, name = cellaap_widget.name
    )
    cellaap_widget.viewer.add_labels(
        np.array(mask_array), name=f"{cellaap_widget.name}_{model_name}_infmask", opacity=0.2
    )
    if points != ():
        points_array = np.vstack(points)
        cellaap_widget.viewer.add_points(
            points_array,
            ndim=points_array.shape[1],
            name= f"{cellaap_widget.name}_{model_name}_infcents",
            size=10,
        )
    cellaap_widget.mask_array = np.array(mask_array)
    


def inference(
    cellaap_widget: ui.cellAAPWidget, img: np.ndarray, frame_num: int = None
) -> tuple[np.ndarray, list[np.ndarray[tuple]]]:
    """
    Runs the actual inference -> Detectron2 -> masks
    ------------------------------------------------
    INPUTS:
        cellaap_widget: instance of ui.cellAAPWidget()
    """
    
    if cellaap_widget.model_type == 'yacs':
        if img.shape[0] < 2048:
            qdiff = (2048 - img.shape[0]) // 4
            img = np.pad(img, [(qdiff, qdiff), (qdiff, qdiff), (0,0)], mode = 'constant', constant_values=img.mean())
        output = cellaap_widget.predictor(img)
    
    else:
        if img.shape[0] < 1024:
            qdiff = (1024 - img.shape[0]) // 4
            img = np.pad(img, [(qdiff, qdiff), (qdiff, qdiff), (0,0)], mode = 'constant', constant_values=img.mean())
        elif img.shape[0] > 2048:
            img = au.binImage(img, (1024, 1024))

        img_perm = np.moveaxis(img, -1, 0)
        with torch.inference_mode():
            output = cellaap_widget.predictor([{'image': torch.from_numpy(img_perm)}])[0]

    segmentations = output["instances"].pred_masks.to("cpu")
    labels = output["instances"].pred_classes.to("cpu")
    seg_labeled = color_masks(segmentations, labels, method = 'custom', custom_dict = {0:1, 1:100})

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
    # segmentations = np.logical_xor.reduce(segmentations, axis = 0)
    # seg_labeled = label(segmentations)

    return seg_labeled, mitotic_centroids


def configure(cellaap_widget: ui.cellAAPWidget):
    """
    Configures some tunable parameters for Detectron2
    ------------------------------------------------
    INPUTS:
        cellaap_widget: instance of ui.cellAAPWidget()
    """

    model, model_type = get_model(cellaap_widget)
    if model_type == 'yacs':
        cellaap_widget.model_type = 'yacs'
        cellaap_widget.cfg = get_cfg()
        cellaap_widget.cfg.merge_from_file(model.fetch("config.yaml"))
        cellaap_widget.cfg.MODEL.WEIGHTS = model.fetch("model_0004999.pth")
    
        if torch.cuda.is_available():
            cellaap_widget.cfg.MODEL.DEVICE = "cuda"
        else:
            cellaap_widget.cfg.MODEL.DEVICE = "cpu"


        if cellaap_widget.confluency_est.value():
            cellaap_widget.cfg.TEST.DETECTIONS_PER_IMAGE = (
                cellaap_widget.confluency_est.value()
            )
        if cellaap_widget.thresholder.value():
            cellaap_widget.cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = (
                cellaap_widget.thresholder.value()
            )
        predictor = DefaultPredictor(cellaap_widget.cfg)

    else:
        cellaap_widget.model_type = 'lazy'
        cellaap_widget.cfg = LazyConfig.load(model.fetch("config.yml"))
        cellaap_widget.cfg.train.init_checkpoint = model.fetch("model_0008399.pth")

        if torch.cuda.is_available():
            cellaap_widget.cfg.train.device = 'cuda'
        else:
            cellaap_widget.cfg.train.device = 'cpu'

        if cellaap_widget.confluency_est.value():
            cellaap_widget.cfg.model.proposal_generator.post_nms_topk[1] = cellaap_widget.confluency_est.value()
        
        if cellaap_widget.thresholder.value():
            cellaap_widget.cfg.model.roi_heads.box_predictor.test_score_thresh = cellaap_widget.thresholder.value()
            
        predictor = instantiate(cellaap_widget.cfg.model)
        predictor.to(cellaap_widget.cfg.train.device)
        predictor = create_ddp_model(predictor)
        DetectionCheckpointer(predictor).load(cellaap_widget.cfg.train.init_checkpoint)
        predictor.eval()

    napari.utils.notifications.show_info(f"Configurations successfully saved")
    cellaap_widget.configured = True
    cellaap_widget.predictor = predictor
    
    


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
        layer_data = cv2.imread(str(cellaap_widget.file_grabber), cv2.IMREAD_GRAYSCALE)
    else:
        layer_data = tiff.imread(str(cellaap_widget.file_grabber))

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
        napari.utils.notifications.show_error("No Image has been selected")
        return

    name = name.replace(".", "/").split("/")[-2]
    cellaap_widget.viewer.add_image(layer_data, name=name)


def grab_file(cellaap_widget):
    """
    Initiates a QtWidget.QFileDialog instance and grabs a file
    -----------------------------------------------------------
    INPUTS:
        cellaap_widget: instance of ui.cellAAPWidget()
    """

    file_filter = "TIFF (*.tiff, *.tif);; JPEG (*.jpg);; PNG (*.png)"
    file_grabber = QtWidgets.QFileDialog.getOpenFileName(
        parent=cellaap_widget,
        caption="Select a file",
        directory=os.getcwd(),
        filter=file_filter,
    )

    cellaap_widget.file_grabber = file_grabber[0]
    napari.utils.notifications.show_info(
        f"File: {file_grabber[0]} is queued for inference/display"
    )


def grab_directory(cellaap_widget):
    """
    Initiates a QtWidget.QFileDialog instance and grabs a directory
    -----------------------------------------------------------
    INPUTS:
        cellaap_widget: instance of ui.cellAAPWidget()I
    """

    dir_grabber = QtWidgets.QFileDialog.getExistingDirectory(
        parent=cellaap_widget, caption="Select a directory to save inference result"
    )

    cellaap_widget.dir_grabber = dir_grabber
    napari.utils.notifications.show_info(f"Directory: {dir_grabber} has been selected")


def get_model(cellaap_widget):
    """
    Instaniates POOCH instance containing model files from the model_registry
    --------------------------------------------------------------------------
    INPUTS:
        cellaap_widget: instance of ui.cellAAPWidget()I
    """

    model_name = cellaap_widget.model_selector.currentText()

    url_registry = {
        "HeLa": "doi:10.5281/zenodo.11387359",
        "HeLaViT": "doi:10.5281/zenodo.11951629",
        "HeLaViT(focal)": "doi:10.5281/zenodo.12206896"
                    }

    weights_registry = {
        "HeLa": ("model_0004999.pth", "md5:8cccf01917e4f04e4cfeda0878bc1f8a"),
        "HeLaViT": ("model_0008399.pth", "md5:9dd789fab740d976c27f9d179128629d"),
        "HeLaViT(focal)": ("model_0014699.pth", "md5:36fd39cf3b053d9e540403fb0e9ca2c7")
    }

    configs_registry = {
        "HeLa": ("config.yaml", "md5:cf1532e9bc0ed07285554b1e28f942de", 'yacs'),
        "HeLaViT": ("config.yml", "md5:0d2c6dd677ff7bcda80e0e297c1b6766" , 'lazy'),
        "HeLaViT(focal)": ("vitb_bin_focal.yaml", "md5:a0d3a54ef2c67d1a09dc5dde3d603b1c", 'lazy')
        }

    model = pooch.create(
        path=pooch.os_cache("cell_aap"),
        base_url=url_registry[f"{model_name}"],
        registry={
            weights_registry[f"{model_name}"][0]: weights_registry[f"{model_name}"][1],
            configs_registry[f"{model_name}"][0]: configs_registry[f"{model_name}"][1],
        },
    )

    model_type = configs_registry[f"{model_name}"][2]

    return model, model_type


def save(cellaap_widget):
    """
    Saves a given napari layer
    """

    try:
        filepath = cellaap_widget.dir_grabber
    except AttributeError:
        napari.utils.notifications.show_error(
            "No Directory has been selected - will save output to current working directory"
        )
        filepath = os.getcwd()
        pass

    tiff.imwrite(
        os.path.join(filepath, f"{cellaap_widget.name}_inf.tif"),
        np.array(cellaap_widget.mask_array).astype("uint16"),
    )


def color_masks(segmentations: np.ndarray, labels, method:Optional[str] = 'random', custom_dict:Optional[dict[int:int]] = None) -> np.ndarray:
    """
    Takes an array of segmentation masks and colors them by some pre-defined metric. If metric is not given masks are colored randomely
    -------------------------------------------------------------------------------------------------------------------------------------
    INPUTS:
        segmentations: np.ndarray
        labels: list
        method: str
        custom_dict: dict
    OUTPUTS:
        seg_labeled: np.ndarray
    """

    seg_labeled = np.zeros_like(segmentations[0], int)
    if method == 'custom':
        for i, mask in enumerate(segmentations):
            for j in custom_dict.keys():
                if labels[i] == j:
                    seg_labeled[mask] = custom_dict[j]
    
    if method == 'random':
        for i, mask in enumerate(segmentations):
            seg_labeled[mask] = i

    return seg_labeled

