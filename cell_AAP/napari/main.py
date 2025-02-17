from __future__ import annotations
import logging
import sys
import napari
import napari.utils.notifications
import cell_AAP.napari.ui as ui  # type:ignore
import cell_AAP.annotation.annotation_utils as au  # type:ignore
import cell_AAP.napari.fileio as fileio  # type: ignore
import cell_AAP.napari.analysis as analysis  # type: ignore

import numpy as np
import cv2
import tifffile as tiff
import re
import os
import torch
import skimage.measure
from skimage.filters import gaussian
from skimage.morphology import binary_erosion, disk
import pooch

from detectron2.utils.logger import setup_logger
from detectron2.engine import DefaultPredictor
from detectron2.engine.defaults import create_ddp_model
from detectron2.config import get_cfg
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import LazyConfig, instantiate
from qtpy import QtWidgets, _warn_old_minor_version
import timm
from typing import Optional
import itertools

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


def create_cellAAP_widget(batch: Optional[bool] = False) -> ui.cellAAPWidget:
    "Creates instance of ui.cellAAPWidget and sets callbacks"

    cellaap_widget = ui.cellAAPWidget(
        napari_viewer=napari.current_viewer(), cfg=None, batch=batch
    )

    cellaap_widget.inference_button.clicked.connect(
        lambda: run_inference(cellaap_widget)
    )

    cellaap_widget.display_button.clicked.connect(
        lambda: fileio.display(cellaap_widget)
    )

    cellaap_widget.image_selector.clicked.connect(
        lambda: fileio.grab_file(cellaap_widget)
    )

    cellaap_widget.path_selector.clicked.connect(
        lambda: fileio.grab_directory(cellaap_widget)
    )

    cellaap_widget.save_selector.clicked.connect(lambda: fileio.save(cellaap_widget))

    cellaap_widget.set_configs.clicked.connect(lambda: configure(cellaap_widget))

    cellaap_widget.results_display.clicked.connect(lambda: disp_inf_results(cellaap_widget))

    return cellaap_widget


def create_batch_widget(batch: Optional[bool] = True) -> ui.cellAAPWidget:
    "Creates instance of ui.cellAAPWidget and sets callbacks"

    cellaap_widget = ui.cellAAPWidget(
        napari_viewer=napari.current_viewer(), cfg=None, batch=batch
    )

    cellaap_widget.inference_button.clicked.connect(
        lambda: batch_inference(cellaap_widget)
    )

    cellaap_widget.set_configs.clicked.connect(lambda: configure(cellaap_widget))

    cellaap_widget.add_button.clicked.connect(lambda: fileio.add(cellaap_widget))

    cellaap_widget.remove_button.clicked.connect(lambda: fileio.remove(cellaap_widget))

    cellaap_widget.path_selector.clicked.connect(
        lambda: fileio.grab_directory(cellaap_widget)
    )

    cellaap_widget.results_display.clicked.connect(lambda: disp_inf_results(cellaap_widget))

    return cellaap_widget


def inference(
    cellaap_widget: ui.cellAAPWidget, img: np.ndarray, frame_num: Optional[int] = None
) -> tuple[np.ndarray, np.ndarray, list, np.ndarray, np.ndarray, np.ndarray]:
    """
    Runs the actual inference -> Detectron2 -> masks
    ------------------------------------------------
    INPUTS:
        cellaap_widget: instance of ui.cellAAPWidget()
    """

    if cellaap_widget.model_type == "yacs":
        if img.shape != (2048, 2048):
            img = au.square_reshape(img, (2048, 2048))
        output = cellaap_widget.predictor(img.astype("float32"))

    else:
        if img.shape != (1024, 1024):
            img = au.square_reshape(img, (1024, 1024))
        img_perm = np.moveaxis(img, -1, 0)

        with torch.inference_mode():
            output = cellaap_widget.predictor(
                [{"image": torch.from_numpy(img_perm).type(torch.float32)}]
            )[0]

    segmentations = output["instances"].pred_masks.to("cpu")
    labels = output["instances"].pred_classes.to("cpu")
    scores = output["instances"].scores.to("cpu").numpy()
    classes = output['instances'].pred_classes.to("cpu").numpy()

    seg_fordisp = color_masks(
        segmentations, labels, method="custom", custom_dict={0: 1, 1: 100}
    )

    seg_fortracking = color_masks(segmentations, labels, method="random")

    centroids = []
    for i, _ in enumerate(labels):
        labeled_mask = skimage.measure.label(segmentations[i])
        centroid = skimage.measure.centroid(labeled_mask)
        if frame_num != None:
            centroid = np.array([frame_num, centroid[0], centroid[1]])

        centroids.append(centroid)

    return seg_fordisp, seg_fortracking, centroids, img, scores, classes


def run_inference(cellaap_widget: ui.cellAAPWidget):
    """
    Runs inference on image returned by self.image_select(), saves inference result if save selector has been checked
    ----------------------------------------------------------------------------------------------------------------
    Inputs:
        cellapp_widget: instance of ui.cellAAPWidget()
    """
    prog_count = 0
    instance_movie = []
    semantic_movie = []
    scores_list = []
    classes_list =[]
    points = ()

    try:
        name, im_array = fileio.image_select(cellaap_widget)
        name = name.replace(".", "/").split("/")[-2]
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

    cellaap_widget.progress_bar.setMaximum(
        cellaap_widget.range_slider.value()[1]
        - cellaap_widget.range_slider.value()[0]
        + 1
    )
    if len(im_array.shape) == 3:
        movie = []
        for frame in range(
            cellaap_widget.range_slider.value()[1]
            - cellaap_widget.range_slider.value()[0]
            + 1
        ):
            prog_count += 1
            frame += cellaap_widget.range_slider.value()[0]
            cellaap_widget.progress_bar.setValue(prog_count)
            img = au.bw_to_rgb(im_array[frame])
            semantic_seg, instance_seg, centroids, img, scores, classes= inference(
                cellaap_widget, img, frame - cellaap_widget.range_slider.value()[0]
            )
            movie.append(img)
            semantic_movie.append(semantic_seg.astype("uint16"))
            instance_movie.append(instance_seg.astype("uint16"))
            scores_list.append(scores)
            classes_list.append(classes)
            if len(centroids) != 0:
                points += (centroids,)

    elif len(im_array.shape) == 2:
        prog_count += 1
        cellaap_widget.progress_bar.setValue(prog_count)
        img = au.bw_to_rgb(im_array)
        semantic_seg, instance_seg, centroids, img, scores, classes= inference(cellaap_widget, img)
        semantic_movie.append(semantic_seg.astype("uint16"))
        instance_movie.append(instance_seg.astype("uint16"))
        scores_list.append(scores)
        classes_list.append(classes)
        if len(centroids) != 0:
            points += (centroids,)

    model_name = cellaap_widget.model_selector.currentText()
    cellaap_widget.progress_bar.reset()

    semantic_movie = np.asarray(semantic_movie)
    instance_movie = np.asarray(instance_movie)
    points_array = np.vstack(points)
    scores_array = np.concatenate(scores_list, axis =0)
    classes_array = np.concatenate(classes_list, axis =0)

    cache_entry_name = f"{name}_{model_name}_{cellaap_widget.confluency_est.value()}_{round(cellaap_widget.thresholder.value(), ndigits = 2)}"
    if cellaap_widget.batch == False:

        try:
            cellaap_widget.viewer.add_image(np.asarray(movie)[:, :, :, 0], name=name)
        except UnboundLocalError:
            try:
                cellaap_widget.viewer.add_image(img[:, :, 0], name=name)
            except:
                pass


    already_cached = [
        cellaap_widget.save_combo_box.itemText(i)
        for i in range(cellaap_widget.save_combo_box.count())
    ]

    if cache_entry_name in already_cached:
        only_cache_entry = [
            entry
            for _, entry in enumerate(already_cached)
            if entry == cache_entry_name
        ]
        cache_entry_name += f"_{len(only_cache_entry)}"

    cellaap_widget.save_combo_box.insertItem(0, cache_entry_name)
    cellaap_widget.save_combo_box.setCurrentIndex(0)

    cellaap_widget.inference_cache.append(
        {
            "name": cache_entry_name,
            "semantic_movie": semantic_movie,
            "instance_movie": instance_movie,
            "centroids": points_array,
            "scores": scores_array,
            "classes": classes_array
        }
    )


def batch_inference(cellaap_widget: ui.cellAAPWidget):
    """
    Runs inference on group of movies through the batch worker
    ----------------------------------------------------------
    Inputs:
        cellapp_widget: instance of ui.cellAAPWidget()
    """

    num_movies = len(cellaap_widget.full_spectrum_files)
    movie_tally = 0
    while movie_tally < num_movies:
        run_inference(cellaap_widget)
        fileio.save(cellaap_widget)
        movie_tally += 1


def configure(cellaap_widget: ui.cellAAPWidget):
    """
    Configures some tunable parameters for Detectron2
    ------------------------------------------------
    INPUTS:
        cellaap_widget: instance of ui.cellAAPWidget()
    """

    model, model_type, weights_name, config_name = get_model(cellaap_widget)
    if model_type == "yacs":
        cellaap_widget.model_type = "yacs"
        cellaap_widget.cfg = get_cfg()
        cellaap_widget.cfg.merge_from_file(model.fetch(f"{config_name}"))
        cellaap_widget.cfg.MODEL.WEIGHTS = model.fetch(f"{weights_name}")

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
        cellaap_widget.model_type = "lazy"
        cellaap_widget.cfg = LazyConfig.load(model.fetch(f"{config_name}"))
        cellaap_widget.cfg.train.init_checkpoint = model.fetch(f"{weights_name}")

        if torch.cuda.is_available():
            cellaap_widget.cfg.train.device = "cuda"
        else:
            cellaap_widget.cfg.train.device = "cpu"

        if cellaap_widget.confluency_est.value():
            cellaap_widget.cfg.model.proposal_generator.post_nms_topk[1] = (
                cellaap_widget.confluency_est.value()
            )

        if cellaap_widget.thresholder.value():
            cellaap_widget.cfg.model.roi_heads.box_predictor.test_score_thresh = (
                cellaap_widget.thresholder.value()
            )

        predictor = instantiate(cellaap_widget.cfg.model)
        predictor.to(cellaap_widget.cfg.train.device)
        predictor = create_ddp_model(predictor)
        DetectionCheckpointer(predictor).load(cellaap_widget.cfg.train.init_checkpoint)
        predictor.eval()

    napari.utils.notifications.show_info(f"Configurations successfully saved")
    cellaap_widget.configured = True
    cellaap_widget.predictor = predictor


def get_model(cellaap_widget):
    """
    Instaniates POOCH instance containing model files from the model_registry
    --------------------------------------------------------------------------
    INPUTS:
        cellaap_widget: instance of ui.cellAAPWidget()I
    """

    model_name = cellaap_widget.model_selector.currentText()

    url_registry = {
        "HeLa": "doi:10.5281/zenodo.14226948",
        "U2OS": "doi:10.5281/zenodo.14226985",
        "HeLa_oof": "doi:10.5281/zenodo.14884515",
        "U2OS_new": "doi:10.5281/zenodo.14879267"
    }

    weights_registry = {
        "HeLa": (
            "model_0043499.pth",
            "md5:ad056dc159ea8fd12f7d5d4562c368a9",
        ),
        "U2OS": (
            "model_0030449.pth",
            "md5:4d65600b92560d7fcda6c6fd59fa0fe8"
        ),
        "HeLa_oof": (
            "model_0004349.pth",
            "md5:01ac361196356fe1d7a1fb2c125ad2f5"
        ),
        "U2OS_new": (
            "model_0034799.pth",
            "md5:9a1d6fdf47e3cfbc909a5c9a275b3838"
        )

    }

    configs_registry = {
        "HeLa": (
            "config.yaml",
            "md5:319ec68250d7ae499a274f7c4f151513",
            "lazy",
        ),
        "U2OS" : (
            "config.yaml",
            "md5:ad80d579860c53a84ab076c4db2604fd",
            "lazy"
        ),
        "HeLa_oof": (
            "config.yaml",
            "md5:eb68946e8f8a1676b3a7bfc8c235d1db",
            "lazy"
        ),
        "U2OS_new": (
            "config.yaml",
            "md5:b20065699f593e406fa30d5fa3bc0085",
            "lazy"
        )
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
    weights_name = weights_registry[f"{model_name}"][0]
    config_name = configs_registry[f"{model_name}"][0]

    return model, model_type, weights_name, config_name


def color_masks(
    segmentations: np.ndarray,
    labels,
    method: Optional[str] = "random",
    custom_dict: Optional[dict[int, int]] = None,
    erode = False
) -> np.ndarray:
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

    if segmentations.size(dim=0) == 0:
        seg_labeled = np.zeros(
            (segmentations.size(dim=1), segmentations.size(dim=2)), dtype="uint8"
        )
        return seg_labeled

    seg_labeled = np.zeros_like(segmentations[0], int)
    for i, mask in enumerate(segmentations):
        loc_mask = seg_labeled[mask]
        mask_nonzero = list(filter(lambda x: x != 0, loc_mask))
        if len(mask_nonzero) < (loc_mask.shape[0] / 4):  # Roughly IOU < 0.5
            if method == "custom" and custom_dict != None:
                for j in custom_dict.keys():
                    if labels[i] == j:
                        seg_labeled[mask] += custom_dict[j]

            else:
                if erode == True:
                    mask = binary_erosion(mask, disk(3))
                if labels[i] == 0:
                    seg_labeled[mask] = 2 * i
                else:
                    seg_labeled[mask] = 2 * i + 1

    return seg_labeled


def disp_inf_results(cellaap_widget) -> None:

    "Displays inference/analysis results when called"

    result_name = cellaap_widget.save_combo_box.currentText()
    result = list(
        filter(
            lambda x: x["name"] in f"{result_name}",
            cellaap_widget.inference_cache,
        )
    )[0]

    cellaap_widget.viewer.add_labels(
        result['semantic_movie'],
        name=f"semantic_{result_name}",
        opacity=0.2,
    )


    cellaap_widget.viewer.add_points(
        result['centroids'],
        ndim=result['centroids'].shape[1],
        name=f"centroids_{result_name}",
        size=int(result['semantic_movie'].shape[1] / 200),
    )

    try:
        data = result['data']
        properties = result['properties']
        graph = result['graph']
        cellaap_widget.viewer.add_tracks(data, properties=properties, graph=graph, name = f"tracks_{result_name}")
    except KeyError:
        napari.utils.notifications.show_info("Tracks layer will not be shown, user has likely not analyzed inference results")
