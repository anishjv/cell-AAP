from __future__ import annotations
import logging
import sys
import napari
import napari.utils.notifications
import cell_AAP.napari.ui as ui  # type:ignore
import cell_AAP.annotation.annotation_utils as au  # type:ignore
import cell_AAP.napari.fileio as fileio  # type: ignore

import numpy as np
import cv2
import tifffile as tiff
import re
import os
import torch
import skimage.measure
from skimage.morphology import binary_erosion, disk
import pooch

sys.path.insert(0, "/Users/whoisv/detectron2_focal/")
from detectron2_loc.utils.logger import setup_logger
from detectron2_loc.engine import DefaultPredictor
from detectron2_loc.engine.defaults import create_ddp_model
from detectron2_loc.config import get_cfg
from detectron2_loc.checkpoint import DetectionCheckpointer
from detectron2_loc.config import LazyConfig, instantiate
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
        lambda: fileio.grab_file(cellaap_widget, wavelength="full_spectrum")
    )

    cellaap_widget.flourescent_image_selector.clicked.connect(
        lambda: fileio.grab_file(cellaap_widget, wavelength="flourescent")
    )

    cellaap_widget.path_selector.clicked.connect(
        lambda: fileio.grab_directory(cellaap_widget)
    )

    cellaap_widget.save_selector.clicked.connect(lambda: fileio.save(cellaap_widget))

    cellaap_widget.set_configs.clicked.connect(lambda: configure(cellaap_widget))

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

    cellaap_widget.file_list_toggle.clicked.connect(
        lambda: fileio.toggle_wavelength(cellaap_widget)
    )

    cellaap_widget.path_selector.clicked.connect(
        lambda: fileio.grab_directory(cellaap_widget)
    )

    cellaap_widget.set_configs.clicked.connect(lambda: configure(cellaap_widget))

    return cellaap_widget


def inference(
    cellaap_widget: ui.cellAAPWidget, img: np.ndarray, frame_num: Optional[int] = None
) -> tuple[np.ndarray, np.ndarray, list, np.ndarray]:
    """
    Runs the actual inference -> Detectron2 -> masks
    ------------------------------------------------
    INPUTS:
        cellaap_widget: instance of ui.cellAAPWidget()
    """

    if cellaap_widget.model_type == "yacs":
        if img.shape != (2048, 2048):
            img = au.square_reshape(img, (2048, 2048))
        output = cellaap_widget.predictor(img)

    else:
        if img.shape != (1024, 1024):
            img = au.square_reshape(img, (1024, 1024))
        img_perm = np.moveaxis(img, -1, 0)

        with torch.inference_mode():
            output = cellaap_widget.predictor([{"image": torch.from_numpy(img_perm).type(torch.float32)}])[
                0
            ]

    segmentations = output["instances"].pred_masks.to("cpu")
    labels = output["instances"].pred_classes.to("cpu")

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

    return seg_fordisp, seg_fortracking, centroids, img


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
    points = ()

    try:
        name, im_array = fileio.image_select(cellaap_widget, wavelength="full_spectrum")
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
        cellaap_widget.range_slider.value()[1] - cellaap_widget.range_slider.value()[0]
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
            img = au.bw_to_rgb(im_array[frame].astype("float32"))
            semantic_seg, instance_seg, centroids, img = inference(
                cellaap_widget, img, frame - cellaap_widget.range_slider.value()[0]
            )
            movie.append(img)
            semantic_movie.append(semantic_seg.astype("uint8"))
            instance_movie.append(instance_seg.astype("uint8"))
            if len(centroids) != 0:
                points += (centroids,)
        cellaap_widget.viewer.add_image(np.asarray(movie)[:, :, :, 0], name=name)

    elif len(im_array.shape) == 2:
        prog_count += 1
        cellaap_widget.progress_bar.setValue(prog_count)
        img = au.bw_to_rgb(im_array.astype("float32"))
        semantic_seg, instance_seg, centroids, img = inference(cellaap_widget, img)
        semantic_movie.append(semantic_seg.astype("uint8"))
        instance_movie.append(instance_seg.astype("uint8"))
        if len(centroids) != 0:
            points += (centroids,)
        cellaap_widget.viewer.add_image(img[:, :, 0], name=name)

    model_name = cellaap_widget.model_selector.currentText()
    cellaap_widget.progress_bar.reset()

    cellaap_widget.viewer.add_labels(
        np.asarray(semantic_movie),
        name=f"{name}_{model_name}_semantic_{cellaap_widget.confluency_est.value()}_{round(cellaap_widget.thresholder.value(), ndigits = 2)}",
        opacity=0.2,
    )

    points_array = np.vstack(points)
    cellaap_widget.viewer.add_points(
        points_array,
        ndim=points_array.shape[1],
        name=f"{name}_{model_name}_centroids_{cellaap_widget.confluency_est.value()}_{round(cellaap_widget.thresholder.value(), ndigits = 2)}",
        size=int(img.shape[1] / 200),
    )

    already_cached = [
        cellaap_widget.save_combo_box.itemText(i)
        for i in range(cellaap_widget.save_combo_box.count())
    ]
    cache_entry_name = f"{name}_{model_name}_{cellaap_widget.confluency_est.value()}_{round(cellaap_widget.thresholder.value(), ndigits = 2)}"

    if cache_entry_name in already_cached:
        only_cache_entry = [
            entry for _, entry in enumerate(already_cached) if entry == cache_entry_name
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
        }
    )


def batch_inference(cellaap_widget: ui.cellAAPWidget):
    """
    Runs inference on group of movies through the batch worker
    ----------------------------------------------------------
    Inputs:
        cellapp_widget: instance of ui.cellAAPWidget()
    """

    # sort files in cellaapwidget.file_list into tuples of (full_spec, flouro)
    full_spec_naming_conv = cellaap_widget.full_spec_format.text()
    flouro_naming_conv = cellaap_widget.flouro_format.text()

    full_spec_file_prefixes = [
        file.split(full_spec_naming_conv)[0]
        for file in cellaap_widget.full_spectrum_files
    ]
    flouro_file_prefixes = [
        file.split(flouro_naming_conv)[0] for file in cellaap_widget.flouro_files
    ]
    flouro_file_suffix = cellaap_widget.flouro_files[0].split(flouro_naming_conv)[1]

    try:
        assert sorted(full_spec_file_prefixes) == sorted(flouro_file_prefixes)
    except AssertionError:
        raise Exception(
            "List of full_spectrum movies does not correspond with list of flourescent movies"
        )

    cellaap_widget.flouro_files = [
        prefix + "Cy5" + flouro_file_suffix for prefix in full_spec_file_prefixes
    ]
    num_movie_pairs = len(cellaap_widget.full_spectrum_files)

    movie_tally = 0
    while movie_tally < num_movie_pairs:
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
        "ResNet-1.8": "doi:10.5281/zenodo.11387359",
        "ViTb-1.8": "doi:10.5281/zenodo.11951629",
        "ViTbFocal-1.8": "doi:10.5281/zenodo.12585190",
        "ViTb-1.9": "doi:10.5281/zenodo.12627315",
        "ViTlFocal-1.9": "doi:10.5281/zenodo.12700955",
    }

    weights_registry = {
        "ResNet-1.8": ("model_0004999.pth", "md5:8cccf01917e4f04e4cfeda0878bc1f8a"),
        "ViTb-1.8": ("model_0008399.pth", "md5:9dd789fab740d976c27f9d179128629d"),
        "ViTbFocal-1.8": (
            "model_0014699.pth",
            "md5:36fd39cf3b053d9e540403fb0e9ca2c7",
        ),
        "ViTb-1.9": ("model_0019574.pth", "md5:0417118a914ad279c827faf6e6d8ddcb"),
        "ViTlFocal-1.9": (
            "model_0043499.pth",
            "md5:ad056dc159ea8fd12f7d5d4562c368a9",
        ),
    }

    configs_registry = {
        "ResNet-1.8": ("config.yaml", "md5:cf1532e9bc0ed07285554b1e28f942de", "yacs"),
        "ViTb-1.8": (
            "config.yml",
            "md5:0d2c6dd677ff7bcda80e0e297c1b6766",
            "lazy",
        ),
        "ViTbFocal-1.8": (
            "vitb_bin_focal.yaml",
            "md5:bcd2efa5ffa1d8fa3ba12a1cb2d187fd",
            "lazy",
        ),
        "ViTb-1.9": (
            "vitb_1.9.yaml",
            "md5:aab66adc5a1b3f126c96c01fac5e8b4d",
            "lazy",
        ),
        "ViTlFocal-1.9": (
            "vitl_focal_1.9.yaml",
            "md5:809bc09220a8dea515c58e2ddb3cfe77",
            "lazy",
        ),
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
                mask = binary_erosion(mask, disk(3))
                if labels[i] == 0:
                    seg_labeled[mask] = 2 * i
                else:
                    seg_labeled[mask] = 2 * i - 1

    return seg_labeled
