"""
Core inference functions shared between GUI and headless inference modes.
-------------------------------------------------------------------------------------------------------
This module contains the core inference logic that is used by both the Napari GUI
and headless batch processing scripts.
"""

import logging
import numpy as np
import torch
import skimage.measure
from skimage.morphology import binary_erosion, disk
import pooch
from typing import Optional

from detectron2.utils.logger import setup_logger
from detectron2.engine import DefaultPredictor
from detectron2.engine.defaults import create_ddp_model
from detectron2.config import get_cfg
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import LazyConfig, instantiate
import detectron2.data.transforms as T
import torch.nn.functional as F
import cell_AAP.annotation.annotation_utils as au  # type:ignore
import cv2

setup_logger()

# Get the logger instance
logger = logging.getLogger(__name__)

# Patch torch.load to handle weights_only parameter compatibility
_original_torch_load = torch.load


def patched_torch_load(*args, **kwargs):
    """
    Patches torch.load to set weights_only=False by default for compatibility
    -------------------------------------------------------------------------
    INPUTS:
        *args: positional arguments passed to torch.load
        **kwargs: keyword arguments passed to torch.load
    OUTPUTS:
        result: return value from torch.load
    """
    if 'weights_only' not in kwargs:
        kwargs['weights_only'] = False
    return _original_torch_load(*args, **kwargs)


def color_masks(
    segmentations: np.ndarray,
    labels,
    method: Optional[str] = "random",
    custom_dict: Optional[dict[int, int]] = None,
    erode: bool = False,
) -> np.ndarray:
    """
    Colors segmentation masks by a chosen method or mapping
    -------------------------------------------------------
    INPUTS:
        segmentations: np.ndarray
            Array of binary masks to color
        labels: np.ndarray
            Labels corresponding to each mask
        method: Optional[str]
            Coloring method: "random", "straight", or "custom"
        custom_dict: Optional[dict[int, int]]
            Custom mapping dictionary for "custom" method
        erode: bool
            Whether to erode masks before coloring (for "random" method)
    OUTPUTS:
        seg_labeled: np.ndarray
            Labeled segmentation image with colored masks
    """

    if method == "custom":
        try:
            assert custom_dict != None
            assert np.isin(labels, list(custom_dict.keys())).all() == True
        except AssertionError:
            logger.warning(
                'Input labels and mapping dictionary did not match when coloring movie, reverting to straight coloring'
            )
            method = "straight"

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

            if method == "custom":
                seg_labeled[mask] += custom_dict[labels[i]]

            elif method == "straight":
                seg_labeled[mask] += labels[i]

            else:
                if erode == True:
                    mask = binary_erosion(mask, disk(3))
                if labels[i] == 0:
                    seg_labeled[mask] = 2 * i
                else:
                    seg_labeled[mask] = 2 * i + 1

    return seg_labeled


def get_model(model_name: str) -> tuple:
    """
    Instantiates a pooch registry for model files and returns model info
    ---------------------------------------------------------------------
    INPUTS:
        model_name: str
            Name of the model to load (e.g., "HeLa", "HeLa_focal")
    RETURNS:
        model: pooch.Pooch
            Pooch instance for downloading model files
        model_type: str
            Type of model config ("yacs" or "lazy")
        weights_name: str
            Name of the weights file
        config_name: str
            Name of the config file
    """

    url_registry = {
        "HeLa": "doi:10.5281/zenodo.15587924",
        "HeLa_focal": "doi:10.5281/zenodo.15587884",
        "HT1080_focal": "doi:10.5281/zenodo.15632609",
        "HT1080": "doi:10.5281/zenodo.15632636",
        "RPE1_focal": "doi:10.5281/zenodo.15632647",
        "RPE1": "doi:10.5281/zenodo.15632661",
        "U2OS_focal": "doi:10.5281/zenodo.15632668",
        "U2OS": "doi:10.5281/zenodo.15632681",
        "general_focal": "doi:10.5281/zenodo.15707118",
        "HeLa_dead": "doi:10.5281/zenodo.17026586",
        "general_dead_focal": "doi:10.5281/zenodo.17178783",
    }

    weights_registry = {
        "HeLa": (
            "model_0040499.pth",
            "md5:62a043db76171f947bfa45c31d7984fe"
        ),
        "HeLa_focal": (
            "model_0053999.pth",
            "md5:40eb9490f3b66894abef739c151c5bfe"
        ),
        "HT1080_focal": (
            "model_0052199.pth",
            "md5:f454095e8891a694905bd2b12a741274"
        ),
        "HT1080": (
            "model_0034799.pth",
            "md5:e5ec71a532d5ad845eb6af37fc785e82"
        ),
        "RPE1_focal": (
            "model_final.pth",
            "md5:f3cc3196470493bba24b05f49773c00e"
        ),
        "RPE1": (
            "model_0048299.pth",
            "md5:5d04462ed4d680b85fd5525d8efc0fc9"
        ),
        "U2OS_focal": (
            "model_final.pth",
            "md5:8fbe8dab57cd96e72537449eb490fa6f"
        ),
        "U2OS": (
            "model_final.pth",
            "md5:8fbe8dab57cd96e72537449eb490fa6f"
        ),
        "general_focal": (
            "model_0061499.pth",
            "md5:62e5f4be12227146f6a9841ada46526a"
        ),
        "HeLa_dead": (
            "model_0080999.pth",
            "md5:9d286376f1b07402023e82f824b2a677"
        ),
        "general_dead_focal": (
            "model_0121499.pth",
            "md5:6e33ab492df6ca1f6b3ae468ea137728"
        ),
    }

    configs_registry = {
        "HeLa": (
            "config.yaml",
            "md5:3e7a6a92045434e4fb7fe25b321749bb",
            "lazy"
        ),
        "HeLa_focal": (
            "config.yaml",
            "md5:320852546ed1390ed2e8fa91008e8bf7",
            "lazy"
        ),
        "HT1080_focal": (
            "config.yaml",
            "md5:cea383632378470aa96dc46adac5d645",
            "lazy"
        ),
        "HT1080": (
            "config.yaml",
            "md5:71674a29e9d5daf3cc23648539c2d0c6",
            "lazy"
        ),
        "RPE1_focal": (
            "config.yaml",
            "md5:78878450ef4805c53b433ff028416510",
            "lazy"
        ),
        "RPE1": (
            "config.yaml",
            "md5:9abb7fcafdb953fff72db7642824202a",
            "lazy"
        ),
        "U2OS_focal": (
            "config.yaml",
            "md5:ab202fd7e0494fce123783bf564a8cde",
            "lazy"
        ),
        "U2OS": (
            "config.yaml",
            "md5:2ab6cd0635b02ad24bcb03371839b807",
            "lazy"
        ),
        "general_focal": (
            "config.yaml",
            "md5:ad609c147ea2cd7d7fde0d734de2e166",
            "lazy"
        ),
        "HeLa_dead": (
            "config.yaml",
            "md5:2bb2594730432a1cc30a6a5fd556df6b",
            "lazy"
        ),
        "general_dead_focal": (
            "config.yaml",
            "md5:eb5281bd93b37e8505846e7b75dba596",
            "lazy"
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


def configure_predictor(
    model_name: str,
    confluency_est: Optional[int] = None,
    conf_threshold: Optional[float] = None,
) -> tuple:
    """
    Configures model parameters and initializes the predictor
    ---------------------------------------------------------
    INPUTS:
        model_name: str
            Name of the model to configure
        confluency_est: Optional[int]
            Maximum detections per image (typically in range 0-2000)
        conf_threshold: Optional[float]
            Confidence threshold for detections (typically in range 0-1)
    RETURNS:
        predictor: object
            Configured predictor instance
        model_type: str
            Type of model config ("yacs" or "lazy")
        cfg: object
            Configuration object
    """

    model, model_type, weights_name, config_name = get_model(model_name)

    if model_type == "yacs":
        cfg = get_cfg()
        cfg.merge_from_file(model.fetch(f"{config_name}"))
        cfg.MODEL.WEIGHTS = model.fetch(f"{weights_name}")

        if torch.cuda.is_available():
            cfg.MODEL.DEVICE = "cuda"
        else:
            cfg.MODEL.DEVICE = "cpu"

        if confluency_est is not None:
            cfg.TEST.DETECTIONS_PER_IMAGE = confluency_est
        if conf_threshold is not None:
            cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = conf_threshold
        predictor = DefaultPredictor(cfg)

    else:
        cfg = LazyConfig.load(model.fetch(f"{config_name}"))
        cfg.train.init_checkpoint = model.fetch(f"{weights_name}")

        if torch.cuda.is_available():
            cfg.train.device = "cuda"
        else:
            cfg.train.device = "cpu"

        if confluency_est is not None:
            cfg.model.proposal_generator.post_nms_topk[1] = confluency_est

        if conf_threshold is not None:
            cfg.model.roi_heads.box_predictor.test_score_thresh = conf_threshold

        predictor = instantiate(cfg.model)
        predictor.to(cfg.train.device)
        predictor = create_ddp_model(predictor)
        torch.load = patched_torch_load
        DetectionCheckpointer(predictor).load(cfg.train.init_checkpoint)
        torch.load = _original_torch_load
        predictor.eval()

    return predictor, model_type, cfg


def run_inference_on_image(
    predictor,
    model_type: str,
    img: np.ndarray,
    frame_num: Optional[int] = None,
    keep_resized_output: bool = False,
) -> tuple[np.ndarray, np.ndarray, list, np.ndarray, np.ndarray, np.ndarray]:
    """
    Runs inference on a single image and produces masks
    ---------------------------------------------------
    INPUTS:
        predictor: object
            Configured Detectron2 predictor instance
        model_type: str
            Type of model ("yacs" or "lazy")
        img: np.ndarray
            Input image to run inference on
        frame_num: Optional[int]
            Frame number for centroid tracking (None for single images)
        keep_resized_output: bool
            If True, returns 1024x1024 output; if False, projects back to original size
    OUTPUTS:
        seg_fordisp: np.ndarray
            Semantic segmentation for display
        seg_fortracking: np.ndarray
            Instance segmentation for tracking
        centroids: list
            List of centroid coordinates
        img: np.ndarray
            Image (original or resized depending on keep_resized_output)
        seg_scores: np.ndarray
            Confidence scores as mask image
        labels: np.ndarray
            Class labels for each detection
    """

    # 1. Capture Original Dimensions and Prepare Image
    print(img.shape, img.dtype)
    orig_h, orig_w = img.shape[:2]
    img_original = img.copy()  # Keep a reference to the original
    img_uint8 = au.to_uint8(img)
    print(img.shape, img.dtype)
    # Ensure 3-channel RGB for the transform logic (handle grayscale inputs)
    
    if img_uint8.ndim == 2:
        img_input = np.stack([img_uint8, img_uint8, img_uint8], axis=-1)
    else:
        img_input = img_uint8

    # 2. Resize Image to Model Requirements (1024x1024)
    # We use Detectron2's transform to ensure consistency with training
    aug = T.Resize((1024, 1024))
    transform = aug.get_transform(img_input)
    img_resized = transform.apply_image(img_input)
    print(img_resized.shape)

    # 3. Run Inference on Resized Image
    if model_type == "yacs":
        output = predictor(img_resized.astype("float32"))
    else:
        # LazyConfig/ViT expect (C, H, W) tensors
        img_perm = np.moveaxis(img_resized, -1, 0)
        with torch.inference_mode():
            output = predictor(
                [{"image": torch.from_numpy(img_perm).type(torch.float32)}]
            )[0]

    # 4. Project Results Back to Original Size (if requested)
    instances = output["instances"].to("cpu")

    # Logic: If keep_resized is FALSE, and the image isn't 1024x1024, we project back
    if not keep_resized_output and (orig_h != 1024 or orig_w != 1024):

        # We use nearest interpolation to maintain binary nature of masks without blurring edges
        # Shape: (N, H, W) -> (N, 1, H, W) for interpolation -> (N, H, W)
        if instances.has("pred_masks") and len(instances.pred_masks) > 0:
            masks = instances.pred_masks.unsqueeze(1).float()
            masks = F.interpolate(masks, size=(orig_h, orig_w), mode="nearest")
            instances.pred_masks = masks.squeeze(1).bool()

        if instances.has("pred_boxes"):
            instances.pred_boxes.tensor = instances.pred_boxes.tensor.clone()
            scale_x = orig_w / 1024.0
            scale_y = orig_h / 1024.0
            instances.pred_boxes.scale(scale_x, scale_y)

        # Update Metadata
        instances._image_size = (orig_h, orig_w)

        # Ensure we return the original image so overlays match in Napari
        img_to_return = img_original
    else:
        # If keeping resized output, return the resized image
        img_to_return = img_resized

    # 5. Extract Results (Standard Flow)
    segmentations = instances.pred_masks
    labels = instances.pred_classes.numpy()
    scores = instances.scores.numpy()
    scores = (scores * 100).astype('uint16')
    classes = instances.pred_classes.numpy()

    custom_dict = {key: key + 99 for key in np.unique(labels)}
    seg_fordisp = color_masks(
        segmentations, labels, method="custom", custom_dict=custom_dict
    )

    scores_mov = color_masks(segmentations, scores, method="straight")
    seg_fortracking = color_masks(segmentations, labels, method="random")

    centroids = []
    for i, _ in enumerate(labels):
        # Ensure mask is numpy
        mask_np = segmentations[i].numpy() if hasattr(segmentations[i], 'numpy') else segmentations[i]
        labeled_mask = skimage.measure.label(mask_np)
        centroid = skimage.measure.centroid(labeled_mask)
        if frame_num is not None:
            centroid = np.array([frame_num, centroid[0], centroid[1]])

        centroids.append(centroid)

    return seg_fordisp, seg_fortracking, centroids, img_to_return, scores_mov, classes

