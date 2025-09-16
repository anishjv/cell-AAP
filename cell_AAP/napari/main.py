from __future__ import annotations
import logging
import napari
import napari.utils.notifications
import cell_AAP.napari.ui as ui  # type:ignore
import cell_AAP.annotation.annotation_utils as au  # type:ignore
import cell_AAP.napari.fileio as fileio  # type: ignore

import numpy as np
import torch
import skimage.measure
from skimage.morphology import binary_erosion, disk
import pooch

from detectron2.utils.logger import setup_logger
from detectron2.engine import DefaultPredictor
from detectron2.engine.defaults import create_ddp_model
from detectron2.config import get_cfg
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import LazyConfig, instantiate
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


_original_torch_load = torch.load
def patched_torch_load(*args, **kwargs):
    if 'weights_only' not in kwargs:
        kwargs['weights_only'] = False
    return _original_torch_load(*args, **kwargs)


def create_cellAAP_widget(batch: Optional[bool] = False) -> ui.cellAAPWidget:
    """
    Creates instance of `ui.cellAAPWidget` and sets callbacks
    ---------------------------------------------------------
    INPUTS:
        batch: Optional[bool]
            If True, configures widget for batch mode; otherwise single-image mode
    RETURNS:
        ui.cellAAPWidget
    """

    cellaap_widget = ui.cellAAPWidget(
        napari_viewer=napari.current_viewer(), cfg=None, batch=batch
    )

    # Connect callbacks with state management
    cellaap_widget.inference_button.clicked.connect(
        lambda: run_inference(cellaap_widget)
    )

    cellaap_widget.display_button.clicked.connect(
        lambda: fileio.display(cellaap_widget)
    )

    cellaap_widget.image_selector.clicked.connect(
        lambda: select_image_and_update_state(cellaap_widget)
    )

    cellaap_widget.save_selector.clicked.connect(lambda: save_and_update_state(cellaap_widget))

    cellaap_widget.set_configs.clicked.connect(lambda: configure_and_update_state(cellaap_widget))

    cellaap_widget.results_display.clicked.connect(lambda: disp_inf_results(cellaap_widget))

    # Initialize button states
    update_button_states(cellaap_widget)

    return cellaap_widget


def create_batch_widget(batch: Optional[bool] = True) -> ui.cellAAPWidget:
    """
    Creates instance of `ui.cellAAPWidget` in batch mode and sets callbacks
    ----------------------------------------------------------------------
    INPUTS:
        batch: Optional[bool]
            Batch mode flag (defaults True for clarity)
    RETURNS:
        ui.cellAAPWidget
    """

    cellaap_widget = ui.cellAAPWidget(
        napari_viewer=napari.current_viewer(), cfg=None, batch=batch
    )

    cellaap_widget.inference_button.clicked.connect(
        lambda: batch_inference(cellaap_widget)
    )

    cellaap_widget.set_configs.clicked.connect(lambda: configure_and_update_state(cellaap_widget))

    cellaap_widget.add_button.clicked.connect(lambda: fileio.add(cellaap_widget))

    cellaap_widget.remove_button.clicked.connect(lambda: fileio.remove(cellaap_widget))

    cellaap_widget.results_display.clicked.connect(lambda: disp_inf_results(cellaap_widget))

    # Initialize button states
    update_button_states(cellaap_widget)

    return cellaap_widget


def select_image_and_update_state(cellaap_widget: ui.cellAAPWidget):
    """
    Selects image and updates button states accordingly
    ---------------------------------------------------
    INPUTS:
        cellaap_widget: `ui.cellAAPWidget`
    RETURNS:
        None
    """
    try:
        fileio.grab_file(cellaap_widget)
        # Enable display button after image selection
        cellaap_widget.display_button.setEnabled(True)
        update_button_states(cellaap_widget)
    except Exception as e:
        napari.utils.notifications.show_error(f"Error selecting image: {str(e)}")


def configure_and_update_state(cellaap_widget: ui.cellAAPWidget):
    """
    Configures model and updates button states accordingly
    ------------------------------------------------------
    INPUTS:
        cellaap_widget: `ui.cellAAPWidget`
    RETURNS:
        None
    """
    try:
        configure(cellaap_widget)
        update_button_states(cellaap_widget)
    except Exception as e:
        napari.utils.notifications.show_error(f"Error configuring model: {str(e)}")


def save_and_update_state(cellaap_widget: ui.cellAAPWidget):
    """
    Saves results and updates button states accordingly
    ---------------------------------------------------
    INPUTS:
        cellaap_widget: `ui.cellAAPWidget`
    RETURNS:
        None
    """
    try:
        # First, prompt user to select save directory
        dir_grabber = fileio.grab_directory(cellaap_widget)
        if dir_grabber:
            # Then save the results
            fileio.save(cellaap_widget)
            # Enable results display after saving
            cellaap_widget.results_display.setEnabled(True)
            update_button_states(cellaap_widget)
    except Exception as e:
        napari.utils.notifications.show_error(f"Error saving results: {str(e)}")


def update_button_states(cellaap_widget: ui.cellAAPWidget):
    """
    Updates button states based on current workflow progress
    --------------------------------------------------------
    INPUTS:
        cellaap_widget: `ui.cellAAPWidget`
    RETURNS:
        None
    """
    
    # Check if image is selected
    has_image = hasattr(cellaap_widget, 'image_path') and cellaap_widget.image_path is not None
    
    # Check if model is configured
    is_configured = cellaap_widget.configured
    
    # Check if inference has been run
    has_inference_results = len(cellaap_widget.inference_cache) > 0
    
    # Update button states based on workflow progress
    if not cellaap_widget.batch:
        # Single image mode
        cellaap_widget.display_button.setEnabled(has_image)
        cellaap_widget.inference_button.setEnabled(has_image and is_configured)
        cellaap_widget.save_selector.setEnabled(has_inference_results)
        cellaap_widget.results_display.setEnabled(has_inference_results)
    else:
        # Batch mode
        has_files = len(cellaap_widget.full_spectrum_files) > 0
        cellaap_widget.inference_button.setEnabled(has_files and is_configured)
        cellaap_widget.save_selector.setEnabled(has_inference_results)
        cellaap_widget.results_display.setEnabled(has_inference_results)


def inference(
    cellaap_widget: ui.cellAAPWidget, img: np.ndarray, frame_num: Optional[int] = None
) -> tuple[np.ndarray, np.ndarray, list, np.ndarray, np.ndarray, np.ndarray]:
    """
    Runs the actual inference (Detectron2) and produces masks
    ---------------------------------------------------------
    INPUTS:
        cellaap_widget: `ui.cellAAPWidget`
        img: np.ndarray
        frame_num: Optional[int]
    OUTPUTS:
        seg_fordisp: np.ndarray
        seg_fortracking: np.ndarray
        centroids: list
        img: np.ndarray
        seg_scores: np.ndarray
        labels: np.ndarray
    """

    if cellaap_widget.model_type == "yacs": 
        img = au.bw_to_rgb(img)
        output = cellaap_widget.predictor(img.astype("float32"))

    else:
        if img.shape == (2048, 2048):
            img = au.square_reshape(img, (1024, 1024))
            img = au.bw_to_rgb(img)
        img_perm = np.moveaxis(img, -1, 0)

        with torch.inference_mode():
            output = cellaap_widget.predictor(
                [{"image": torch.from_numpy(img_perm).type(torch.float32)}]
            )[0]

    segmentations = output["instances"].pred_masks.to("cpu")
    labels = output["instances"].pred_classes.to("cpu").numpy()
    scores = output["instances"].scores.to("cpu").numpy()
    scores = (scores*100).astype('uint16')
    
    custom_dict  = {key: key+99 for key in np.unique(labels)}
    seg_fordisp = color_masks(
        segmentations, labels, method="custom", custom_dict=custom_dict
    )

    seg_fortracking = color_masks(segmentations, labels, method="random")

    seg_scores = color_masks(segmentations, scores, method = "straight")

    centroids = []
    for i, _ in enumerate(labels):
        labeled_mask = skimage.measure.label(segmentations[i])
        centroid = skimage.measure.centroid(labeled_mask)
        if frame_num != None:
            centroid = np.array([frame_num, centroid[0], centroid[1]])

        centroids.append(centroid)

    return seg_fordisp, seg_fortracking, centroids, img, seg_scores, labels


def run_inference(cellaap_widget: ui.cellAAPWidget):
    """
    Runs inference on the selected image and caches results
    -------------------------------------------------------
    INPUTS:
        cellaap_widget: `ui.cellAAPWidget`
    RETURNS:
        None
    """
    prog_count = 0
    instance_movie = []
    semantic_movie = []
    scores_movie = []
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
    
    # Configure frame progress for batch vs single-image mode
    if len(im_array.shape) == 3:
        frame_total = im_array.shape[0] if cellaap_widget.batch else (
            cellaap_widget.range_slider.value()[1] - cellaap_widget.range_slider.value()[0] + 1
        )
        cellaap_widget.progress_bar.setValue(0)
        cellaap_widget.progress_bar.setMaximum(frame_total)
        movie = []
        for i in range(frame_total):
            # In batch mode, process all frames; otherwise respect slider range
            frame_index = i if cellaap_widget.batch else i + cellaap_widget.range_slider.value()[0]
            img = im_array[frame_index]
            semantic_seg, instance_seg, centroids, img, scores_seg, classes= inference(
                cellaap_widget, img, i
            )
            movie.append(img)
            semantic_movie.append(semantic_seg.astype("uint16"))
            instance_movie.append(instance_seg.astype("uint16"))
            scores_movie.append(scores_seg)
            classes_list.append(classes)
            if len(centroids) != 0:
                points += (centroids,)
            prog_count += 1
            cellaap_widget.progress_bar.setValue(prog_count)

    elif len(im_array.shape) == 2:
        cellaap_widget.progress_bar.setValue(0)
        cellaap_widget.progress_bar.setMaximum(1)
        semantic_seg, instance_seg, centroids, img, scores, classes= inference(cellaap_widget, im_array)
        semantic_movie.append(semantic_seg.astype("uint16"))
        instance_movie.append(instance_seg.astype("uint16"))
        scores_movie.append(scores)
        classes_list.append(classes)
        if len(centroids) != 0:
            points += (centroids,)
        prog_count = 1
        cellaap_widget.progress_bar.setValue(prog_count)

    model_name = cellaap_widget.model_selector.currentText()
    cellaap_widget.progress_bar.reset()

    semantic_movie = np.asarray(semantic_movie)
    instance_movie = np.asarray(instance_movie)
    points_array = np.vstack(points)
    scores_movie = np.asarray(scores_movie)
    classes_array = np.concatenate(classes_list, axis =0)

    cache_entry_name = f"{name}_{model_name}_{cellaap_widget.confluency_est.value()}_{round(cellaap_widget.thresholder.value(), ndigits = 2)}"

    already_cached = [
        cellaap_widget.save_combo_box.itemText(i)
        for i in range(cellaap_widget.save_combo_box.count())
    ]

    if cache_entry_name in already_cached:
        only_cache_entry = [
            entry
            for _, entry in enumerate(already_cached)
            if entry in cache_entry_name
        ]
        cache_entry_name += f"_{len(only_cache_entry)}"

    cellaap_widget.save_combo_box.insertItem(0, cache_entry_name)
    cellaap_widget.save_combo_box.setCurrentIndex(0)

    cache_entry = {
        "name": cache_entry_name,
        "semantic_movie": semantic_movie,
        "instance_movie": instance_movie,
        "centroids": points_array,
        "scores_movie": scores_movie,
        "classes": classes_array,
    }
    # Store the raw image/movie for later display in disp_inf_results
    try:
        cache_entry["movie"] = np.asarray(movie)
    except UnboundLocalError:
        cache_entry["movie"] = img
    # Store a reasonable image layer name
    cache_entry["image_layer_name"] = name

    cellaap_widget.inference_cache.append(cache_entry)
    
    # Update button states after inference is complete
    update_button_states(cellaap_widget)


def batch_inference(cellaap_widget: ui.cellAAPWidget):
    """
    Runs inference on a group of movies in batch mode and caches results
    --------------------------------------------------------------------
    INPUTS:
        cellaap_widget: `ui.cellAAPWidget`
    RETURNS:
        None
    """

    num_movies = len(cellaap_widget.full_spectrum_files)
    movie_tally = 0
    # Configure image-level progress bar if present
    cellaap_widget.progress_bar_images.setMaximum(num_movies)
    cellaap_widget.progress_bar_images.setValue(0)

    while movie_tally < num_movies:
        run_inference(cellaap_widget)
        movie_tally += 1
        cellaap_widget.progress_bar_images.setValue(movie_tally)



def configure(cellaap_widget: ui.cellAAPWidget):
    """
    Configures tunable parameters and initializes the predictor
    ----------------------------------------------------------
    INPUTS:
        cellaap_widget: `ui.cellAAPWidget`
    RETURNS:
        None
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
        torch.load = patched_torch_load
        DetectionCheckpointer(predictor).load(cellaap_widget.cfg.train.init_checkpoint)
        torch.load = _original_torch_load
        predictor.eval()

    napari.utils.notifications.show_info(f"Configurations successfully saved")
    cellaap_widget.configured = True
    cellaap_widget.predictor = predictor


def get_model(cellaap_widget):
    """
    Instantiates a `pooch` registry for model files and returns model info
    ----------------------------------------------------------------------
    INPUTS:
        cellaap_widget: `ui.cellAAPWidget`
    RETURNS:
        model: pooch.Pooch
        model_type: str
        weights_name: str
        config_name: str
    """

    model_name = cellaap_widget.model_selector.currentText()

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
            "HeLa_dead": "doi:10.5281/zenodo.17123228",
            "general_dead_focal": "doi:10.5281/zenodo.17026595"
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
            "model_0121499.pth",
            "md5:2ddfa097d5b357094c5191e03ef73cce"
        ),
        "general_dead_focal": (
            "model_0143499.pth",
            "md5:37c7e70599c6be6721c55227739a96bc"
        )

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
            "md5:7117f5b9473658e438b2e56ff91baac8",
            "lazy"
        ),
        "general_dead_focal": (
            "config.yaml",
            "md5:3fb65fcc6c3f06e52b36ef53282266c6",
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


def color_masks(
    segmentations: np.ndarray,
    labels,
    method: Optional[str] = "random",
    custom_dict: Optional[dict[int, int]] = None,
    erode = False
) -> np.ndarray:
    """
    Colors segmentation masks by a chosen method or mapping
    -------------------------------------------------------
    INPUTS:
        segmentations: np.ndarray
        labels: np.ndarray
        method: Optional[str]
        custom_dict: Optional[dict[int, int]]
        erode: bool
    OUTPUTS:
        seg_labeled: np.ndarray
    """

    if method == "custom":
        try:
            assert custom_dict != None
            assert np.isin(labels, list(custom_dict.keys())).all() == True
        except AssertionError:
            print('Input labels and mapping dictionary did not match when coloring movie, reverting to straight coloring')
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

def disp_inf_results(cellaap_widget) -> None:
    """
    Displays inference and analysis results for the selected cache entry
    --------------------------------------------------------------------
    INPUTS:
        cellaap_widget: `ui.cellAAPWidget`
    RETURNS:
        None
    """

    result_name = cellaap_widget.save_combo_box.currentText()
    result = list(
        filter(
            lambda x: x["name"] in f"{result_name}",
            cellaap_widget.inference_cache,
        )
    )[0]

    # Add underlying raw image/movie first if available
    try:
        movie = result['movie']
        image_layer_name = result.get('image_layer_name', result_name)
        if isinstance(movie, np.ndarray) and movie.ndim == 4:
            # shape (frames, H, W, C) -> display first channel as image stack
            cellaap_widget.viewer.add_image(movie[:, :, :, 0], name=image_layer_name)
        else:
            # 2D or 3D image
            cellaap_widget.viewer.add_image(movie if movie.ndim != 3 else movie[:, :, 0], name=image_layer_name)
    except Exception:
        # If anything goes wrong, continue with labels/points
        pass

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
