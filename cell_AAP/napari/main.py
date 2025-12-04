from __future__ import annotations
import logging
from typing import Optional
import napari
import napari.utils.notifications
import cell_AAP.napari.ui as ui  # type:ignore
import cell_AAP.annotation.annotation_utils as au  # type:ignore
import cell_AAP.napari.fileio as fileio  # type: ignore
import cell_AAP.core.inference_core as inference_core  # type: ignore

import numpy as np
import torch

from detectron2.utils.logger import setup_logger

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


# Use shared patched_torch_load from core module
from cell_AAP.core.inference_core import _original_torch_load, patched_torch_load


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
    cellaap_widget: ui.cellAAPWidget, 
    img: np.ndarray, 
    frame_num: Optional[int] = None
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
    # Read Parameter from GUI
    # Because ui.py automatically sets attributes from the widget dict keys,
    # we can access the checkbox directly via the name we gave it in sub_widgets.py
    keep_resized_output = cellaap_widget.keep_resized_checkbox.isChecked()

    return inference_core.run_inference_on_image(
        cellaap_widget.predictor,
        cellaap_widget.model_type,
        img,
        frame_num,
        keep_resized_output,
    )


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
    model_name = cellaap_widget.model_selector.currentText()
    confluency_est = cellaap_widget.confluency_est.value() if cellaap_widget.confluency_est.value() else None
    conf_threshold = cellaap_widget.thresholder.value() if cellaap_widget.thresholder.value() else None

    predictor, model_type, cfg = inference_core.configure_predictor(
        model_name, confluency_est, conf_threshold
    )

    cellaap_widget.model_type = model_type
    cellaap_widget.cfg = cfg
    cellaap_widget.predictor = predictor
    cellaap_widget.configured = True

    napari.utils.notifications.show_info(f"Configurations successfully saved")


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
    return inference_core.get_model(model_name)


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
