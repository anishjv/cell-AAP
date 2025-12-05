import re
import cv2
import os
import tifffile as tiff
import numpy as np
import pandas as pd
from cell_AAP.napari import ui  # type:ignore
from qtpy import QtWidgets
import napari
import napari.utils.notifications
from typing import Optional


def image_select(
    cellaap_widget: ui.cellAAPWidget, pop: Optional[bool] = True
):
    """
    Returns the selected path and its image array
    --------------------------------------------
    INPUTS:
        cellaap_widget: `ui.cellAAPWidget`
        pop: Optional[bool]
            If True, removes the file from `full_spectrum_files`
    OUTPUTS:
        name: str
        layer_data: np.ndarray
    """

    file = cellaap_widget.full_spectrum_files[0]
    if pop:
        cellaap_widget.full_spectrum_files.pop(0)

    if (
        re.search(
            r"^.+\.(?:(?:[tT][iI][fF][fF]?)|(?:[tT][iI][fF]))$",
            str(file),
        )
        == None
    ):
        layer_data = cv2.imread(str(file), cv2.IMREAD_GRAYSCALE)
    else:
        layer_data = tiff.imread(str(file))

    return str(file), layer_data


def display(cellaap_widget: ui.cellAAPWidget):
    """
    Displays the selected file in the Napari GUI
    --------------------------------------------
    INPUTS:
        cellaap_widget: `ui.cellAAPWidget`
    RETURNS:
        None
    """
    try:
        name, layer_data = image_select(
            cellaap_widget, pop=False
        )
    except AttributeError or TypeError:
        napari.utils.notifications.show_error("No Image has been selected")
        return

    name = name.replace(".", "/").split("/")[-2]
    cellaap_widget.viewer.add_image(layer_data, name=name)


def grab_file(cellaap_widget: ui.cellAAPWidget):
    """
    Initiates a file dialog and grabs one or more files
    ---------------------------------------------------
    INPUTS:
        cellaap_widget: `ui.cellAAPWidget`
    RETURNS:
        None
    """
    file_filter = "TIFF (*.tiff, *.tif);; JPEG (*.jpg)"
    file_names, filter = QtWidgets.QFileDialog.getOpenFileNames(
        parent=cellaap_widget,
        caption="Select file(s)",
        directory=os.getcwd(),
        filter=file_filter,
    ) #type:ignore

    if file_names:
        cellaap_widget.full_spectrum_files = file_names
        cellaap_widget.image_path = file_names[0]  # Store the selected image path
        
        try:
            if 'JPEG' in filter: 
                shape = cv2.imread(str(file_names[0]), cv2.IMREAD_GRAYSCALE).shape
            else:
                shape = tiff.imread(str(file_names[0])).shape
            napari.utils.notifications.show_info(
                f"File: {file_names[0]} is queued for inference/analysis"
            )
            # Only set range slider for single-movie mode
            if not getattr(cellaap_widget, 'batch', False):
                if len(shape) == 3: 
                    cellaap_widget.range_slider.setRange(min=0, max=shape[0] - 1)
                    cellaap_widget.range_slider.setValue((0, shape[1]))
                else:
                    cellaap_widget.range_slider.setRange(min=0, max=0)
                    cellaap_widget.range_slider.setValue((0, 0))
        except AttributeError or IndexError:
            napari.utils.notifications.show_error("No file was selected")
    else:
        napari.utils.notifications.show_error("No file was selected")


def grab_directory(cellaap_widget):
    """
    Initiates a directory selection dialog
    --------------------------------------
    INPUTS:
        cellaap_widget: `ui.cellAAPWidget`
    OUTPUTS:
        dir_grabber: str
    """

    dir_grabber = QtWidgets.QFileDialog.getExistingDirectory(
        parent=cellaap_widget, caption="Select a directory to save inference result"
    )

    if dir_grabber:
        cellaap_widget.dir_grabber = dir_grabber
        cellaap_widget.save_path = dir_grabber  # Store the save path
        napari.utils.notifications.show_info(f"Directory: {dir_grabber} has been selected")
        return dir_grabber
    else:
        napari.utils.notifications.show_error("No directory was selected")
        return ""


def save(cellaap_widget):
    """
    Saves inference results to disk
    -------------------------------
    INPUTS:
        cellaap_widget: `ui.cellAAPWidget`
    BEHAVIOR:
        - In batch mode, saves all cached results
        - In single-image mode, saves the selected cache entry
    RETURNS:
        None
    """

    try:
        filepath = cellaap_widget.dir_grabber
    except AttributeError:
        napari.utils.notifications.show_error(
            "No Directory has been selected - will save output to current working directory"
        )
        filepath = os.getcwd()

    # Check cache availability
    if len(cellaap_widget.inference_cache) == 0:
        napari.utils.notifications.show_error("No inference results to save")
        return

    # Determine which entries to save
    if getattr(cellaap_widget, 'batch', False):
        cache_entries = cellaap_widget.inference_cache
    else:
        # Use selected entry from combo box
        if cellaap_widget.save_combo_box.count() == 0:
            napari.utils.notifications.show_error("No inference results to save")
            return
        selected_name = cellaap_widget.save_combo_box.currentText()
        cache_entries = [
            entry for entry in cellaap_widget.inference_cache if entry["name"] == selected_name
        ]
        if len(cache_entries) == 0:
            napari.utils.notifications.show_error("Selected inference result not found")
            return

    model_name = cellaap_widget.model_selector.currentText()

    for inference_result in cache_entries:
        inference_result_name = inference_result["name"]
        try:
            position = re.search(r"_s\d_", inference_result_name).group()
            analysis_file_prefix = inference_result_name.split(position)[0] + position
        except Exception:
            analysis_file_prefix = inference_result_name.split(model_name)[0]

        inference_folder_path = os.path.join(filepath, inference_result_name + "_inference")
        os.makedirs(inference_folder_path, exist_ok=True)

        # Save scores movie
        tiff.imwrite(
            os.path.join(
                inference_folder_path, analysis_file_prefix + "scores_movie.tif"
            ),
            inference_result["scores_movie"],
        )

        # Save semantic movie
        tiff.imwrite(
            os.path.join(
                inference_folder_path, analysis_file_prefix + "semantic_movie.tif"
            ),
            inference_result["semantic_movie"],
            dtype="uint16",
        )

        # Save instance movie (tracking results)
        tiff.imwrite(
            os.path.join(
                inference_folder_path, analysis_file_prefix + "instance_movie.tif"
            ),
            inference_result["instance_movie"],
            dtype="uint16",
        )

        napari.utils.notifications.show_info(f"Results saved to: {inference_folder_path}")

def add(cellaap_widget: ui.cellAAPWidget):
    """
    Adds a movie to the batch worker
    --------------------------------
    INPUTS:
        cellaap_widget: `ui.cellAAPWidget`
    RETURNS:
        None
    """

    grab_file(cellaap_widget)
    for file in cellaap_widget.full_spectrum_files:
        cellaap_widget.full_spectrum_file_list.addItem(file)


def remove(cellaap_widget: ui.cellAAPWidget):
    """
    Removes a movie from the batch worker
    -------------------------------------
    INPUTS:
        cellaap_widget: `ui.cellAAPWidget`
    RETURNS:
        None
    """
    current_row = cellaap_widget.full_spectrum_file_list.currentRow()
    if current_row >= 0:
        current_item = cellaap_widget.full_spectrum_file_list.takeItem(current_row)
        del current_item
        #cellaap_widget.full_spectrum_files.pop(current_row)


def clear(cellaap_widget: ui.cellAAPWidget):
    """
    Clears the batch worker of all movies
    ------------------------------------
    INPUTS:
        cellaap_widget: `ui.cellAAPWidget`
    RETURNS:
        None
    """

    cellaap_widget.full_spectrum_file_list.clear()
    cellaap_widget.full_spectrum_files = []
