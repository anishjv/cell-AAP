import re
import cv2
import os
import tifffile as tiff
import numpy as np
import pandas as pd
from cell_AAP.napari import ui  # type:ignore
from cell_AAP.napari import analysis  # type:ignore
from qtpy import QtWidgets
import napari
import btrack
import napari.utils.notifications
from typing import Optional
from skimage.filters import gaussian
import cell_AAP.annotation.annotation_utils as au


def image_select(
    cellaap_widget: ui.cellAAPWidget, attribute: str, pop: Optional[bool] = True
):
    """
    Returns the path selected in the image selector box and the array corresponding the to path
    -------------------------------------------------------------------------------------------
    """

    match attribute.split():
        case ["full_spectrum"]:
            file = cellaap_widget.full_spectrum_files[0]
            if pop:
                cellaap_widget.full_spectrum_files.pop(0)
        case ["flouro"]:
            file = cellaap_widget.flouro_files[0]
            if pop:
                cellaap_widget.flouro_files.pop(0)
        case ["flouro_blank"]:
            file = cellaap_widget.flouro_blank
        case ["trans_blank"]:
            file = cellaap_widget.trans_blank
        case _:
            print("Attribute of assignment was not valid")


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
    Displays file in Napari gui if file has been selected, also returns the 'name' of the image file
    ------------------------------------------------------------------------------------------------
    INPUTS:
        cellaap_widget: instance of ui.cellAAPWidget()
    """
    try:
        name, layer_data = image_select(
            cellaap_widget, attribute="full_spectrum", pop=False
        )
    except AttributeError or TypeError:
        napari.utils.notifications.show_error("No Image has been selected")
        return

    name = name.replace(".", "/").split("/")[-2]
    cellaap_widget.viewer.add_image(layer_data, name=name)


def grab_file(cellaap_widget: ui.cellAAPWidget, attribute: str):
    """
    Initiates a QtWidget.QFileDialog instance and grabs a file
    -----------------------------------------------------------
    INPUTS:
        cellaap_widget: instance of ui.cellAAPWidget()
    """
    file_filter = "TIFF (*.tiff, *.tif);; JPEG (*.jpg);; PNG (*.png)"
    file_names, _ = QtWidgets.QFileDialog.getOpenFileNames(
        parent=cellaap_widget,
        caption="Select file(s)",
        directory=os.getcwd(),
        filter=file_filter,
    ) #type:ignore

    match attribute.split():
        case ["full_spectrum"]:
            cellaap_widget.full_spectrum_files = file_names
        case ["flouro"]:
            cellaap_widget.flouro_files = file_names
        case ["flouro_blank"]:
            try:
                cellaap_widget.flouro_blank = file_names[0]
                napari.utils.notifications.show_info(
                    f"File: {cellaap_widget.flouro_blank} is queued as the flourescent blank "
                )
            except IndexError:
                napari.utils.notifications.show_error("No File was selected")
        case ["trans_blank"]:
            try:
                cellaap_widget.trans_blank = file_names[0]
                napari.utils.notifications.show_info(
                    f"File: {cellaap_widget.trans_blank} is queued as the full_spectrum blank "
                )
            except IndexError:
                napari.utils.notifications.show_error("No File was selected")
        case _:
            napari.utils.notifications.show_error("Attribute of assignment was invalid")
            return

    if attribute in ["full_spectrum", "flouro"]:
        try:
            shape = tiff.imread(file_names[0]).shape
            napari.utils.notifications.show_info(
                f"File: {file_names[0]} is queued for inference/analysis"
            )
            cellaap_widget.range_slider.setRange(min=0, max=shape[0] - 1)
            cellaap_widget.range_slider.setValue((0, shape[1]))
        except AttributeError or IndexError:
            napari.utils.notifications.show_error("No file was selected")


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


def save(cellaap_widget):
    """
    Saves and analyzes an inference result
    """

    try:
        filepath = cellaap_widget.dir_grabber
    except AttributeError:
        napari.utils.notifications.show_error(
            "No Directory has been selected - will save output to current working directory"
        )
        filepath = os.getcwd()
        pass


    inference_result_name = cellaap_widget.save_combo_box.currentText()
    inference_result = list(
        filter(
            lambda x: x["name"] in f"{inference_result_name}",
            cellaap_widget.inference_cache,
        )
    )[0]


    model_name = cellaap_widget.model_selector.currentText()
    try:
        position = re.search(r"_s\d_", inference_result_name).group()
        analysis_file_prefix = inference_result_name.split(position)[0] + position
    except Exception as error:
        analysis_file_prefix = inference_result_name.split(model_name)[0]

    inference_folder_path = os.path.join(filepath, inference_result_name + "_inference")
    os.mkdir(inference_folder_path)

    instance_movie = np.asarray(inference_result["instance_movie"])
    if cellaap_widget.analyze_check_box.isChecked():
        try:
            intensity_movie_path, intensity_movie = image_select(
                cellaap_widget, attribute="flouro"
            )
        except AttributeError:
            napari.utils.notifications.show_error(
                "A Flourescence image has not been selected"
            )
            return

        intensity_movie = intensity_movie[
            cellaap_widget.range_slider.value()[
                0
            ] : cellaap_widget.range_slider.value()[1]
            + 1
        ]
        tracks, data, properties, graph, cfg = analysis.track(
            instance_movie, intensity_movie
        )

        tracking_update = {
            'data': data,
            'properties': properties,
            'graph': graph
        }

        list(
            filter(
                lambda x: x["name"] in f"{inference_result_name}",
                cellaap_widget.inference_cache,
            )
        )[0].update(tracking_update)


        state_matrix, intensity_matrix, x_coords, y_coords = analysis.analyze_raw(
            tracks, instance_movie
        )

        to_save = [state_matrix, intensity_matrix, x_coords, y_coords]
        names = ["State Matrix", "Intensity Matrix", "X Coordinates", "Y Coordinates"]
        analysis.write_output(
            to_save,
            inference_folder_path,
            names,
            file_name=analysis_file_prefix + "analysis.xlsx",
        )

        with btrack.io.HDF5FileHandler(
            os.path.join(inference_folder_path, analysis_file_prefix + "tracks.h5"),
            "w",
            obj_type="obj_type_1",
        ) as writer:
            writer.write_tracks(tracks)

    tiff.imwrite(
        os.path.join(
            inference_folder_path, analysis_file_prefix + "semantic_movie.tif"
        ),
        inference_result["semantic_movie"],
        dtype="uint16",
    )

    tiff.imwrite(
        os.path.join(
            inference_folder_path, analysis_file_prefix + "instance_movie.tif"
        ),
        inference_result["instance_movie"],
        dtype="uint16",
    )

def add(cellaap_widget: ui.cellAAPWidget):
    "Adds a movie to the batch worker"

    grab_file(cellaap_widget, attribute="full_spectrum")
    for file in cellaap_widget.full_spectrum_files:
        cellaap_widget.full_spectrum_file_list.addItem(file)


def remove(cellaap_widget: ui.cellAAPWidget):
    "Removes a movie from the batch worker"
    current_row = cellaap_widget.full_spectrum_file_list.currentRow()
    if current_row >= 0:
        current_item = cellaap_widget.full_spectrum_file_list.takeItem(current_row)
        del current_item
        cellaap_widget.full_spectrum_files.pop(current_row)


def clear(cellaap_widget: ui.cellAAPWidget):
    "Clears the batchworker of all movies"

    cellaap_widget.full_spectrum_file_list.clear()
    cellaap_widget.full_spectrum_files = []
