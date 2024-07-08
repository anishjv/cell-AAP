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


def image_select(cellaap_widget: ui.cellAAPWidget, wavelength: str, pop: Optional[bool] = True):
    """
    Returns the path selected in the image selector box and the array corresponding the to path
    -------------------------------------------------------------------------------------------
    """
    if wavelength == 'full_spectrum':
        file = cellaap_widget.full_spectrum_files[0]
        if pop:
             cellaap_widget.full_spectrum_files.pop(0)
    else:
        file = cellaap_widget.flouro_files[0]
        if pop:
             cellaap_widget.flouro_files.pop(0)

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
        name, layer_data = image_select(cellaap_widget, wavelength='full_spectrum', pop = False)
    except AttributeError:
        napari.utils.notifications.show_error("No Image has been selected")
        return

    name = name.replace(".", "/").split("/")[-2]
    cellaap_widget.viewer.add_image(layer_data, name=name)


def grab_file(cellaap_widget: ui.cellAAPWidget, wavelength : str):
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
    )

    if wavelength == 'full_spectrum':
        cellaap_widget.full_spectrum_files = file_names

    else:
        cellaap_widget.flouro_files = file_names

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

    inference_result_name = cellaap_widget.save_combo_box.currentText()
    inference_result = list(
        filter(
            lambda x: x["name"] in f"{inference_result_name}",
            cellaap_widget.inference_cache,
        )
    )[0]

    inference_folder_path = os.path.join(filepath, inference_result_name)

    os.mkdir(inference_folder_path)

    # TODO
    # Make it possible to add other configs or features from within the gui

    if cellaap_widget.analyze_check_box.isChecked():
        instance_movie = np.asarray(inference_result["instance_movie"])
        try:
            intensity_movie_path, intensity_movie = image_select(cellaap_widget, wavelength='flourescent')
        except AttributeError:
            napari.utils.notifications.show_error("A Flourescence image has not been selected")
            return

        intensity_movie = intensity_movie[
            cellaap_widget.range_slider.value()[
                0
            ] : cellaap_widget.range_slider.value()[1] + 1
        ]
        tracks, data, properties, graph, cfg = analysis.track(
            instance_movie, intensity_movie
        )

        (
            state_duration_vec,
            avg_time_in_mitosis,
            intensity_matrix,
            avg_intensity_vec,
            mitotic_intensity_vec,
            state_matrix,
        ) = analysis.analyze(
            tracks,
            instance_movie,
            cellaap_widget.interframe_duration.value(),
        )

        init_vec, term_vec, init_vec_mitotic, term_vec_mitotic = (
            analysis.compile_tracking_coords(tracks, state_duration_vec)
        )

        mitosis_data = np.concatenate(
            (
                init_vec_mitotic.T,
                term_vec_mitotic.T,
                [state_duration_vec[state_duration_vec > 0]],
                [mitotic_intensity_vec[state_duration_vec > 0]],
            ),
            axis=0,
        )

        general_data = np.concatenate(
            (init_vec.T, term_vec.T, [state_duration_vec], [avg_intensity_vec])
        )

        to_save = [mitosis_data.T, general_data.T, np.asarray([avg_time_in_mitosis])]
        columns = [
            [
                "x_i",
                "y_i",
                "x_f",
                "y_f",
                "Time in Mitosis",
                "Mean Intensity during Mitosis",
            ],
            ["x_i", "y_i", "x_f", "y_f", "Time in Mitosis", "Mean Intensity"],
            ["Average Time in Mitosis (across movie)"],
        ]
        names = ["Mitosis", "General", "Average Time in Mitosis"]

        analysis.write_output(to_save, inference_folder_path, names, columns)

        state_matrix_df = pd.DataFrame(state_matrix)
        state_matrix_df.to_json(
            os.path.join(inference_folder_path, "state_matrix.json"), orient="table"
        )

        intensity_matrix_df = pd.DataFrame(intensity_matrix)
        intensity_matrix_df.to_json(
            os.path.join(inference_folder_path, "intensity_matrix.json"), orient="table"
        )

        with btrack.io.HDF5FileHandler(
            os.path.join(inference_folder_path, "tracks.h5"), "w", obj_type="obj_type_1"
        ) as writer:
            writer.write_tracks(tracks)

    tiff.imwrite(
        os.path.join(inference_folder_path, "semantic_movie.tif"),
        inference_result["semantic_movie"],
    )
    tiff.imwrite(
        os.path.join(inference_folder_path, "instance_movie.tif"),
        inference_result["instance_movie"],
    )

    centroids_df = pd.DataFrame(
        inference_result["centroids"], columns=["Frame", "x", "y"]
    )
    centroids_df.to_json(
        os.path.join(inference_folder_path, "centroids.json"), orient="records"
    )

#for next three functions, infer wavelength from file_list_toggle

def add(cellaap_widget : ui.cellAAPWidget):

    grab_file(cellaap_widget, wavelength=cellaap_widget.batch_list_wavelength)
    if cellaap_widget.batch_list_wavelength == 'full_spectrum':
        for file in cellaap_widget.full_spectrum_files:
            cellaap_widget.full_spectrum_file_list.addItem(file)
    else:
        for file in cellaap_widget.flouro_files:
            cellaap_widget.flouro_file_list.addItem(file)


def remove(cellaap_widget: ui.cellAAPWidget):

    if cellaap_widget.batch_list_wavelength == "full_spectrum":
        current_row = cellaap_widget.full_spectrum_file_list.currentRow()
        if current_row >= 0:
            current_item = cellaap_widget.full_spectrum_file_list.takeItem(current_row)
            del current_item
            cellaap_widget.full_spectrum_files.pop(current_row)
    else:
        current_row = cellaap_widget.flouro_file_list.currentRow()
        if current_row >= 0:
            current_item = cellaap_widget.flouro_file_list.takeItem(current_row)
            del current_item
            cellaap_widget.flouro_files.pop(current_row)


def clear(cellaap_widget: ui.cellAAPWidget):

    if cellaap_widget.batch_list_wavelength == "full_spectrum":
        cellaap_widget.full_spectrum_file_list.clear()
    else:
        cellaap_widget.flouro_file_list.clear()



def toggle_wavelength(cellaap_widget: ui.cellAAPWidget):

    if cellaap_widget.batch_list_wavelength == "full_spectrum":
        cellaap_widget.batch_list_wavelength = "flourescence"
    else:
        cellaap_widget.batch_list_wavelength = "full_spectrum"

    napari.utils.notifications.show_info(f"{cellaap_widget.batch_list_wavelength} file list is selected for editing")
