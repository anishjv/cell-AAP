import re
import cv2
import os
import tifffile as tiff
import numpy as np
from cell_AAP.napari import ui #type:ignore
from cell_AAP.napari import analysis #type:ignore
from qtpy import QtWidgets
import napari
import napari.utils.notifications


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


def grab_file(cellaap_widget: ui.cellAAPWidget):
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
        f"File: {file_grabber[0]} is queued for inference/analysis"
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
        grab_file(cellaap_widget)
        instance_movie = np.asarray(inference_result['instance_movie'])
        intensity_movie_path, intensity_movie = image_select(cellaap_widget)
        tracks, data, properties, graph, cfg = analysis.track(
            instance_movie, intensity_movie
        )

        (
            state_duration_vec,
            avg_time_in_mitosis,
            intensity_matrix,
            mitotic_intensity_vec,
            state_matrix,
        ) = analysis.analyze(
            tracks,
            instance_movie,
            cellaap_widget.interframe_duration.value(),
        )
        vecs_to_save = np.concatenate(
            ([state_duration_vec], [mitotic_intensity_vec]), axis=0
        )
        matrices_to_save = np.concatenate((intensity_matrix, state_matrix))

        to_save = [vecs_to_save.T, matrices_to_save, np.asarray([avg_time_in_mitosis])]
        columns = [
            ["Time in mitosis", "Aeverage Intensity during Mitosis"],
            None,
            ["Average Time in Mitosis (across movie)"],
        ]
        names = ["Mitosis", "Raw", "Average Time in Mitosis"]

        analysis.write_output(to_save, inference_folder_path, names, columns)

    tiff.imwrite(
        os.path.join(inference_folder_path, "semantic_movie.tif"),
        inference_result["semantic_movie"],
    )
    tiff.imwrite(
        os.path.join(inference_folder_path, "instance_movie.tif"),
        inference_result["instance_movie"],
    )
    np.save(
        os.path.join(inference_folder_path, "centroids.npy"),
        inference_result["centroids"],
    )
