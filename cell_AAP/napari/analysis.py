import numpy as np
import os
from skimage import segmentation
import btrack #type: ignore
from btrack import datasets # type:ignore
import pandas as pd
from typing import Optional
import tifffile as tiff
import cell_AAP.napari.ui as ui
import cell_AAP.napari.fileio as fileio
import cell_AAP.annotation.annotation_utils as au


def track(
    instance_movie: np.ndarray,
    intensity_movie: np.ndarray,
    config_file: Optional[str] = datasets.cell_config(),
    features: Optional[list[str]] = None,
):
    """
    Utilizes btrack to track cells through time, assigns class_id labels to each track, 0: non-mitotic, 1: mitotic
    --------------------------------------------------------------------------------------------------------------
    INPUTS:
        instance_movie: np.ndarray,
        intensity_movie: np.ndarray,
        config_file: str,
        features: list
    """

    if features == None:
        features = [
            "area",
            "major_axis_length",
            "minor_axis_length",
            "orientation",
            "solidity",
            "intensity_mean"
        ]

    if intensity_movie.shape[1] != instance_movie.shape[1]:
        intensity_movie_binned = [
            au.square_reshape(np.asarray(intensity_movie[i]), desired_shape= instance_movie.shape[1:])
            for i, _ in enumerate(intensity_movie)
        ]

        intensity_movie = np.asarray(intensity_movie_binned)

    objects = btrack.utils.segmentation_to_objects(
        instance_movie,
        intensity_image=intensity_movie,
        properties=tuple(features),
        assign_class_ID=True,
        num_workers=1,
    )

    for object in objects:
        if object.properties['class_id'] % 2 == 0:
            object.properties['class_id'] = 0
        else:
            object.properties['class_id'] = 1

    with btrack.BayesianTracker() as tracker:

        tracker.configure(config_file)
        tracker.max_search_radius = 50
        tracker.tracking_updates = ["MOTION", "VISUAL"]
        tracker.features = features

        tracker.append(objects)
        tracker.volume = ((0, intensity_movie.shape[1]), (0, intensity_movie.shape[0]))
        tracker.track(step_size=100)
        tracker.optimize()

        data, properties, graph = tracker.to_napari()
        tracks = tracker.tracks
        cfg = tracker.configuration

        return tracks, data, properties, graph, cfg


def time_in_mitosis(
    tracks, interframe_duration: float, time_points: int
) -> tuple[np.ndarray, np.ndarray, int]:
    """
    Takes in a tracks object from btrack and an interframe duration, returns a vector containing the time spent in mitosis for each track
    --------------------------------------------------------------------------------------------------------------------------------------
    INPUTS:
        tracks
        interframe_duration: float
    OUTPUTS:
        avg_time_in_mitosis: int
        state_duration_vec: np.ndarray, indexed like "state_duration_vec[cell]"
    """

    state_matrix = []
    for cell in tracks:
        state_matrix_row = np.zeros(shape=(time_points,))
        state_matrix_row[0 : len(cell.properties["class_id"])] = cell.properties[
            "class_id"
        ]
        state_matrix.append(state_matrix_row)

    state_matrix = np.asarray(state_matrix)

    mask = np.isnan(state_matrix)  # may not be the optimal way to handle NaN values
    state_matrix[mask] = np.interp(
        np.flatnonzero(mask), np.flatnonzero(~mask), state_matrix[~mask]
    )

    state_duration_vec = np.sum(state_matrix, axis=1) * interframe_duration
    total_time = np.sum(state_duration_vec)
    num_mitotic_cells = state_duration_vec[state_duration_vec > 0].shape[
        0
    ]  # removing entries that were never mitotic
    avg_time_in_mitosis = (total_time) / (num_mitotic_cells + np.finfo(float).eps)

    return state_matrix, state_duration_vec, avg_time_in_mitosis


def cell_intensity(tracks, time_points: int) -> np.ndarray:
    """
    Takes in a tracks object from btrack and an interframe duration, returns a matrix containing the average intensity of each cell at each timepoint
    ---------------------------------------------------------------------------------------------------------------------------------------------------
    INPUTS:
        tracks
    OUTPUTS:
        intensity_matric: np.ndarray, indexed like "intensity_matrix[cell, timepoint]"
    """
    intensity_matrix = []
    for cell in tracks:
        intensity_matrix_row = np.zeros(shape=(time_points,))
        intensity_matrix_row[0 : len(cell.properties["intensity_mean"])] = (
            cell.properties["intensity_mean"]
        )
        intensity_matrix.append(intensity_matrix_row)

    intensity_matrix = np.asarray(intensity_matrix)

    mask = np.isnan(intensity_matrix)  # may not be the optimal way to handle NaN values
    intensity_matrix[mask] = np.interp(
        np.flatnonzero(mask), np.flatnonzero(~mask), intensity_matrix[~mask]
    )

    return intensity_matrix

def mitotic_intensity(
    state_duration_vec: np.ndarray,
    state_matrix: np.ndarray,
    intensity_matrix: np.ndarray,
    interframe_duration: float,
) -> np.ndarray:
    """
    Takes the results of time_in_mitosis and cell_intensity and correlates the two, returning a vector containing the averga intensity of each cell during mitosis
    --------------------------------------------------------------------------------------------------------------------------------------------------------------
    INPUTS:
        state_duration_vec: np.ndarray
        state_matrix: np.ndarray
        intensity_matrix: np.ndarray
        interframe_duration: int
    OUTPUTS:
        mitotic_intensity_vec: np.ndarray
    """

    try:
        state_matrix.shape == intensity_matrix.shape
    except Exception as error:
        raise AssertionError(
            "The state matrix and intensity matrix must be of the same shape"
        )

    state_matrix[state_matrix < 0.5] = 0
    state_matrix[state_matrix >= 0.5] = 1
    mitotic_intensity_matrix = np.multiply(intensity_matrix, state_matrix)
    mitotic_intensitysum_vec = mitotic_intensity_matrix.sum(axis=1)
    mitotic_intensity_vec = np.divide(
        mitotic_intensitysum_vec,
        (state_duration_vec + np.finfo(float).eps) / interframe_duration,
    )

    return mitotic_intensity_vec

def write_output(data: list[np.ndarray], directory: str, names: list[str], columns: Optional[list] = None) -> None:
    """
    Writes analysis output to an excel file
    ---------------------------------------
    INPUTS:
        data: list[np.ndarray]
        directory: str
        names: list[str]
        columns: list
    """

    df_cache = []
    for i, array in enumerate(data):

        if columns[i] == None:
            df = pd.DataFrame(array)

        else:
            df = pd.DataFrame(
                array, columns = columns[i]
            )
        df_cache.append(df)

    filename = os.path.join(directory, 'analysis.xlsx')
    with pd.ExcelWriter(filename) as writer:
        [df.to_excel(writer, sheet_name=names[i]) for i, df in enumerate(df_cache)]


def analyze(
        tracks,
        instance_movie: np.ndarray,
        interframe_duration: float,
    ) -> tuple[np.ndarray, int, np.ndarray, np.ndarray, np.ndarray]:
        """
        Composite function that facillitates the joint running of
            - time_in_mitosis()
            - cell_intensity()
            - mitotic_intensity()

        See docstrings of aforemtioned functions for inputs and outputs
        """

        num_timepoints = instance_movie.shape[0]
        state_matrix, state_duration_vec, avg_time_in_mitosis = time_in_mitosis(
            tracks, interframe_duration, num_timepoints
        )
        intensity_matrix = cell_intensity(tracks, num_timepoints)
        mitotic_intensity_vec = mitotic_intensity(
            state_duration_vec, state_matrix, intensity_matrix, interframe_duration
        )

        return state_duration_vec, avg_time_in_mitosis, intensity_matrix, mitotic_intensity_vec, state_matrix
