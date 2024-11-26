import numpy as np
import numpy.typing as npt
import os
import skimage
import scipy
from skimage import segmentation
import btrack  # type: ignore
from btrack import datasets  # type:ignore
import pandas as pd
from typing import Optional
import tifffile as tiff
import cell_AAP.napari.ui as ui
import cell_AAP.napari.fileio as fileio
import cell_AAP.annotation.annotation_utils as au


def projection(im_array: np.ndarray, projection_type: str):

    if im_array.shape[0] % 2 == 0:
        center_index = im_array.shape[0] // 2 - 1
    else:
        center_index = im_array.shape[0] // 2

    range = center_index // 2

    try:
        assert projection_type in ["max", "min", "average"]
    except AssertionError:
        print("Projection type was not valid, valid types include: max, min, mean")

    if projection_type == "max":
        projected_image = np.max(
            im_array[center_index - range : center_index + range], axis=0
        )
    elif projection_type == "average":
        projected_image = np.mean(
            im_array[center_index - range : center_index + range], axis=0
        )
    elif projection_type == "min":
        projected_image = np.min(
            im_array[center_index - range : center_index + range], axis=0
        )

    return np.array(projected_image)


def track(
    instance_movie: npt.NDArray,
    intensity_movie: npt.NDArray,
    config_file: Optional[str] = datasets.cell_config(),
    features: Optional[list[str]] = None,
):
    """
    Utilizes btrack to track cells through time, assigns class_id labels to each track, 0: non-mitotic, 1: mitotic
    --------------------------------------------------------------------------------------------------------------
    INPUTS:
        instance_movie: npt.NDArray,
        intensity_movie: npt.NDArray,
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
            "intensity_mean",
            "intensity_mean",
        ]

    if intensity_movie.shape[1] != instance_movie.shape[1]:
        intensity_movie_binned = [
            au.square_reshape(
                np.asarray(intensity_movie[i]), desired_shape=instance_movie.shape[1:]
            )
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

    for i, object in enumerate(objects):
        if object.properties["class_id"] % 2 == 0:
            object.properties["class_id"] = 0
        else:
            object.properties["class_id"] = 1
        if object.properties['area'] < 500:
                objects.pop(i)
        if object.properties['solidity'] < 0.90:
            objects.pop(i)

    with btrack.BayesianTracker() as tracker:

        tracker.configure(config_file)
        tracker.max_search_radius = 10
        tracker.tracking_updates = ["MOTION", "VISUAL"]
        tracker.features = features

        tracker.append(objects)
        tracker.volume = ((0, intensity_movie.shape[1]), (0, intensity_movie.shape[2]))
        tracker.track(step_size=100)
        tracker.optimize()

        data, properties, graph = tracker.to_napari()
        tracks = tracker.tracks
        cfg = tracker.configuration

        return tracks, data, properties, graph, cfg


def time_in_mitosis(
    state_matrix : npt.NDArray, interframe_duration: float
) -> tuple[ npt.NDArray, npt.NDArray, int]:
    """
    Takes in a tracks object from btrack and an interframe duration, returns a vector containing the time spent in mitosis for each track
    --------------------------------------------------------------------------------------------------------------------------------------
    INPUTS:
        tracks
        interframe_duration: float
    OUTPUTS:
        state_matric: npt.NDArray
        state_duration_vec: npt.NDArray, indexed like "state_duration_vec[cell]"
        avg_time_in_mitosis: int
    """

    last_frame_mitotic = [
        row_index
        for row_index in range(state_matrix.shape[0])
        if state_matrix[row_index, -1] == 1
    ]

    state_matrix_cleaned = state_matrix
    for row_index in last_frame_mitotic:
        state_matrix_cleaned[row_index, :] = [
            0 for state in state_matrix_cleaned[row_index, :]
        ]

    state_duration_vec = np.sum(state_matrix_cleaned, axis=1) * interframe_duration
    total_time = np.sum(state_duration_vec)
    num_mitotic_cells = state_duration_vec[state_duration_vec > 0].shape[
        0
    ]  # removing entries that were never mitotic
    avg_time_in_mitosis = (total_time) / (num_mitotic_cells + np.finfo(float).eps)

    return state_matrix_cleaned, state_duration_vec, avg_time_in_mitosis


def cell_intensity(tracks, time_points: int) -> tuple[npt.NDArray, npt.NDArray]:
    """
    Takes in a tracks object from btrack and an interframe duration, returns a matrix containing the average intensity of each cell at each timepoint
    ---------------------------------------------------------------------------------------------------------------------------------------------------
    INPUTS:
        tracks: btrack tracks object
        time_points: int
    OUTPUTS:
        intensity_matric: npt.NDArray, indexed like "intensity_matrix[cell, timepoint]"
        avg_intensity_vec: npt.NDArray
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

    avg_intensity_vec = np.sum(intensity_matrix, axis=1) / time_points

    return intensity_matrix, avg_intensity_vec


def mitotic_intensity(
    state_duration_vec: npt.NDArray,
    state_matrix: npt.NDArray,
    intensity_matrix: npt.NDArray,
    interframe_duration: float,
) -> npt.NDArray:
    """
    Takes the results of time_in_mitosis and cell_intensity and correlates the two, returning a vector containing the average intensity of each cell during mitosis
    --------------------------------------------------------------------------------------------------------------------------------------------------------------
    INPUTS:
        state_duration_vec: npt.NDArray
        state_matrix: npt.NDArray
        intensity_matrix: npt.NDArray
        interframe_duration: int
    OUTPUTS:
        mitotic_intensity_vec: npt.NDArray
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


def write_output(
    data: list[npt.NDArray],
    directory: str,
    names: list[str],
    file_name: Optional[str] = "analysis.xlsx",
    columns: Optional[list] = None,
) -> None:
    """
    Writes analysis output to an excel file
    ---------------------------------------
    INPUTS:
        data: list[npt.NDArray]
        directory: str
        names: list[str]
        columns: list
    """

    df_cache = []
    for i, array in enumerate(data):

        if columns == None:
            df = pd.DataFrame(array)

        else:
            df = pd.DataFrame(array, columns=columns[i])
        df_cache.append(df)

    filename = os.path.join(directory, file_name)
    with pd.ExcelWriter(filename) as writer:
        [df.to_excel(writer, sheet_name=names[i]) for i, df in enumerate(df_cache)]


def compile_tracking_coords(
    tracks, state_duration_vec: npt.NDArray
) -> tuple[npt.NDArray, npt.NDArray, npt.NDArray, npt.NDArray]:
    """
    Compliles the inititation and termination of all tracks from a btrack tracks instance, requires a vector specifying whether or not each track was ever mitotic (class_id == 1)
    -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    INPUTS:
        tracks: btrack tracking instance
        state_duration_vec: npt.NDArray, vector of shape == len(tracks) specifying whether or not each track was every a given class id
    OUTPUTS:
        init_vec: npt.NDArray,  vector containing (x_i, y_i) where i stands for initial
        term_vec: npt.NDArray, vector containing (x_f, y_f) where f stands for final
        init_vec_mitotic: npt.NDArray, init_vec trimmed to only include mitotic cells
        term_vec_mitotic, npt.NDArray, term_vec trimmed to only include mitotic cells
    """

    init_vec = [[cell.x[0], cell.y[0]] for cell in tracks]
    term_vec = [[cell.x[-1], cell.y[-1]] for cell in tracks]
    init_vec = np.asarray(init_vec)
    term_vec = np.asarray(term_vec)

    double_duration_vec = np.concatenate(
        ([state_duration_vec], [state_duration_vec]), axis=0
    )
    mitotic_vec_mask = double_duration_vec.T > 0

    init_vec_mitotic = init_vec[mitotic_vec_mask[:, 0] == True, :]
    term_vec_mitotic = term_vec[mitotic_vec_mask[:, 0] == True, :]

    return init_vec, term_vec, init_vec_mitotic, term_vec_mitotic


def timepoints_in_mitosis(state_matrix: npt.NDArray):
    """
    Reads a matrix specifying whether or not a track was in mitosis at a given timepoint and outputs a vector of strings, where the string specifying all the timepoints at which that track was in mitosis'
    --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    INPUTS:
        state_matrix: nd.array, matrix specifying the state of alls tracks at all timepoints ( > 0 = mitotic)
    OUTPUTS:
        index_vec: list[list], vector of strings, each string specifies the timepoints at which the track was mitotic, i.e -14-15-16-17 corresponds to the track being mitotic from the 14-17th frames
    """

    index_vec = []
    for row in range(state_matrix.shape[0]):
        mitotic_index_str = ""
        mitotic_index_vec = [
            index for index, value in enumerate(state_matrix[row]) if value > 0
        ]
        for index in mitotic_index_vec:
            mitotic_index_str += f"-{index}"
        if mitotic_index_str == "":
            mitotic_index_str = "None"
        index_vec.append([mitotic_index_str])

    return index_vec


def analyze_raw(
    tracks, instance_movie
) -> tuple[npt.NDArray, npt.NDArray, npt.NDArray, npt.NDArray]:

    state_matrix = []
    intensity_matrix = []
    x_coords = []
    y_coords = []
    time_points = instance_movie.shape[0]
    for cell in tracks:
        state_matrix_row = np.zeros(shape=(time_points,))
        state_matrix_row[0 : len(cell.properties["class_id"])] = cell.properties[
            "class_id"
        ]
        state_matrix.append(state_matrix_row)

        intensity_matrix_row = np.zeros(shape=(time_points,))
        intensity_matrix_row[0 : len(cell.properties["intensity_mean"])] = (
            cell.properties["intensity_mean"]
        )
        intensity_matrix.append(intensity_matrix_row)

        x_coords_row = np.zeros(shape=(time_points,))
        x_coords_row[0 : len(cell.x)] = cell.x
        x_coords.append(x_coords_row)

        y_coords_row = np.zeros(shape=(time_points,))
        y_coords_row[0 : len(cell.y)] = cell.y
        y_coords.append(y_coords_row)

    state_matrix = np.asarray(state_matrix)
    mask = np.isnan(state_matrix)  # may not be the optimal way to handle NaN values
    state_matrix[mask] = np.interp(
        np.flatnonzero(mask), np.flatnonzero(~mask), state_matrix[~mask]
    )

    intensity_matrix = np.asarray(intensity_matrix)
    mask = np.isnan(intensity_matrix)  # may not be the optimal way to handle NaN values
    intensity_matrix[mask] = np.interp(
        np.flatnonzero(mask), np.flatnonzero(~mask), intensity_matrix[~mask]
    )

    x_coords = np.asarray(x_coords)
    mask = np.isnan(x_coords)  # may not be the optimal way to handle NaN values
    x_coords[mask] = np.interp(
        np.flatnonzero(mask), np.flatnonzero(~mask), x_coords[~mask]
    )

    y_coords = np.asarray(y_coords)
    mask = np.isnan(y_coords)  # may not be the optimal way to handle NaN values
    y_coords[mask] = np.interp(
        np.flatnonzero(mask), np.flatnonzero(~mask), y_coords[~mask]
    )

    return state_matrix, intensity_matrix, x_coords, y_coords


def gen_intensitymap(image: npt.NDArray) -> npt.NDArray:
    """
    Computes the intensity map for flouresence microscopy intensity normalization if the input is a blank with flourescent media
    ----------------------------------------------------------------------------------------------------------------------------
    INPUTS:
        image: npt.NDArray
    OUTPUTPS:
        intensity_map: npt.NDArray
    """

    mean_plane = projection(image, "average")
    med_filtered_mean_plane = scipy.ndimage.median_filter(mean_plane, 9)
    smoothed_mean_plane = skimage.filters.gaussian(med_filtered_mean_plane, 45)
    intensity_map = smoothed_mean_plane / (np.max(smoothed_mean_plane))

    return intensity_map
