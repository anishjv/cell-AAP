import numpy as np
import numpy.typing as npt
import os
import skimage
import scipy
from skimage import segmentation
import btrack  # type: ignore
from btrack import datasets  # type:ignore
import pandas as pd
import re
from typing import Optional
import tifffile as tiff
import cell_AAP.napari.analysis as analysis
import cell_AAP.napari.fileio as fileio
import cell_AAP.napari.graphing as graph
import cell_AAP.annotation.annotation_utils as au

"""
Script that searches for standard cell-AAP inference directories and computes time_in_mitosis/flourescence metrics
for each.

The script is not general, it assumes some things about the inference directory and the xlsx sheet within it.

    1) We assume that the inference folder ends in "inference" and contains the well and position name respectively,
       i.e. B10_s2 for well B10 and position 2.
    2) We assume that the analysis file ends in "analysis"
    3) We assume that the analysis file contains at least 2 sheets (State Matrix, Intensity Matrix)
    4) Optionally the analysis file should contain two other sheets (X Coordinates, Y Coordinates)
"""


def main():
    root_dir = input(
        "Please enter path to home directory where analysis folders are stored: "
    )
    interframe_duration = float(
        input("Please enter the duration between frames in minutes: ")
    )
    home_dirs = [
        f.path
        for f in os.scandir(root_dir)
        if f.is_dir() and str(f.path).split("_")[-1] == "inference"
    ]
    background_map = [
        f.path
        for f in os.scandir(root_dir)
        if re.search(r"background_map\.ti(f|ff)", str(f.path)) != None
    ]
    intensity_map = [
        f.path
        for f in os.scandir(root_dir)
        if re.search(r"intensity_map\.ti(f|ff)", str(f.path)) != None
    ]

    for home_dir in home_dirs:
        well = re.search(r"[A-G]([0][1-9]|[1][1-2])", home_dir).group()
        position = re.search(r"[s]\d", home_dir).group()
        prefix = str(home_dir.split("/")[-1].split(position)[0] + position)

        home_dir_validity = os.path.exists(home_dir)
        analysis_file_validity = os.path.exists(f"{home_dir}/{prefix}_analysis.xlsx")

        if home_dir_validity and analysis_file_validity:
            state_matrix = pd.read_excel(
                f"{home_dir}/{prefix}_analysis.xlsx", "State Matrix"
            ).to_numpy(dtype="float")[:, 1:]
            state_matrix_cleaned, state_duration_vec, avg_time_in_mitosis = (
                analysis.time_in_mitosis(state_matrix, interframe_duration)
            )

            num_timepoints = tiff.imread(
                f"{home_dir}/{prefix}_semantic_movie.tif"
            ).shape[0]
            intensity_matrix = pd.read_excel(
                f"{home_dir}/{prefix}_analysis.xlsx", "Intensity Matrix"
            ).to_numpy(dtype="float")[:, 1:]

            if (len(intensity_map), len(background_map)) == (1, 1):
                intensity_map = tiff.imread(intensity_map[0])
                background_map = tiff.imread(background_map[0])

                x_coords = pd.read_excel(
                    f"{home_dir}/{prefix}_analysis.xlsx", "X Coordinates"
                ).to_numpy(dtype="float")[:, 1:]
                y_coords = pd.read_excel(
                    f"{home_dir}/{prefix}_analysis.xlsx", "Y Coordinates"
                ).to_numpy(dtype="float")[:, 1:]

                for i in range(intensity_matrix.shape[0]):  # tracks
                    for j in range(intensity_matrix.shape[1]):  # frames
                        x = round(x_coords[i, j])
                        y = round(y_coords[i, j])
                        intensity_matrix[i, j] /= intensity_map[x, y]
                        intensity_matrix[i, j] -= background_map[j, x, y]

            mitotic_intensity_vec = analysis.mitotic_intensity(
                state_duration_vec,
                state_matrix_cleaned,
                intensity_matrix,
                interframe_duration,  #
            )

            index_vec = analysis.timepoints_in_mitosis(state_matrix)

            mitotic_duration_vec = state_duration_vec[state_duration_vec > 0]
            mitotic_intensity_vec = mitotic_intensity_vec[mitotic_intensity_vec > 0]

            arr1 = np.asarray([mitotic_duration_vec, mitotic_intensity_vec]).T
            columns1 = ["Duration in Mitosis", "Intensity in Mitosis"]
            df1 = pd.DataFrame(data=arr1, columns=columns1)

            arr2 = np.asarray(index_vec)
            columns2 = ["Timepoints in Mitosis"]
            df2 = pd.DataFrame(data=arr2, columns=columns2)

            writer = pd.ExcelWriter(
                path=f"{home_dir}/{prefix}_analysis.xlsx",
                engine="openpyxl",
                mode="a",
                if_sheet_exists="overlay",
            )

            df1.to_excel(
                writer,
                sheet_name="Post Analysis",
                columns=columns1,
            )

            df2.to_excel(
                writer,
                sheet_name="Post Analysis",
                columns=columns2,
                startcol=5,
            )

            writer.close()

        else:
            (
                print(home_dir + " could not be found")
                if home_dir_validity == False
                else print(
                    home_dir + f"/{prefix}_analysis.xslx" + " could not be found"
                )
            )
            return


if __name__ == "__main__":
    main()
