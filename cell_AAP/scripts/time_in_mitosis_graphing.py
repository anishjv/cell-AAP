import os
import re
import pandas as pd
import cell_AAP.napari.graphing as graph


"""
Script that takes the standard cell-AAP inference output and the output of time_in_mitosis.py and plots a scatter plot (Time in Mitosis vs. Flourescence).
Again, this script is not general and assumes some things about the root directory
    1) We assume that the inference directories end in "inference"
    2) We assume that the inferencen directories contain well information, i.e. A11 or B09
    3) We assume that the analysis file in the inference directory follows the standard cell-AAP analysis file naming convention
    4) We assume that the columns B:C in the sheet "Post Analysis" within the analysis file correspond to the variables "Duration in Mitosis" and "Flourescence in Mitosis" respectively
"""



def main():
    root_dir = input(
        "Please enter path to home directory where analysis folders are stored: "
    )

    inference_dirs = [
        f.path
        for f in os.scandir(root_dir)
        if f.is_dir() and str(f.path).split("_")[-1] == "inference"
    ]


    well_pairs = []
    for dir in sorted(inference_dirs):
        well = re.search(r"[A-G]([0][1-9]|[1][1-2])", dir).group() #find the well name of the first inference directory
        well_pair = [ dir for dir in inference_dirs if well in dir ]
        if well_pair != []:
            well_pairs.append(
                well_pair
            )   #search the through the inference directory, and grab all directories containing the well name
            inference_dirs = [dir for dir in inference_dirs if dir not in well_pairs[-1]]  #remove the directories added to the list of well pairs

    for pair in well_pairs:
        positions = [re.search(r"[s]\d", pair_entry).group() for pair_entry in pair]
        print(pair)
        data = []
        wells = []

        for i, pair_entry in enumerate(pair):
            prefix = str(pair_entry.split("/")[-1].split(positions[i])[0] + positions[i])
            analysis_file_validity = os.path.exists(f"{pair_entry}/{prefix}_analysis.xlsx")

            wells.append(
                re.search(r"[A-G]([0][1-9]|[1][1-2])", pair_entry).group()
            )

            print(analysis_file_validity)
            if analysis_file_validity:
                data.append(
                    pd.read_excel(
                        f"{pair_entry}/{prefix}_analysis.xlsx",
                        "Post Analysis",
                        usecols = "B:C"
                    )
                )
            else:
                print(f"Analsysis file: {pair_entry}/{prefix}_analysis.xlsx, was not found")

        print(data)
        try:
            data_table = pd.concat(
                data,
                ignore_index = True
            )
        except ValueError:
            print('data table was empty, ending script')
            return


        alt_title = str(
            input(f"Would you like to assign an alternate title to the graph for wells {pair[0]}- {pair[-1]} (y/n): ")
        )

        wells_str = [
            "" + str(well) for well in wells
            ][0]
        if "y" in alt_title:
            title = input("Enter: title: ")
        else:
            title = wells_str

        fig = graph.time_in_mitosis(
            df = data_table,
            x = "Intensity in Mitosis",
            y = 'Duration in Mitosis',
            bin = True,
            alt_xlabel = "mCherry Flourescence (abt. units)",
            alt_ylabel = "Time in Mitosis (mins.)",
            title = title
        )


        fig.savefig(f"{pair[-1]}/{wells_str}_time_in_mitosis.png")


if __name__ == "__main__":
    main()
