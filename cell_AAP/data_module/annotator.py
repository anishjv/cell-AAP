import re
import numpy as np
import tifffile as tiff
from skimage.measure import regionprops_table
from annotation_utils import *


class Annotator:
    def __init__(
        self,
        dna_image_list,
        dna_image_stack,
        phase_image_list,
        phase_image_stack,
        props_list,
    ):
        self.dna_image_list = dna_image_list
        self.dna_image_stack = dna_image_stack
        self.phase_image_list = phase_image_list
        self.phase_image_stack = phase_image_stack
        self.props_list = props_list
        self.point_prompts = True
        self.box_prompts = False
        self.frame_count = None
        self.cell_count = None
        self.cleaned_binary_roi = None
        self.cleaned_scalar_roi = None
        self.masks = None
        self.roi = None
        self.labels = None
        self.coords = None
        self.cropped = False
        self.df_generated = False
        self.segmentations = None
        self.to_segment = True

    def __str__(self):
        return "Instance of class, Processor, implemented to process microscopy images into regions of interest"

    @classmethod
    def get(cls, props_list:list[str], dna_image_list:list[str], phase_image_list:list[str], frame_step:Optional[int]=1):

        try:
            assert len(dna_image_list) == len(phase_image_list)
        except Exception as error:
            raise AssertionError(
                "dna_image_list and phase_image_list must be of the same length (number of files)"
            ) from error

        if len(dna_image_list) > 1:
            dna_image_tensor = tiff.imread(dna_image_list)
            phase_image_tensor = tiff.imread(phase_image_list)
            dna_tup = ()
            phase_tup = ()
            if dna_image_tensor.shape[0] == phase_image_tensor.shape[0]:
                for i in range(dna_image_tensor.shape[0]):
                    dna_tup = dna_tup + dna_image_tensor[i][0::frame_step, :, :]
                    phase_tup = phase_tup + phase_image_tensor[i][0::frame_step, :, :]

            dna_image_stack = np.concatenate(dna_tup, axis=0)
            phase_image_stack = np.concatenate(phase_tup, axis=0)

        else:
            dna_image_stack = tiff.imread(dna_image_list[0])[0::frame_step, :, :]
            phase_image_stack = tiff.imread(phase_image_list[0])[0::frame_step, :, :]

        return cls(
            dna_image_list,
            dna_image_stack,
            phase_image_list,
            phase_image_stack,
            props_list,
        )

    @property
    def dna_image_list(self):
        return self._dna_image_list

    @dna_image_list.setter
    def dna_image_list(self, dna_image_list):
        for i in dna_image_list:
            if (
                re.search(r"^.+\.(?:(?:[tT][iI][fF][fF]?)|(?:[tT][iI][fF]))$", i)
                == None
            ):
                raise ValueError("Image must be a tiff file")
            else:
                pass
        self._dna_image_list = dna_image_list

    @property
    def dna_image_stack(self):
        return self._dna_image_stack

    @dna_image_stack.setter
    def dna_image_stack(self, dna_image_stack):
        self._dna_image_stack = dna_image_stack

    def crop(self, threshold_division, sigma, segment=True, predictor=None):
        if predictor == None:
            self.to_segment == False
        (
            self.roi,
            self.discarded_box_counter,
            region_props_stack,
            self.segmentations,
        ) = crop_regions_predict(
            self.dna_image_stack,
            self.phase_image_stack,
            predictor,
            threshold_division,
            sigma,
            self.point_prompts,
            self.box_prompts,
            self.to_segment,
        )

        self.frame_count, self.cell_count = counter(
            region_props_stack, self.discarded_box_counter
        )
        self.cleaned_binary_roi, self.cleaned_scalar_roi, self.masks = clean_regions(
            self.roi, self.frame_count, self.cell_count, threshold_division, sigma
        )
        self.cropped = True
        return self

    def gen_df(self, extra_props):
        """
        Given a dictionary of ROI's, this function will generate a dataframe containing values of selected skimage properties, one per ROI.
        -----------------------------------------------------------------------------------------------------------------------------------
        INPUTS:
            props_list: a list of all the properties (that can be generated from boolean masks) wished to be included in the final dataframe
            intense_props_list: a list of all the properties (that can be generated from scalar values images) wished to be included in the final dataframe
            frame_count: an int with a value equal to the number of frames in the image stack of interest
            cell_count: list, vector containing one coloumn per frame of the image stack of interest, the value of each key is the number of cells on that frame
            cleaned_regions: list, rank 4 tensor containing cleaned, binary DNA image ROIs, can be indexed as cleaned_regions[mu][nu] where mu represents the frame and nu represents the cell
            cleaned_intensity_regions: list, rank 4 tensor containing cleaned, sclar valued DNA image ROIs, can be indexed in the same manner as cleaned_regions

        OUTPUTS:
            main_df: a vectorized dataframe containing the values for each property for each cell in 'cleaned_regions'. The dataframe stores no knowledge of the frame from which a cell came.
        """
        try:
            assert self.cropped == True
        except Exception as error:
            raise AssertionError(
                "the method, crop(), must be called before the method gen_df()"
            )
        try:
            assert isinstance(self.props_list, list)
        except Exception as error:
            raise AssertionError("props_list must be of type 'list'") from error
        try:
            assert len(self.cell_count) == self.frame_count
        except Exception as error:
            raise AssertionError(
                "cell_count must contain the same number of frames as specified by frame_count"
            ) from error

        main_df = np.empty(shape=(0, len(self.props_list) + 3 + len(extra_props)))

        for i in range(self.frame_count):
            for j in range(self.cell_count[i]):
                if self.cleaned_binary_roi[i][j].any() != 0:
                    props = regionprops_table(
                        self.cleaned_binary_roi[i][j].astype("uint8"),
                        intensity_image=self.cleaned_scalar_roi[i][j],
                        properties=self.props_list,
                        extra_properties=extra_props,
                    )

                    df = np.array(list(props.values())).T
                    if df.shape == (1, len(self.props_list) + 1 + len(extra_props)):
                        tracker = [[i, j]]
                        df = np.append(df, tracker, axis=1)
                        main_df = np.append(main_df, df, axis=0)
                    else:
                        self.cell_count[i] -= 1
                        pass

                else:
                    self.cell_count[i] -= 1
                    pass

        return main_df
