import re
import numpy as np
import cv2
import tifffile as tiff
import gc
from skimage.measure import regionprops_table
from annotation_utils import *
from typing import Optional
from cell_AAP import configs #type: ignore


class Annotator:
    def __init__(
        self,
        dna_image_list,
        dna_image_stack,
        phase_image_list,
        phase_image_stack,
        configs:configs.Cfg,
    ):
        self.dna_image_list = dna_image_list
        self.dna_image_stack = dna_image_stack
        self.phase_image_list = phase_image_list
        self.phase_image_stack = phase_image_stack
        self.configs = configs
        self.cell_count = None
        self.cleaned_binary_roi = self.cleaned_scalar_roi = None
        self.roi = self.labels = self.coords = self.segmentations = None
        self.cropped = False
        self.df_generated = False
        self.to_segment = True

    def __str__(self):
        return "Instance of class, Processor, implemented to process microscopy images into regions of interest"

    @classmethod
    def get(cls, configs:configs.Cfg, dna_image_list:list[str], phase_image_list:list[str]):

        try:
            assert len(dna_image_list) == len(phase_image_list)
        except Exception as error:
            raise AssertionError(
                "dna_image_list and phase_image_list must be of the same length (number of files)"
            ) from error
        
        if len(dna_image_list) > 1:
            if (re.search(r"^.+\.(?:(?:[tT][iI][fF][fF]?)|(?:[tT][iI][fF]))$", str(dna_image_list[0]))== None):
                dna_image_stack = [
                    cv2.imread(str(dna_image_list[i]), cv2.IMREAD_GRAYSCALE) for i, _ in enumerate(dna_image_list)
                ]
                phase_image_stack = [
                    cv2.imread(str(phase_image_list[i]), cv2.IMREAD_GRAYSCALE) for i, _ in enumerate(phase_image_list)
                ]
            else:
                dna_image_stack = [
                    tiff.imread(dna_image_list[i]) for i,_ in enumerate(dna_image_list)
                ]
                phase_image_stack = [
                    tiff.imread(phase_image_list[i]) for i,_ in enumerate(phase_image_list)
                ]
                
                # Check for multi-frame TIFF images in the list
                for i, (dna_img, phase_img) in enumerate(zip(dna_image_stack, phase_image_stack)):
                    if len(dna_img.shape) == 3 or len(phase_img.shape) == 3:
                        raise ValueError(
                            f"Multi-frame TIFF image detected at index {i}: {dna_img.shape}. "
                            "Time-series data is not supported as it can lead to cell duplication in datasets. "
                            "Please provide single-frame images or extract individual frames from your time-series data."
                        )
            
            # Stack multiple images into a single 3D array
            dna_image_stack = np.stack(dna_image_stack, axis=0)
            phase_image_stack = np.stack(phase_image_stack, axis=0)
                     
        else:
            if (re.search(r"^.+\.(?:(?:[tT][iI][fF][fF]?)|(?:[tT][iI][fF]))$", str(dna_image_list[0]))== None):
                # Non-TIFF files (JPEG, PNG, etc.) - cv2.imread always returns 2D arrays
                dna_image_stack = cv2.imread(str(dna_image_list[0]), cv2.IMREAD_GRAYSCALE)
                phase_image_stack = cv2.imread(str(phase_image_list[0]), cv2.IMREAD_GRAYSCALE)
                                
                # Convert to 3D stack format
                dna_image_stack = np.expand_dims(dna_image_stack, axis=0)
                phase_image_stack = np.expand_dims(phase_image_stack, axis=0)
            else:
                # TIFF files - can be 2D or 3D
                dna_image_stack = tiff.imread(dna_image_list[0])
                phase_image_stack = tiff.imread(phase_image_list[0])
                
                # Handle single 2D TIFF images
                if len(dna_image_stack.shape) == 2 and len(phase_image_stack.shape) == 2:
                    dna_image_stack = np.expand_dims(dna_image_stack, axis=0)
                    phase_image_stack = np.expand_dims(phase_image_stack, axis=0)
                    
                # Multi-frame TIFF images are not supported due to cell duplication concerns
                else:
                    if len(dna_image_stack.shape) == 3 and len(phase_image_stack.shape) == 3:
                        raise ValueError(
                            f"Multi-frame TIFF image detected: DNA shape {dna_image_stack.shape}, Phase shape {phase_image_stack.shape}. "
                            "Time-series data is not supported as it can lead to cell duplication in datasets. "
                            "Please provide single-frame images or extract individual frames from your time-series data."
                        )
                    else:
                        raise ValueError(
                            f"Invalid image format: DNA shape {dna_image_stack.shape}, Phase shape {phase_image_stack.shape}. "
                            "Please provide lists of 2-D images."
                        )

        return cls(
            dna_image_list,
            dna_image_stack,
            phase_image_list,
            phase_image_stack,
            configs,
        )

    @property
    def dna_image_list(self):
        return self._dna_image_list

    @dna_image_list.setter
    def dna_image_list(self, dna_image_list):
        self._dna_image_list = dna_image_list

    @property
    def dna_image_stack(self):
        return self._dna_image_stack

    @dna_image_stack.setter
    def dna_image_stack(self, dna_image_stack):
        self._dna_image_stack = dna_image_stack

    def crop(self, predictor=None):
        if predictor == None:
            self.to_segment == False
        (
            self.roi,
            self.discarded_box_counter,
            self.segmentations,
            self.phs_roi,
            self.cleaned_binary_roi
        ) = crop_regions_predict(
            self.dna_image_stack,
            self.phase_image_stack,
            predictor,
            self.configs.threshold_division,
            self.configs.gaussian_sigma,
            self.configs.erosionstruct, 
            self.configs.tophatstruct,
            self.configs.box_size,
            self.configs.point_prompts,
            self.configs.box_prompts,
            self.to_segment,
            self.configs.threshold_type,
            self.configs.iou_thresh
        )


        self.cleaned_scalar_roi = []

        for i in range(self.cleaned_binary_roi.shape[0]):
            cleaned_scalar_roi_temp = []

            for binary_mask, intensity in zip(self.cleaned_binary_roi[i], self.roi[i]):
                cleaned_scalar_roi = binary_mask * intensity
                cleaned_scalar_roi_temp.append(cleaned_scalar_roi)

            self.cleaned_scalar_roi.append(cleaned_scalar_roi_temp)

        self.cleaned_scalar_roi = np.asarray(self.cleaned_scalar_roi, dtype='object')

        # Clear predictor memory after processing
        if predictor is not None:
            try:
                predictor.reset_image()
            except:
                pass
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        
        # Force garbage collection
        gc.collect()

        self.cropped = True
        return self


    def gen_df(self, extra_props):
        """
        Given a dictionary of ROI's, this function will generate a dataframe containing values of selected skimage properties, one per ROI.
        -----------------------------------------------------------------------------------------------------------------------------------
        INPUTS:
            props_list: a list of all the properties (that can be generated from boolean masks) wished to be included in the final dataframe
            intense_props_list: a list of all the properties (that can be generated from scalar values images) wished to be included in the final dataframe
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
            assert isinstance(self.configs.propslist, list)
        except Exception as error:
            raise AssertionError("props_list must be of type 'list'") from error


        main_df = []

        for i in range(self.cleaned_binary_roi.shape[0]):
            for j, region in enumerate(self.cleaned_binary_roi[i]):
                if region.any() != 0:
                    props = regionprops_table(
                        region.astype("uint8"),
                        intensity_image=self.cleaned_scalar_roi[i][j],
                        properties=self.configs.propslist,
                        extra_properties=extra_props,
                    )

                    df = np.asarray(list(props.values())).T[0]
                    tracker = [i, j]
                    df = np.append(df, tracker)
                    main_df.append(df)

        return np.asarray(main_df)
