import re
import cv2
import numpy as np
from PIL import Image
import tifffile as tiff
from skimage.measure import regionprops, regionprops_table, label
from skimage.morphology import white_tophat, square, disk, erosion
from skimage.segmentation import clear_border
from skimage.filters import (
    gaussian,
    threshold_isodata,
)  # pylint: disable=no-name-in-module


def preprocess_2d(image, threshold_division, sigma, strel_cell=square(71)):
    """
    Preprocesses a specified image
    ------------------------------
    INPUTS:
        image: n-darray
        strel_cell: n-darray, structuring element for white_tophat
        threshold_division: float or int
        sigma: float or int
    OUTPUTS:
        redseg: n-darray, segmented targetstack
        labels: n-darray, labeled redseg
    """

    im = gaussian(image, sigma)  # 2D gaussian smoothing filter to reduce noise
    im = white_tophat(
        im, strel_cell
    )  # Background subtraction + uneven illumination correction
    thresh_im = threshold_isodata(im) 
    redseg = im > (
        thresh_im / threshold_division
    )  # only keep pixels above the threshold
    lblred = label(redseg)
    labels = label(lblred)

    return labels, redseg


def preprocess_3d(targetstack, threshold_division, sigma, strel_cell=square(71)):
    """
    Preprocesses a stack of images
    ------------------------------
    INPUTS:
        targetstach: n-darray, stack of (n x n) images
        strel_cell: n-darray, structuring element for white_tophat
        threshold_division: float or int
        sigma: float or int
    OUTPUTS:
        region_props: skimage object, region properties for each cell in each stack of a given image, can be indexed as 'region_props['Frame_i']'
    """

    region_props = {}

    for i in range(targetstack.shape[0]):
        im = targetstack[i, :, :].copy()
        im = gaussian(im, sigma=3)  # 2D gaussian smoothing filter to reduce noise
        im = white_tophat(
            im, strel_cell
        )  # Background subtraction + uneven illumination correction
        thresh_im = threshold_isodata(im)
        redseg = im > (thresh_im / threshold_division)  # only keep pixels above the threshold
        lblred = label(redseg)

        labels = label(lblred)
        region_props[f"Frame_{i}"] = regionprops(labels)

    return region_props


def bw_to_rgb(image, max_pixel_value=255, min_pixel_value=0):
    """
    Converts a tiffile of shape (x, y) to a file of shape (3, x, y) where each (x, y) frame of the first dimension corresponds to a color
    --------------------------------------------------------------------------------------------------------------------------------------
    INPUTS:
        image: n-darray, an image of shape (x, y)
        max_pixel_value: int, the maximum desired pixel value for the output array
        min_pixel_value: int, the minimum desired pixel value for the output array
    """
    if len(np.array(image).shape) == 2:
        image = cv2.normalize(
            np.array(image),
            None,
            max_pixel_value,
            min_pixel_value,
            cv2.NORM_MINMAX,
            cv2.CV_8U,
        )
        rgb_image = np.zeros((image.shape[0], image.shape[1], 3), "uint8")
        rgb_image[:, :, 0] = image
        rgb_image[:, :, 1] = image
        rgb_image[:, :, 2] = image

    return rgb_image


def get_box_size(region_props, scaling_factor: float):
    """
    Given a skimage region props object from a flouresence microscopy image, computes the bounding box size to be used in crop_regions or crop_regions_predict
    -----------------------------------------------------------------------------------------------------------------------------------------------------------
    INPUTS:
            region_props: skimage object, each index represents a grouping of properties about a given cell
            scaling factor: float,  the average area of a cell divided by the average area of a nuclei
                            If an ideal bb_side_length is known compute the scaling factor with the equation: scaling_factor = l^2 / A
                            Where l is your ideal bb_side_length and A is the mean or median area of a nuclei
    OUTPUTS:
            half the side length of a bounding box
    """
    areas = []
    for i in range(len(region_props)):
        areas.append(region_props[i].area)

    dna_area = np.median(np.array(areas))
    phase_area = scaling_factor * dna_area
    bb_side_length = np.sqrt(phase_area)

    return bb_side_length // 2


def crop_regions(dna_image_stack, threshold_division, sigma):
    """
    Given a stack of flouresence microscopy images, D, and corresponding phase images, P, returns regions cropped from D for each cell
    -----------------------------------------------------------------------------------------------------------------------------------
    INPUTS:
           dna_image_stack: n-darray, an array of shape (frame_count, x, y) where each (x, y) frame in the first dimension corresponds to one image
           box_size: 1/2 the side length of boxes to be cropped from the input image
           threshold_division: float or int
            sigma: float or int

    OUTPUTS:
            dna_regions: list, rank 4 tensor of cropped roi's which can be indexed as dna_regions[mu][nu] where mu is the frame number and nu is the cell number
            discarded_box_counter: n-darray, vector of integers corresponding to the number of roi's that had to be discarded due to 'incomplete' bounding boxes
            i.e. spilling out of the image. can be indexed as discarded_box_counter[mu] where mu is the frame number
            image_region_props: skimage object, region properties for each frame as computed by skimage
    """

    image_region_props = preprocess_3d(dna_image_stack, threshold_division, sigma)
    discarded_box_counter = np.array([])
    dna_regions = []

    for i in range(len(list(image_region_props))):
        frame_props = image_region_props[f"Frame_{i}"]
        box_size = get_box_size(image_region_props, scaling_factor= (5 * np.pi)/3)
        dna_regions_temp = []
        discarded_box_counter = np.append(discarded_box_counter, 0)

        for j in range(len(image_region_props[f"Frame_{i}"])):
            y, x = frame_props[j].centroid

            x1, y1 = x - box_size, y + box_size  # top left
            x2, y2 = x + box_size, y - box_size  # bottom right

            coords_temp = [x1, y1, x2, y2]

            if all(k >= 0 and k <= 2048 for k in coords_temp) == True:
                image = Image.fromarray(dna_image_stack[i, :, :])
                dna_region = np.array(image.crop((x1, y2, x2, y1)))
                dna_regions_temp.append(dna_region)

            else:
                discarded_box_counter[i] += 1
        dna_regions.append(dna_regions_temp)

    dna_regions = np.array(dna_regions, dtype=object)

    return dna_regions, discarded_box_counter, image_region_props


def crop_regions_predict(dna_image_stack, phase_image_stack, predictor, threshold_division, sigma):
    """
    Given a stack of flouresence microscopy images, D, and corresponding phase images, P, returns regions cropped from D and masks from P, for each cell
    ------------------------------------------------------------------------------------------------------------------------------------------------------
    INPUTS:
           dna_image_stack: n-darray, an array of shape (frame_count, x, y) where each (x, y) frame in the first dimension corresponds to one image
           phase_image_stack: n-darray, an array of shape (frame_count, x, y) where each (x, y) frame in the first dimension corresponds to one image
           box_size: 1/2 the side length of boxes to be cropped from the input image
           predictor: SAM, predicitive algorithm for segmenting cells
           threshold_division: float or int
            sigma: float or int


    OUTPUTS:
            dna_regions: list, rank 4 tensor of cropped roi's which can be indexed as dna_regions[mu][nu] where mu is the frame number and nu is the cell number
            discarded_box_counter: n-darray, vector of integers corresponding to the number of roi's that had to be discarded due to 'incomplete' bounding boxes
            i.e. spilling out of the image. can be indexed as discarded_box_counter[mu] where mu is the frame number
            image_region_props: skimage object, region properties for each frame as computed by skimage
            segmentations: rank 4 tensor containing one mask per cell per frame. It can be indexed as segmentations[mu][nu] where mu is the frame number and nu is the cell number
                           Note: segmentations must converted back to masks in the following way
                                1) mask = np.unpackbits(instance.segmentations[1][i], axis = 0, count = 2048)
                                2) mask = np.array([mask])
    """
    try:
        assert dna_image_stack.shape[0] == phase_image_stack.shape[0]
    except Exception as error:
        raise AssertionError(
            "there must be the same number of frames in the dna image and the corresponding phase image"
        ) from error

    sam_current_image = None
    discarded_box_counter = np.array([])
    dna_regions = []
    segmentations = []
    dna_image_region_props = preprocess_3d(dna_image_stack, threshold_division, sigma)

    for i in range(len(list(dna_image_region_props))):
        frame_props = dna_image_region_props[f"Frame_{i}"]
        box_size = get_box_size(frame_props, scaling_factor= (5 * np.pi)/3)
        dna_regions_temp = []
        segmentations_temp = []
        discarded_box_counter = np.append(discarded_box_counter, 0)
        sam_current_image = i
        sam_previous_image = None

        for j in range(len(dna_image_region_props[f"Frame_{i}"])):
            y, x = frame_props[j].centroid

            x1, y1 = x - box_size, y + box_size  # top left
            x2, y2 = x + box_size, y - box_size  # bottom right

            coords_temp = [x1, y1, x2, y2]
            phase_coords = [x1, y2, x2, y1]

            if all(k >= 0 and k <= 2048 for k in coords_temp) == True:
                dna_image = Image.fromarray(dna_image_stack[i, :, :])
                dna_region = np.array(dna_image.crop((x1, y2, x2, y1)))
                dna_regions_temp.append(dna_region)

                if (
                    sam_current_image != sam_previous_image
                    or sam_previous_image == None
                ):
                    phase_image_rgb = bw_to_rgb(
                        phase_image_stack[sam_current_image, :, :]
                    )
                    predictor.set_image(phase_image_rgb)
                    sam_previous_image = sam_current_image

                mask, __, __ = predictor.predict(
                    point_coords= None,
                    point_labels= None,
                    box=np.array(phase_coords),
                    multimask_output=False,
                )
                segmentations_temp.append(np.packbits(mask[0], axis=0))

            else:
                discarded_box_counter[i] += 1

        dna_regions.append(dna_regions_temp)
        segmentations.append(segmentations_temp)

    dna_regions = np.array(dna_regions, dtype=object)
    segmentations = np.array(segmentations, dtype=object)

    return dna_regions, discarded_box_counter, dna_image_region_props, segmentations


def counter(image_region_props, discarded_box_counter):
    """
    Counts the number of cells per frame and number of frames processed through either crop_regions or crop_regions_predict
    ------------------------------------------------------------------------------------------------------------------------
    INPUTS:
      image_region_props: dict, initial region props dictionary generated within the crop_regions function
      discarded_box_counter: vector of integers corresponding to the number of roi's that had to be discarded due to 'incomplete' bounding boxes
                             i.e. spilling out of the image. can be indexed as discarded_box_counter[mu] where mu is the frame number

    OUTPUTS:
      frame_count: int, number of frames in the original image stack
      cell_count: n-darray, vector containing the number of cropped cells in a given frame, it can be indexed as cell_count[mu] where mu is the frame number
    """

    frame_count = len(list(image_region_props))
    cell_count = []
    for i in range(frame_count):
        cell_count.append(
            int(len((image_region_props[f"Frame_{i}"])) - discarded_box_counter[i])
        )

    cell_count = np.array(cell_count)
    return frame_count, cell_count


def clean_regions(regions, frame_count, cell_count, threshold_division, sigma):
    """
    INPUTS:
          regions: must the output of 'crop_regions', is a dict containg all cropped regions
          region_props: must be the output of preprocess_3D, is only used in this function for the purpose of indexing
          discarded_box_counter: must be the output of 'crop_regions' is a dict containing the number of discared boxes per frame,
                                 is only used in this function for the purposes of indexing
          threshold_division: float or int
          sigma: float or int

    OUTPUTS:
           cleaned_regions: list, rank 4 tensor containing cleaned, binary DNA image ROIs, can be indexed as cleaned_regions[mu][nu] where mu represents the frame and nu represents the cell
           masks: list, rank 4 tensor containing masks of the central connected region in DNA image ROIs, can be indexed in the same manner as cleaned_regions
           cleaned_intensity_regions: list, rank 4 tensor containing cleaned, sclar valued DNA image ROIs, can be indexed in the same manner as cleaned_regions
    """
    masks = []
    cleaned_regions = []
    cleaned_intensity_regions = []

    for i in range(frame_count):
        masks_temp = []
        cleaned_regions_temp = []
        cleaned_intensity_regions_temp = []

        for j in range(int(cell_count[i])):
            mask = preprocess_2d(regions[i][j], threshold_division, sigma)[1]
            cleaned_mask = clear_border(mask)
            cleaned_intensity_regions_temp.append(
                np.multiply(regions[i][j], cleaned_mask)
            )
            cleaned_regions_temp.append(label(cleaned_mask))
            masks_temp.append(cleaned_mask)

        masks.append(masks_temp)
        cleaned_regions.append(cleaned_regions_temp)
        cleaned_intensity_regions.append(cleaned_intensity_regions_temp)

    masks = np.array(masks, dtype="object")
    cleaned_regions = np.array(cleaned_regions, dtype="object")
    cleaned_intensity_regions = np.array(cleaned_intensity_regions, dtype="object")

    return cleaned_regions, cleaned_intensity_regions, masks


def add_labels(data_frame, labels):
    if len(labels.shape) == len(data_frame.shape):
        if labels.shape[0] == data_frame.shape[0]:
            data_frame = np.append(data_frame, labels, axis=1)
    else:
        data_frame = np.append(
            data_frame, np.reshape(labels, (data_frame.shape[0], 1)), axis=1
        )

    return data_frame


class ROI:
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

    def __str__(self):
        return "Instance of class, Processor, implemented to process microscopy images into regions of interest"

    @classmethod
    def get(cls, props_list, dna_image_list, phase_image_list, frame_step=1):
        dna_image_stack = tiff.imread(dna_image_list[0])[0::frame_step, :, :]
        phase_image_stack = tiff.imread(phase_image_list[0])[0::frame_step, :, :]
        if len(dna_image_list) > 1:
            for i in range(len(dna_image_list) - 1):
                dna_image_stack = np.concatenate(
                    (
                        dna_image_stack,
                        tiff.imread(dna_image_list[i + 1])[0::frame_step, :, :],
                    ),
                    axis=0,
                )

        if len(phase_image_list) > 1:
            for i in range(len(phase_image_list) - 1):
                phase_image_stack = np.concatenate(
                    (
                        phase_image_stack,
                        tiff.imread(phase_image_list[i + 1])[0::frame_step, :, :],
                    ),
                    axis=0,
                )

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
        if segment == True and predictor != None:
            (
                self.roi,
                self.discarded_box_counter,
                region_props_stack,
                self.segmentations,
            ) = crop_regions_predict(
                self.dna_image_stack, self.phase_image_stack, predictor, threshold_division, sigma
            )

        else:
            (
                self.roi,
                self.discarded_box_counter,
                region_props_stack,
                self.coords,
            ) = crop_regions(self.dna_image_list, self.dna_image_stack, threshold_division, sigma)

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
