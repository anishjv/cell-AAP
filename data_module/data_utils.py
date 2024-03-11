import re
import cv2
import numpy as np
from PIL import Image
import tifffile as tiff
from skimage.measure import regionprops, regionprops_table, label
from skimage.morphology import white_tophat, square
from skimage.segmentation import clear_border
from skimage.filters import gaussian, threshold_isodata # pylint: disable=no-name-in-module

def preprocess_2d(
    image, strel_cell=square(71), nucsize_min=int(500), nucsize_max=int(2800), threshold_division =4
):
    """
    Preprocesses a specified image
    INPUTS:
        image
        strel_cell  = structuring element for white_tophat
    OUTPUTS:
        redseg      = segmented targetstack
        trackseg    = label stack filtered by size to remove non-nuclei
    """

    try:
        assert nucsize_min > 0 and nucsize_max > 0
    except Exception as error:
        raise AssertionError("Sizes must be > 0") from error

    if nucsize_min > nucsize_max:
        print("Swapping max and min size values!")
        nucsize_min, nucsize_max = nucsize_max, nucsize_min

    im = gaussian(image, sigma=3)  # 2D gaussian smoothing filter to reduce noise
    im = white_tophat(
        im, strel_cell
    )  # Background subtraction + uneven illumination correction
    thresh_im = threshold_isodata(im)  # find threshold value
    redseg = im > (thresh_im / threshold_division) # only keep pixels above the threshold
    lblred = label(redseg)
    labels = label(lblred)

    return labels, redseg


def preprocess_3d(
    targetstack, strel_cell=square(71), nucsize_min=int(500), nucsize_max=int(2800)
):
    """
    Preprocesses the specified plane from the targetstack
    INPUTS:
        targetstack = [t, x, y] image stack
        plane       = uint(plane number)
        strel_cell  = structuring element for white_tophat
    OUTPUTS:
        redseg      = segmented targetstack
        trackseg    = label stack filtered by size to remove non-nuclei
    """

    try:
        assert nucsize_min > 0 and nucsize_max > 0
    except Exception as error:
        raise AssertionError("Sizes must be > 0") from error

    if nucsize_min > nucsize_max:
        print("Swapping max and min size values!")
        nucsize_min, nucsize_max = nucsize_max, nucsize_min

    region_props = {}

    for i in range(targetstack.shape[0]):
        im = targetstack[i, :, :].copy()
        im = gaussian(im, sigma=3)  # 2D gaussian smoothing filter to reduce noise
        im = white_tophat(
            im, strel_cell
        )  # Background subtraction + uneven illumination correction
        thresh_im = threshold_isodata(im)  # find threshold value
        redseg = im > thresh_im  # only keep pixels above the threshold
        lblred = label(redseg)

        labels = label(lblred)
        region_props[f"Frame_{i}"] = regionprops(labels)

    return region_props


def bw_to_rgb(image, max_pixel_value = 255, min_pixel_value = 0):
    '''
    Converts a tiffile of shape (x, y) to a file of shape (3, x, y) where each (x, y) frame of the first dimension corresponds to a color
    INPUTS:
        image: n-darray, an image of shape (x, y)
        max_pixel_value: int, the maximum desired pixel value for the output array
        min_pixel_value: int, the minimum desired pixel value for the output array
    '''
    if len(np.array(image).shape) == 2:
        image = cv2.normalize(np.array(image), None, max_pixel_value, min_pixel_value, cv2.NORM_MINMAX, cv2.CV_8U)
        rgb_image = np.zeros((image.shape[0], image.shape[1], 3), 'uint8' ) 
        rgb_image[:,:,0] = image
        rgb_image[:,:,1] = image
        rgb_image[:,:,2] = image
        
    return rgb_image
    

def crop_regions(dna_image_list, dna_image_stack, box_size):
    """
    INPUTS:
           dna_image_list: list, a list containg the paths of all images to be processed 
           dna_image_stack: n-darray, an array of shape (frame_count, x, y) where each (x, y) frame in the first dimension corresponds to one image
           box_size: 1/2 the side length of boxes to be cropped from the input image

    OUTPUTS:
            regions: dict, dictionary of cropped roi's, labeled 'Frame_i_cell_j
            discarded_box_counter: dict, dictionary of integers, integer corresponds to the number of roi's that had to be discarded due to be 'incomplete',
            i.e. spilling out of the image
            image_region_props: skimage object, region properties for each frame as computed by skimage
            coords: dict, a dictionary specifying the locational coordinates (x_bottom_left, y_bottom_left) and (x_top_right, y_top_right) for each ROI
    """

    assert isinstance(box_size, int)
    assert isinstance(dna_image_list, list)

    pil_image_dict = {}

    for i in range(dna_image_stack.shape[0]):
        image_to_append = Image.fromarray(dna_image_stack[i, :, :])
        pil_image_dict[f"Frame_{i}"] = image_to_append

    image_region_props = preprocess_3d(dna_image_stack)

    regions = {}
    discarded_box_counter = {}
    coords = {}

    for i in range(len(list(image_region_props))):
        discarded_box_counter[f"Frame_{i}"] = 0

        for j in range(len(image_region_props[f"Frame_{i}"])):
            y, x = image_region_props[f"Frame_{i}"][j].centroid

            x1, y1 = x - box_size, y + box_size  # top left
            x2, y2 = x + box_size, y - box_size  # bottom right

            coords_temp = [x1, y1, x2, y2]

            if all(k >= 0 and k <= 2048 for k in coords_temp) == True:
                image = pil_image_dict[f"Frame_{i}"]
                region = np.array(image.crop((x1, y2, x2, y1)))
                regions[
                    f"Frame_{i}_cell_{j - discarded_box_counter[f'Frame_{i}']}"
                ] = region

                coords[
                    f"Frame_{i}_cell_{j - discarded_box_counter[f'Frame_{i}']}"
                ] = coords_temp
            else:
                discarded_box_counter[f"Frame_{i}"] += 1

    return regions, discarded_box_counter, image_region_props, coords




def crop_regions_predict(dna_image_stack, box_size, phase_image_stack, predictor):
    """
    INPUTS:
           dna_image_stack: n-darray, an array of shape (frame_count, x, y) where each (x, y) frame in the first dimension corresponds to one image
           phase_image_stack: n-darray, an array of shape (frame_count, x, y) where each (x, y) frame in the first dimension corresponds to one image
           box_size: 1/2 the side length of boxes to be cropped from the input image
           predictor: SAM, predicitive algorithm for segmenting cells


    OUTPUTS:
            dna_regions: dictionary of cropped roi's, labeled 'Frame_i_cell_j
            discarded_box_counter: dictionary of integers, integer corresponds to the number of roi's that had to be discarded due to be 'incomplete',
            i.e. spilling out of the image
            image_region_props: skimage object, region properties for each frame as computed by skimage
            coords: a dictionary specifying the locational coordinates (x_bottom_left, y_bottom_left) and (x_top_right, y_top_right) for each ROI
            segmentations: dict, dictionary containing one mask per cell per frame, keys are of the form 'Frame_i_cell_j'
    """

    assert isinstance(box_size, int)
    assert(dna_image_stack.shape[0] == phase_image_stack.shape[0])

    sam_current_image = None
    pil_dna_image_dict = {}
    pil_phase_image_dict = {}
    dna_regions = {}
    coords = {}
    discarded_box_counter = {}
    segmentations = {}

    for i in range(dna_image_stack.shape[0]):
        dna_image_to_append = Image.fromarray(dna_image_stack[i, :, :])
        pil_dna_image_dict[f"Frame_{i}"] = dna_image_to_append
        phase_image_to_append = Image.fromarray(phase_image_stack[i, :, :])
        pil_phase_image_dict[f"Frame_{i}"] = phase_image_to_append

    dna_image_region_props = preprocess_3d(dna_image_stack)


    for i in range(len(list(dna_image_region_props))):
        discarded_box_counter[f"Frame_{i}"] = 0
        sam_current_image = f'Frame_{i}'
        sam_previous_image = None
        
        for j in range(len(dna_image_region_props[f"Frame_{i}"])):
            y, x = dna_image_region_props[f"Frame_{i}"][j].centroid

            x1, y1 = x - box_size, y + box_size  # top left
            x2, y2 = x + box_size, y - box_size  # bottom right

            coords_temp = [x1, y1, x2, y2]
            phase_coords = [x1, y2, x2, y1]

            if all(k >= 0 and k <= 2048 for k in coords_temp) == True:
                dna_image = pil_dna_image_dict[f"Frame_{i}"]
                dna_region = np.array(dna_image.crop((x1, y2, x2, y1)))
                dna_regions[f"Frame_{i}_cell_{j - discarded_box_counter[f'Frame_{i}']}"] = dna_region
                coords[f"Frame_{i}_cell_{j - discarded_box_counter[f'Frame_{i}']}"] = coords_temp
         
                if sam_current_image != sam_previous_image or sam_previous_image == None:
                    phase_image_rgb = bw_to_rgb(pil_phase_image_dict[sam_current_image])
                    predictor.set_image(phase_image_rgb)
                    sam_previous_image = sam_current_image                    
                
                mask, __, __ = predictor.predict(point_coords=None,
                                                  point_labels= None,
                                                  box = np.array(phase_coords),
                                                  multimask_output=False,
                                                  )
                segmentations[f"Frame_{i}_cell_{j - discarded_box_counter[f'Frame_{i}']}"] = mask

            else:
                discarded_box_counter[f"Frame_{i}"] += 1
                

    return dna_regions, discarded_box_counter, dna_image_region_props, coords, segmentations


def counter(image_region_props, discarded_box_counter):
    """
    INPUTS:
      image_region_props: dict, initial region props dictionary generated within the crop_regions function
      discarded_box_counter: dict, number of ROI's discared per frame. Has keys of the form Frame_i and integer values.
                             ROI's may be discarded for exceeding the dimensions of the image from which they were cropped

    OUTPUTS:
      frame_count: int, number of frames in the original image stack
      cell_count_dict: dict, contains the number or cells per frame with keys in the form Frame_i and integer values
    """

    frame_count = len(list(image_region_props))

    cell_count_dict = {}
    for i in range(frame_count):
        cell_count_dict[f"Frame_{i}"] = (
            len((image_region_props[f"Frame_{i}"]))
            - discarded_box_counter[f"Frame_{i}"]
        )

    return frame_count, cell_count_dict


def clean_regions(regions, frame_count, cell_count):
    """
    INPUTS:
          regions: must the output of 'crop_regions', is a dict containg all cropped regions
          region_props: must be the output of preprocess_3D, is only used in this function for the purpose of indexing
          discarded_box_counter: must be the output of 'crop_regions' is a dict containing the number of discared boxes per frame,
                                 is only used in this function for the purposes of indexing

    OUTPUTS:
           cleaned_regions: a dict containing the cleaned regions, labeled 'Frame_i_cell_j' to indicate which frame of the original image they were cropped from
    """

    cleaned_intensity_regions = {}
    cleaned_regions = {}
    masks = {}

    for i in range(frame_count):
        for j in range(cell_count[f"Frame_{i}"]):
            mask = preprocess_2d(regions[f"Frame_{i}_cell_{j}"])[1]
            cleaned_mask = clear_border(mask)
            cleaned_intensity_regions[f"Frame_{i}_cell_{j}"] = np.multiply(
                regions[f"Frame_{i}_cell_{j}"], cleaned_mask
            )
            cleaned_regions[f"Frame_{i}_cell_{j}"] = label(cleaned_mask)
            masks[f'Frame_{i}_cell_{j}'] = cleaned_mask

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
    def __init__(self, dna_image_list, dna_image_stack, phase_image_list, phase_image_stack, roi_size, props_list, predictor):
        self.dna_image_list = dna_image_list
        self.dna_image_stack = dna_image_stack
        self.phase_image_list = phase_image_list
        self.phase_image_stack = phase_image_stack
        self.roi_size = roi_size
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
        self.predictor = predictor
        self.segmentations = None

    def __str__(self):
        return "Instance of class, Processor, implemented to process microscopy images into regions of interest"

    @classmethod
    def get(cls, props_list, dna_image_list, phase_image_list, roi_size, predictor, frame_step = 1):
        dna_image_stack = tiff.imread(dna_image_list[0])[0::frame_step, :, :]
        phase_image_stack = tiff.imread(phase_image_list[0])[0::frame_step, :, :]
        if len(dna_image_list) > 1:
            for i in range(len(dna_image_list) - 1):
                dna_image_stack = np.concatenate( (dna_image_stack, tiff.imread(dna_image_list[i + 1])[0::frame_step, :, :]), axis = 0)

        if len(phase_image_list) > 1:
            for i in range(len(phase_image_list) - 1):
                phase_image_stack = np.concatenate( (phase_image_stack, tiff.imread(phase_image_list[i + 1])[0::frame_step, :, :]), axis = 0)

        return cls(dna_image_list, dna_image_stack, phase_image_list, phase_image_stack, roi_size, props_list, predictor)

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

    def crop(self, segment = True):
        if segment == True:
            self.roi, self.discarded_box_counter, region_props_stack, self.coords, self.segmentations = crop_regions_predict(
                self.dna_image_stack, self.roi_size, self.phase_image_stack, self.predictor
            )
        
        else:
            self.roi, discarded_box_counter, region_props_stack, self.coords = crop_regions(
                self.dna_image_list, self.dna_image_stack, self.roi_size
            )  

        self.frame_count, self.cell_count = counter(
            region_props_stack, self.discarded_box_counter
        )
        self.cleaned_binary_roi, self.cleaned_scalar_roi, self.masks = clean_regions(
            self.roi,
            self.frame_count,
            self.cell_count,
        )
        self.cropped = True
        return self


    def gen_df(self):
        """
        INPUTS:
            props_list: a list of all the properties (that can be generated from boolean masks) wished to be included in the final dataframe
            intense_props_list: a list of all the properties (that can be generated from scalar values images) wished to be included in the final dataframe
            frame_count: an int with a value equal to the number of frames in the image stack of interest
            cell_count: a dict containing one key per frame of the image stack of interest, the value of each key is the number of cells on that frame
            cleaned_regions: a dict with one key per cell, the value of each key being an 84X84 boolean array representation of a cell. Each key is labeled, 'Frame_i_cell_j'
            cleaned_intense_regions: a dict with one key per cell, the value of each key being an 84X84 scalar values array representation of a cell. Each key is labeled 'Frame_i_cell_j'

        OUTPUTS:
            main_df: a pandas dataframe containing the values for each property for each cell in 'cleaned_regions'. The dataframe stores no knowledge of the frame from which a cell came.

        SUMMARY:
            Given a dictionary of ROI's, this function will generate a dataframe containing values of selected skimage properties, one per ROI.

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
            assert isinstance(self.cell_count, dict)
        except Exception as error:
            raise AssertionError(
                "cell_count must of type 'dict', each entry in the dictionary should contain the number of cells in a corresponding frame"
            ) from error
        try:
            assert len(self.cell_count) == self.frame_count
        except Exception as error:
            raise AssertionError(
                "cell_count must contain the same number of frames as specified by frame_count"
            ) from error
        try:
            assert self.props_list[0] == "area"
        except Exception as error:
            raise AssertionError(
                "area must be the first element of props_list"
            ) from error

        main_df = np.empty(shape=(0, len(self.props_list) + 3))

        for i in range(self.frame_count):
            for j in range(self.cell_count[f"Frame_{i}"]):
                props = regionprops_table(
                    self.cleaned_binary_roi[f"Frame_{i}_cell_{j}"],
                    intensity_image=self.cleaned_scalar_roi[f"Frame_{i}_cell_{j}"],
                    properties=self.props_list,
                )

                df = np.array(list(props.values())).T
                if df.shape == (1, 15):
                    tracker = [[i, j]]
                    df = np.append(df, tracker, axis=1)
                    main_df = np.append(main_df, df, axis=0)
                else:
                    pass

        return main_df

