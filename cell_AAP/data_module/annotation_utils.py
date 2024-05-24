import cv2
import numpy as np
import torch
from PIL import Image
from skimage.measure import regionprops, label
from skimage.morphology import white_tophat, square, disk, erosion
from skimage.segmentation import clear_border
from skimage.filters import (
    gaussian,
    threshold_isodata,
)  # pylint: disable=no-name-in-module
from typing import Optional


def preprocess_2d(image, threshold_division : float, sigma : float, strel_cell=square(71)):
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


def preprocess_3d(targetstack, threshold_division : float, sigma : float, strel_cell=square(71)):
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
        im = gaussian(im, sigma)  # 2D gaussian smoothing filter to reduce noise
        im = white_tophat(
            im, strel_cell
        )  # Background subtraction + uneven illumination correction
        thresh_im = threshold_isodata(im)
        im = erosion(im, disk(8))
        redseg = im > (
            thresh_im / (threshold_division + 0.25)
        )  # only keep pixels above the threshold
        lblred = label(redseg)

        labels = label(lblred)
        region_props[f"Frame_{i}"] = regionprops(labels)

    return region_props


def bw_to_rgb(image, max_pixel_value:Optional[int]=255, min_pixel_value:Optional[int]=0):
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
    major_axis = []
    for i in range(len(region_props)):
        major_axis.append(region_props[i].axis_major_length)

    dna_major_axis = np.median(np.array(major_axis))
    bb_side_length = scaling_factor * dna_major_axis
   
    return bb_side_length // 2


def iou_with_list(input_box: list, boxes_list: list[list]):
    ious = []
    for box in boxes_list:

        intersection_width = min(input_box[2], box[2]) - max(input_box[0], box[0])
        intersection_height = min(input_box[1], box[1]) - max(input_box[3], box[3])

        if intersection_width == 0 and intersection_height == 0:
            ious.append(1)
        elif intersection_width <= 0 or intersection_height <= 0:
            ious.append(0)
        else:
            intersection_area = intersection_width * intersection_height
            box1_area = (input_box[2] - input_box[0]) * (input_box[1] - input_box[3])
            box2_area = (box[2] - box[0]) * ((box[1] - box[3]))
            union_area = box1_area + box2_area - intersection_area
    
            ious.append(intersection_area / (union_area + np.finfo("float").eps))
    return ious


def predict(
    predictor,
    image,
    boxes: Optional[list[list]] = None,
    points: Optional[list] = None,
    box_prompts=False,
    point_prompts=True,
):
    segmentations = []
    if box_prompts == True:

        try:
            assert boxes != None
        except Exception as error:
            raise AssertionError(
                "Must provide input bounding boxes if box_propmts = True has been selected"
            ) from error

        input_boxes = torch.tensor(boxes, device=predictor.device)
        transformed_boxes = predictor.transform.apply_boxes_torch(
            input_boxes, image.shape[:2]
        )
        masks, _, _ = predictor.predict_torch(
            point_coords=None,
            point_labels=None,
            boxes=transformed_boxes,
            multimask_output=False,
        )
        masks = masks.detach().cpu().numpy()

    elif point_prompts == True:

        try:
            assert points != None
        except Exception as error:
            raise AssertionError(
                "Failed to provide input centroids, please select box_prompts = True if attempeting to provide bouding box prompts"
            ) from error
        masks, _, _ = predictor.predict(
            point_coords=np.array([points]),
            point_labels=np.array([1]),
            box = None,
            multimask_output=False,
        )

    if len(masks.shape) == 4:
        for h in range(masks.shape[0]):
            packed_mask = np.packbits(masks[h, 0, :, :], axis=0)
            segmentations.append(packed_mask)
    else:
        packed_mask = np.packbits(masks[0, :, :], axis=0)
        segmentations.append(packed_mask)

    return segmentations


def crop_regions_predict(
    dna_image_stack,
    phase_image_stack,
    predictor,
    threshold_division : float,
    sigma : float,
    point_prompts: bool = True,
    box_prompts: bool = False,
    to_segment: bool =True,
):
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
    batch_size = 50
    discarded_box_counter = np.array([])
    dna_regions = []
    segmentations = []
    boxes = []
    dna_image_region_props = preprocess_3d(dna_image_stack, threshold_division, sigma)

    for i in range(len(list(dna_image_region_props))):
        frame_props = dna_image_region_props[f"Frame_{i}"]
        box_size = get_box_size(frame_props, scaling_factor=2.5)
        dna_regions_temp = []
        segmentations_temp = []
        discarded_box_counter = np.append(discarded_box_counter, 0)
        sam_current_image = i
        sam_previous_image = None

        for j in range(len(dna_image_region_props[f"Frame_{i}"])):
            y, x = frame_props[j].centroid

            x1, y1 = x - box_size, y + box_size  # top left
            x2, y2 = x + box_size, y - box_size  # bottom right

            coords_temp = [x1, y2, x2, y1]

            if (
                all(k >= 0 and k <= 2048 for k in coords_temp)
                and any(iou_with_list(coords_temp, boxes)) < 0.8 #experimental iou thresholding
            ):
                dna_image = Image.fromarray(dna_image_stack[i, :, :])
                dna_region = np.array(dna_image.crop((x1, y2, x2, y1)))
                dna_regions_temp.append(dna_region)
                boxes.append(coords_temp)

                if to_segment == True:
                    if (
                        sam_current_image != sam_previous_image
                        or sam_previous_image == None
                    ):
                        phase_image_rgb = bw_to_rgb(
                            phase_image_stack[sam_current_image, :, :]
                        )
                        predictor.set_image(phase_image_rgb)
                        sam_previous_image = sam_current_image

                    if box_prompts == True:
                        if len(boxes) == batch_size or (j + 1) == len(
                            dna_image_region_props[f"Frame_{i}"]
                        ):
                            segmentations_temp = predict(
                                predictor, phase_image_rgb, boxes = boxes, box_prompts=True
                            )
                            boxes = []
                    elif point_prompts == True:
                        points = [x, y]
                        segmentations_temp = predict(
                            predictor, phase_image_rgb, points = points, point_prompts=True
                        )
                else:
                    pass

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


def clean_regions(regions, frame_count, cell_count, threshold_division : float, sigma: float):
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
    """
    Adds labels to a dataframe in the labels and dataframe are of the same dimension and have the same number of rows
    ------------------------------------------------------------------------------------------------------------------
    INPUTS:
            data_frame: n-darray
            labels: n-darray
    OUTPUTS:
            data-frame: n-darray of 1 extra coloumn as compared to the input
    """
    if len(labels.shape) == len(data_frame.shape):
        if labels.shape[0] == data_frame.shape[0]:
            data_frame = np.append(data_frame, labels, axis=1)
    else:
        data_frame = np.append(
            data_frame, np.reshape(labels, (data_frame.shape[0], 1)), axis=1
        )

    return data_frame
