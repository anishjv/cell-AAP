import cv2
import numpy as np
import numpy.typing as npt
import torch
from PIL import Image
import skimage
from skimage.measure import regionprops, label
from skimage.morphology import white_tophat, square, disk, erosion
from skimage.segmentation import clear_border
from skimage.filters import (
    gaussian,
    threshold_isodata,
    threshold_multiotsu,
)  # pylint: disable=no-name-in-module
from typing import Optional, Union
import scipy
import itertools


def preprocess_2d(
    image: npt.NDArray,
    threshold_division: float,
    sigma: float,
    threshold_type: str = "single",
    tophatstruct=square(71),
) -> tuple[npt.NDArray, npt.NDArray]:
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
        im, tophatstruct
    )  # Background subtraction + uneven illumination correction
    if threshold_type == "multi":
        thresholds = threshold_multiotsu(im)
        redseg = np.digitize(im, bins=thresholds)
    else:
        thresh = threshold_isodata(im)
        redseg = im > (thresh / threshold_division)
    lblred = label(redseg)
    labels = label(lblred)

    return labels, redseg


def preprocess_3d(
    targetstack: npt.NDArray,
    threshold_division: float,
    sigma: int,
    threshold_type: str,
    erosionstruct,
    tophatstruct,
) -> tuple[npt.NDArray, skimage.measure.regionprops]:
    """
    Preprocesses a stack of images
    ------------------------------
    INPUTS:
        targetstach: n-darray, stack of (n x n) images, i.e. a (z, n, n) dimensional asarray
        strel_cell: n-darray, structuring element for white_tophat
        threshold_division: float or int
        sigma: float or int
    OUTPUTS:
        region_props: skimage object, region properties for each cell in each stack of a given image, can be indexed as 'region_props['Frame_i']'
    """

    region_props = {}
    labels_whole = []

    for i in range(targetstack.shape[0]):
        im = targetstack[i, :, :].copy()
        im = gaussian(im, sigma)  # 2D gaussian smoothing filter to reduce noise
        im = white_tophat(
            im, tophatstruct
        )  # Background subtraction + uneven illumination correction

        im = erosion(im, erosionstruct)
        if threshold_type == "multi":
            thresholds = threshold_multiotsu(im)
            redseg = np.digitize(im, bins=thresholds)

        else:
            thresh = threshold_isodata(im)
            redseg = im > (thresh / threshold_division)

        lblred = label(redseg)
        labels = label(lblred)
        region_props[f"Frame_{i}"] = regionprops(labels, intensity_image=labels * im)
        labels_whole.append(labels)

    labels_whole = np.asarray(labels_whole)

    return labels_whole, region_props


def bw_to_rgb(
    image: npt.NDArray,
    max_pixel_value: Optional[int] = 255,
    min_pixel_value: Optional[int] = 0,
) -> npt.NDArray:
    """
    Converts a tiffile of shape (x, y) to a file of shape (3, x, y) where each (x, y) frame of the first dimension corresponds to a color
    --------------------------------------------------------------------------------------------------------------------------------------
    INPUTS:
        image: n-darray, an image of shape (x, y)
        max_pixel_value: int, the maximum desired pixel value for the output asarray
        min_pixel_value: int, the minimum desired pixel value for the output asarray
    """
    if len(np.asarray(image).shape) == 2:
        image = cv2.normalize(
            np.asarray(image),
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


def get_box_size(
    region_props: skimage.measure.regionprops, scaling_factor: float
) -> float:
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

    major_axis = [region_props[i].axis_major_length for i, _ in enumerate(region_props)]

    dna_major_axis = np.median(np.asarray(major_axis))
    bb_side_length = scaling_factor * dna_major_axis

    print("The bounding box side length was", bb_side_length, "pixels")
    return bb_side_length // 2


def get_box_size_scaled(region_props, max_size: float) -> list[float]:
    """
    Given a skimage region props object from a flouresence microscopy image, computes the bounding box size to be used in crop_regions or crop_regions_predict
    -----------------------------------------------------------------------------------------------------------------------------------------------------------
    INPUTS:
            region_props: skimage object, each index represents a grouping of properties about a given cell
            min_size: float, the approximate minimum cell size
    OUTPUTS:
            half the side length of a bounding box
    """

    major_axis = [region_props[i].axis_major_length for i, _ in enumerate(region_props)]
    intensity = [region_props[i].intensity for i, _ in enumerate(region_props)]

    std_intensity = np.std(intensity)
    std_major_axis = np.std(major_axis)
    mean_intensity = np.mean(intensity)
    mean_major_axis = np.mean(major_axis)

    bb_side_lengths = []
    for i, _ in enumerate(region_props):
        z_score = 0.5 * (
            (major_axis[i] - mean_major_axis) / std_major_axis
            + (intensity[i] - mean_intensity) / std_intensity
        )
        percentile = scipy.integrate.quad(
            lambda x: (1 / 2 * np.pi) * np.e ** (-(x**2) / 2), -np.inf, z_score
        )
        bb_side_lengths.append(max_size * percentile)

    print(np.asarray(bb_side_lengths))
    return np.asarray(bb_side_lengths) // 2


def square_box(centroid: npt.NDArray, box_size: float) -> npt.NDArray:
    """
    Draws an upright bounding box given a centroud and box size
    ------------------------------------------------------------

    INPUTS:
        centroid: ndarray
        box_size: Union(int, float)
    RETURNS:
        coords: ndarray
    """

    x, y = centroid[1], centroid[0]  # centroid must be of the form (y, x)
    x1, y1 = x - box_size, y + box_size  # top left
    x2, y2 = x + box_size, y - box_size  # bottom right

    return np.asarray([x1, y2, x2, y1])


def box_size_wrapper(func, frame_props, args):
    "Facillitates the usage of different box size determination functions"

    try:
        return func(frame_props, *args)
    except Exception as error:
        raise AttributeError("args do not match function") from error


def bbox_wrapper(
    func, centroid, box_size: Optional[float] = None, args: Optional[list] = None
):
    "Facillitates the usage of different box drawing determination functions"

    try:
        if box_size and args:
            return func(centroid, box_size, *args)
        elif box_size:
            return func(centroid, box_size)
        elif args:
            return func(centroid, *args)
        else:
            return func(centroid)
    except Exception as error:
        raise AttributeError("args do not match function") from error


def iou_with_list(
    input_list: list[npt.NDArray], iou_thresh: float
) -> tuple[list[npt.NDArray], list[int]]:
    """
    Computes the IOU of all combinations of arrays in a list and returns a new list with arrays for which iou > iou_thresh are removed
    -------------------------------------------------------------------------------------------------------------------------------------
    INPUTS:
        input_list: list[npt.NDArray]
        iou_thresh: float

    OUTPUTS:
        input_list: list[npt.NDArray]
    """
    poped_indices = []
    sorted_list = sorted(input_list, key=lambda x: np.count_nonzero(x == 1))

    combinations = list(itertools.combinations(sorted_list, 2))
    for combo in combinations:
        combo_1 = combo[0] > 0
        combo_2 = combo[1] > 0

        combo_1_area = np.count_nonzero(combo_1 == 1)
        combo_2_area = np.count_nonzero(combo_2 == 1)
        intersection = np.count_nonzero(np.logical_and(combo_1, combo_2))
        iou = intersection / (combo_1_area + combo_2_area - intersection)

        if iou >= iou_thresh:
            for i, _ in enumerate(input_list):
                if np.array_equal(input_list[i], combo[0]):
                    poped_indices.append(i)
    input_list = [seg for i, seg in enumerate(input_list) if i not in poped_indices]

    return input_list, poped_indices


def predict(
    predictor,
    image,
    boxes: Optional[list[list]] = None,
    points: Optional[list] = None,
    box_prompts=False,
    point_prompts=True,
) -> np.ndarray:
    """
    Implementation of FAIR's SAM using box or point prompts:
    --------------------------------------------------------
    """
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
                "Failed to provide input centroids, please select box_prompts = True if attempting to provide bouding box prompts"
            ) from error
        masks, _, _ = predictor.predict(
            point_coords=np.array([points]),
            point_labels=np.array([1]),
            box=None,
            multimask_output=False,
        )

    if len(masks.shape) == 4:
        for h in range(masks.shape[0]):
            packed_mask = np.packbits(masks[h, 0, :, :], axis=0)
            segmentations.append(packed_mask)

    else:
        segmentations = np.packbits(masks[0, :, :], axis=0)

    return segmentations


def crop_regions_predict(
    dna_image_stack,
    phase_image_stack,
    predictor,
    threshold_division: float,
    sigma: float,
    erosionstruct,
    tophatstruct,
    box_size: tuple,
    point_prompts: bool = True,
    box_prompts: bool = False,
    to_segment: bool = True,
    threshold_type: str = "single",
    iou_thresh: Optional[bool] = 0.85,
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

    try:
        assert box_prompts != point_prompts
    except Exception as error:
        raise AssertionError(
            "You must use only one of box prompts and point prompts"
        ) from error

    batch_size = 50
    discarded_box_counter = np.array([])
    dna_regions = []
    phs_regions = []
    segmentations = []
    boxes = []
    box_size_func = box_size[0]
    box_size_args = box_size[1]
    _, dna_image_region_props = preprocess_3d(
        dna_image_stack,
        threshold_division,
        sigma,
        threshold_type,
        erosionstruct,
        tophatstruct,
    )

    for i, _ in enumerate(dna_image_region_props):  # for each image

        frame_props = dna_image_region_props[f"Frame_{i}"]
        box_sizes = box_size_wrapper(box_size_func, frame_props, box_size_args)
        dna_regions_temp = []
        phs_regions_temp = []
        segmentations_temp = []
        discarded_box_counter = np.append(discarded_box_counter, 0)
        sam_current_image = i
        sam_previous_image = None

        for j, _ in enumerate(dna_image_region_props[f"Frame_{i}"]):  # for each cell

            y, x = frame_props[j].centroid
            if isinstance(box_sizes, list):
                box_sizes = box_sizes[j]

            x1, y1 = x - box_sizes, y + box_sizes  # top left
            x2, y2 = x + box_sizes, y - box_sizes  # bottom right

            coords_temp = [x1, y2, x2, y1]

            dna_image = Image.fromarray(dna_image_stack[i, :, :])
            dna_region = np.asarray(dna_image.crop((x1, y2, x2, y1)))
            dna_regions_temp.append(dna_region)
            phs_image = Image.fromarray(phase_image_stack[i, :, :])
            phs_region = np.asarray(phs_image.crop((x1, y2, x2, y1)))
            phs_regions_temp.append(phs_region)

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

                    if all(
                        k >= 0 and k <= dna_image_stack.shape[1] for k in coords_temp
                    ):
                        boxes.append(coords_temp)
                    else:
                        discarded_box_counter[i] += 1

                    if len(boxes) == batch_size or (j + 1) == len(
                        dna_image_region_props[f"Frame_{i}"]
                    ):
                        masks = predict(
                            predictor,
                            phase_image_rgb,
                            boxes=boxes,
                            box_prompts=True,
                        )

                        for l in range(masks.shape[0]):
                            segmentations_temp.append(masks[l])
                        boxes = []

                elif point_prompts == True:
                    points = [x, y]
                    mask = predict(
                        predictor,
                        phase_image_rgb,
                        points=points,
                        point_prompts=True,
                    )
                    segmentations_temp.append(mask)

        segmentations_temp, poped_indices = iou_with_list(
            segmentations_temp, iou_thresh
        )
        segmentations.append(segmentations_temp)
        dna_regions_temp = [
            roi for i, roi in enumerate(dna_regions_temp) if i not in poped_indices
        ]
        phs_regions_temp = [
            roi for i, roi in enumerate(phs_regions_temp) if i not in poped_indices
        ]
        discarded_box_counter[i] += len(poped_indices)
        dna_regions.append(dna_regions_temp)
        phs_regions.append(phs_regions_temp)

    dna_regions = np.asarray(dna_regions, dtype=object)
    phs_regions = np.asarray(phs_regions, dtype=object)
    segmentations = np.asarray(segmentations, dtype=object)

    return (
        dna_regions,
        discarded_box_counter,
        dna_image_region_props,
        segmentations,
        phs_regions,
    )


def counter(
    image_region_props: skimage.measure.regionprops, discarded_box_counter: npt.NDArray
) -> tuple[float, npt.NDArray]:
    """
    Counts the number of cells per frame and number of frames processed through either crop_regions or crop_regions_predict
    ------------------------------------------------------------------------------------------------------------------------
    INPUTS:
      image_region_props: skimage.measure.region_rops, initial region props dictionary generated within the crop_regions function
      discarded_box_counter: vector of integers corresponding to the number of roi's that had to be discarded due to 'incomplete' bounding boxes
                             i.e. spilling out of the image. can be indexed as discarded_box_counter[mu] where mu is the frame number

    OUTPUTS:
      frame_count: int, number of frames in the original image stack
      cell_count: n-darray, vector containing the number of cropped cells in a given frame, it can be indexed as cell_count[mu] where mu is the frame number
    """

    frame_count = len(list(image_region_props))
    cell_count = [
        int(len(image_region_props[f"Frame_{i}"]) - discarded_box_counter[i])
        for i in range(frame_count)
    ]

    cell_count = np.asarray(cell_count)
    return frame_count, cell_count


def clean_regions(
    regions: npt.NDArray,
    frame_count: float,
    cell_count: npt.NDArray,
    threshold_division: float,
    sigma: float,
    threshold_type: str = "single",
) -> tuple[npt.NDArray, npt.NDArray, npt.NDArray]:
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
            mask = preprocess_2d(
                regions[i][j], threshold_division, sigma, threshold_type
            )[1]
            cleaned_mask = clear_border(mask)
            cleaned_intensity_regions_temp.append(
                np.multiply(regions[i][j], cleaned_mask)
            )
            cleaned_regions_temp.append(label(cleaned_mask))
            masks_temp.append(cleaned_mask)

        masks.append(masks_temp)
        cleaned_regions.append(cleaned_regions_temp)
        cleaned_intensity_regions.append(cleaned_intensity_regions_temp)

    masks = np.asarray(masks, dtype="object")
    cleaned_regions = np.asarray(cleaned_regions, dtype="object")
    cleaned_intensity_regions = np.asarray(cleaned_intensity_regions, dtype="object")

    return cleaned_regions, cleaned_intensity_regions, masks


def add_labels(data_frame: npt.NDArray, labels: npt.NDArray) -> npt.NDArray:
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


def binImage(img: npt.NDArray, new_shape: tuple, method: str = "mean") -> npt.NDArray:
    """
    img = Original asarray to be binned
    new_shape = final desired shape of the asarray
    method = 'min' - minimum binned
             'max' - max. binned
             'mean' - mean binned; default
    """
    if len(img.shape) == 3:
        shape = (
            new_shape[0],
            img.shape[0] // new_shape[0],
            new_shape[1],
            img.shape[1] // new_shape[1],
            3,
        )
        index0 = -2
        index1 = 1
    elif len(img.shape) == 2:
        shape = (
            new_shape[0],
            img.shape[0] // new_shape[0],
            new_shape[1],
            img.shape[1] // new_shape[1],
        )
        index0 = -1
        index1 = 1
    else:
        print(
            "Input image must be either RGB like, (3 dimensional) or black and white (2 dimensional)"
        )
        return
    img = img.reshape(shape)
    if method == "min":
        out = img.min(index0).min(index1)
    elif method == "max":
        out = img.max(index0).max(index1)
    elif method == "mean":
        out = img.mean(index0).mean(index1)
    return out


def write_clusters(
    dataframe: npt.NDArray, cluster_coloumn: int
) -> dict[Union[str, int], npt.NDArray]:
    """
    Takes in a dataframe containing cluster labels, and writes new arrays, one for each label
    -------------------------------------------------------------------------------------------
    INPUTS:
        dataframe: npt.NDArray, dataframe containing one coloumn that corresponds to a cluster
        cluster_column: int, the index of the coloumn that corresponds to the clustering

    OUTPUTS:
        clusters: dict, dictionary containing the cluster number as a key and the asarray containing [frame, cell] as values.
    """

    num_clusters = int(dataframe[:, cluster_coloumn].max() + 1)
    clusters = {i: np.zeros(shape=(0, 2)) for i in range(num_clusters)}
    clusters["noise"] = np.zeros(shape=(0, 2))

    for i in range(dataframe.shape[0]):
        for j in range(num_clusters):
            if dataframe[i, cluster_coloumn] == j:
                cluster_temp = np.asarray([[dataframe[i, -3], dataframe[i, -2]]])
                clusters[j] = np.append(clusters[j], cluster_temp, axis=0)

        if dataframe[i, cluster_coloumn] == -1:
            cluster_temp = np.asarray([[dataframe[i, -3], dataframe[i, -2]]])
            clusters["noise"] = np.append(clusters["noise"], cluster_temp, axis=0)

    return clusters


def square_reshape(img: npt.NDArray, desired_shape: tuple) -> npt.NDArray:
    """ "
    Reshapes a square image
    -----------------------
    INPUTS:
        image: npt.NDArray
        desired_shape: tuple
    OUTPUTS:
        image: npt.NDArray
    """

    if img.shape[0] < desired_shape[0]:
        qdiff = (2048 - img.shape[0]) // 4
        img = np.pad(
            img,
            [(qdiff, qdiff), (qdiff, qdiff), (0, 0)],
            mode="constant",
            constant_values=img.mean(),
        )
    elif img.shape[0] > desired_shape[0]:
        img = binImage(img, desired_shape)

    return img
