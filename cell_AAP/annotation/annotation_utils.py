import cv2
import numpy as np
import numpy.typing as npt
import torch
import skimage
from skimage.measure import regionprops, label
from skimage.morphology import white_tophat, square, erosion
from skimage.filters import (
    gaussian,
    threshold_isodata,  
    threshold_multiotsu,
)  # pylint: disable=no-name-in-module
from skimage.segmentation import watershed
from skimage.feature import peak_local_max
from typing import Optional, Union
import scipy
from scipy.ndimage import distance_transform_edt
import gc  # Add garbage collection import


def preprocess_2d(
    image: npt.NDArray,
    threshold_division: float,
    sigma: float,
    threshold_type: Optional[str] = "single",
    erosionstruct: Optional[any] = False,
    tophatstruct: Optional[any] = square(71),
) -> tuple[npt.NDArray, npt.NDArray]:
    """
    Preprocesses a 2D image with Gaussian smoothing, background subtraction, and thresholding.
    ------------------------------------------------------------------------------------------------------
    INPUTS:
    	image: npt.NDArray, input image to preprocess
    	threshold_division: float, division factor for threshold calculation
    	sigma: float, sigma value for Gaussian smoothing
    	threshold_type: Optional[str], type of thresholding ("single", "multi", or "watershed")
    	erosionstruct: Optional[any], structuring element for erosion operation
    	tophatstruct: Optional[any], structuring element for white top-hat operation
    OUTPUTS:
    	labels: npt.NDArray, labeled connected components
    	redseg: npt.NDArray, binary segmented image
    """

    im = gaussian(image, sigma)  # 2D gaussian smoothing filter to reduce noise
    
    if isinstance(tophatstruct, np.ndarray):
        im = white_tophat(im, tophatstruct)
    else:
        pass

    if isinstance(erosionstruct, np.ndarray):
        im = erosion(im, erosionstruct)
    else:
        pass

    # Common thresholding logic for both "single" and "watershed" methods
    if threshold_type in ["single", "watershed"]:
        thresh = threshold_isodata(im)
        redseg = im > (thresh / threshold_division)
        labels = label(redseg)
        
        if threshold_type == "watershed":
            # Use the labeled masks from "single" method to estimate cell count and spacing
            num_cells = labels.max() if labels.max() > 0 else 1
            image_area = im.shape[0] * im.shape[1]
            
            # Estimate average cell area and diameter
            avg_cell_area = np.sum(redseg) / num_cells
            avg_cell_diameter = np.sqrt(avg_cell_area / np.pi) * 2
            
            # Set minimum distance as a fraction of average cell diameter
            min_distance = int(avg_cell_diameter * 0.4)  # 40% of average cell diameter
            min_distance = max(5, min(min_distance, int(image_area**0.5 * 0.1)))  # Clamp between 5 and 10% of image diagonal
            
            # Create distance transform for watershed markers
            binary_mask = redseg.astype(bool)
            distance = distance_transform_edt(binary_mask)
            
            # Find local maxima as watershed markers
            coords = peak_local_max(
                distance, 
                min_distance=min_distance,
                threshold_abs=0.1 * distance.max()
            )
            
            # Create markers image
            markers = np.zeros_like(binary_mask, dtype=int)
            markers[coords[:, 0], coords[:, 1]] = np.arange(1, len(coords) + 1)
            
            # Apply watershed
            labels = watershed(-distance, markers, mask=binary_mask)
            redseg = (labels > 0).astype(np.uint8)
    
    elif threshold_type == "multi":
        thresholds = threshold_multiotsu(im)
        redseg = np.digitize(im, bins=thresholds)
        labels = label(redseg)

    return labels, redseg


def preprocess_3d(
    targetstack: npt.NDArray,
    threshold_division: float,
    sigma: int,
    threshold_type: Optional[str] = "single",
    erosionstruct: Optional[any] = False,
    tophatstruct: Optional[any] = square(71),
) -> tuple[npt.NDArray, dict]:
    """
    Preprocesses a stack of images and computes region properties for each frame.
    ------------------------------------------------------------------------------------------------------
    INPUTS:
    	targetstack: npt.NDArray, stack of images with shape (z, n, n)
    	threshold_division: float, division factor for threshold calculation
    	sigma: int, sigma value for Gaussian smoothing
    	threshold_type: Optional[str], type of thresholding to use ("single", "multi", or "watershed")
    	erosionstruct: Optional[any], structuring element for erosion operation
    	tophatstruct: Optional[any], structuring element for white top-hat operation
    OUTPUTS:
    	labels_whole: npt.NDArray, labeled stack of processed images
    	region_props: dict, region properties for each frame, indexed as 'region_props['Frame_i']'
    """

    region_props = {}
    labels_whole = []

    for i in range(targetstack.shape[0]):
        im = targetstack[i, :, :].copy()

        labels, _ = preprocess_2d(
            im, threshold_division, sigma, threshold_type, erosionstruct, tophatstruct
        )
        region_props[f"Frame_{i}"] = regionprops(labels, intensity_image=labels * im)
        labels_whole.append(labels)
        
        # Clear intermediate variables to reduce memory usage
        del im
        if i % 10 == 0:  # Garbage collect every 10 frames
            gc.collect()

    labels_whole = np.asarray(labels_whole)

    return labels_whole, region_props


def bw_to_rgb(
    image: npt.NDArray,
    max_pixel_value: Optional[int] = 255,
    min_pixel_value: Optional[int] = 0,
) -> npt.NDArray:
    """
    Converts a black and white image to RGB format by replicating the grayscale values across all channels.
    ------------------------------------------------------------------------------------------------------
    INPUTS:
    	image: npt.NDArray, input grayscale image of shape (x, y)
    	max_pixel_value: Optional[int], maximum pixel value for normalization (default 255)
    	min_pixel_value: Optional[int], minimum pixel value for normalization (default 0)
    OUTPUTS:
    	rgb_image: npt.NDArray, RGB image of shape (x, y, 3) with identical values across all channels
    """
    if len(np.asarray(image).shape) == 2:
        # Normalize image in-place to save memory
        image_normalized = cv2.normalize(
            np.asarray(image),
            None,
            max_pixel_value,
            min_pixel_value,
            cv2.NORM_MINMAX,
            cv2.CV_8U,
        )
        # Use broadcasting for more memory-efficient RGB conversion
        rgb_image = np.stack([image_normalized] * 3, axis=-1)
        
        # Clear intermediate variable
        del image_normalized

    return rgb_image


def get_box_size(
    region_props: skimage.measure.regionprops, scaling_factor: float
) -> float:
    """
    Computes bounding box size for cell segmentation based on region properties.
    ------------------------------------------------------------------------------------------------------
    INPUTS:
    	region_props: skimage.measure.regionprops, region properties for detected nuclei
    	scaling_factor: float, scaling factor to convert nucleus size to cell size (l²/A where l is ideal box length, A is mean nucleus area)
    OUTPUTS:
    	bb_side_length: float, half the side length of a bounding box
    """

    major_axis = [region_props[i].axis_major_length for i, _ in enumerate(region_props)]

    dna_major_axis = np.median(np.asarray(major_axis))
    bb_side_length = scaling_factor * dna_major_axis

    return bb_side_length // 2


def get_box_size_scaled(region_props, max_size: float) -> list[float]:
    """
    Computes scaled bounding box sizes based on region properties and intensity.
    ------------------------------------------------------------------------------------------------------
    INPUTS:
    	region_props: skimage.measure.regionprops, region properties for detected nuclei
    	max_size: float, maximum bounding box size for scaling
    OUTPUTS:
    	bb_side_lengths: list[float], list of half side lengths for bounding boxes, scaled based on nucleus properties
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


def square_box(centroid: list[float], box_size: float) -> npt.NDArray:
    """
    Draws an upright bounding box given a centroid and box size.
    ------------------------------------------------------------------------------------------------------
    INPUTS:
    	centroid: list[float], centroid coordinates in the form (y, x)
    	box_size: float, half the side length of the bounding box
    OUTPUTS:
    	coords: npt.NDArray, bounding box coordinates in the form [x1, y2, x2, y1] where (x1, y1) is top-left and (x2, y2) is bottom-right
    """

    x, y = centroid[1], centroid[0]  # centroid must be of the form (y, x)
    x1, y1 = x - box_size, y + box_size  # top left
    x2, y2 = x + box_size, y - box_size  # bottom right

    return np.asarray([x1, y2, x2, y1])


def box_size_wrapper(func, frame_props, args):
    """
    Facilitates the usage of different boxes size determination functions.
    ------------------------------------------------------------------------------------------------------
    INPUTS:
    	func: callable, function to determine box sizes
    	frame_props: skimage.measure.regionprops, region properties for a frame
    	args: tuple, additional arguments to pass to the function
    OUTPUTS:
    	result: any, result from the box size determination function
    """
    try:
        return func(frame_props, *args)
    except Exception as error:
        raise AttributeError("args do not match function") from error


def bbox_wrapper(
    func, centroid, box_size: Optional[float] = None, args: Optional[list] = None
):
    """
    Facilitates the usage of different box drawing determination functions.
    ------------------------------------------------------------------------------------------------------
    INPUTS:
    	func: callable, function to draw bounding boxes
    	centroid: npt.NDArray, centroid coordinates in the form (y, x)
    	box_size: Optional[float], half the side length of the bounding box
    	args: Optional[list], additional arguments to pass to the function
    OUTPUTS:
    	result: any, result from the box drawing function
    """
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
) -> list[int]:
    """
    Return indices (w.r.t. original order) of non-overlapping masks (IoU < iou_thresh);
    preferentially keep the larger mask. Uses PyTorch NMS for optimal speed.
    ------------------------------------------------------------------------------------------------------
    INPUTS:
    	input_list: list[npt.NDArray], list of packed binary masks
    	iou_thresh: float, IoU threshold above which masks are considered overlapping
    OUTPUTS:
    	kept_idx: list[int], indices of masks to keep (in original order) to eliminate overlaps
    """

    # Precompute popcount table for fast area computation
    _POPCOUNT = np.array([bin(i).count("1") for i in range(256)], dtype=np.uint8)

    def _area_packed(mask: np.ndarray) -> int:
        m = mask.view(np.uint8)
        return int(_POPCOUNT[m].astype(np.uint32).sum())

    def _bbox_from_packed(mask: np.ndarray) -> tuple[int, int, int, int]:
        """Get bounding box from packed mask"""
        # Unpack just enough to get bounding box
        mask_bool = np.unpackbits(mask.view(np.uint8), axis=0).astype(bool, copy=False)
        rows = np.any(mask_bool, axis=1)
        cols = np.any(mask_bool, axis=0)
        y1, y2 = np.where(rows)[0][[0, -1]] if len(rows) > 0 else (0, 0)
        x1, x2 = np.where(cols)[0][[0, -1]] if len(cols) > 0 else (0, 0)
        return x1, y1, x2, y2

    if not input_list:
        return []

    # Compute areas and bounding boxes
    areas = np.array([_area_packed(m) for m in input_list], dtype=np.float32)
    bboxes = [_bbox_from_packed(m) for m in input_list]
    
    # Convert to PyTorch tensors
    boxes_tensor = torch.tensor(bboxes, dtype=torch.float32)
    scores_tensor = torch.tensor(areas, dtype=torch.float32)
    
    # Use PyTorch NMS - it keeps the highest scoring boxes (largest areas)
    # NMS returns indices of kept boxes
    kept_indices = torch.ops.torchvision.nms(
        boxes_tensor, 
        scores_tensor, 
        iou_threshold=iou_thresh
    ).numpy()
    
    # Convert to list of kept indices in original order
    kept_idx = kept_indices.tolist()

    return kept_idx


def predict(
    predictor,
    image,
    boxes: Optional[list[list]] = None,
    points: Optional[list] = None,
    box_prompts=False,
    point_prompts=True,
    point_labels: Optional[list] = None,
) -> np.ndarray:
    """
    Implementation of FAIR's SAM using box or point prompts.
    ------------------------------------------------------------------------------------------------------
    INPUTS:
    	predictor: SAM predictor object, predictive algorithm for segmenting cells
    	image: npt.NDArray, input image for segmentation
    	boxes: Optional[list[list]], list of bounding box coordinates
    	points: Optional[list], list of point coordinates
    	box_prompts: bool, whether to use box prompts for segmentation
    	point_prompts: bool, whether to use point prompts for segmentation
    	point_labels: Optional[list], labels for point prompts (1 for positive, 0 for negative)
    OUTPUTS:
    	segmentations: np.ndarray, packed binary masks from SAM prediction
    """
    segmentations = []
    if box_prompts == True:

        try:
            assert boxes != None
        except Exception as error:
            raise AssertionError(
                "Must provide input bounding boxes if box_propmts = True has been selected"
            ) from error

        input_boxes = torch.tensor(np.asarray(boxes), device=predictor.device)
        # Get image shape from predictor if image is None
        if image is None:
            # SAM already has the image set, get shape from predictor
            image_shape = predictor.original_size
        else:
            image_shape = image.shape[:2]
            
        transformed_boxes = predictor.transform.apply_boxes_torch(
            input_boxes, image_shape
        )
        masks, _, _ = predictor.predict_torch(
            point_coords=None,
            point_labels=None,
            boxes=transformed_boxes,
            multimask_output=False,
        )

        masks = masks.detach().cpu().numpy()

    elif point_prompts == True:

        if not point_labels:
            point_labels = np.ones(len(points))

        try:
            assert points != None
        except Exception as error:
            raise AssertionError(
                "Failed to provide input centroids, please select box_prompts = True if attempting to provide bouding box prompts"
            ) from error
        masks, _, _ = predictor.predict(
            point_coords=np.array(points),
            point_labels=point_labels,
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
    iou_thresh: Optional[float] = 0.85,
):
    """
    Crops regions from DNA and phase microscopy images and generates cell segmentations.
    ------------------------------------------------------------------------------------------------------
    INPUTS:
    	dna_image_stack: npt.NDArray, array of shape (image_count, x, y) where each image corresponds to one DNA image
    	phase_image_stack: npt.NDArray, array of shape (image_count, x, y) where each image corresponds to one phase image
    	predictor: SAM predictor object, predictive algorithm for segmenting cells
    	threshold_division: float, division factor for threshold calculation
    	sigma: float, sigma value for Gaussian smoothing
    	erosionstruct: any, structuring element for erosion operation
    	tophatstruct: any, structuring element for white top-hat operation
    	box_size: tuple, tuple containing (function, args) for box size determination
    	point_prompts: bool, whether to use point prompts for segmentation
    	box_prompts: bool, whether to use box prompts for segmentation
    	to_segment: bool, whether to perform segmentation
    	threshold_type: str, type of thresholding to use ("single" or "multi")
    	iou_thresh: Optional[float], IoU threshold for removing overlapping masks
    OUTPUTS:
    	dna_regions: npt.NDArray, rank 4 tensor of cropped ROI's indexed as dna_regions[frame][cell]
    	discarded_box_counter: npt.NDArray, vector of integers corresponding to the number of ROI's discarded due to incomplete bounding boxes
    	segmentations: npt.NDArray, rank 4 tensor containing one mask per cell per frame, indexed as segmentations[frame][cell]
    	phs_regions: npt.NDArray, rank 4 tensor of cropped phase image ROI's
    	dna_seg: npt.NDArray, regions containing nucleus masks
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

    discarded_box_counter = np.array([])
    dna_regions = []
    phs_regions = []
    segmentations = []
    dna_seg = []
    prompts = []
    box_size_func = box_size[0]
    box_size_args = box_size[1]

    # Process images one at a time to reduce memory usage
    labeled_stack, dna_image_region_props = preprocess_3d(
        dna_image_stack,
        threshold_division,
        sigma,
        threshold_type,
        erosionstruct,
        tophatstruct,
    )

    total_frames = dna_image_stack.shape[0]

    for i, _ in enumerate(dna_image_region_props):  # for each image

        frame_props = dna_image_region_props[f"Frame_{i}"]
        box_sizes = box_size_wrapper(box_size_func, frame_props, box_size_args)
        print(f"Image {i+1}/{total_frames}: found {len(frame_props)} candidate cells")
        dna_regions_temp = []
        phs_regions_temp = []
        dna_seg_temp = []
        segmentations_temp = []
        prompts_temp = []
        discarded_box_counter = np.append(discarded_box_counter, 0)
        sam_current_image = i
        sam_previous_image = None
        
        # Clear SAM memory before processing new image
        if to_segment and predictor is not None:
            try:
                predictor.reset_image()
            except:
                pass  # Some SAM versions don't have reset_image
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()

        # Print box size info for non-list case (single box size for all cells)
        if not isinstance(box_sizes, list):
            print(f"Image {i+1}/{total_frames}: current box size is  {2* box_sizes} pixels")
        

        for j, _ in enumerate(dna_image_region_props[f"Frame_{i}"]):  # for each cell

            cell_props = frame_props[j]
            # Get the box size for this specific cell
            if isinstance(box_sizes, list):
                current_box_size = box_sizes[j]
            else:
                current_box_size = box_sizes
            
            y, x = cell_props.centroid
            coords_temp = square_box([y,x], current_box_size)
            x1, y2, x2, y1 = coords_temp

            if not all(k >= 0 and k <= dna_image_stack.shape[1] for k in coords_temp):
                discarded_box_counter[i] += 1
                continue

            dna_region = dna_image_stack[i, int(y2) : int(y1), int(x1) : int(x2)]
            dna_regions_temp.append(dna_region)

            og_seg_region = labeled_stack[i, int(y2) : int(y1), int(x1) : int(x2)]
            og_seg_region = (og_seg_region == cell_props.label).astype(np.uint8)
            dna_seg_temp.append(og_seg_region)

            phs_region = phase_image_stack[i, int(y2) : int(y1), int(x1) : int(x2)]
            phs_regions_temp.append(phs_region)

            if to_segment == True:
                if (
                    sam_current_image != sam_previous_image
                    or sam_previous_image == None
                ):
                    # Convert to RGB only when needed and clear after use
                    phase_image_rgb = bw_to_rgb(
                        phase_image_stack[sam_current_image, :, :]
                    )
                    predictor.set_image(phase_image_rgb)
                    sam_previous_image = sam_current_image
                    
                    # Clear the RGB image from memory after setting it in predictor
                    del phase_image_rgb
                    gc.collect()

                if box_prompts == True:

                    mask = predict(
                        predictor,
                        None,  # Don't pass RGB image again, SAM already has it
                        boxes=[coords_temp],
                        box_prompts=True,
                    )
                    # Append prompts in same order as masks
                    segmentations_temp.extend(mask)
                    prompts_temp.append(np.array(coords_temp, dtype=float))

                elif point_prompts == True:

                    points = [[x, y]]
                    mask = predict(
                        predictor,
                        None,  # Don't pass RGB image again, SAM already has it
                        points=points,
                        point_prompts=True,
                    )
                    segmentations_temp.append(mask)
                    prompts_temp.append([x, y])

        kept_indices = iou_with_list(
            segmentations_temp, iou_thresh
        )

        discarded_box_counter[i] += (len(segmentations_temp) - len(kept_indices))

        segmentations_temp = [
            seg for i, seg in enumerate(segmentations_temp) if i in kept_indices
        ]
        prompts_temp = [
            prm for i, prm in enumerate(prompts_temp) if i in kept_indices
        ]
        dna_regions_temp = [
            roi for i, roi in enumerate(dna_regions_temp) if i in kept_indices
        ]
        phs_regions_temp = [
            roi for i, roi in enumerate(phs_regions_temp) if i in kept_indices
        ]
        dna_seg_temp = [
            roi for i, roi in enumerate(dna_seg_temp) if i in kept_indices
        ]

        segmentations.append(segmentations_temp)
        dna_regions.append(dna_regions_temp)
        phs_regions.append(phs_regions_temp)
        dna_seg.append(dna_seg_temp)
        prompts.append(prompts_temp)

        kept = len(segmentations_temp)
        print(f"Image {i+1}/{total_frames}: kept {kept} cells")

    # Clear large intermediate data structures
    del labeled_stack, dna_image_region_props
    gc.collect()
    
    dtype = object if len(dna_regions) > 1 else 'uint16'
    dna_regions = np.asarray(dna_regions, dtype=dtype)
    phs_regions = np.asarray(phs_regions, dtype=dtype)
    segmentations = np.asarray(segmentations, dtype=dtype)
    dna_seg = np.asarray(dna_seg, dtype=dtype)
    prompts = np.asarray(prompts, dtype=dtype)

    return (dna_regions, discarded_box_counter, segmentations, phs_regions, dna_seg, prompts)


def add_labels(data_frame: npt.NDArray, labels: npt.NDArray) -> npt.NDArray:
    """
    Adds labels to a dataframe when labels and dataframe have the same number of rows.
    ------------------------------------------------------------------------------------------------------
    INPUTS:
    	data_frame: npt.NDArray, input dataframe
    	labels: npt.NDArray, labels to add to the dataframe
    OUTPUTS:
    	data_frame: npt.NDArray, dataframe with one extra column containing the labels
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
    Bins an image to a new shape using specified method.
    ------------------------------------------------------------------------------------------------------
    INPUTS:
    	img: npt.NDArray, original array to be binned
    	new_shape: tuple, final desired shape of the array
    	method: str, binning method ('min', 'max', or 'mean')
    OUTPUTS:
    	out: npt.NDArray, binned image with the specified new shape
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
    Separates dataframe rows into clusters based on cluster labels.
    ------------------------------------------------------------------------------------------------------
    INPUTS:
    	dataframe: npt.NDArray, dataframe containing one column that corresponds to cluster labels
    	cluster_coloumn: int, the index of the column that corresponds to the clustering
    OUTPUTS:
    	clusters: dict[Union[str, int], npt.NDArray], dictionary containing cluster numbers as keys and arrays of [frame, cell] coordinates as values
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
    """
    Reshapes a square image to desired dimensions.
    ------------------------------------------------------------------------------------------------------
    INPUTS:
    	img: npt.NDArray, input image to be reshaped
    	desired_shape: tuple, target shape for the image
    OUTPUTS:
    	img: npt.NDArray, reshaped image with the desired dimensions
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

def to_uint8(img: np.ndarray) -> np.ndarray:
    """
    Reliably converts any image (uint16, float, int) to uint8 (0-255).
    -------------------------------------------------------------------
    INPUTS:
    	img: np.ndarray, input image to be scaled
    OUTPUTS:
    	img: np.ndarray, scaled image
    """
            
    img = img.astype("float32")
    # Min-Max Normalization (Contrast Stretching)
    # This makes the darkest pixel 0 and brightest 255
    img_min = img.min()
    img_max = img.max()
    
    # Avoid division by zero if the image is solid color (e.g., all black)
    if img_max > img_min:
        img = (img - img_min) / (img_max - img_min) * 255.0
    else:
        # Image is constant value; mapping to 0 or 255 depending on intensity is risky
        # Safer to map to 0 usually, or keep relative brightness if known.
        img = img - img_min # becomes 0
            
    # 3. Clip and Cast
    # Ensure no values fall outside 0-255 due to float rounding errors
    img = np.clip(img, 0, 255)
    return img.astype("uint8")