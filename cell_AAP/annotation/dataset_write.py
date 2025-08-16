import os
import cv2
from PIL import Image
import numpy as np
from typing import Optional
# TODO
# cannot import this module without installing cell-AAP, this should not be the case, throws "no module 'cell-AAP' error"
from cell_AAP.annotation import annotation_utils  # type:ignore



def write_dataset_ranges(
    parent_dir: str,
    phase_image_stack,
    segmentations,
    labeled_data_frame,
    splits: list[tuple],
    name: str,
    label_to_class: dict,
    bin_size: tuple = (1024, 1024),
    bin_method: str = "max",
):
    """
    Write a dataset split by inclusive frame ranges: create per-split images/annotations folders and
    save binned JPG images and unpacked PNG masks for each frame and cell.
    -------------------------------------------------------------------------------------------------------

    INPUTS:
    	parent_dir: str, parent directory within which the dataset directory is created
    	phase_image_stack: n-darray, stack of grayscale frames indexed by frame (converted to RGB per frame)
    	segmentations: n-darray or list, per-frame per-cell packed bitmask masks (unpacked via np.unpackbits with count=2048)
    	labeled_data_frame: n-darray, rows with columns where -3=frame index, -2=cell index, -1=label id
    	splits: list[tuple[int,int]], inclusive frame ranges (start, end) for each split directory
    	name: str, name of the dataset directory to create under parent_dir
    	label_to_class: dict[int,str], maps numeric label ids to class name strings
    	bin_size: tuple[int,int], output width and height for binning/resizing images and masks
    	bin_method: str, binning method to use (e.g., "max")

    OUTPUTS:
    	images: JPEG files, saved to <parent_dir>/<name>/<split>/images/{frame}.jpg
    	annotations: PNG files, saved to <parent_dir>/<name>/<split>/annotations/{frame}_{class}_frame{frame}cell{cell}.png
    """

    # Create main dataset directory
    main_path = os.path.join(parent_dir, name)
    os.makedirs(main_path, exist_ok=True)

    # Create subdirectories for each split
    for i in range(len(splits)):
        os.makedirs(os.path.join(main_path, str(i), "images"), exist_ok=True)
        os.makedirs(os.path.join(main_path, str(i), "annotations"), exist_ok=True)
    
    # Debug: Check label distribution and mapping
    unique_labels = np.unique(labeled_data_frame[:, -1].astype(int))
    print(f"Unique label IDs in data: {unique_labels}")
    print(f"label_to_class mapping: {label_to_class}")
    
    # Count -1 labels
    minus_one_count = np.sum(labeled_data_frame[:, -1] == -1)
    print(f"Number of cells with label_id -1: {minus_one_count}")

    # Save images
    num_frames = int(np.max(labeled_data_frame[:, -3])) + 1
    for frame in range(num_frames):
        for split_idx, (start, end) in enumerate(splits):
            if start <= frame <= end:
                image_dir = os.path.join(main_path, str(split_idx), "images")
                image = annotation_utils.bw_to_rgb(phase_image_stack[frame])
                print('before binning', image.shape)
                binned_image = annotation_utils.binImage(image, bin_size, bin_method)
                print('after binning', binned_image.shape)
                image_path = os.path.join(image_dir, f"{frame}.jpg")
                Image.fromarray(binned_image).save(image_path)
                break  # Only one split should match

    # Save annotation masks
    for j in range(labeled_data_frame.shape[0]):
        frame = int(labeled_data_frame[j, -3])
        cell = int(labeled_data_frame[j, -2])
        label_id = int(labeled_data_frame[j, -1])
        
        # Skip invalid label IDs or handle them appropriately
        if label_id == -1:
            print(f"Warning: Skipping cell {cell} in frame {frame} with invalid label_id {label_id}")
            continue
        
        classi = label_to_class[label_id]
        safe_class = str(classi).replace('/', '-')

        for split_idx, (start, end) in enumerate(splits):
            if start <= frame <= end:
                annotation_dir = os.path.join(main_path, str(split_idx), "annotations")
                packed = segmentations[frame][cell]
                # Determine original height from packed rows (rows*8)
                unpack_count = packed.shape[0] * 8
                mask = np.unpackbits(packed.astype(np.uint8), axis=0, count=unpack_count) * 255
                mask = annotation_utils.binImage(mask, bin_size, bin_method)
                mask_path = os.path.join(annotation_dir, f"{frame}_{safe_class}_frame{frame}cell{cell}.png")
                cv2.imwrite(mask_path, mask)
                break  # Only one split should match