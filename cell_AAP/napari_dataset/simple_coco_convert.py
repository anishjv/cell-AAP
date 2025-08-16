from __future__ import annotations
import os
import re
import fnmatch
import json
from typing import List, Dict, Any
from PIL import Image
import numpy as np
from cell_AAP.annotation.pycococreator.pycococreatortools import pycococreatortools #type: ignore


def _filter_for_jpeg(root: str, files: list[str]) -> list[str]:
    file_types = ['*.jpeg', '*.jpg']
    file_types = r'|'.join([fnmatch.translate(x) for x in file_types])
    files = [os.path.join(root, f) for f in files]
    files = [f for f in files if re.match(file_types, f)]
    return files


def _filter_for_annotations(root: str, files: list[str], image_filename: str) -> list[str]:
    file_types = ['*.png']
    file_types = r'|'.join([fnmatch.translate(x) for x in file_types])
    basename_no_extension = os.path.splitext(os.path.basename(image_filename))[0]
    file_name_prefix = basename_no_extension + '_.*'
    files = [os.path.join(root, f) for f in files]
    files = [f for f in files if re.match(file_types, f)]
    files = [f for f in files if re.match(file_name_prefix, os.path.splitext(os.path.basename(f))[0])]
    return files


def write_coco_json(
    images_dir: str,
    annotations_dir: str,
    out_json: str,
    categories: List[Dict[str, Any]],
    dataset_info: Dict[str, Any],
):
    """
    Create a COCO JSON using pycococreatortools (same logic as annotation/dataset_convert.py).
    -------------------------------------------------------------------------------------------------------
    INPUTS:
		images_dir: str, directory with binned JPG images
		annotations_dir: str, directory with binned PNG masks per image
		out_json: str, output JSON path to write
		categories: list[dict], COCO categories (e.g., [{"id":1,"name":"class"}])
		dataset_info: dict, COCO info section metadata
    OUTPUTS:
		None: None, writes COCO JSON to disk
    """
    print(f"COCO convert: images_dir={images_dir}")
    print(f"COCO convert: annotations_dir={annotations_dir}")
    print(f"COCO convert: out_json={out_json}")
    try:
        print(f"COCO convert: categories={[c['name'] for c in categories]}")
    except Exception:
        print("COCO convert: categories could not be printed")

    os.makedirs(os.path.dirname(out_json), exist_ok=True)

    coco_output = {
        "info": dataset_info,
        "licenses": [],
        "categories": categories,
        "images": [],
        "annotations": []
    }

    image_id = 1
    segmentation_id = 1

    for root, _, files in os.walk(images_dir):
        image_files = _filter_for_jpeg(root, files)
        for image_filename in image_files:
            image = Image.open(image_filename)
            print(f"Processing image_id={image_id}: {os.path.basename(image_filename)} size={image.size}")
            image_info = pycococreatortools.create_image_info(
                image_id, os.path.basename(image_filename), image.size
            )
            coco_output["images"].append(image_info)

            # find all annotation masks for this image
            for ann_root, _, ann_files in os.walk(annotations_dir):
                annotation_files = _filter_for_annotations(ann_root, ann_files, image_filename)
                print(f"  Found {len(annotation_files)} masks for image_id={image_id}")
                skipped = 0
                for annotation_filename in annotation_files:
                    try:
                        print(f"  Mask file: {os.path.basename(annotation_filename)}")
                        # infer class id by name match
                        class_id = [x['id'] for x in categories if x['name'] in annotation_filename]
                        if not class_id:
                            print("    No matching category; skipping mask")
                            skipped += 1
                            continue
                        class_id = class_id[0]

                        category_info = {'id': class_id, 'is_crowd': 'crowd' in image_filename}

                        # Load mask robustly as 2D uint8 in {0,1}
                        bm = Image.open(annotation_filename).convert('1')
                        binary_mask = np.asarray(bm)
                        if binary_mask.ndim == 3:
                            binary_mask = binary_mask[..., 0]
                        binary_mask = (binary_mask > 0).astype(np.uint8, copy=False)
                        print(f"    mask shape={binary_mask.shape} dtype={binary_mask.dtype} nnz={int(binary_mask.sum())}")

                        annotation_info = pycococreatortools.create_annotation_info(
                            segmentation_id, image_id, category_info, binary_mask,
                            image.size, tolerance=2
                        )
                        if annotation_info is not None:
                            coco_output["annotations"].append(annotation_info)
                            print(f"    Added annotation_id={segmentation_id}")
                        else:
                            print("    annotation_info is None; likely empty polygon/area")
                            skipped += 1
                        segmentation_id += 1
                    except Exception as e:
                        # Skip problematic annotation but continue converting others
                        print(f"WARN: Failed to convert annotation '{annotation_filename}' for image '{image_filename}': {e}")
                        skipped += 1
                print(f"  Finished image_id={image_id}; skipped={skipped}")

            image_id += 1

    print(f"Writing COCO JSON: {out_json}")
    print(f"Totals: images={len(coco_output['images'])}, annotations={len(coco_output['annotations'])}")
    with open(out_json, 'w') as f:
        json.dump(coco_output, f, indent=2)


