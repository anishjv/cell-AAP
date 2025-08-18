from __future__ import annotations
import logging
import napari
import napari.utils.notifications
import cell_AAP.napari_dataset.ui as ui  # type:ignore
import cell_AAP.napari_dataset.fileio as fileio  # type: ignore
import cell_AAP.napari_dataset.config_editor as config_editor  # type: ignore
import cell_AAP.napari_dataset.results_viewer as results_viewer  # type: ignore
import cell_AAP.annotation.annotator as annotator  # type:ignore
import cell_AAP.configs as configs  # type:ignore
import cell_AAP.defaults as defaults  # type:ignore

import numpy as np
import os
import json
from typing import Optional, List, Dict, Any
from qtpy.QtWidgets import QMessageBox
import skimage.morphology as morph
from cell_AAP.annotation import annotation_utils as au  # type: ignore
from cell_AAP.napari_dataset import zenodo_uploader  # type: ignore

__all__ = [
    "create_dataset_generation_widget",
]

# get the logger instance
logger = logging.getLogger(__name__)

# if we don't have any handlers, set one up
if not logger.handlers:
    # configure stream handler
    log_fmt = logging.Formatter(
        "[%(levelname)s][%(asctime)s] %(message)s",
        datefmt="%Y/%m/%d %I:%M:%S %p",
    )
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(log_fmt)

    logger.addHandler(console_handler)
    logger.setLevel(logging.DEBUG)


def create_dataset_generation_widget() -> ui.DatasetGenerationWidget:
    """
    Create the dataset generation widget and connect all callbacks.
    -------------------------------------------------------------------------------------------------------
    INPUTS:
		None: None, uses current napari viewer
    OUTPUTS:
		dataset_widget: ui.DatasetGenerationWidget, fully wired widget instance
    """

    dataset_widget = ui.DatasetGenerationWidget(
        napari_viewer=napari.current_viewer()
    )

    # Populate SAM model selector dynamically from registry
    try:
        registry = get_zenodo_registry()
        dataset_widget.sam_model_selector.clear()
        dataset_widget.sam_model_selector.addItems(list(registry.keys()))
    except Exception:
        pass

    # Connect file selection buttons
    dataset_widget.dna_selector.clicked.connect(
        lambda: fileio.select_dna_files(dataset_widget)
    )
    
    dataset_widget.phase_selector.clicked.connect(
        lambda: fileio.select_phase_files(dataset_widget)
    )

    # Connect configuration buttons
    dataset_widget.config_editor_button.clicked.connect(
        lambda: config_editor.open_config_editor(dataset_widget)
    )
    dataset_widget.config_select_button.clicked.connect(
        lambda: fileio.select_existing_config(dataset_widget)
    )
    
    # Simplified SAM: select and load
    dataset_widget.sam_load_button.clicked.connect(
        lambda: load_sam_predictor(dataset_widget)
    )

    # Connect processing buttons
    dataset_widget.process_button.clicked.connect(
        lambda: run_dataset_generation(dataset_widget)
    )

    # Connect results navigation
    dataset_widget.prev_result_button.clicked.connect(
        lambda: results_viewer.show_previous_result(dataset_widget)
    )
    
    dataset_widget.next_result_button.clicked.connect(
        lambda: results_viewer.show_next_result(dataset_widget)
    )

    # Connect save button
    dataset_widget.save_button.clicked.connect(
        lambda: fileio.save_dataset_results(dataset_widget)
    )

    # Connect COCO assembly
    dataset_widget.coco_button.clicked.connect(
        lambda: assemble_coco_dataset(dataset_widget)
    )
    
    # Connect Share to Zenodo (PAT flow)
    dataset_widget.share_button.clicked.connect(
        lambda: share_dataset_to_zenodo(dataset_widget)
    )

    return dataset_widget


def run_dataset_generation(dataset_widget: ui.DatasetGenerationWidget):
    """
    Run the complete dataset generation pipeline for all selected image pairs.
    -------------------------------------------------------------------------------------------------------
    INPUTS:
		dataset_widget: ui.DatasetGenerationWidget, active widget containing file lists and config
    OUTPUTS:
		None: None, updates widget state and viewer layers, stores results in widget.results
    """
    
    # Validate inputs
    if not dataset_widget.dna_files or not dataset_widget.phase_files:
        napari.utils.notifications.show_error(
            "Please select both DNA and phase image files"
        )
        return
    
    if len(dataset_widget.dna_files) != len(dataset_widget.phase_files):
        napari.utils.notifications.show_error(
            "Number of DNA and phase files must match"
        )
        return

    try:
        # Load configuration
        config = load_configuration(dataset_widget)
        
        
        # Require a SAM predictor to be loaded
        if not hasattr(dataset_widget, 'predictor') or dataset_widget.predictor is None:
            napari.utils.notifications.show_error(
                "Please select and load a SAM model before running."
            )
            return
        
        # Initialize results storage
        dataset_widget.results = {}
        
        napari.utils.notifications.show_info(f"Processing {len(dataset_widget.dna_files)} image pairs")
        
        # Create annotator instance with all image pairs
        annotator_instance = annotator.Annotator.get(
            configs=config,
            dna_image_list=dataset_widget.dna_files,
            phase_image_list=dataset_widget.phase_files
        )
        
        # Run crop() if SAM predictor is available
        if hasattr(dataset_widget, 'predictor') and dataset_widget.predictor is not None:
            annotator_instance.crop(predictor=dataset_widget.predictor)
        else:
            annotator_instance.crop()
        
        # Generate dataframe with empty extra_props
        df = annotator_instance.gen_df(extra_props=[])
        
        # Store results
        dataset_widget.results['df_whole'] = df
        dataset_widget.results['roi_data'] = annotator_instance.roi
        dataset_widget.results['phase_roi_data'] = annotator_instance.phs_roi
        dataset_widget.results['segmentations'] = annotator_instance.segmentations
        dataset_widget.results['cleaned_binary_roi'] = annotator_instance.cleaned_binary_roi
        dataset_widget.results['cleaned_scalar_roi'] = annotator_instance.cleaned_scalar_roi
        dataset_widget.results['file_names'] = [os.path.basename(f) for f in dataset_widget.dna_files]
        dataset_widget.results['full_dna_images'] = annotator_instance.dna_image_stack
        dataset_widget.results['full_phase_images'] = annotator_instance.phase_image_stack
        dataset_widget.results['prompts'] = annotator_instance.prompts
        
        # Initialize results viewer
        dataset_widget.current_result_index = 0
        results_viewer.display_current_result(dataset_widget)
        
        napari.utils.notifications.show_info("Dataset generation completed successfully!")
        
    except Exception as e:
        napari.utils.notifications.show_error(f"Error during dataset generation: {str(e)}")
        logger.error(f"Dataset generation error: {str(e)}", exc_info=True)


def load_configuration(dataset_widget: ui.DatasetGenerationWidget) -> configs.Cfg:
    """
    Load configuration from JSON (if provided) and construct configs.Cfg.
    -------------------------------------------------------------------------------------------------------
    INPUTS:
		dataset_widget: ui.DatasetGenerationWidget, provides optional config file path
    OUTPUTS:
		config: configs.Cfg, configuration object ready for Annotator
    """
    
    config_file = dataset_widget.config_file_path
    
    def build_cfg_from_json(cfg_json: Dict[str, Any]) -> configs.Cfg:
        # Start from DEFAULTS
        d = defaults._DEFAULT
        version = cfg_json.get("version", d["VERSION"])
        threshold_type = cfg_json.get("threshold_type", d["THRESHOLD_TYPE"])
        threshold_division = cfg_json.get("threshold_division", d["THRESHOLD_DIVISION"])
        gaussian_sigma = cfg_json.get("gaussian_sigma", d["GAUSSIAN_SIGMA"]) 
        point_prompts = cfg_json.get("point_prompts", d["POINTPROMPTS"]) 
        box_prompts = cfg_json.get("box_prompts", d["BOXPROMPTS"]) 
        propslist = cfg_json.get("propslist", d["PROPSLIST"]) 
        iou_thresh = cfg_json.get("iou_thresh", d["IOU_THRESH"]) 

        # Morphology (function + args)
        morph_json = cfg_json.get("morphology", {})
        def build_kernel(spec: Dict[str, Any], fallback):
            # Handle None case - user wants to disable this operation
            if spec is None:
                return None
            
            func_name = spec.get("func")
            args = spec.get("args", {})
            
            # Handle None case for func_name - user wants to disable this operation
            if func_name is None:
                return None
                
            if func_name and hasattr(morph, func_name):
                try:
                    return getattr(morph, func_name)(**args)
                except Exception:
                    pass
            return fallback
        tophatstruct = build_kernel(morph_json.get("tophatstruct", {}), morph.square(71))
        erosionstruct = build_kernel(morph_json.get("erosionstruct", {}), morph.disk(8))

        # BBox settings
        bbox_json = cfg_json.get("bbox", {})
        box_size_scale = float(bbox_json.get("box_size_scale", 2.5))
        box_size = (au.get_box_size, (box_size_scale,))
        bbox_func_name = bbox_json.get("bbox_func", "square_box")
        bbox_func = getattr(au, bbox_func_name, au.square_box)

        return configs.Cfg(
            version=version,
            threshold_type=threshold_type,
            threshold_division=threshold_division,
            tophatstruct=tophatstruct,
            erosionstruct=erosionstruct,
            gaussian_sigma=gaussian_sigma,
            point_prompts=point_prompts,
            box_prompts=box_prompts,
            propslist=propslist,
            box_size=box_size,
            bbox_func=bbox_func,
            iou_thresh=iou_thresh,
        )

    if config_file and os.path.exists(config_file):
        try:
            with open(config_file, 'r') as f:
                config_data = json.load(f)
            cfg = build_cfg_from_json(config_data)
            # Debug: log the actual propslist being used
            napari.utils.notifications.show_info(f"Loaded config with propslist: {cfg.propslist}")
            return cfg
        except Exception as e:
            napari.utils.notifications.show_warning(
                f"Failed to load custom config, using defaults: {str(e)}"
            )

    # Use default configuration, but build it through our JSON converter to get consistent property names
    return build_cfg_from_json({})


def assemble_coco_dataset(dataset_widget: ui.DatasetGenerationWidget) -> None:
    """
    Prompt user for dataset assembly parameters and write COCO-style splits and JSONs.
    -------------------------------------------------------------------------------------------------------
    INPUTS:
		dataset_widget: ui.DatasetGenerationWidget, provides results and UI context
    OUTPUTS:
		None: None, writes dataset folders and JSONs to disk
    """
    from qtpy.QtWidgets import QInputDialog, QFileDialog
    import cell_AAP.annotation.dataset_write as dw  # type: ignore
    from cell_AAP.napari_dataset.simple_coco_convert import write_coco_json  # type: ignore

    if not dataset_widget.results or not dataset_widget.results.get('df_whole') is not None:
        napari.utils.notifications.show_error("No results to assemble from. Run generation first.")
        return

    # 1. Dataset name
    name, ok = QInputDialog.getText(dataset_widget, "Dataset Name", "Enter dataset name:")
    if not ok or not name:
        return

    # 2. Train/Test/Validation splits as inclusive ranges; allow 'none'
    def parse_range(label: str) -> Optional[tuple[int, int]]:
        text, ok_local = QInputDialog.getText(dataset_widget, f"{label} Split", f"Enter {label.lower()} range as start,end or 'none':")
        if not ok_local or not text:
            return None
        if text.strip().lower() == 'none':
            return None
        try:
            s0, s1 = [int(x.strip()) for x in text.split(',')]
        except Exception:
            napari.utils.notifications.show_error("Invalid range. Use integers: start,end or 'none'")
            return None
        if not (s0 <= s1):
            napari.utils.notifications.show_error("Each range must satisfy start <= end")
            return None
        return (s0, s1)

    tr = parse_range('Train')
    te = parse_range('Test')
    va = parse_range('Validation')
    ranges_map: Dict[str, Optional[tuple[int, int]]] = {"train": tr, "test": te, "validation": va}
    # ensure non-overlap among provided ranges
    provided = [(k, v) for k, v in ranges_map.items() if v is not None]
    for i in range(len(provided)):
        for j in range(i + 1, len(provided)):
            (k1, (a0, a1)) = provided[i]
            (k2, (b0, b1)) = provided[j]
            if max(a0, b0) <= min(a1, b1):
                napari.utils.notifications.show_error(f"{k1} and {k2} ranges must not overlap")
                return

    # 3. Class name (single)
    class_name, ok = QInputDialog.getText(dataset_widget, "Class Name", "Enter single class name:")
    if not ok or not class_name:
        return
    label_to_class = {0: class_name}

    # 4. Stage in temp dir, zip to Desktop
    import tempfile
    desktop = os.path.join(os.path.expanduser('~'), 'Desktop')
    os.makedirs(desktop, exist_ok=True)
    temp_parent = tempfile.mkdtemp(prefix="cellapp_coco_")

    # Prepare labeled df: append label=0 to each row
    df_whole = dataset_widget.results['df_whole']
    label_col = np.zeros((df_whole.shape[0], 1), dtype=int)
    df_whole_labeled = np.concatenate([df_whole, label_col], axis=1)

    # Build phase stack and segmentations from full-size images
    phase_image_stack = dataset_widget.results.get('full_phase_images')
    segmentations = dataset_widget.results.get('segmentations')

    if phase_image_stack is None or segmentations is None:
        napari.utils.notifications.show_error("Missing full phase images or segmentations for dataset assembly")
        return

    # Ensure numpy array for image stack
    phase_image_stack = np.asarray(phase_image_stack)

    try:
        # Build splits list in a deterministic order of present splits
        split_keys = [k for k in ["train", "validation", "test"] if ranges_map[k] is not None]
        split_ranges = [ranges_map[k] for k in split_keys]
        # write using numeric split indices then rename
        dw.write_dataset_ranges(
            temp_parent,
            phase_image_stack,
            segmentations,
            df_whole_labeled,
            split_ranges,  # type: ignore
            name,
            label_to_class,
        )
    except Exception as e:
        napari.utils.notifications.show_error(f"Dataset write failed: {str(e)}")
        return

    # Rename numeric split dirs to their names and write COCO JSONs for present splits
    try:
        base_path = os.path.join(temp_parent, name)
        # rename 0,1,2 -> ordered present split names
        ordered_present = [k for k in ["train", "validation", "test"] if ranges_map[k] is not None]
        for idx, split_name in enumerate(ordered_present):
            old = os.path.join(base_path, str(idx))
            new = os.path.join(base_path, split_name)
            if os.path.isdir(old) and not os.path.isdir(new):
                os.rename(old, new)
        # write JSONs for each present split
        for split_name in ordered_present:
            split_dir = os.path.join(base_path, split_name)
            write_coco_json(
                images_dir=os.path.join(split_dir, 'images'),
                annotations_dir=os.path.join(split_dir, 'annotations'),
                out_json=os.path.join(split_dir, f"{name}_{'val' if split_name=='validation' else split_name}.json"),
                categories=[{"id": 1, "name": class_name}],
                dataset_info={"description": name, "version": "1.0", "split": split_name}
            )
        napari.utils.notifications.show_info("COCO dataset assembled")
    except Exception as e:
        napari.utils.notifications.show_error(f"COCO conversion failed: {str(e)}")
        return

    # Zip to Desktop
    try:
        zip_path = _zip_dataset_folder(temp_parent, name)
        # Move resulting zip to Desktop
        final_zip = os.path.join(desktop, os.path.basename(zip_path))
        if os.path.abspath(zip_path) != os.path.abspath(final_zip):
            if os.path.exists(final_zip):
                os.remove(final_zip)
            os.replace(zip_path, final_zip)
        dataset_widget.last_assembled_zip_path = final_zip
        dataset_widget.last_assembled_name = name
        dataset_widget.share_button.setEnabled(True)
        QMessageBox.information(dataset_widget, "COCO Assembled", f"Saved zip to Desktop:\n{final_zip}")
    except Exception as e:
        napari.utils.notifications.show_error(f"Failed to zip/move dataset: {str(e)}")
        return


def _zip_dataset_folder(base_dir: str, dataset_name: str) -> str:
    import zipfile
    archive_path = os.path.join(base_dir, f"{dataset_name}.zip")
    root_dir = os.path.join(base_dir, dataset_name)
    with zipfile.ZipFile(archive_path, 'w', compression=zipfile.ZIP_DEFLATED) as zf:
        for dirpath, _, filenames in os.walk(root_dir):
            for fname in filenames:
                full = os.path.join(dirpath, fname)
                rel = os.path.relpath(full, base_dir)
                zf.write(full, rel)
    return archive_path


def share_dataset_to_zenodo(dataset_widget: ui.DatasetGenerationWidget) -> None:
    from qtpy.QtWidgets import QInputDialog
    # Ensure a recently assembled zip exists
    if not getattr(dataset_widget, 'last_assembled_zip_path', None):
        napari.utils.notifications.show_error("No assembled dataset zip found. Please run 'Assemble COCO Dataset' first.")
        return
    name = getattr(dataset_widget, 'last_assembled_name', None) or 'CellAPP_Dataset'
    zip_path = dataset_widget.last_assembled_zip_path

    # Prompt for Zenodo PAT & minimal metadata
    pat, ok = QInputDialog.getText(dataset_widget, "Zenodo Personal Access Token", "Enter your Zenodo PAT (deposit scope):")
    if not ok or not pat:
        return
    title, ok = QInputDialog.getText(dataset_widget, "Title", "Dataset title:")
    if not ok or not title:
        return
    description, ok = QInputDialog.getMultiLineText(dataset_widget, "Description", "Short description:")
    if not ok:
        return
    creators_str, ok = QInputDialog.getText(dataset_widget, "Creators", "Creators as 'Name1;Name2' (optional affiliations in description):")
    if not ok:
        return
    creators: List[Dict[str, str]] = []
    for part in [c.strip() for c in creators_str.split(';') if c.strip()]:
        creators.append({"name": part})
    if not creators:
        creators = [{"name": "Unknown"}]
    
    try:
        record_url = zenodo_uploader.upload_dataset_zip(
            pat=pat,
            zip_path=zip_path,  # assembled zip on Desktop
            title=title or name,
            description=description or name,
            creators=creators,
            keywords=["Cell-APP", "cell segmentation", "COCO"],
            license_id="cc-by-4.0",
            community_identifier="cellapp_external",
            publish=True,
        )
        napari.utils.notifications.show_info(f"Uploaded to Zenodo: {record_url}")
        QMessageBox.information(dataset_widget, "Zenodo Upload", f"Upload complete.\n{record_url}")
    except Exception as e:
        napari.utils.notifications.show_error(f"Zenodo upload failed: {str(e)}")
        QMessageBox.critical(dataset_widget, "Zenodo Upload Failed", str(e))

def load_sam_predictor(dataset_widget: ui.DatasetGenerationWidget) -> None:
    """
    Resolve, download, and load the selected SAM model; attach predictor to the widget.
    -------------------------------------------------------------------------------------------------------
    INPUTS:
		dataset_widget: ui.DatasetGenerationWidget, provides selected model key
    OUTPUTS:
		None: None, sets dataset_widget.predictor on success
    """
    try:
        selection = dataset_widget.sam_model_selector.currentText()
        registry = get_zenodo_registry()
        entry = registry.get(selection)
        if selection not in registry:
            napari.utils.notifications.show_error("Invalid model selection")
            return
        # Determine SAM backbone (first two tokens joined: vit_h, vit_l, vit_b)
        parts = selection.split("_")
        backbone = "_".join(parts[:2])
        ckpt = resolve_and_fetch_checkpoint(selection)
        from segment_anything import sam_model_registry, SamPredictor  # type: ignore
        sam = sam_model_registry[backbone](checkpoint=ckpt)
        import torch
        use_cuda = torch.cuda.is_available()
        if use_cuda:
            sam.to(device="cuda")
        predictor = SamPredictor(sam)
        dataset_widget.predictor = predictor
        napari.utils.notifications.show_info(f"Loaded SAM ({selection})")
    except Exception as e:
        import traceback
        print(f"[SAM DEBUG] Failed to load SAM: {e}")
        traceback.print_exc()
        napari.utils.notifications.show_error(f"Failed to load SAM: {str(e)}")


def get_zenodo_registry() -> Dict[str, Dict[str, str]]:
    """
    Return local registry mapping for Zenodo-hosted SAM checkpoints.
    -------------------------------------------------------------------------------------------------------
    INPUTS:
		None: None
    OUTPUTS:
		registry: dict, maps model_key -> {doi: str, filename: str}
    """
    # May 2024 entries (LM/EM x {vit_h, vit_l, vit_b}). Placeholder DOIs; update as needed.
    return {
        "vit_l_lm": {"doi": "10.5281/zenodo.11111177", "filename": "vit_l.pt"},
        "vit_b_lm": {"doi": "10.5281/zenodo.11103798", "filename": "vit_b.pt"},
        "vit_t_lm": {"doi": "10.5281/zenodo.11111329", "filename": "vit_t.pt"},
        "vit_l_em": {"doi": "10.5281/zenodo.11111055", "filename": "vit_l.pt"},
        "vit_b_em": {"doi": "10.5281/zenodo.11111294", "filename": "vit_b.pt"},
        "vit_t_em": {"doi": "10.5281/zenodo.11110951", "filename": "vit_t.pt"},
    }


def resolve_and_fetch_checkpoint(model_key: str) -> str:
    """
    Fetch a checkpoint for the given model key from Zenodo and return the local path.
    -------------------------------------------------------------------------------------------------------
    INPUTS:
		model_key: str, key from registry (e.g., "vit_h_lm")
    OUTPUTS:
		local_path: str, filesystem path to the downloaded checkpoint file
    """
    import pooch
    reg = get_zenodo_registry()
    entry = reg.get(model_key)
    if not entry:
        raise RuntimeError("No registry entry for selected model")
    doi = entry["doi"]
    filename = entry["filename"]
    fetcher = pooch.create(path=pooch.os_cache("cell_aap_sam"), base_url=f"doi:{doi}", registry={filename: None})
    local_path = fetcher.fetch(filename)
    return local_path