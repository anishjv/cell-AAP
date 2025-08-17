from __future__ import annotations
from qtpy import QtWidgets
from qtpy.QtWidgets import QFileDialog
import json
import os
import napari.utils.notifications
from cell_AAP import defaults  # type: ignore


def open_config_editor(widget) -> None:
    """
    Open configuration editor pre-populated with attributes from defaults for configs.Cfg.
    -------------------------------------------------------------------------------------------------------
    INPUTS:
		widget: QtWidgets.QWidget, parent widget used for save dialog and status
    OUTPUTS:
		None: None, writes JSON file and opens it in the default editor
    """
    # Check if user already has a config file selected
    if widget.config_file_path and os.path.exists(widget.config_file_path):
        config_file = widget.config_file_path
        try:
            os.system(f"open {config_file}")
            napari.utils.notifications.show_info(f"Opened existing configuration: {config_file}")
            return
        except Exception as e:
            napari.utils.notifications.show_error(f"Error opening existing config: {str(e)}")
            return
    
    # Pull defaults from cell_AAP.defaults
    d = defaults._DEFAULT
    # Build a JSON-serializable configuration reflecting configs.Cfg fields
    default_config = {
        "version": d["VERSION"],
        "threshold_type": d["THRESHOLD_TYPE"],
        "threshold_division": d["THRESHOLD_DIVISION"],
        "gaussian_sigma": d["GAUSSIAN_SIGMA"],
        "point_prompts": d["POINTPROMPTS"],
        "box_prompts": d["BOXPROMPTS"],
        "propslist": [
            "area",
            "solidity", 
            "perimeter_crofton",
            "convex_area",
            "eccentricity",
            "major_axis_length",
            "minor_axis_length", 
            "perimeter",
            "local_centroid",
            "euler_number",
            "extent",
            "intensity_max",
            "intensity_min",
            "filled_area",
        ],
        "iou_thresh": d["IOU_THRESH"],
        # Morphology parameters (serialized as function + args)
        # To disable an operation, set it to null/None: {"tophatstruct": null}
        "morphology": {
            "tophatstruct": {"func": "square", "args": {"size": 71}},
            "erosionstruct": {"func": "disk", "args": {"radius": 8}}
        },
        # Bounding box parameters (serialized)
        "bbox": {
            "box_size_scale": float(d["BOX_SIZE"][1][0]) if isinstance(d["BOX_SIZE"], tuple) else 2.5,
            "bbox_func": getattr(d["BBOX_FUNC"], "__name__", "square_box")
        }
    }
    
    # Get config file path
    config_file, _ = QFileDialog.getSaveFileName(
        widget,
        "Save Configuration File",
        "",
        "JSON Files (*.json);;All Files (*)"
    )
    
    if not config_file:
        return
    
    try:
        # Save default configuration
        with open(config_file, 'w') as f:
            json.dump(default_config, f, indent=4)
        
        # Update widget
        widget.update_config_path(config_file)
        
        # Open file in default editor
        os.system(f"open {config_file}")
        
        napari.utils.notifications.show_info(f"Configuration file created: {config_file}")
        
    except Exception as e:
        napari.utils.notifications.show_error(f"Error creating configuration file: {str(e)}")
