from __future__ import annotations
from qtpy import QtWidgets
from qtpy.QtWidgets import QFileDialog
import numpy as np
import os
from typing import List, Optional
import napari.utils.notifications


def select_dna_files(widget) -> None:
    """
    Open a file dialog to select one or more DNA image files and update the list.
    -------------------------------------------------------------------------------------------------------
    INPUTS:
		widget: QtWidgets.QWidget, parent widget receiving the selection
    OUTPUTS:
		None: None, updates widget.dna_files and list display
    """
    files, _ = QFileDialog.getOpenFileNames(
        widget,
        "Select DNA Image Files",
        "",
        "Image Files (*.tif *.tiff *.jpg *.jpeg *.png *.bmp);;All Files (*)"
    )
    
    if files:
        widget.dna_files = files
        widget.update_file_lists()


def select_phase_files(widget) -> None:
    """
    Open a file dialog to select one or more phase image files and update the list.
    -------------------------------------------------------------------------------------------------------
    INPUTS:
		widget: QtWidgets.QWidget, parent widget receiving the selection
    OUTPUTS:
		None: None, updates widget.phase_files and list display
    """
    files, _ = QFileDialog.getOpenFileNames(
        widget,
        "Select Phase Image Files",
        "",
        "Image Files (*.tif *.tiff *.jpg *.jpeg *.png *.bmp);;All Files (*)"
    )
    
    if files:
        widget.phase_files = files
        widget.update_file_lists()


def save_dataset_results(widget) -> None:
    """
    Save all generated dataset arrays (df, ROIs, segmentations, filenames) to disk.
    -------------------------------------------------------------------------------------------------------
    INPUTS:
		widget: QtWidgets.QWidget, provides results and triggers directory picker
    OUTPUTS:
		None: None, writes .npy files to selected directory
    """
    if not widget.results:
        napari.utils.notifications.show_error("No results to save")
        return
    
    # Get save directory
    save_dir = QFileDialog.getExistingDirectory(
        widget,
        "Select Directory to Save Results"
    )
    
    if not save_dir:
        return
    
    try:
        # Save main dataframe
        if 'df_whole' in widget.results and len(widget.results['df_whole']) > 0:
            np.save(os.path.join(save_dir, 'df_whole.npy'), widget.results['df_whole'])
        
        # Save ROI data
        if 'roi_data' in widget.results:
            np.save(os.path.join(save_dir, 'roi_data.npy'), np.array(widget.results['roi_data'], dtype=object))
        
        if 'phase_roi_data' in widget.results:
            np.save(os.path.join(save_dir, 'phase_roi_data.npy'), np.array(widget.results['phase_roi_data'], dtype=object))
        
        # Save segmentations
        if 'segmentations' in widget.results:
            np.save(os.path.join(save_dir, 'segmentations.npy'), np.array(widget.results['segmentations'], dtype=object))
        
        # Save cleaned ROIs
        if 'cleaned_binary_roi' in widget.results:
            np.save(os.path.join(save_dir, 'cleaned_binary_roi.npy'), np.array(widget.results['cleaned_binary_roi'], dtype=object))
        
        if 'cleaned_scalar_roi' in widget.results:
            np.save(os.path.join(save_dir, 'cleaned_scalar_roi.npy'), np.array(widget.results['cleaned_scalar_roi'], dtype=object))
        
        # Save file names
        if 'file_names' in widget.results:
            np.save(os.path.join(save_dir, 'file_names.npy'), np.array(widget.results['file_names']))
        
        napari.utils.notifications.show_info(f"Results saved to {save_dir}")
        
    except Exception as e:
        napari.utils.notifications.show_error(f"Error saving results: {str(e)}")


def select_existing_config(widget) -> None:
    """
    Browse for an existing configuration JSON file and set it.
    -------------------------------------------------------------------------------------------------------
    INPUTS:
		widget: QtWidgets.QWidget, parent widget receiving the selection
    OUTPUTS:
		None: None, updates widget config path
    """
    file, _ = QFileDialog.getOpenFileName(
        widget,
        "Select Existing Configuration File",
        "",
        "JSON Files (*.json);;All Files (*)"
    )
    if file:
        widget.update_config_path(file)
