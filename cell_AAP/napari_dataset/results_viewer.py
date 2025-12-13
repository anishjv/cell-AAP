from __future__ import annotations
import numpy as np
import napari.utils.notifications
from typing import Optional


def display_current_result(widget) -> None:
    """
    Display the current result (phase image + segmentations, DNA + prompts) in napari.
    -------------------------------------------------------------------------------------------------------
    INPUTS:
		widget: QtWidgets.QWidget, dataset widget with results and viewer
    OUTPUTS:
		None: None, updates viewer layers and result counter
    """
    if not widget.results or 'file_names' not in widget.results:
        return
    
    if not widget.results['file_names']:
        return
    
    try:
        # Clear existing layers
        widget.viewer.layers.clear()
        
        current_idx = widget.current_result_index
        
        # Display full phase image with segmentations
        if 'full_phase_images' in widget.results and len(widget.results['full_phase_images']) > current_idx:
            phase_data = widget.results['full_phase_images'][current_idx]
            if phase_data is not None:
                widget.viewer.add_image(phase_data, name="Transmitted-light image", colormap="gray")
                
                # Display segmentations if available
                if 'segmentations' in widget.results and len(widget.results['segmentations']) > current_idx:
                    frame_segmentations = widget.results['segmentations'][current_idx]
                    if frame_segmentations is not None and len(frame_segmentations) > 0:
                        combined_seg = np.zeros_like(phase_data, dtype=np.uint8)
                        for cell_idx, seg in enumerate(frame_segmentations):
                            if seg is not None:
                                # Dynamically unpack using packed shape
                                # seg has shape (packed_rows, width); original height = packed_rows * 8
                                unpack_count = seg.shape[0] * 8
                                mask = np.unpackbits(seg.astype(np.uint8), axis=0, count=unpack_count).astype(bool, copy=False)
                                combined_seg[mask] = cell_idx + 1
                        widget.viewer.add_labels(combined_seg.astype(np.uint32), name="Segmentations", opacity=0.1)
        
        # Display full DNA image and prompts
        if 'full_dna_images' in widget.results and len(widget.results['full_dna_images']) > current_idx:
            dna_data = widget.results['full_dna_images'][current_idx]
            if dna_data is not None:
                widget.viewer.add_image(dna_data, name="Prompt-creation image", colormap="gray")
                
                prompts_all = widget.results.get('prompts')
                if prompts_all is not None and len(prompts_all) > current_idx:
                    frame_prompts = prompts_all[current_idx]
                    if frame_prompts is not None:
                        arr = frame_prompts if isinstance(frame_prompts, np.ndarray) else np.asarray(frame_prompts)
                        if arr.size > 0:
                            if arr.ndim == 1:
                                arr = arr.reshape(1, -1)
                            ncols = arr.shape[-1]
                            if ncols == 2:
                                points = np.stack([arr[:, 1], arr[:, 0]], axis=1).astype(float)
                                widget.viewer.add_points(points, name="Prompts (points)", size=6, face_color='red')
                            elif ncols == 4:
                                # Boxes [x1, y2, x2, y1] -> rectangle corners
                                shapes = []
                                for row in arr:
                                    row_list = row.tolist() if isinstance(row, np.ndarray) else list(row)
                                    if len(row_list) != 4:
                                        continue
                                    x1, y2, x2, y1 = row_list
                                    shapes.append([[y1, x1], [y1, x2], [y2, x2], [y2, x1]])
                                widget.viewer.add_shapes(shapes, shape_type='rectangle', name="Prompts (boxes)", edge_color='red', face_color='transparent')
        
        # Update result counter
        widget.update_result_counter()
        
    except Exception as e:
        napari.utils.notifications.show_error(f"Error displaying result: {str(e)}")


def show_previous_result(widget) -> None:
    """
    Navigate to and display the previous result, if available.
    -------------------------------------------------------------------------------------------------------
    INPUTS:
		widget: QtWidgets.QWidget, dataset widget with results and viewer
    OUTPUTS:
		None: None
    """
    if widget.results and 'file_names' in widget.results:
        if widget.current_result_index > 0:
            widget.current_result_index -= 1
            display_current_result(widget)


def show_next_result(widget) -> None:
    """
    Navigate to and display the next result, if available.
    -------------------------------------------------------------------------------------------------------
    INPUTS:
		widget: QtWidgets.QWidget, dataset widget with results and viewer
    OUTPUTS:
		None: None
    """
    if widget.results and 'file_names' in widget.results:
        if widget.current_result_index < len(widget.results['file_names']) - 1:
            widget.current_result_index += 1
            display_current_result(widget)
