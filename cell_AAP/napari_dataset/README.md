# Cell-APP Dataset Generation GUI

This napari plugin provides a graphical user interface for generating Cell-APP datasets.

## Features

1. **File Selection**: Load DNA and phase image files (supports TIFF, JPEG, PNG, BMP)
2. **Configuration Management**: Edit default parameters through JSON configuration files
3. **Extra Properties**: Add custom property functions through Python files
4. **Processing Pipeline**: Run `get()`, `crop()`, and `gen_df()` methods
5. **Results Visualization**: Display phase images with segmentations and DNA images with prompts
6. **Results Navigation**: Scroll through results for each image pair
7. **Data Export**: Save all outputs as NumPy arrays

## Usage

1. **Launch napari** and install this plugin
2. **Select Files**: Use the "Select DNA Images" and "Select Phase Images" buttons to choose your image files
3. **Configure Parameters**: Click "Edit Configuration" to create/modify a JSON configuration file
4. **Add Extra Properties**: Click "Edit Extra Properties" to create/modify a Python file with custom property functions
5. **Generate Dataset**: Click "Generate Dataset" to run the complete pipeline
6. **View Results**: Use the navigation buttons to scroll through results
7. **Save Results**: Click "Save Results" to export all data as NumPy arrays

## File Structure

- `main.py`: Main entry point and processing logic
- `ui.py`: Main GUI widget class
- `fileio.py`: File selection and saving functions
- `config_editor.py`: Configuration and extra properties editors
- `results_viewer.py`: Results display and navigation
- `napari.yaml`: Plugin configuration
- `__init__.py`: Package initialization

## Configuration

The configuration file should be a JSON file with the following structure:

```json
{
    "iou_thresh": 0.5,
    "threshold_method": "otsu",
    "min_area": 100,
    "max_area": 10000,
    "gaussian_sigma": 1.0,
    "background_subtraction": true,
    "sam_model_type": "vit_h",
    "sam_checkpoint": "sam_vit_h_4b8939.pth"
}
```

## Extra Properties

Extra properties should be defined in a Python file with functions that take a region as input and return a scalar value:

```python
def custom_area_ratio(region):
    """Example custom property"""
    bbox_area = (region.bbox[2] - region.bbox[0]) * (region.bbox[3] - region.bbox[1])
    return region.area / bbox_area if bbox_area > 0 else 0.0
```

## Output Files

The plugin saves the following NumPy arrays:
- `df_whole.npy`: Combined dataframe with all results
- `roi_data.npy`: DNA ROI data
- `phase_roi_data.npy`: Phase ROI data
- `segmentations.npy`: Segmentation masks
- `cleaned_binary_roi.npy`: Cleaned binary ROIs
- `cleaned_scalar_roi.npy`: Cleaned scalar ROIs
- `file_names.npy`: Original file names
