from __future__ import annotations
from qtpy import QtWidgets
from superqt import QLabeledRangeSlider
from typing import Optional


def create_file_selector_widgets() -> dict[str, QtWidgets.QWidget]:
    """
    Creates File Selector Widgets
    ------------------------------
    RETURNS:
        widgets: dict
            - image_selector: QtWidgets.QPushButton
            - path_selector: QtWidgets.QPushButtons (These push buttons connect to a function that creates an instance of QtWidgets.QFileDialog)
    """

    image_selector = QtWidgets.QPushButton("Select Movie")
    image_selector.setToolTip("Select an image to ultimately run inference on")
    widgets = {"image_selector": image_selector}

    display_button = QtWidgets.QPushButton("Display Movie")
    display_button.setToolTip("Display selected image")
    widgets["display_button"] = display_button

    return widgets


def create_save_widgets(
    batch: Optional[bool] = False,
) -> dict[str, QtWidgets.QWidget]:
    """
    Creates Inference Saving Widgets
    ------------------------------
    RETURNS:
        widgets: dict
            - save_selector: QtWidgets.QPushButton
            - save_combo_box: QtWidgets.QPushButton
    """




    save_combo_box = QtWidgets.QComboBox()
    widgets = {"save_combo_box": save_combo_box}

    path_selector = QtWidgets.QPushButton("Select Directory")
    path_selector.setToolTip(
        "Select a directory to ultimately store the inference results at"
    )
    widgets["path_selector"] = path_selector

    save_selector = QtWidgets.QPushButton("Save Results")
    save_selector.setToolTip("Click to save the inference results")

    widgets["save_selector"] = save_selector

    results_display = QtWidgets.QPushButton("Display Results")
    results_display.setToolTip('Display Inference and tracking results if they exist')

    widgets['results_display'] = results_display

    return widgets


def create_config_widgets() -> dict[str, tuple[str, QtWidgets.QWidget]]:
    """
    Creates Configuration Widgets
    ------------------------------
    RETURNS:
        widgets: dict
            - thresholder: QtWidgets.QDoubleSpinBox
            - confluency_est: QtWidgets.QSpinBox
            - set_configs: QtWidgets.QPushButton
            - model_selector: QtWigets.QComboxBox
    """

    model_selector = QtWidgets.QComboBox()
    model_selector.addItem("HeLa")
    model_selector.addItem("HeLa_focal")
    model_selector.addItem("U2OS")
    model_selector.addItem("U2OS_focal")
    model_selector.addItem("HT1080")
    model_selector.addItem("HT1080_focal")
    model_selector.addItem("RPE1")
    model_selector.addItem("RPE1_focal")
    model_selector.addItem("general_focal")
    model_selector.addItem("HeLa_dead")
    model_selector.addItem("general_dead_focal")
    widgets = {"model_selector": ("Select Model", model_selector)}

    thresholder = QtWidgets.QDoubleSpinBox()
    thresholder.setRange(0, 100)
    thresholder.setValue(0.25)
    thresholder.setStepType(QtWidgets.QAbstractSpinBox.AdaptiveDecimalStepType)
    thresholder.setToolTip("Set Confidence Hyperparameter")
    thresholder.setWrapping(True)
    widgets["thresholder"] = ("Confidence", thresholder)

    confluency_est = QtWidgets.QSpinBox()
    confluency_est.setRange(100, 5000)
    confluency_est.setValue(2000)
    confluency_est.setStepType(QtWidgets.QAbstractSpinBox.AdaptiveDecimalStepType)
    confluency_est.setToolTip("Estimate the number of cells in a frame")

    widgets["confluency_est"] = ("Cell Quantity", confluency_est)

    keep_resized_checkbox = QtWidgets.QCheckBox("Keep 1024x1024")
    keep_resized_checkbox.setToolTip(
        "If checked, the output will remain 1024x1024 (Model Native).\n"
        "If unchecked, the output is projected back to the original image size."
    )
    keep_resized_checkbox.setChecked(False) # Default to False (Project back)
    
    # We add it to the widgets dict. 
    # The tuple format is (Label Text, Widget Object).
    widgets["keep_resized_checkbox"] = ("Output Size", keep_resized_checkbox)

    set_configs = QtWidgets.QPushButton("Push Configurations")
    set_configs.setToolTip("Set Configurations")

    widgets["set_configs"] = ("Set Configurations", set_configs)

    return widgets


def create_inf_widgets() -> dict[str, QtWidgets.QWidget]:
    """
    Creates Display and Inference Widgets
    ------------------------------
    RETURNS:
        widgets: dict
            - inference_button: QtWidgets.QPushButton
            - display_button: QtWidgets.QPushButton
            - pbar: QtWidgets.QProgressBar
    """

    inference_button = QtWidgets.QPushButton()
    inference_button.setText("Inference")
    inference_button.setToolTip("Run Inference")

    widgets = {"inference_button": inference_button}

    range_slider = QLabeledRangeSlider()
    range_slider.setToolTip(
        "Select the frames of the movie over which to run inference"
    )
    widgets["range_slider"] = range_slider

    pbar = QtWidgets.QProgressBar()
    widgets["progress_bar"] = pbar

    return widgets


def create_batch_widgets() -> dict[str, QtWidgets.QWidget]:
    """
    Creates Batch Worker Widgets
    ------------------------------
    RETURNS:
        widgets: dict
            - full_spectrum_file_list: QtWidgets.QListWidget
            - flouro_file_list: QtWidgets.QListWidget
            - add_button: QtWidgets.QPushButton
            - remove_button: QtWigets.QPushButton
            - file_list_toggle: QtWidgets.QPushButton
    """

    full_spectrum_file_list = QtWidgets.QListWidget()
    full_spectrum_file_list.setToolTip(
        "Add full spectrum files to the list, we will infer the flourescent files for you"
    )
    widgets = {"full_spectrum_file_list": full_spectrum_file_list}

    add_button = QtWidgets.QPushButton("Add Movie")
    widgets["add_button"] = add_button

    remove_button = QtWidgets.QPushButton("Remove Movie")
    widgets["remove_button"] = remove_button

    return widgets
