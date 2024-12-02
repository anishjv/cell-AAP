from __future__ import annotations
from qtpy import QtWidgets
from superqt import QLabeledRangeSlider
from typing import Union, Optional


def create_file_selector_widgets() -> dict[str, QtWidgets.QWidget]:
    """
    Creates File Selector Widgets
    ------------------------------
    RETURNS:
        widgets: dict
            - image_selector: QtWidgets.QPushButton
            - path_selector: QtWidgets.QPushButtons (These push buttons connect to a function that creates an instance of QtWidgets.QFileDialog)
    """

    image_selector = QtWidgets.QPushButton("\u03bb" + " ~ Full Spectrum")
    image_selector.setToolTip("Select an image to ultimately run inference on")
    widgets = {"image_selector": image_selector}

    flourescent_image_selector = QtWidgets.QPushButton("\u03bb" + " ~ Flourescent")
    image_selector.setToolTip(
        "Optionally select a flourescent image to use for analysis"
    )
    widgets["flourescent_image_selector"] = flourescent_image_selector

    display_button = QtWidgets.QPushButton("Display")
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

    analyze_check_box = QtWidgets.QCheckBox("Analyze")
    analyze_check_box.setToolTip(
        "Check to perform analysis when saving inference results, requires intensity image"
    )
    widgets = {"analyze_check_box": analyze_check_box}

    save_combo_box = QtWidgets.QComboBox()
    widgets["save_combo_box"] = save_combo_box

    path_selector = QtWidgets.QPushButton("Select Directory")
    path_selector.setToolTip(
        "Select a directory to ultimately store the inference results at"
    )
    widgets["path_selector"] = path_selector

    save_selector = QtWidgets.QPushButton("Save && Analyze")
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
    model_selector.addItem("U2OS")
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


def create_naming_convention_widgets() -> dict[str, tuple[str, QtWidgets.QWidget]]:
    """
    Creates Naming Convention Widgets
    ------------------------------
    RETURNS:
        widgets: dict
            - full_spec_format: QtWidgets.QLineEdit
            - flouro_format: QtWidgets.QLineEdit
    """

    full_spec_format = QtWidgets.QLineEdit()
    widgets = {
        "full_spec_format": ("Full Spectrum Naming Convention", full_spec_format)
    }

    flouro_format = QtWidgets.QLineEdit()
    widgets["flouro_format"] = ("Flourescence Naming Convention", flouro_format)

    flouro_media_blank = QtWidgets.QPushButton("Select Image")
    flouro_media_blank.setToolTip("Optional, add a stack of (x, y, z) equal to that of the stack being segmented")
    widgets["flouro_media_blank"] = ("Flourescence Media Blank", flouro_media_blank)

    trans_media_blank = QtWidgets.QPushButton("Select Image")
    trans_media_blank.setToolTip("Optional, add a stack of (x, y, z) equal to that of the stack being segmented")
    widgets["trans_media_blank"] = ("Translucent Media Blank", trans_media_blank)

    return widgets
