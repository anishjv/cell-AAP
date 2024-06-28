from __future__ import annotations
from qtpy import QtWidgets
from typing import Union


def create_file_selector_widgets() -> dict[str, QtWidgets.QWidget]:
    """
    Creates File Selector Widgets
    ------------------------------
    RETURNS:
        widgets: dict
            - image_selector: QtWidgets.QPushButton
            - path_selector: QtWidgets.QPushButtons (These push buttons connect to a function that creates an instance of QtWidgets.QFileDialog)
    """

    image_selector = QtWidgets.QPushButton("Select Image")
    image_selector.setToolTip("Select an image to ultimately run inference on")

    widgets = {"image_selector": image_selector}

    display_button = QtWidgets.QPushButton("Display Image")
    display_button.setToolTip("Display selected image")

    widgets["display_button"] = display_button

    path_selector = QtWidgets.QPushButton("Select Directory")
    path_selector.setToolTip(
        "Select a directory to ultimately store the inference results at"
    )

    widgets["path_selector"] = path_selector

    return widgets


def create_save_widgets() -> tuple[dict[str, QtWidgets.QWidget], dict[str, tuple[str, QtWidgets.QWidget]]]:
    """
    Creates Inference Saving Widgets
    ------------------------------
    RETURNS:
        widgets: dict
            - save_selector: QtWidgets.QPushButton
            - save_combo_box: QtWidgets.QPushButton
    """

    analyze_check_box = QtWidgets.QCheckBox('Analyze')
    analyze_check_box.setToolTip('Check to perform analysis when saving inference results, requires intensity image')
    widgets = {'analyze_check_box': analyze_check_box}

    save_combo_box = QtWidgets.QComboBox()
    widgets['save_combo_box'] =  save_combo_box

    interframe_duration = QtWidgets.QSpinBox()
    interframe_duration.setRange(0, 100)
    interframe_duration.setValue(10)
    interframe_duration.setStepType(QtWidgets.QAbstractSpinBox.AdaptiveDecimalStepType)
    interframe_duration.setToolTip("Time between frames of the movie for analysis, in any units ")
    named_widgets = {'interframe_duration': ('Interframe Duration', interframe_duration)}

    save_selector = QtWidgets.QPushButton("Save/Analyze")
    save_selector.setToolTip("Click to save the inference results")

    widgets['save_selector'] = save_selector

    return widgets, named_widgets


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
    model_selector.addItem('HeLaViT')
    widgets = {"model_selector": ("Select Model", model_selector)}

    thresholder = QtWidgets.QDoubleSpinBox()
    thresholder.setRange(0, 100)
    thresholder.setValue(0.25)
    thresholder.setStepType(QtWidgets.QAbstractSpinBox.AdaptiveDecimalStepType)
    thresholder.setToolTip("Set Confidence Hyperparameter")
    thresholder.setWrapping(True)
    widgets["thresholder"] = ("Confidence Threshold", thresholder)

    confluency_est = QtWidgets.QSpinBox()
    confluency_est.setRange(100, 5000)
    confluency_est.setValue(2000)
    confluency_est.setStepType(QtWidgets.QAbstractSpinBox.AdaptiveDecimalStepType)
    confluency_est.setToolTip("Estimate the number of cells in a frame")

    widgets["confluency_est"] = ("Number of Cells (Approx.)", confluency_est)

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

    pbar = QtWidgets.QProgressBar()
    widgets["progress_bar"] = pbar

    return widgets
