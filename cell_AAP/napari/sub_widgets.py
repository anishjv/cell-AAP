from __future__ import annotations
from qtpy import QtWidgets


def create_file_selector_widgets() -> dict[str, QtWidgets.QWidget]:
    """
    Creates File Selector Widgets
    ------------------------------
    RETURNS:
        widgets: dict
            - image_selector: QtWidgets.QPushButton
            - path_selector: QtWidgets.QPushButtons (These push buttons connect to a function that creates an instance of QtWidgets.QFileDialog)
            - save_selector: QtWidgets.QCheckBox
    """

    image_selector = QtWidgets.QPushButton("Select Image")
    image_selector.setToolTip("Select an image to ultimately run inference on")

    widgets = {"image_selector": image_selector}

    path_selector = QtWidgets.QPushButton("Select Directory")
    path_selector.setToolTip(
        "Select a directory to ultimately store the inference results at"
    )

    widgets["path_selector"] = path_selector

    save_selector = QtWidgets.QPushButton("Save Inference")
    save_selector.setToolTip("Click to save the inference results")

    widgets["save_selector"] = save_selector

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
    model_selector.addItem('HeLa_former')
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

    set_configs = QtWidgets.QPushButton("Push")
    set_configs.setToolTip("Set Configurations")

    widgets["set_configs"] = ("Set Configurations", set_configs)

    return widgets


def create_disp_inf_widgets() -> dict[str, QtWidgets.QWidget]:
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

    display_button = QtWidgets.QPushButton("Display")
    display_button.setToolTip("Display selected image")

    widgets["display_button"] = display_button

    pbar = QtWidgets.QProgressBar()
    widgets["progress_bar"] = pbar

    return widgets
