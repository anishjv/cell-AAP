from __future__ import annotations
from napari.viewer import Viewer
from qtpy import QtWidgets
from qtpy.QtCore import Qt
from typing import Optional
from cell_AAP.napari import sub_widgets  # type: ignore


class cellAAPWidget(QtWidgets.QScrollArea):
    """
    cellAAPWidget GUI Class
    -----------------------
    INPUTS:
        napari_viewer: Viewer
        cfg: Any
        batch: Optional[bool]
    """

    def __getitem__(self, key: str):
        return self._widgets[key]

    def __init__(
        self, napari_viewer: Viewer, cfg, batch: Optional[bool] = False
    ) -> None:
        """
        Instantiates the primary widget in napari
        -----------------------------------------
        INPUTS:
            napari_viewer: Viewer
            cfg: Any
            batch: Optional[bool]
        RETURNS:
            None
        """
        super().__init__()

        self.viewer = napari_viewer
        self.cfg = cfg
        self.configured = False
        self.inference_cache = []
        self.batch = batch

        self.setWidgetResizable(True)  # noqa: FBT003
        self.full_spectrum_files = []
        self.flouro_files = []

        self._main_layout = QtWidgets.QVBoxLayout()
        self._main_widget = QtWidgets.QWidget()
        self._main_widget.setLayout(self._main_layout)
        self.setWidget(self._main_widget)

        # Create widgets and add to layout in temporal order
        self._widgets = {}
        self._add_file_selection_widgets()
        self._add_range_selection_widgets()
        self._add_config_widgets()
        self._add_inference_widgets()
        self._add_results_widgets()

        for name, widget in self._widgets.items():
            self.__setattr__(name, widget)

    def _add_file_selection_widgets(self):
        """
        Adds file selection widgets in temporal order (first step)
        ---------------------------------------------------------
        RETURNS:
            None
        """
        
        # File selection group
        file_group = QtWidgets.QGroupBox("1. Select Image")
        file_layout = QtWidgets.QVBoxLayout()
        
        if not self.batch:
            # Single image selection
            self.image_selector = QtWidgets.QPushButton("Select Movie")
            self.image_selector.setToolTip("Select an image to run inference on")
            self.image_selector.setStyleSheet(
                "QPushButton { background-color: #4A90E2; color: white; font-weight: bold; padding: 8px; }\n"
                "QPushButton:hover { background-color: #357ABD; }\n"
                "QPushButton:disabled { background-color: rgba(74, 144, 226, 0.3); color: rgba(255,255,255,0.6); }"
            )
            file_layout.addWidget(self.image_selector)
            
            self.display_button = QtWidgets.QPushButton("Display Movie")
            self.display_button.setToolTip("Display selected image")
            self.display_button.setEnabled(False)
            self.display_button.setStyleSheet(
                "QPushButton { background-color: #4A90E2; color: white; font-weight: bold; padding: 8px; }\n"
                "QPushButton:hover { background-color: #357ABD; }\n"
                "QPushButton:disabled { background-color: rgba(74, 144, 226, 0.3); color: rgba(255,255,255,0.6); }"
            )
            file_layout.addWidget(self.display_button)
            
            self._widgets.update({
                "image_selector": self.image_selector,
                "display_button": self.display_button
            })
        else:
            # Batch file selection
            batch_widgets = sub_widgets.create_batch_widgets()
            self._widgets.update(batch_widgets)
            
            # Style batch widgets
            for widget_name, widget in batch_widgets.items():
                if isinstance(widget, QtWidgets.QPushButton):
                    widget.setStyleSheet(
                        "QPushButton { background-color: #4A90E2; color: white; font-weight: bold; padding: 8px; }\n"
                        "QPushButton:hover { background-color: #357ABD; }\n"
                        "QPushButton:disabled { background-color: rgba(74, 144, 226, 0.3); color: rgba(255,255,255,0.6); }"
                    )
            
            file_layout.addWidget(batch_widgets["full_spectrum_file_list"])
            
            # Add/Remove buttons in a horizontal layout
            btn_layout = QtWidgets.QHBoxLayout()
            btn_layout.addWidget(batch_widgets["add_button"])
            btn_layout.addWidget(batch_widgets["remove_button"])
            file_layout.addLayout(btn_layout)
        
        file_group.setLayout(file_layout)
        self._main_layout.addWidget(file_group)

    def _add_range_selection_widgets(self):
        """
        Adds range selection widgets (second step)
        -----------------------------------------
        RETURNS:
            None
        """
        
        # Always create inference widgets (button + progress bar), but
        # only show the frame range slider for single-image mode.
        range_widgets = sub_widgets.create_inf_widgets()
        self._widgets.update(range_widgets)

        if not self.batch:
            # Range selection group (single image mode only)
            range_group = QtWidgets.QGroupBox("2. Set Frame Range")
            range_layout = QtWidgets.QVBoxLayout()

            # Style range slider
            range_slider = range_widgets["range_slider"]
            range_slider.setToolTip("Select the frames of the movie over which to run inference")
            range_layout.addWidget(range_slider)

            range_group.setLayout(range_layout)
            self._main_layout.addWidget(range_group)

    def _add_config_widgets(self):
        """
        Adds configuration widgets (third step)
        --------------------------------------
        RETURNS:
            None
        """
        
        # Configuration group
        config_group = QtWidgets.QGroupBox("3. Configure Model")
        config_layout = QtWidgets.QFormLayout()
        
        config_widgets = sub_widgets.create_config_widgets()
        self._widgets.update({key: value[1] for key, value in config_widgets.items()})
        
        # Style config widgets
        for label, widget in config_widgets.values():
            label_widget = QtWidgets.QLabel(label)
            label_widget.setToolTip(widget.toolTip())
            config_layout.addRow(label_widget, widget)
            
            # Style buttons
            if isinstance(widget, QtWidgets.QPushButton):
                widget.setStyleSheet(
                    "QPushButton { background-color: #4A90E2; color: white; font-weight: bold; padding: 8px; }\n"
                    "QPushButton:hover { background-color: #357ABD; }\n"
                    "QPushButton:disabled { background-color: rgba(74, 144, 226, 0.3); color: rgba(255,255,255,0.6); }"
                )
        
        config_layout.setLabelAlignment(Qt.AlignLeft)
        config_group.setLayout(config_layout)
        self._main_layout.addWidget(config_group)

    def _add_inference_widgets(self):
        """
        Adds inference widgets (fourth step)
        -----------------------------------
        RETURNS:
            None
        """
        
        # Inference group
        inference_group = QtWidgets.QGroupBox("4. Run Inference")
        inference_layout = QtWidgets.QVBoxLayout()
        
        # Get inference button from range widgets
        inference_button = self._widgets["inference_button"]
        inference_button.setText("Run Inference")
        inference_button.setToolTip("Run Inference")
        inference_button.setStyleSheet(
            "QPushButton { background-color: #4A90E2; color: white; font-weight: bold; padding: 12px; font-size: 14px; }\n"
            "QPushButton:hover { background-color: #357ABD; }\n"
            "QPushButton:disabled { background-color: rgba(74, 144, 226, 0.3); color: rgba(255,255,255,0.6); }"
        )
        inference_layout.addWidget(inference_button)
        
        # Progress bar(s)
        progress_bar = self._widgets["progress_bar"]
        inference_layout.addWidget(progress_bar)

        # In batch mode, add a second progress bar to indicate image progress
        if self.batch:
            progress_bar_images = QtWidgets.QProgressBar()
            progress_bar_images.setTextVisible(True)
            progress_bar_images.setFormat("Image %v/%m")
            self._widgets["progress_bar_images"] = progress_bar_images
            inference_layout.addWidget(progress_bar_images)
        
        inference_group.setLayout(inference_layout)
        self._main_layout.addWidget(inference_group)

    def _add_results_widgets(self):
        """
        Adds results widgets (fifth step)
        ---------------------------------
        RETURNS:
            None
        """
        
        # Results group
        results_group = QtWidgets.QGroupBox("5. Save Results")
        results_layout = QtWidgets.QVBoxLayout()
        
        # Results combo box for selecting which inference result to save
        save_combo_box = QtWidgets.QComboBox()
        save_combo_box.setToolTip("Select which inference result to save")
        self._widgets["save_combo_box"] = save_combo_box
        results_layout.addWidget(save_combo_box)
        

        
        # Save button
        self.save_selector = QtWidgets.QPushButton("Save Results")
        self.save_selector.setToolTip("Click to save the inference results")
        self.save_selector.setEnabled(False)
        self.save_selector.setStyleSheet(
            "QPushButton { background-color: #4A90E2; color: white; font-weight: bold; padding: 8px; }\n"
            "QPushButton:hover { background-color: #357ABD; }\n"
            "QPushButton:disabled { background-color: rgba(74, 144, 226, 0.3); color: rgba(255,255,255,0.6); }"
        )
        self._widgets["save_selector"] = self.save_selector
        results_layout.addWidget(self.save_selector)
        
        # Display results button
        self.results_display = QtWidgets.QPushButton("Display Results")
        self.results_display.setToolTip('Display Inference and tracking results if they exist')
        self.results_display.setEnabled(False)
        self.results_display.setStyleSheet(
            "QPushButton { background-color: #4A90E2; color: white; font-weight: bold; padding: 8px; }\n"
            "QPushButton:hover { background-color: #357ABD; }\n"
            "QPushButton:disabled { background-color: rgba(74, 144, 226, 0.3); color: rgba(255,255,255,0.6); }"
        )
        self._widgets["results_display"] = self.results_display
        results_layout.addWidget(self.results_display)
        
        results_group.setLayout(results_layout)
        self._main_layout.addWidget(results_group)
