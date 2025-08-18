from __future__ import annotations
from napari.viewer import Viewer
from qtpy import QtWidgets
from qtpy.QtCore import Qt
from typing import Optional, List, Dict, Any


class DatasetGenerationWidget(QtWidgets.QScrollArea):
    """
    Dataset generation widget for Cell-APP with file selection, config, SAM, and results UI.
    -------------------------------------------------------------------------------------------------------
    INPUTS:
		napari_viewer: Viewer, active napari viewer instance
    OUTPUTS:
		None: None, initializes widget state and child controls
    """

    def __init__(self, napari_viewer: Viewer) -> None:
        """
        Instantiate the primary dataset generation widget in napari.
        -------------------------------------------------------------------------------------------------------
        INPUTS:
		napari_viewer: Viewer, active napari viewer instance
        OUTPUTS:
		None: None
        """
        super().__init__()

        self.viewer = napari_viewer
        self.dna_files = []
        self.phase_files = []
        self.config_file_path = None
        self.results = {}
        self.current_result_index = 0
        self.last_assembled_zip_path = None
        self.last_assembled_name = None

        self.setWidgetResizable(True)
        self._main_layout = QtWidgets.QVBoxLayout()
        self._main_widget = QtWidgets.QWidget()
        self._main_widget.setLayout(self._main_layout)
        self.setWidget(self._main_widget)

        # Create widgets and add to layout
        self._create_file_selection_widgets()
        self._create_configuration_widgets()
        self._create_processing_widgets()
        self._create_results_widgets()
        self._create_footer_widgets()

    def _create_footer_widgets(self):
        """
        Create a footer row with a small Share Dataset button aligned to the bottom-right.
        """
        footer_layout = QtWidgets.QHBoxLayout()
        footer_layout.setContentsMargins(0, 0, 0, 0)
        footer_layout.addStretch(1)
        # Share dataset (Zenodo upload via PAT) in the bottom-right
        self.share_button = QtWidgets.QPushButton("Share dataset")
        self.share_button.setToolTip("Package and upload your dataset to Zenodo (requires a Personal Access Token)")
        self.share_button.setEnabled(False)
        self.share_button.setStyleSheet(
            "QPushButton { background-color: #0B8457; color: white; font-weight: bold; padding: 4px 10px; font-size: 10px; }\n"
            "QPushButton:disabled { background-color: rgba(154, 51, 36, 0.15); color: rgba(255,255,255,0.6); }"
        )
        self.share_button.setFixedHeight(28)
        footer_layout.addWidget(self.share_button)
        self._main_layout.addLayout(footer_layout)

    def _create_file_selection_widgets(self):
        """
        Create file selection controls for DNA and Phase image lists.
        -------------------------------------------------------------------------------------------------------
        INPUTS:
		None: None
        OUTPUTS:
		None: None
        """
        
        # File selection group
        file_group = QtWidgets.QGroupBox("File Selection")
        file_layout = QtWidgets.QFormLayout()
        
        # DNA file selection
        self.dna_selector = QtWidgets.QPushButton("Select prompt-creation images")
        self.dna_selector.setToolTip("Select one or more files")
        file_layout.addRow(self.dna_selector)
        
        # DNA file list display
        self.dna_file_list = QtWidgets.QListWidget()
        self.dna_file_list.setMaximumHeight(100)
        # Enable drag-and-drop reordering
        self.dna_file_list.setSelectionMode(QtWidgets.QAbstractItemView.SingleSelection)
        self.dna_file_list.setDragEnabled(True)
        self.dna_file_list.setAcceptDrops(True)
        self.dna_file_list.setDropIndicatorShown(True)
        self.dna_file_list.setDragDropMode(QtWidgets.QAbstractItemView.InternalMove)
        file_layout.addRow(self.dna_file_list)
        
        # Phase file selection
        self.phase_selector = QtWidgets.QPushButton("Select transmitted-light images")
        self.phase_selector.setToolTip("Select one or more files")
        file_layout.addRow(self.phase_selector)
        
        # Phase file list display
        self.phase_file_list = QtWidgets.QListWidget()
        self.phase_file_list.setMaximumHeight(100)
        # Enable drag-and-drop reordering
        self.phase_file_list.setSelectionMode(QtWidgets.QAbstractItemView.SingleSelection)
        self.phase_file_list.setDragEnabled(True)
        self.phase_file_list.setAcceptDrops(True)
        self.phase_file_list.setDropIndicatorShown(True)
        self.phase_file_list.setDragDropMode(QtWidgets.QAbstractItemView.InternalMove)
        file_layout.addRow(self.phase_file_list)
        
        file_group.setLayout(file_layout)
        self._main_layout.addWidget(file_group)
        
        # Connect model row-moved to update underlying file lists
        try:
            self.dna_file_list.model().rowsMoved.connect(self._on_dna_list_reordered)
        except Exception:
            pass
        try:
            self.phase_file_list.model().rowsMoved.connect(self._on_phase_list_reordered)
        except Exception:
            pass

    def _create_configuration_widgets(self):
        """
        Create configuration, extra properties, and SAM model controls.
        -------------------------------------------------------------------------------------------------------
        INPUTS:
		None: None
        OUTPUTS:
		None: None
        """
        
        # Configuration group
        config_group = QtWidgets.QGroupBox("Configuration")
        config_layout = QtWidgets.QFormLayout()
        
        # Configuration editor button
        self.config_editor_button = QtWidgets.QPushButton("Create new config")
        self.config_editor_button.setToolTip("Open configuration editor to modify default parameters")
        
        # Select existing config button
        self.config_select_button = QtWidgets.QPushButton("Use existing config")
        self.config_select_button.setToolTip("Browse for an existing configuration JSON file")
        
        # Place both buttons side-by-side
        config_btns_container = QtWidgets.QWidget()
        config_btns_layout = QtWidgets.QHBoxLayout()
        config_btns_layout.setContentsMargins(0, 0, 0, 0)
        config_btns_layout.setSpacing(8)
        config_btns_layout.addWidget(self.config_editor_button)
        config_btns_layout.addWidget(self.config_select_button)
        config_btns_container.setLayout(config_btns_layout)
        config_layout.addRow("Configuration:", config_btns_container)
        
        # Instructional hint
        config_choice_hint = QtWidgets.QLabel("Choose one: Create new or Use existing")
        config_choice_hint.setStyleSheet("color: gray; font-style: italic;")
        config_layout.addRow("", config_choice_hint)
        
        # Configuration file path display
        self.config_path_label = QtWidgets.QLabel("No configuration file selected")
        self.config_path_label.setStyleSheet("color: red; font-style: italic;")
        config_layout.addRow("", self.config_path_label)

        # SAM model controls (Simplified)
        self.sam_model_selector = QtWidgets.QComboBox()
        self.sam_model_selector.setToolTip("Select a SAM model to load (LM/EM variants)")

        self.sam_load_button = QtWidgets.QPushButton("Load selected SAM")
        self.sam_load_button.setToolTip("Download (if needed) and load the selected SAM model")

        # Place selector and load button side-by-side
        sam_btns_container = QtWidgets.QWidget()
        sam_btns_layout = QtWidgets.QHBoxLayout()
        sam_btns_layout.setContentsMargins(0, 0, 0, 0)
        sam_btns_layout.setSpacing(8)
        sam_btns_layout.addWidget(self.sam_model_selector)
        sam_btns_layout.addWidget(self.sam_load_button)
        sam_btns_container.setLayout(sam_btns_layout)
        config_layout.addRow("SAM:", sam_btns_container)
        
        config_group.setLayout(config_layout)
        self._main_layout.addWidget(config_group)

    def _create_processing_widgets(self):
        """
        Create processing controls (run button and progress bar).
        -------------------------------------------------------------------------------------------------------
        INPUTS:
		None: None
        OUTPUTS:
		None: None
        """
        
        # Processing group
        processing_group = QtWidgets.QGroupBox("Processing")
        processing_layout = QtWidgets.QVBoxLayout()
        
        # Process button
        self.process_button = QtWidgets.QPushButton("Generate dataset")
        self.process_button.setToolTip("Run the complete dataset generation pipeline")
        self.process_button.setStyleSheet("QPushButton { background-color: #9A3324; color: white; font-weight: bold; padding: 8px; }")
        processing_layout.addWidget(self.process_button)
        
        processing_group.setLayout(processing_layout)
        self._main_layout.addWidget(processing_group)

    def _create_results_widgets(self):
        """
        Create results navigation and save controls.
        -------------------------------------------------------------------------------------------------------
        INPUTS:
		None: None
        OUTPUTS:
		None: None
        """
        
        # Results group
        results_group = QtWidgets.QGroupBox("Results")
        results_layout = QtWidgets.QVBoxLayout()
        
        # Results navigation
        nav_layout = QtWidgets.QHBoxLayout()
        
        self.prev_result_button = QtWidgets.QPushButton("← Previous")
        self.prev_result_button.setEnabled(False)
        nav_layout.addWidget(self.prev_result_button)
        
        self.result_counter_label = QtWidgets.QLabel("No results")
        self.result_counter_label.setAlignment(Qt.AlignCenter)
        nav_layout.addWidget(self.result_counter_label)
        
        self.next_result_button = QtWidgets.QPushButton("Next →")
        self.next_result_button.setEnabled(False)
        nav_layout.addWidget(self.next_result_button)
        
        results_layout.addLayout(nav_layout)
        
        # Save button
        self.save_button = QtWidgets.QPushButton("Save results")
        self.save_button.setToolTip("Save all generated results as numpy arrays")
        self.save_button.setEnabled(False)
        self.save_button.setStyleSheet(
            "QPushButton { background-color: #9A3324; color: white; font-weight: bold; padding: 8px; }\n"
            "QPushButton:disabled { background-color: rgba(154, 51, 36, 0.15); color: rgba(255,255,255,0.6); }"
        )
        results_layout.addWidget(self.save_button)

        # Assemble COCO Dataset (full row)
        self.coco_button = QtWidgets.QPushButton("Assemble COCO dataset")
        self.coco_button.setToolTip("Assemble COCO-style train/test datasets from current results")
        self.coco_button.setEnabled(False)
        self.coco_button.setStyleSheet(
            "QPushButton { background-color: #9A3324; color: white; font-weight: bold; padding: 6px 12px; font-size: 12px; }\n"
            "QPushButton:disabled { background-color: rgba(154, 51, 36, 0.15); color: rgba(255,255,255,0.6); }"
        )
        
        # Button heights
        self.coco_button.setFixedHeight(34)

        # Original layout: Assemble button full row under Save
        results_layout.addWidget(self.coco_button)

        results_group.setLayout(results_layout)
        self._main_layout.addWidget(results_group)

    def update_file_lists(self):
        """
        Update the DNA and Phase file list widgets with current selections.
        -------------------------------------------------------------------------------------------------------
        INPUTS:
		None: None (uses self.dna_files/self.phase_files)
        OUTPUTS:
		None: None
        """
        
        # Update DNA file list
        self.dna_file_list.clear()
        for file_path in self.dna_files:
            parts = str(file_path).split('/')
            display = '/'.join(parts[-2:]) if len(parts) >= 2 else parts[-1]
            item = QtWidgets.QListWidgetItem(display)
            item.setData(Qt.UserRole, file_path)
            self.dna_file_list.addItem(item)
        
        # Update phase file list
        self.phase_file_list.clear()
        for file_path in self.phase_files:
            parts = str(file_path).split('/')
            display = '/'.join(parts[-2:]) if len(parts) >= 2 else parts[-1]
            item = QtWidgets.QListWidgetItem(display)
            item.setData(Qt.UserRole, file_path)
            self.phase_file_list.addItem(item)

    def update_config_path(self, file_path: str):
        """
        Update the configuration file path display label and internal path.
        -------------------------------------------------------------------------------------------------------
        INPUTS:
		file_path: str, absolute path to the configuration JSON
        OUTPUTS:
		None: None
        """
        if file_path:
            self.config_file_path = file_path
            self.config_path_label.setText(file_path)
            self.config_path_label.setStyleSheet("color: black;")
        else:
            self.config_file_path = None
            self.config_path_label.setText("No configuration file selected")
            self.config_path_label.setStyleSheet("color: gray;")

    # No checkpoint UI; selection is resolved via registry at load time

    def update_result_counter(self):
        """
        Update result counter label and enable/disable navigation/save controls.
        -------------------------------------------------------------------------------------------------------
        INPUTS:
		None: None
        OUTPUTS:
		None: None
        """
        if self.results and 'file_names' in self.results:
            total_results = len(self.results['file_names'])
            if total_results > 0:
                current = self.current_result_index + 1
                self.result_counter_label.setText(f"Result {current} of {total_results}")
                self.prev_result_button.setEnabled(self.current_result_index > 0)
                self.next_result_button.setEnabled(self.current_result_index < total_results - 1)
                self.save_button.setEnabled(True)
                self.coco_button.setEnabled(True)
            else:
                self.result_counter_label.setText("No results")
                self.prev_result_button.setEnabled(False)
                self.next_result_button.setEnabled(False)
                self.save_button.setEnabled(False)
                self.coco_button.setEnabled(False)
                self.share_button.setEnabled(False)
        else:
            self.result_counter_label.setText("No results")
            self.prev_result_button.setEnabled(False)
            self.next_result_button.setEnabled(False)
            self.save_button.setEnabled(False)
            self.coco_button.setEnabled(False)
            self.share_button.setEnabled(False)

    # Handle list reordering to keep underlying arrays in sync
    def _on_dna_list_reordered(self, *args):
        new_order = []
        for i in range(self.dna_file_list.count()):
            item = self.dna_file_list.item(i)
            path = item.data(Qt.UserRole) or item.text()
            new_order.append(path)
        self.dna_files = new_order

    def _on_phase_list_reordered(self, *args):
        new_order = []
        for i in range(self.phase_file_list.count()):
            item = self.phase_file_list.item(i)
            path = item.data(Qt.UserRole) or item.text()
            new_order.append(path)
        self.phase_files = new_order