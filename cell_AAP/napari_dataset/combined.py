from __future__ import annotations
from qtpy import QtWidgets
import napari
from typing import Optional

import cell_AAP.napari.main as inference_main  # type: ignore
import cell_AAP.napari_dataset.main as dataset_main  # type: ignore


def create_combined_widget() -> QtWidgets.QWidget:
	"""
	Create a combined Napari dock widget with two tabs for Inference and Dataset Generation.
	-------------------------------------------------------------------------------------------------------
	INPUTS:
		None: None, no direct inputs required (uses napari.current_viewer())
	OUTPUTS:
		widget: QtWidgets.QWidget, container widget with a QTabWidget hosting both sub-widgets
	"""
	container = QtWidgets.QWidget()
	layout = QtWidgets.QVBoxLayout(container)
	container.setLayout(layout)

	# Create tab widget
	tabs = QtWidgets.QTabWidget(container)
	layout.addWidget(tabs)

	# Inference tab: reuse the original cell-APP inference widget
	inference_widget = inference_main.create_cellAAP_widget(batch=False)
	tabs.addTab(inference_widget, "Inference")

	# Dataset generation tab: use the new dataset-generation widget
	dataset_widget = dataset_main.create_dataset_generation_widget()
	tabs.addTab(dataset_widget, "Dataset Generation")

	return container
