from __future__ import annotations
from napari.viewer import Viewer
from qtpy import QtWidgets
from qtpy.QtCore import Qt
from typing import Optional
from cell_AAP.napari import sub_widgets  # type: ignore


class cellAAPWidget(QtWidgets.QScrollArea):
    "cellAAPWidget GUI Class"

    def __getitem__(self, key: str):
        return self._widgets[key]

    def __init__(
        self, napari_viewer: Viewer, cfg, batch: Optional[bool] = False
    ) -> None:
        """Instantiates the primary widget in napari.

        Args:
            napari_viewer: A napari viewer instance
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
        self._tabs = QtWidgets.QTabWidget()

        # Create widgets and add to layout
        self._widgets = {}
        self._add_inf_widgets()
        if batch:
            self._add_batch_widgets()
            self._add_naming_widgets()
        else:
            self._add_file_widgets()
        self._add_config_widgets()
        self._add_save_widgets()

        self._main_layout.addWidget(self._tabs, stretch=0)

        for name, widget in self._widgets.items():
            self.__setattr__(name, widget)

    def _add_inf_widgets(self):
        "Adds disp_inf_widgets which are the output of sub_widgets.create_disp_ing_widgets. The aforementioned function outputs -> dict[str: QtWidgets.QWidget]"

        disp_inf_widgets = sub_widgets.create_inf_widgets()
        self._widgets.update(disp_inf_widgets)
        widget_holder = QtWidgets.QGroupBox("Inference")
        layout = QtWidgets.QFormLayout()
        for widget in disp_inf_widgets.values():
            layout.addRow(widget)

        widget_holder.setLayout(layout)
        self._main_layout.addWidget(widget_holder, stretch=0)

    def _add_save_widgets(self):
        "Adds save_widgets with are the outputs of sub_widgets.create_save_widgets."

        save_widgets = sub_widgets.create_save_widgets(batch=self.batch)
        layout = QtWidgets.QFormLayout()

        if self.batch:
            save_widgets.pop("save_selector")


        layout.setLabelAlignment(Qt.AlignLeft)
        self._widgets.update(save_widgets)
        for widget in save_widgets.values():
            layout.addRow(widget)

        tab = QtWidgets.QWidget()
        tab.setLayout(layout)
        self._tabs.addTab(tab, "\N{GEAR}" + " Analysis")

    def _add_file_widgets(self):
        "Adds file_widgets which are the output of sub_widgets.create_file_selector_widgets. The aforementioned function outputs -> dict[str: QtWidgets.QWidget]"

        file_widgets = sub_widgets.create_file_selector_widgets()
        self._widgets.update(file_widgets)

        layout = QtWidgets.QFormLayout()
        for widget in file_widgets.values():
            layout.addRow(widget)

        widget_holder = QtWidgets.QGroupBox("Movie Selector")
        widget_holder.setLayout(layout)
        self._main_layout.addWidget(widget_holder, stretch=0)

    def _add_config_widgets(self):
        "Adds config_widgets which are the output of sub_widgets.create_config_widgets. The aforementioned function outputs -> dict[str: tuple[str, QtWidgets.QWidget]]"

        config_widgets = sub_widgets.create_config_widgets()
        self._widgets.update({key: value[1] for key, value in config_widgets.items()})

        layout = QtWidgets.QFormLayout()
        for label, widget in config_widgets.values():
            label_widget = QtWidgets.QLabel(label)
            label_widget.setToolTip(widget.toolTip())
            layout.addRow(label_widget, widget)

        layout.setLabelAlignment(Qt.AlignLeft)

        tab = QtWidgets.QWidget()
        tab.setLayout(layout)
        self._tabs.addTab(tab, "\N{GEAR}" + " Inference")

    def _add_batch_widgets(self):
        "Adds batch_widgets which are the output of sub_widgets.create_batch_widgets. The aforementioned function outputs -> dict[str: QtWidgets.QWidget]"

        batch_widgets = sub_widgets.create_batch_widgets()
        self._widgets.update(batch_widgets)

        layout = QtWidgets.QGridLayout()
        layout.addWidget(
            batch_widgets["full_spectrum_file_list"], 0, 0, 1, 2, Qt.AlignCenter
        )
        batch_widgets.pop("full_spectrum_file_list")

        for i, widget in enumerate(batch_widgets.values()):
            layout.addWidget(widget, 1, i, 1, 1, Qt.AlignCenter)

        widget_holder = QtWidgets.QGroupBox("Batch Worker")
        widget_holder.setLayout(layout)
        self._main_layout.addWidget(widget_holder)

    def _add_naming_widgets(self):
        "Adds naming convention widgets which are the output of sub_widgets.create_naming_convention_widgets. The aforementioned function outputs -> dict[str: tuple[str, QtWidgets.QWidget]]"

        naming_widgets = sub_widgets.create_naming_convention_widgets()
        self._widgets.update({key: value[1] for key, value in naming_widgets.items()})

        layout = QtWidgets.QFormLayout()
        for label, widget in naming_widgets.values():
            label_widget = QtWidgets.QLabel(label)
            label_widget.setToolTip(widget.toolTip())
            layout.addRow(label_widget, widget)

        layout.setLabelAlignment(Qt.AlignLeft)

        widget_holder = QtWidgets.QGroupBox("\u03bb " + "Naming Conventions / Blanks")
        widget_holder.setLayout(layout)
        self._main_layout.addWidget(widget_holder, stretch=0)
