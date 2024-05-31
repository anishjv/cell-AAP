from __future__ import annotations
from napari.viewer import Viewer
from qtpy import QtWidgets
from cell_AAP.napari import sub_widgets  # type: ignore


class cellAAPWidget(QtWidgets.QScrollArea):
    "cellAAPWidget GUI Class"

    def __getitem__(self, key: str) -> QtWidgets.QWidget:
        return self._widgets[key]

    def __init__(self, napari_viewer: Viewer, cfg) -> None:
        """Instantiates the primary widget in napari.

        Args:
            napari_viewer: A napari viewer instance
        """
        super().__init__()

        self.viewer = napari_viewer
        self.cfg = cfg
        self.configured = False

        self.setWidgetResizable(True)  # noqa: FBT003

        self._main_layout = QtWidgets.QVBoxLayout()
        self._main_widget = QtWidgets.QWidget()
        self._main_widget.setLayout(self._main_layout)
        self.setWidget(self._main_widget)
        self._tabs = QtWidgets.QTabWidget()

        # Create widgets and add to layout
        self._widgets = {}
        self._add_disp_inf_widgets()
        self._add_file_widgets()
        self._add_config_widgets()
        self._main_layout.addWidget(self._tabs, stretch=0)

        for name, widget in self._widgets.items():
            self.__setattr__(name, widget)

    def _add_disp_inf_widgets(self):
        "Adds disp_inf_widgets which are the output of sub_widgets.create_disp_ing_widgets. The aforementioned function outputs -> dict[str: QtWidgets.QWidget]"

        disp_inf_widgets = sub_widgets.create_disp_inf_widgets()
        self._widgets.update(disp_inf_widgets)
        widget_holder = QtWidgets.QGroupBox("cell-AAP")
        layout = QtWidgets.QFormLayout()
        for widget in disp_inf_widgets.values():
            layout.addRow(widget)

        widget_holder.setLayout(layout)
        self._main_layout.addWidget(widget_holder, stretch=0)

    def _add_file_widgets(self):
        "Adds file_widgets which are the output of sub_widgets.create_file_selector_widgets. The aforementioned function outputs -> dict[str: QtWidgets.QWidget]"

        file_widgets = sub_widgets.create_file_selector_widgets()
        self._widgets.update(file_widgets)

        layout = QtWidgets.QFormLayout()
        for widget in file_widgets.values():
            layout.addRow(widget)

        tab = QtWidgets.QWidget()
        tab.setLayout(layout)
        self._tabs.addTab(tab, "FileIO")

    def _add_config_widgets(self):
        "Adds config_widgets which are the output of sub_widgets.create_config_widgets. The aforementioned function outputs -> dict[str: tuple[str, QtWidgets.QWidget]]"

        config_widgets = sub_widgets.create_config_widgets()
        self._widgets.update({key: value[1] for key, value in config_widgets.items()})

        layout = QtWidgets.QFormLayout()
        for label, widget in config_widgets.values():
            label_widget = QtWidgets.QLabel(label)
            label_widget.setToolTip(widget.toolTip())
            layout.addRow(label_widget, widget)

        tab = QtWidgets.QWidget()
        tab.setLayout(layout)
        self._tabs.addTab(tab, "Configs")
