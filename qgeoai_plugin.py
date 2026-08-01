# -*- coding: utf-8 -*-

"""
QGeoAI Toolkit

Main plugin class.

Author:
Dimitra Pappa
"""

import os

from qgis.PyQt.QtGui import QIcon
from qgis.PyQt.QtWidgets import QAction

from .qgeoai_dialog import QGeoAIDialog


class QGeoAIPlugin:
    """
    Main QGIS plugin class.
    """

    def __init__(self, iface):
        """
        Initialize the plugin.

        Parameters
        ----------
        iface : QgisInterface
            QGIS application interface.
        """

        self.iface = iface

        self.plugin_dir = os.path.dirname(
            os.path.abspath(__file__)
        )

        self.action = None

        self.dialog = None

        self.menu = "&QGeoAI Toolkit"

        self.toolbar = self.iface.addToolBar(
            "QGeoAI Toolkit"
        )

        self.toolbar.setObjectName(
            "QGeoAI Toolkit"
        )

    def initGui(self):
        """
        Create the plugin action, menu entry and toolbar button.
        """

        icon_path = os.path.join(
            self.plugin_dir,
            "icon.png"
        )

        icon = QIcon(icon_path)

        self.action = QAction(
            icon,
            "QGeoAI Toolkit",
            self.iface.mainWindow()
        )

        self.action.setObjectName(
            "QGeoAI Toolkit"
        )

        self.action.setStatusTip(
            "GeoAI tools for remote sensing "
            "and machine learning"
        )

        self.action.setWhatsThis(
            "Open the QGeoAI Toolkit"
        )

        self.action.triggered.connect(
            self.run
        )

        self.iface.addPluginToMenu(
            self.menu,
            self.action
        )

        self.toolbar.addAction(
            self.action
        )

    def unload(self):
        """
        Remove the plugin action, menu entry and toolbar.
        """

        if self.action is not None:

            self.iface.removePluginMenu(
                self.menu,
                self.action
            )

            self.toolbar.removeAction(
                self.action
            )

            self.action.deleteLater()

            self.action = None

        if self.dialog is not None:

            self.dialog.close()

            self.dialog.deleteLater()

            self.dialog = None

        if self.toolbar is not None:

            self.iface.mainWindow().removeToolBar(
                self.toolbar
            )

            self.toolbar.deleteLater()

            self.toolbar = None

    def run(self):
        """
        Open the main QGeoAI Toolkit dialog.
        """

        if self.dialog is None:

            self.dialog = QGeoAIDialog(
                self.iface
            )

        self.dialog.show()

        self.dialog.raise_()

        self.dialog.activateWindow()