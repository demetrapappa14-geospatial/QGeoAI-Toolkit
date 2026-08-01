# -*- coding: utf-8 -*-

"""
QGeoAI Toolkit
QGIS plugin initialization file.

This file is automatically read by QGIS when the plugin is loaded.
"""


def classFactory(iface):
    """
    Create and return the main QGeoAI Toolkit plugin object.

    Parameters
    ----------
    iface : QgisInterface
        The QGIS application interface.

    Returns
    -------
    QGeoAIPlugin
        The main plugin instance.
    """

    from .qgeoai_plugin import QGeoAIPlugin

    return QGeoAIPlugin(iface)