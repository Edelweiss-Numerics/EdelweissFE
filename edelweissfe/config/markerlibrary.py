#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#  ---------------------------------------------------------------------
#
#  _____    _      _              _         _____ _____
# | ____|__| | ___| |_      _____(_)___ ___|  ___| ____|
# |  _| / _` |/ _ \ \ \ /\ / / _ \ / __/ __| |_  |  _|
# | |__| (_| |  __/ |\ V  V /  __/ \__ \__ \  _| | |___
# |_____\__,_|\___|_| \_/\_/ \___|_|___/___/_|   |_____|
#
#
#  Unit of Strength of Materials and Structural Analysis
#  University of Innsbruck,
#  2017 - today
#
#  Matthias Neuner matthias.neuner@uibk.ac.at
#
#  This file is part of EdelweissFE.
#
#  This library is free software; you can redistribute it and/or
#  modify it under the terms of the GNU Lesser General Public
#  License as published by the Free Software Foundation; either
#  version 2.1 of the License, or (at your option) any later version.
#
#  The full text of the license can be found in the file LICENSE.md at
#  the top level directory of EdelweissFE.
#  ---------------------------------------------------------------------

"""Registry of adaptive-mesh-refinement markers.

A marker decides which elements an adaptivity mechanism should refine, from either topological
information (an element/node set, a surface) or a field quantity (a fieldOutput expression, the
Zienkiewicz-Zhu recovered-gradient error). Markers are selected by name from a
``>>marker, type=<name>`` block; see :mod:`edelweissfe.adaptivity.marking` for the marker classes
and their per-type option schemas.

A stress-based criterion (e.g. Rankine, max principal stress >= f_t) needs no dedicated marker: it
is a ``type=fieldOutput`` marker whose ``expression`` uses the ``eigVal`` helper already available in
the expression stack -- see :mod:`edelweissfe.adaptivity.marking`.
"""

from edelweissfe.config import registry


def getMarkerClass(name: str) -> type:
    """Get the class of the requested AMR refinement marker.

    Resolved through the L3 registry (``marker`` category) rather than a hardcoded ``if/elif`` inside
    a single model modifier, so any adaptivity mechanism -- and any third-party package registering a
    marker via an entry point -- reaches the same marker library uniformly. An unknown name raises
    :class:`~edelweissfe.config.registry.RegistryLookupError`, naming the available markers, instead
    of a bare ``ValueError``.

    Parameters
    ----------
    name
        Marker name (case-insensitive): ``fieldOutput``, ``elementSet``, ``nodeSet``, ``surface`` or
        ``recoveryError``.

    Returns
    -------
    type
        The :class:`~edelweissfe.adaptivity.marking.MarkerBase` subclass. Construct it from a
        ``>>marker`` block's options via its
        :meth:`~edelweissfe.adaptivity.marking.MarkerBase.fromOptions` classmethod.
    """

    markerClass, _ = registry.lookup("marker", name)

    return markerClass
