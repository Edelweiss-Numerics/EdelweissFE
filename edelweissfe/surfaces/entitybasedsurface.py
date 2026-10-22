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

from collections.abc import Iterable


class EntityBasedSurface(dict):
    """This is a surface based on underlying entities such as elements.
    It follows the concept like Abaqus element-based surfaces, i.e., defining
    the surface as a compound of element faces defined through face ids of type int and
    a list of entities.

    Parameters
    ----------
    name : str
        The name of the surface.
    faceToEntities : dict[int, Iterable]
        A dictionary mapping face ids (int) to an iterable collection of entities (e.g., elements
        or nodes) that share this face.
    """

    def __init__(self, name: str, faceToEntities: dict[int, Iterable]):

        self.name = name
        self._version = 0  #: bumped on every in-place mutation (replaceData); stable identity for AMR

        self._checkFaceToEntities(faceToEntities)

        super().__init__(faceToEntities)

    def _checkFaceToEntities(self, faceToEntities: dict[int, Iterable]):
        if not isinstance(faceToEntities, dict):
            raise TypeError("face_to_entities must be of type dict[int, list]")
        if not all(isinstance(k, int) for k in faceToEntities.keys()):
            raise TypeError("All keys of faceToEntities must be of type int")
        if not all(isinstance(v, Iterable) for v in faceToEntities.values()):
            raise TypeError("All values of faceToEntities must be iterable")

    def replaceData(self, faceToEntities: dict[int, Iterable]):
        """Replace the face-to-entity mapping in-place, preserving this object's identity so
        consumers that cached a reference (e.g. a distributed load's ``self._surface``) see the
        new mapping without re-fetching from the model.

        Parameters
        ----------
        faceToEntities
            The new dictionary mapping face ids (int) to entities sharing that face.
        """
        self._checkFaceToEntities(faceToEntities)
        self.clear()
        self.update(faceToEntities)
        self._version += 1
