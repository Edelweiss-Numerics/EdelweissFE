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
#  Alexander Dummer alexander.dummer@uibk.ac.at
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
# Created on Wed Aug 31 08:35:06 2022
# @author: matthias

"""
A single quadrature point 'element' which drives a Marmot material directly, such that
materials can be investigated without any spatial discretization.

The element itself is agnostic of the material's base class. Everything base class
specific, i.e., the coupled fields, the layout of the state variables and how the
material response translates into the residual and the tangent, is delegated to a
material driver in :mod:`~edelweissfe.elements.marmotsingleqpelement.materialdrivers`.
"""

import numpy as np

from edelweissfe.elements.base.baseelement import BaseElement
from edelweissfe.elements.marmotsingleqpelement.materialdrivers import (
    materialDriverForMaterialType,
)
from edelweissfe.points.node import Node


class MarmotMaterialWrappingElement(BaseElement):
    def __init__(self, materialType: str, elNumber: int):
        """This element serves as a driver for MarmotMaterials,
        cf. `Marmot <https://github.com/MAteRialMOdelingToolbox/Marmot/>`_.

        It has a single quadrature point, and one (dummy) node.
        For interfacing with the different Marmot material base classes,
        specialized material drivers are used.

        The element allows to run quadrature point simulations investigating materials
        for development purposes.

        Parameters
        ----------
        materialType
            The Marmot material base class which should be driven, e.g.,
            MarmotMaterialHypoElastic.
        elNumber
            The number of the element."""

        self._elNumber = elNumber
        self._materialType = materialType
        self._nNodes = 1
        self._nSpatialDimensions = 0
        self._ensightType = "point"
        self._hasMaterial = False

        self._driver = materialDriverForMaterialType(materialType)()
        self._fields = (self._driver.fields,)
        self._nDof = self._driver.nDof
        self._dofIndicesPermutation = np.arange(0, self._nDof, 1, dtype=int)

    @property
    def elNumber(self):
        return self._elNumber

    @property
    def nNodes(self):
        return self._nNodes

    @property
    def nSpatialDimensions(self):
        return self._nSpatialDimensions

    @property
    def nodes(self):
        return self._nodes

    @property
    def nDof(self):
        return self._nDof

    @property
    def fields(self):
        return self._fields

    @property
    def dofIndicesPermutation(self):
        return self._dofIndicesPermutation

    @property
    def ensightType(self):
        return self._ensightType

    @property
    def hasMaterial(self):
        return self._hasMaterial

    def setNodes(self, nodes: list):
        """Assign the nodes.

        Only the first node is considered.

        Parameters
        ----------
        nodes
            The list of node instances.
        """

        self._nodes = nodes
        self._nodeCoordinates = nodes[0].coordinates
        self._qpCoordinates = nodes[0].coordinates

    def setProperties(self, elementProperties):
        """
        Not used by this driver"""

        raise ValueError("This should not be called for this material driver!")

    def initializeElement(
        self,
    ):
        """
        Not used by this driver"""

    def setMaterial(self, materialName: str, materialProperties: np.ndarray):
        """Assign a material and material properties to the underlying driver.
        Furthermore, create two sets of state vars:

            * the actual set,
            * and a temporary set for backup in nonlinear iteration schemes

        Parameters
        ----------
        materialName
            The name of the requested material.
        materialProperties
            The properties for he requested material.
        """

        self._materialProperties = materialProperties

        self._driver.createMaterial(materialName.upper(), materialProperties)

        self._nStateVars = self._driver.getNumberOfRequiredStateVars()

        self._stateVars = np.zeros(self._nStateVars)
        self._stateVarsTemp = np.zeros(self._nStateVars)

        # The temporary state vars are only ever overwritten in place, hence the driver
        # may hold persistent views into them.
        self._driver.assignStateVars(self._stateVarsTemp)

        self._hasMaterial = True

    def _initializeStateVarsTemp(
        self,
    ):
        self._stateVarsTemp[:] = self._stateVars

    def setInitialCondition(self, stateType, values):
        self._initializeStateVarsTemp()

        if stateType == "initialize material":
            self._driver.initializeYourself()

        if stateType == "characteristic element length":
            self._driver.setCharacteristicElementLength(values[0])

        self.acceptLastState()

    def computeKernels(
        self,
        Ke: np.ndarray,
        Pe: np.ndarray,
        U: np.ndarray,
        dU: np.ndarray,
        time: float,
        dTime: float,
    ):
        self._initializeStateVarsTemp()

        self._driver.computeKernels(Ke, Pe, U, dU, time, dTime)

    def computeKernelsExplicit(
        self,
        Pe: np.ndarray,
        U: np.ndarray,
        dU: np.ndarray,
        time: float,
        dTime: float,
    ):
        self._initializeStateVarsTemp()

        self._driver.computeKernels(np.zeros((self._nDof, self._nDof)), Pe, U, dU, time, dTime)

    def computeLumpedInertia(self, Me: np.ndarray):
        """Not implemented for this driver."""

        raise ValueError("This should not be called for this driver.")

    def computeCriticalTimeStepForExplicitDynamics(self, Q: np.ndarray):
        """Not implemented for this driver."""

        raise ValueError("This should not be called for this driver.")

    def computeDistributedLoad(
        self,
        loadType: str,
        P: np.ndarray,
        K: np.ndarray,
        faceID: int,
        load: np.ndarray,
        U: np.ndarray,
        time: float,
        dTime: float,
    ):
        """Not implemented for this driver."""

        raise ValueError("This should not be called for this driver.")

    def computeInternalEnergy(self) -> float:
        """Not implemented for this driver."""

        raise ValueError("This should not be called for this driver.")

    def computeBodyForce(
        self,
        P: np.ndarray,
        K: np.ndarray,
        load: np.ndarray,
        U: np.ndarray,
        time: float,
        dTime: float,
    ):
        """Not implemented for this driver."""

        raise ValueError("This should not be called for this driver.")

    def acceptLastState(
        self,
    ):
        """Accept the computed state."""

        self._stateVars[:] = self._stateVarsTemp

    def resetToLastValidState(
        self,
    ):
        pass

    def getResultArray(self, result: str, quadraturePoint: int, getPersistentView: bool = True) -> np.ndarray:
        return self._driver.getResultArray(result, getPersistentView)

    def getCoordinatesAtCenter(self) -> np.ndarray:
        """Return the only node's coordinates.

        Returns
        -------
        np.ndarray
            The node's coordinates."""

        return self._nodeCoordinates

    def getCoordinatesAtQuadraturePoints(self) -> np.ndarray:
        """Return the only qp's coordinates.

        Returns
        -------
        np.ndarray
            The qp's coordinates."""

        return self._qpCoordinates

    def getNumberOfQuadraturePoints(self) -> int:
        """Return the only qp's coordinates.

        Returns
        -------
        np.ndarray
            The qp's coordinates."""

        return 1

    @property
    def visualizationNodes(self) -> list[Node]:
        """The nodes for visualization. Commonly, these are the same as the nodes of the entity.
        However, in some cases, the visualization nodes are different from the nodes of the entity, e.g., in case of mixed formulations.
        """

        return self._nodes
