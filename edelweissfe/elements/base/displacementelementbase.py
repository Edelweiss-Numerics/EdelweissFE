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
#  Daniel Reitmair daniel.reitmair@uibk.ac.at
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

import numpy as np
import numpy.linalg as lin

from edelweissfe.config import registry
from edelweissfe.elements.base.baseelement import BaseElement
from edelweissfe.elements.displacementelement._elementcomputationmatrices import (
    computeNOperator,
)
from edelweissfe.elements.library import elLibrary
from edelweissfe.materials.base.basehyperelasticmaterial import BaseHyperElasticMaterial
from edelweissfe.points.node import Node
from edelweissfe.utils.caseinsensitivedict import CaseInsensitiveDict


class DisplacementElementBase(BaseElement):
    """Common base of the isoparametric continuum elements with displacement degrees of freedom.

    Everything that does not depend on the kinematics lives here: the element geometry and
    quadrature, the material and quadrature-point state, loads, mass, and result access.
    A subclass adds the kinematics -- how strain and stress follow from the nodal displacements,
    i.e. :meth:`initializeElement`, :meth:`computeKernels` and :meth:`computeKernelsExplicit`:

    - :class:`~edelweissfe.elements.displacementelement.element.DisplacementElement`: small strain.
    - :class:`~edelweissfe.elements.displacementtlelement.element.DisplacementTLElement`: finite
      strain, total Lagrangian formulation.

    Parameters
    ----------
    elementType
        A string identifying the requested element formulation, see the subclasses.
    elNumber
        A unique integer label used for all kinds of purposes.
    """

    def __init__(self, elementType: str, elNumber: int):
        self._elType = elementType
        properties = elLibrary[elementType]
        # Guard against being handed an element type belonging to a *different* formulation, e.g.
        # `DisplacementElement("CPE4TL", 1)`: `elLibrary` supplies quadrature data for both
        # formulations, so nothing else here would notice. The type -> class mapping is looked up
        # in the `element` category of the registry; a subclass of that class is accepted too.
        if not isinstance(self, registry.lookup("element", elementType)[0]):
            raise Exception("Something went wrong with the element initialization!")
        self._elNumber = elNumber
        self._nNodes = properties["nNodes"]
        self._nDof = properties["nDof"]
        self._dofIndices = properties["dofIndices"]
        self._ensightType = properties["ensightType"]
        self.nSpatialDimensions = properties["nSpatialDimensions"]
        self._nInt = properties["nInt"]
        self._xi = properties["xi"]
        self._eta = properties["eta"]
        self._zeta = properties["zeta"]
        self._weight = properties["w"]
        self._matrixSize = properties["matSize"]
        self._activeVoigtIndices = properties["index"]
        self.planeStrain = properties["plStrain"]
        if self.nSpatialDimensions == 3:
            self._t = 1  # "thickness" for 3D elements
        self._fields = [["displacement"] for i in range(self._nNodes)]
        self._dStrain = np.zeros([self._nInt, 6])

    @property
    def elNumber(self) -> int:
        """The unique number of this element"""

        return self._elNumber  # return number

    @property
    def elType(self) -> str:
        """The type of this element."""

        return self._elType

    @property
    def nNodes(self) -> int:
        """The number of nodes this element requires"""

        return self._nNodes

    @property
    def nodes(self) -> int:
        """The list of nodes this element holds"""

        return self._nodes

    @property
    def nDof(self) -> int:
        """The total number of degrees of freedom this element has"""

        return self._nDof

    @property
    def fields(self) -> list[list[str]]:
        """The list of fields per nodes."""

        return self._fields

    @property
    def dofIndicesPermutation(self) -> np.ndarray:
        """The permutation pattern for the residual vector and the stiffness matrix to
        aggregate all entries in order to resemble the defined fields nodewise.
        In this case it stays the same because we use the nodes exactly like they are."""

        return self._dofIndices

    @property
    def ensightType(self) -> str:
        """The shape of the element in Ensight Gold notation."""

        return self._ensightType

    @property
    def visualizationNodes(self) -> str:
        """The nodes for visualization."""

        return self._nodes

    @property
    def hasMaterial(self) -> str:
        """Flag to check if a material was assigned to this element."""

        return self._hasMaterial

    def setNodes(self, nodes: list[Node]):
        """Assign the nodes to the element.

        Parameters
        ----------
        nodes
            A list of nodes.
        """

        self._nodes = nodes
        _nodesCoordinates = np.array([n.coordinates for n in nodes])  # get node coordinates
        self._nodesCoordinates = _nodesCoordinates.transpose()  # nodes given column-wise: x-coordinate - y-coordinate

    def setProperties(self, elementProperties: np.ndarray):
        """Assign a set of properties to the element.

        Parameters
        ----------
        elementProperties
            A numpy array containing the element properties.

        Attributes
        ----------
        thickness
            Thickness of 2D elements.
        """

        if self.nSpatialDimensions == 2:
            self._t = elementProperties[0]  # thickness

    def assignProperty(self, propertyName: str, properties: np.ndarray):
        """Assign a property of the element by name."""
        if propertyName.lower() == "thickness":
            if self.nSpatialDimensions == 2:
                self._t = float(properties[0])
            else:
                raise Exception("Thickness property is only supported for 2D elements.")
        else:
            raise NotImplementedError(f"Property '{propertyName}' is not supported by this element.")

    def getPropertyNames(self) -> list[str]:
        """Get the names of all the valid properties of the element."""
        if self.nSpatialDimensions == 2:
            return ["thickness"]
        return []

    def setInitialCondition(self, stateType: str, values: np.ndarray):
        """Assign initial conditions.

        Parameters
        ----------
        stateType
            The type of initial state.
        values
            The numpy array describing the initial state.
        """

        raise Exception("Setting an initial condition is not possible with this element provider.")

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
        """Evaluate residual and stiffness for given time, field, and field increment due to a surface load.

        Parameters
        ----------
        loadType
            The type of load.
        P
            The external load vector to be defined.
        K
            The stiffness matrix to be defined.
        faceID
            The number of the elements face this load acts on.
        load
            The magnitude (or vector) describing the load.
        U
            The current solution vector.
        time
            The current time.
        dTime
            The time increment.
        """

        raise Exception("Applying a distributed load is currently not possible with this element provider.")

    def computeBodyForce(
        self, P: np.ndarray, K: np.ndarray, load: np.ndarray, U: np.ndarray, time: float, dTime: float
    ):
        """Evaluate residual and stiffness for given time, field, and field increment due to a body force load.

        Parameters
        ----------
        P
            The external load vector to be defined.
        K
            The stiffness matrix to be defined.
        load
            The magnitude (or vector) describing the load.
        U
            The current solution vector.
        time
            The current time.
        dTime
            The time increment.
        """

        N = computeNOperator(self._xi, self._eta, self._zeta, self._nInt, self.nNodes, self.nSpatialDimensions)
        for i in range(self._nInt):
            P += np.outer(N[i], load).flatten() * lin.det(self.J[i]) * self._t * self._weight[i]

    def computeConsistentMassMatrix(self, M: np.ndarray):
        """Compute the consistent mass matrix.

        Parameters
        ----------
        M
            The mass matrix to be defined.
        """
        N = computeNOperator(self._xi, self._eta, self._zeta, self._nInt, self.nNodes, self.nSpatialDimensions)
        nDoFPerNode = int(self._nDof / self.nNodes)
        for i in range(self._nInt):
            # compute element volume
            detJ = lin.det(self.J[i])
            # compute mass matrix for element j in point i
            N_ = np.zeros((self.nSpatialDimensions, self._nDof))
            for j in range(self.nNodes):
                for k in range(nDoFPerNode):
                    N_[k, nDoFPerNode * j + k] = N[i][j]

            M += self.material.getDensity() * N_.T @ N_ * detJ * self._weight[i] * self._t

    def computeLumpedInertia(self, M: np.ndarray):
        """Compute the lumped mass matrix with simple row summing of the consistent mass matrix.

        Parameters
        ----------
        M
            The mass matrix to be defined.
        """
        # compute element volume
        cmm = np.zeros((self._nDof, self._nDof))
        self.computeConsistentMassMatrix(cmm)

        # compute lumped mass matrix by summing up the rows
        M[:] = np.sum(cmm, axis=1)

    def getCharacteristicElementLength(self, qp: int = 0):
        """Compute the characteristic element length.
        Parameters
        ----------
        qp
            The number of the quadrature point for which the characteristic element length should be computed. If not given, it is computed for the first quadrature point.

        Returns
        -------
        l
            The characteristic element length.
        """
        if self.nSpatialDimensions == 1:
            return self._nodesCoordinates[0, 1] - self._nodesCoordinates[0, 0]
        elif self.nSpatialDimensions == 2:
            return np.sqrt(4 * self.detJ[qp])
        elif self.nSpatialDimensions == 3:
            return np.cbrt(8 * self.detJ[qp])

    def resetToLastValidState(
        self,
    ):
        """Reset to the last valid state."""

    def getResultArray(self, result: str, quadraturePoint: int, getPersistentView: bool = True) -> np.ndarray:
        """Get the array of a result, possibly as a persistent view which is continiously
        updated by the element.

        Parameters
        ----------
        result
            The name of the result.
        quadraturePoint
            The number of the quadrature point.
        getPersistentView
            If true, the returned array should be continiously updated by the element.

        Returns
        -------
        np.ndarray
            The result.
        """

        try:
            return self._stateVars[quadraturePoint][result]
        except KeyError:  # result in material
            self.material.assignCurrentStateVars(self._stateVarsRef[quadraturePoint][12:])
            return self.material.getResult(result)

    def getCoordinatesAtCenter(self) -> np.ndarray:
        """Compute the underlying MarmotElement centroid coordinates.

        Returns
        -------
        np.ndarray
            The element's central coordinates.
        """

        x = self._nodesCoordinates
        return np.average(x, axis=1)

    def getNumberOfQuadraturePoints(self) -> int:
        """Get the number of Quadrature points the element has.

        Returns
        -------
        nInt
            The number of Quadrature points.
        """

        return self._nInt

    def getCoordinatesAtQuadraturePoints(self) -> np.ndarray:
        """Compute the coordinates of the quadrature points.

        Returns
        -------
        np.ndarray
            The coordinates, one row per quadrature point: shape ``(nInt, nSpatialDimensions)``.
        """

        N = computeNOperator(self._xi, self._eta, self._zeta, self._nInt, self.nNodes, self.nSpatialDimensions)
        return N @ self._nodesCoordinates.T

    def setMaterial(self, material: type):
        """Assign a material.

        Parameters
        ----------
        material
            An initalized instance of a material.
        """

        self.material = material
        if self.planeStrain:  # use 3D
            self._matrixVoigtIndices = np.array([0, 1, 3])
            self._matrixSize = 6
        else:
            self._matrixVoigtIndices = np.arange(self._matrixSize)
        stateVarsSize = 12 + self.material.getNumberOfRequiredStateVars()
        self._hasMaterial = True
        self._stateVarsRef = np.zeros([self._nInt, stateVarsSize])
        self._stateVars = [
            CaseInsensitiveDict(
                {
                    "stress": self._stateVarsRef[i][0:6],
                    "strain": self._stateVarsRef[i][6:12],
                    "materialstate": self._stateVarsRef[i][12:],
                }
            )
            for i in range(self._nInt)
        ]
        self._stateVarsTemp = np.zeros([self._nInt, stateVarsSize])
        if issubclass(type(self.material), BaseHyperElasticMaterial):  # check if material is hyperelastic
            self._isHyperelastic = True
        else:
            self._isHyperelastic = False

    def updateMaterialProperty(self, index: int, value: float):
        """Change one entry of the property vector of the assigned material, keeping the state.

        The material instance is rebuilt from its modified property vector; the state lives in the
        arrays of this element, not in the material, and is handed to the material before every
        evaluation, so it is preserved as is.

        Parameters
        ----------
        index
            The index of the property in the material's property vector.
        value
            The new value of the property.
        """

        materialProperties = self.material.materialProperties.copy()
        materialProperties[index] = value
        self.material = type(self.material)(materialProperties)

    def acceptLastState(
        self,
    ):
        """Accept the computed state (in nonlinear iteration schemes)."""

        # copy every array in array (complete copying)
        self._stateVarsRef[:] = [self._stateVarsTemp[i][:] for i in range(self._nInt)]

    def getStateVars(self) -> np.ndarray:
        """Return a copy of the converged quadrature-point state-variable buffer."""

        return self._stateVarsRef.reshape(-1).copy()

    def setStateVars(self, values: np.ndarray):
        """Overwrite the converged quadrature-point state-variable buffer in place, so
        the ``_stateVars`` per-quadrature-point views (stress/strain/materialstate) stay valid.

        The working buffer is seeded from it as well. :meth:`acceptLastState` copies
        ``_stateVarsTemp`` *over* ``_stateVarsRef``, and it can run before any element computation
        has populated that working buffer: both explicit solvers process a zero increment first,
        for which :meth:`computeYourself` is never called, and then accept it. On a resumed run
        that copied a freshly allocated buffer of zeros over the state just restored, silently
        discarding the whole quadrature-point history. A cold start never noticed, because there
        ``_stateVarsRef`` is zero anyway.
        """

        self._stateVarsRef[:] = np.asarray(values).reshape(self._stateVarsRef.shape)
        self._stateVarsTemp = self._stateVarsRef.copy()
