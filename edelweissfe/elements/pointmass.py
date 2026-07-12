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

import numpy as np

from edelweissfe.elements.base.baseelement import BaseElement


class PointMass(BaseElement):
    """
    A 1-node element that adds lumped mass and rotary inertia to a node.
    """

    def __init__(self, elNumber: int, nodes: list, model, mass: float, inertia: list = None):
        super().__init__("PointMass", elNumber)
        self._elNumber = elNumber
        self._nodes = nodes
        self.model = model
        self.domainSize = model.domainSize
        self.mass = mass
        self._use_rotation = inertia is not None

        # Normalize the rotary inertia to one value per rotational DOF. PointMass is only ever
        # used by DiscreteRigidBody, which is 3D-only (enforced by the generator and constraint).
        if inertia is None:
            self.inertia = np.zeros(3)
        else:
            inertia = np.atleast_1d(np.asarray(inertia, dtype=float))
            if inertia.shape[0] != 3:
                raise ValueError("PointMass in 3D requires a diagonal inertia [Ixx, Iyy, Izz].")
            self.inertia = inertia

    def computeLumpedInertia(self, Me: np.ndarray):
        """
        Populate the lumped mass matrix for this element.
        Me is a 1D array of size self.nDof.
        """
        Me[:] = 0.0

        n_disp = self.domainSize
        Me[:n_disp] = self.mass

        if self._use_rotation:
            Me[n_disp : n_disp + self.inertia.shape[0]] = self.inertia

    # Dummy implementations for abstract methods of BaseElement
    @property
    def ensightType(self) -> str:
        return "point"

    @property
    def elType(self) -> str:
        return "PointMass"

    @property
    def fields(self) -> list:
        active_fields = ["displacement"]
        if self._use_rotation:
            active_fields.append("rotation")
        return [active_fields]

    @property
    def nDof(self) -> int:
        ndof = self.domainSize
        if self._use_rotation:
            ndof += 1 if self.domainSize == 2 else 3
        return ndof

    @property
    def nNodes(self) -> int:
        return 1

    @property
    def elNumber(self) -> int:
        return self._elNumber

    @property
    def nodes(self) -> list:
        return self._nodes

    @property
    def dofIndicesPermutation(self):
        return None

    def setNodes(self, nodes: list):
        self._nodes = nodes

    def acceptLastState(self):
        pass

    def computeBodyForce(
        self,
        P: np.ndarray,
        K: np.ndarray,
        load: np.ndarray,
        U: np.ndarray,
        time: float,
        dTime: float,
        *args,
        **kwargs,
    ):
        """
        Apply a body force (e.g., gravity) to the point mass.

        Unlike continuum elements, where ``load`` is a force *per unit volume* that gets integrated
        over the element, a point mass has no volume: ``load`` is interpreted as a force *per unit
        mass* (i.e., an acceleration vector, such as gravitational acceleration), and scaled directly
        by :attr:`mass` to obtain the total translational force. This is what lets a rigid body left
        without a Dirichlet BC on its reference point ("free") settle to quasi-static equilibrium
        under self-weight and contact reaction.
        """
        P[: self.domainSize] += self.mass * load[: self.domainSize]

    def computeCriticalTimeStepForExplicitDynamics(self, Q=None, *args, **kwargs) -> float:
        return 1e99


    def computeDistributedLoad(
        self,
        loadType: str,
        P: np.ndarray,
        K: np.ndarray,
        faceID: int,
        load: np.ndarray,
        U: np.ndarray,
        time: float,
        dT: float,
        *args,
        **kwargs,
    ):
        pass

    def computeInternalEnergy(self) -> float:
        return 0.0

    def computeKernels(
        self, P: np.ndarray, K: np.ndarray, U: np.ndarray, dU: np.ndarray, time: float, dT: float, *args, **kwargs
    ):
        pass

    def computeKernelsExplicit(
        self, P: np.ndarray, U: np.ndarray, dU: np.ndarray, time: float, dT: float, *args, **kwargs
    ):
        pass

    def computeYourself(self, *args, **kwargs):
        pass

    def computeYourselfExplicit(self, *args, **kwargs):
        pass

    def getCoordinatesAtCenter(self) -> np.ndarray:
        return self.nodes[0].coordinates

    def getCoordinatesAtQuadraturePoints(self) -> list:
        return [self.nodes[0].coordinates]

    def getNumberOfQuadraturePoints(self) -> int:
        return 1

    def getResultArray(self, result: str, quadraturePoint: int, getPersistentView: bool = True) -> np.ndarray:
        return np.zeros(1)

    @property
    def hasMaterial(self) -> bool:
        # Deliberately True, not an oversight: this bypasses FEModel._prepareElements' blanket
        # materialAssigned check (which would otherwise reject any model containing a PointMass),
        # while setMaterial below still raises loudly if a section is ever actually assigned to one.
        return True

    def initializeElement(self):
        pass

    def resetToLastValidState(self):
        pass

    def setInitialCondition(self, prop: str, value: float):
        pass

    def setMaterial(self, materialName: str, materialProperties: np.ndarray):
        raise TypeError("PointMass elements cannot have materials assigned to them.")

    def setProperties(self, properties: list):
        pass

    @property
    def visualizationNodes(self) -> list:
        return self.nodes
