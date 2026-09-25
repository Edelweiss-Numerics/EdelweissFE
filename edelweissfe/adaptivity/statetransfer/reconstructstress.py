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

"""Rebuild the children's stress from their own compatible strain instead of transferring it."""

import numpy as np

from edelweissfe.adaptivity.statetransfer.base import (
    StateTransferStrategy,
    perQuadraturePointBlockSize,
    quadraturePointReferenceCoordinates,
)
from edelweissfe.materials.linearelastic.linearelastic import LinearElasticMaterial


def compatibleStrainAtReferenceCoordinates(topology, parentNodeCoords, parentNodalDisplacement, refCoords):
    """Small strain of the parent's displacement field, evaluated at points given in the parent
    reference cube. Voigt order and engineering shear as Marmot's B-operator:
    :math:`(\\varepsilon_{11}, \\varepsilon_{22}, \\varepsilon_{33}, \\gamma_{12}, \\gamma_{13}, \\gamma_{23})`.

    Returns
    -------
    np.ndarray
        ``(nPoints, 6)``.
    """
    strains = np.empty((len(refCoords), 6))
    for q, xi in enumerate(refCoords):
        _, dN_dXi = topology.shape_functions_and_grad(*xi)
        dX_dXi = parentNodeCoords.T @ dN_dXi
        dN_dX = dN_dXi @ np.linalg.inv(dX_dXi)
        du_dX = parentNodalDisplacement.T @ dN_dX  # du_dX[i, j] = du_i / dx_j
        strains[q] = (
            du_dX[0, 0],
            du_dX[1, 1],
            du_dX[2, 2],
            du_dX[0, 1] + du_dX[1, 0],
            du_dX[0, 2] + du_dX[2, 0],
            du_dX[1, 2] + du_dX[2, 1],
        )
    return strains


class ReconstructStressFromStrain(StateTransferStrategy):
    """Give an *elastic* child quadrature point the stress its own displacement field implies,
    :math:`\\boldsymbol{\\sigma} = \\mathbb{C} : \\boldsymbol{\\varepsilon}(\\mathbf{u}_{child})`, and
    that strain, instead of a stress transferred from the parent.

    Why: for a hypoelastic material the stress is integrated incrementally, so whatever stress a
    child starts with is kept for good. A copied (piecewise constant) or projected stress is not
    the stress of the children's displacement field; the part of it that the refined mesh cannot
    equilibrate stays locked in as a residual stress that alternates between siblings. The
    children's nodes are warm-started by interpolating the parent's displacement field, so their
    compatible strain is that field's strain at the child quadrature points -- computed here from
    ``parentNodalDisplacement``. For an elastic history this is exactly the state a mesh refined
    from the start would hold.

    Scope: only quadrature points whose transferred *material* state is still virgin (every
    column from the element's ``begin of material state`` on equals the child's initial value --
    no material-specific names) are rebuilt. Every other point keeps the result of ``fallback``,
    which transfers the whole block first; those points are counted and reported. Energy and other
    element-level columns follow ``fallback``.

    The elastic stiffness is passed explicitly (``youngsModulus``, ``poissonRatio``) as a stopgap:
    the material does not yet expose its elastic stiffness to the element wrapper.
    """

    def __init__(self, fallback: StateTransferStrategy, youngsModulus: float, poissonRatio: float):
        self._fallback = fallback
        self._elasticStiffness = LinearElasticMaterial(np.array([youngsModulus, poissonRatio])).elasticityMatrix()
        self._nRebuilt = 0
        self._nFallback = 0

    def transferState(self, parent, children, topology, parentNodalDisplacement):
        # the children's freshly initialised state, before the fallback overwrites it: the
        # reference for "is this point's material state still virgin?"
        virginStates = [
            child.getStateVars().reshape(child.getNumberOfQuadraturePoints(), perQuadraturePointBlockSize(child)).copy()
            for child in children
        ]
        self._fallback.transferState(parent, children, topology, parentNodalDisplacement)

        stressOffset, stressSize = parent.getStateVarSlice("stress")
        strainOffset, strainSize = parent.getStateVarSlice("strain")
        materialStateOffset, _ = parent.getStateVarSlice("begin of material state")
        parentNodeCoords = np.array([n.coordinates for n in parent.nodes], dtype=float)
        displacement = np.asarray(parentNodalDisplacement, dtype=float).reshape(len(parent.nodes), -1)

        for child, virgin in zip(children, virginStates):
            transferred = child.getStateVars().reshape(virgin.shape).copy()
            elastic = np.all(transferred[:, materialStateOffset:] == virgin[:, materialStateOffset:], axis=1)
            childRefCoords = quadraturePointReferenceCoordinates(child, parentNodeCoords, topology)
            strains = compatibleStrainAtReferenceCoordinates(
                topology, parentNodeCoords, displacement, childRefCoords[elastic]
            )
            transferred[elastic, strainOffset : strainOffset + strainSize] = strains
            transferred[elastic, stressOffset : stressOffset + stressSize] = strains @ self._elasticStiffness.T
            child.setStateVars(transferred.reshape(-1))
            self._nRebuilt += int(elastic.sum())
            self._nFallback += int((~elastic).sum())

    def reportAndResetTransferStatistics(self) -> str | None:
        if self._nRebuilt + self._nFallback == 0:
            return None
        report = (
            f"stress rebuilt from compatible strain at {self._nRebuilt} child quadrature point(s); "
            f"{self._nFallback} with non-virgin material state kept the fallback transfer"
        )
        self._nRebuilt = self._nFallback = 0
        return report

    def _transferColumns(self, parentValues, parentRefCoords, childRefCoords, childInitValues, columns):
        raise NotImplementedError("ReconstructStressFromStrain works on whole blocks in transferState().")
