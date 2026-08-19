# cython: language_level=3
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
"""
The compiled inner loop of a finite difference momentum balance cell.

The counterpart of :mod:`edelweissfe.kernels.gradientplasticitykernel` for the single field
case, and the same argument applies: the blocks of a cell are small enough that a numpy call
costs more in dispatch than in arithmetic, so the whole cell is done in C with the interpreter
lock released.

This one talks to :class:`MarmotMaterialHypoElastic` directly rather than through a shim, since
that class is already declared in :mod:`edelweissfe.materials.marmot._marmotmaterials` together
with the Eigen types it exchanges, and the whole declaration block is ``nogil``. All four stress
states are supported, each going to the Marmot routine that condenses out the components it has
to:

===================  ==========================  ================  ============================
stress state         routine                     tangent           Voigt components carried
===================  ==========================  ================  ============================
``3d``               ``computeStress``           6 x 6             all six
``plane strain``     ``computeStress``           6 x 6             all six
``plane stress``     ``computePlaneStress``      3 x 3             11, 22, 12
``uniaxial stress``  ``computeUniaxialStress``   scalar            11
===================  ==========================  ================  ============================

Per material point, with :math:`\\boldsymbol B_p` the strain operator reduced to the components
the stress state carries and :math:`V_p` the material point volume:

.. math::
    \\boldsymbol P \\mathrel{+}= \\boldsymbol B_p^T \\boldsymbol \\sigma \\, V_p,
    \\qquad
    \\boldsymbol K \\mathrel{+}= \\boldsymbol B_p^T \\boldsymbol C_p \\boldsymbol B_p \\, V_p
"""

cimport cython
from libcpp.string cimport string

import numpy as np

from edelweissfe.materials.marmot._marmotmaterials cimport (
    MarmotMaterialHypoElastic,
    MarmotMaterialHypoElasticFactory,
    Matrix3d,
    Matrix6d,
    StateView,
    Vector3d,
    Vector6d,
)

from edelweissfe.utils.exceptions import CutbackRequest


#: The number of Voigt components Marmot always works in.
cdef int nVoigt = 6

#: The routine to call, by number, avoiding a string comparison in the hot loop.
cdef int routineFull = 0
cdef int routinePlaneStress = 1
cdef int routineUniaxial = 2

#: The stress state names this kernel understands, mapped to those numbers.
routines = {
    "3d": 0,
    "plane strain": 0,
    "plane stress": 1,
    "uniaxial stress": 2,
}


@cython.final
@cython.boundscheck(False)
@cython.wraparound(False)
cdef class DisplacementKernel:
    """The compiled kernel of one momentum balance cell.

    One instance owns one Marmot material, since a material carries the state variable storage it
    operates on as mutable state and cells are distributed over threads.

    Parameters
    ----------
    materialName
        The Marmot material name, e.g. ``LINEARELASTIC``.
    materialProperties
        The material property vector.
    stressState
        One of the keys of :data:`routines`.
    characteristicLength
        The characteristic length of a material point, handed to the material.
    strainOperators
        The full six row strain operators, shape ``(nMaterialPoints, 6, nDof)``.
    weightedReducedTransposed
        The transposes of the strain operators reduced to the carried components, already
        multiplied by the material point volume, shape ``(nMaterialPoints, nDof, nCarried)``.
    reducedOperators
        The strain operators reduced to the carried components, shape
        ``(nMaterialPoints, nCarried, nDof)``.
    carriedVoigtIndices
        Which Voigt components the stress state carries. The material point volumes do not appear
        separately: they are already folded into ``weightedReducedTransposed``, which is the only
        place they enter.
    stateVars, stateVarsTemp
        The accepted and trial state variables, shape ``(nMaterialPoints, nStateVarsPerPoint)``,
        laid out as stress, strain and then the material's own variables.
    nStateVarsOverhead
        How many state variables the stencil keeps in front of the material's own ones.
    """

    cdef MarmotMaterialHypoElastic* _material
    cdef double[::1] _materialProperties
    cdef int _routine

    cdef int _nMaterialPoints
    cdef int _nDof
    cdef int _nCarried
    cdef int _nStateVarsOverhead

    cdef double[:, :, ::1] _B
    cdef double[:, :, ::1] _reducedBTv
    cdef double[:, :, ::1] _reducedB
    cdef long[::1] _carried

    cdef double[:, ::1] _stateVars
    cdef double[:, ::1] _stateVarsTemp

    cdef double[::1] _dStrain
    cdef double[::1] _stress
    cdef double[::1] _tangent
    cdef double[:, ::1] _operatorTimesTangent

    def __cinit__(
        self,
        str materialName,
        double[::1] materialProperties,
        str stressState,
        double characteristicLength,
        double[:, :, ::1] strainOperators,
        double[:, :, ::1] weightedReducedTransposed,
        double[:, :, ::1] reducedOperators,
        long[::1] carriedVoigtIndices,
        double[:, ::1] stateVars,
        double[:, ::1] stateVarsTemp,
        int nStateVarsOverhead,
    ):
        if stressState not in routines:
            raise ValueError("Unknown stress state '{:}'".format(stressState))

        cdef string encodedName = materialName.encode("UTF-8")

        # Marmot keeps only the pointer to the properties, so the copy has to outlive the call
        self._materialProperties = np.ascontiguousarray(materialProperties, dtype=float).copy()

        self._material = MarmotMaterialHypoElasticFactory.createMaterial(
            encodedName,
            &self._materialProperties[0],
            self._materialProperties.shape[0],
            1,
        )

        self._material.setCharacteristicElementLength(characteristicLength)

        self._routine = routines[stressState]

        self._B = strainOperators
        self._reducedBTv = weightedReducedTransposed
        self._reducedB = reducedOperators
        self._carried = carriedVoigtIndices

        self._stateVars = stateVars
        self._stateVarsTemp = stateVarsTemp
        self._nStateVarsOverhead = nStateVarsOverhead

        self._nMaterialPoints = strainOperators.shape[0]
        self._nDof = strainOperators.shape[2]
        self._nCarried = carriedVoigtIndices.shape[0]

        self._dStrain = np.zeros(nVoigt)
        self._stress = np.zeros(nVoigt)
        self._tangent = np.zeros(self._nCarried * self._nCarried)
        self._operatorTimesTangent = np.zeros((self._nDof, self._nCarried))

    def __dealloc__(self):
        if self._material != NULL:
            del self._material

    def getNumberOfRequiredStateVars(self) -> int:
        """The number of state variables the material needs per material point."""

        return self._material.getNumberOfRequiredStateVars()

    def initializeMaterialStateVars(self, double[::1] stateVars):
        """Let Marmot initialize the state variables of one material point."""

        self._material.initializeYourself(&stateVars[0], stateVars.shape[0])

    def getStateView(self, str stateName, double[::1] stateVars) -> np.ndarray:
        """A view of a named material state variable of one material point."""

        cdef string encoded = stateName.encode("UTF-8")
        cdef StateView view = self._material.getStateView(encoded, &stateVars[0])
        cdef double[::1] result = <double[: view.stateSize]> view.stateLocation

        return np.asarray(result)

    cdef void _evaluateMaterial(self, int p, double time, double dTime) except *:
        """Evaluate the material of one point, exchanging through Marmot's Eigen types.

        The stress is read from and written back to its state variable slot. The tangent buffer is
        zeroed first, since only the carried block is written for the reduced stress states.
        """

        cdef int i, j, v
        cdef double[3] reducedStress
        cdef double[3] reducedStrain

        # the nested state structs, declared here since Cython wants them at function scope
        cdef MarmotMaterialHypoElastic.state3D state3D
        cdef MarmotMaterialHypoElastic.state2D state2D
        cdef MarmotMaterialHypoElastic.state1D state1D
        cdef MarmotMaterialHypoElastic.timeInfo timeInfo

        cdef Matrix6d tangent6
        cdef Matrix3d tangent3
        cdef Vector6d strain6
        cdef Vector3d strain3

        timeInfo.time = time
        timeInfo.dT = dTime

        for i in range(self._nCarried * self._nCarried):
            self._tangent[i] = 0.0

        if self._routine == routineFull:
            state3D.stress = Vector6d(&self._stress[0])
            state3D.stateVars = &self._stateVarsTemp[p, self._nStateVarsOverhead]

            tangent6 = Matrix6d(&self._tangent[0])
            strain6 = Vector6d(&self._dStrain[0])

            self._material.computeStress(state3D, tangent6, strain6, timeInfo)

            for v in range(nVoigt):
                self._stress[v] = state3D.stress(v)

            for i in range(nVoigt):
                for j in range(nVoigt):
                    self._tangent[i * nVoigt + j] = tangent6(i, j)

        elif self._routine == routinePlaneStress:
            for i in range(3):
                reducedStress[i] = self._stress[self._carried[i]]
                reducedStrain[i] = self._dStrain[self._carried[i]]

            state2D.stress = Vector3d(&reducedStress[0])
            state2D.stateVars = &self._stateVarsTemp[p, self._nStateVarsOverhead]

            tangent3 = Matrix3d(&self._tangent[0])
            strain3 = Vector3d(&reducedStrain[0])

            self._material.computePlaneStress(state2D, tangent3, strain3, timeInfo)

            for i in range(3):
                self._stress[self._carried[i]] = state2D.stress(i)

            for i in range(3):
                for j in range(3):
                    self._tangent[i * 3 + j] = tangent3(i, j)

        else:
            state1D.stress = self._stress[self._carried[0]]
            state1D.stateVars = &self._stateVarsTemp[p, self._nStateVarsOverhead]

            self._material.computeUniaxialStress(
                state1D, self._tangent[0], self._dStrain[self._carried[0]], timeInfo
            )

            self._stress[self._carried[0]] = state1D.stress

    def computeKernels(
        self,
        double[:, :] K,
        double[::1] P,
        const double[::1] U,
        const double[::1] dU,
        double time,
        double dTime,
    ):
        """Add this cell's contribution to the global tangent and internal flux.

        Parameters
        ----------
        K
            The cell's dense block of the global tangent, of any strides.
        P
            The cell's slice of the internal flux.
        U
            The total field values on the cell.
        dU
            Their increment.
        time
            The total time.
        dTime
            The time increment.
        """

        cdef int nMP = self._nMaterialPoints
        cdef int nDof = self._nDof
        cdef int nC = self._nCarried

        cdef int p, i, j, a, b, v
        cdef double accumulated

        # the trial state starts from the last accepted one
        for p in range(nMP):
            for i in range(self._stateVars.shape[1]):
                self._stateVarsTemp[p, i] = self._stateVars[p, i]

        try:
            for p in range(nMP):
                with nogil:
                    for v in range(nVoigt):
                        accumulated = 0.0
                        for j in range(nDof):
                            accumulated += self._B[p, v, j] * dU[j]
                        self._dStrain[v] = accumulated

                    for v in range(nVoigt):
                        self._stress[v] = self._stateVarsTemp[p, v]

                self._evaluateMaterial(p, time, dTime)

                with nogil:
                    for v in range(nVoigt):
                        self._stateVarsTemp[p, v] = self._stress[v]
                        self._stateVarsTemp[p, nVoigt + v] += self._dStrain[v]

                    # -- internal flux --------------------------------------------------
                    for i in range(nDof):
                        accumulated = 0.0
                        for a in range(nC):
                            accumulated += self._reducedBTv[p, i, a] * self._stress[self._carried[a]]
                        P[i] += accumulated

                    # -- tangent, in two steps rather than one quadruple loop -----------
                    for i in range(nDof):
                        for a in range(nC):
                            accumulated = 0.0
                            for b in range(nC):
                                accumulated += self._reducedBTv[p, i, b] * self._tangent[b * nC + a]
                            self._operatorTimesTangent[i, a] = accumulated

                    for i in range(nDof):
                        for j in range(nDof):
                            accumulated = 0.0
                            for a in range(nC):
                                accumulated += self._operatorTimesTangent[i, a] * self._reducedB[p, a, j]
                            K[i, j] += accumulated

        except (RuntimeError, ValueError) as exception:
            raise CutbackRequest(str(exception), 0.25)

    def acceptLastState(self):
        """Accept the trial state variables."""

        cdef int p, i

        for p in range(self._stateVars.shape[0]):
            for i in range(self._stateVars.shape[1]):
                self._stateVars[p, i] = self._stateVarsTemp[p, i]

    def resetToLastValidState(self):
        """Discard the trial state variables."""

        cdef int p, i

        for p in range(self._stateVars.shape[0]):
            for i in range(self._stateVars.shape[1]):
                self._stateVarsTemp[p, i] = self._stateVars[p, i]
