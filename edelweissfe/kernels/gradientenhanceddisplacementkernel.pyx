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
The compiled inner loop of a finite difference gradient enhanced cell.

The two field counterpart for the gradient *enhanced* material family, where -- unlike gradient
plasticity, see :mod:`edelweissfe.kernels.gradientplasticitykernel` -- the discretisation owns the
second balance equation rather than the material. The material returns a local driving variable
:math:`K_{local}` and a nonlocal parameter :math:`c`, and the screened Poisson equation

.. math::
    K - \\nabla \\cdot \\left( c \\nabla K \\right) - K_{local} = 0

is assembled here. Expanded, and with the nonlocal variable collocated as the cell average
:math:`\\boldsymbol N` while its gradient uses the cell gradient operator :math:`\\boldsymbol G`:

.. math::
    P_K = \\left[ \\boldsymbol N \\left( K - K_{local} \\right)
          + c \\, \\boldsymbol G^T \\boldsymbol G \\boldsymbol K_{corners} \\right] V_p

The reason for compiling it is the same as everywhere here: the blocks are small enough that a
numpy call costs more in dispatch than in arithmetic, so the cell is done in C with the
interpreter lock released. This kernel reaches Marmot through the ``double*`` interface of
:mod:`edelweissfe.materials.marmot._gradientenhanced`, whose declarations are already ``nogil``.

``G^T G`` is constant per material point, so the stencil computes it once and passes it in rather
than forming it per evaluation.
"""

cimport cython
from libcpp.string cimport string

import numpy as np

from edelweissfe.materials.marmot._gradientenhanced cimport (
    GradientEnhancedHypoElasticShim1,
)
from edelweissfe.materials.marmot._marmotmaterials cimport StateView

from edelweissfe.utils.exceptions import CutbackRequest


#: The number of Voigt components Marmot always works in.
cdef int nVoigt = 6


@cython.final
@cython.boundscheck(False)
@cython.wraparound(False)
cdef class GradientEnhancedDisplacementKernel:
    """The compiled kernel of one gradient enhanced cell.

    One instance owns one Marmot material, since a material carries the state variable storage it
    operates on as mutable state and cells are distributed over threads.

    Parameters
    ----------
    materialName
        The Marmot material name, e.g. ``AT2PHASEFIELD``.
    materialProperties
        The material property vector.
    planeStress
        Whether the material should condense the out-of-plane components for plane stress.
    strainOperators
        The strain operators of the material points, shape ``(nMaterialPoints, 6, nDisplacementDofs)``.
    weightedTransposedOperators
        Their transposes, already multiplied by the material point volume.
    averageOperator
        The operator averaging the nonlocal field over the cell, shape ``(nNonlocalDofs,)``.
    laplacianProducts
        ``G^T G`` per material point, shape ``(nMaterialPoints, nNonlocalDofs, nNonlocalDofs)``.
    materialPointVolumes
        The volume of each material point.
    displacementDofs, nonlocalDofs
        The local indices of the two fields.
    stateVars, stateVarsTemp
        The accepted and trial state variables, laid out as stress, strain, the nonlocal variable,
        the local driving variable and then the material's own variables.
    nStateVarsOverhead
        How many state variables the stencil keeps in front of the material's own ones.
    """

    cdef GradientEnhancedHypoElasticShim1* _material
    cdef double[::1] _materialProperties
    cdef bint _planeStress

    cdef int _nMaterialPoints
    cdef int _nDisplacementDofs
    cdef int _nNonlocalDofs
    cdef int _nStateVarsOverhead

    cdef double[:, :, ::1] _B
    cdef double[:, :, ::1] _BTv
    cdef double[::1] _N
    cdef double[:, :, ::1] _GTG
    cdef double[::1] _volumes

    cdef long[::1] _displacementDofs
    cdef long[::1] _nonlocalDofs

    cdef double[:, ::1] _stateVars
    cdef double[:, ::1] _stateVarsTemp

    cdef double[::1] _dStrain
    cdef double[::1] _nonlocal
    cdef double[::1] _dNonlocal
    cdef double[::1] _stress
    cdef double[::1] _KLocal
    cdef double[::1] _c
    cdef double[::1] _elasticEnergyDensity
    cdef double[::1] _dissipation
    cdef double[::1] _dStress_dStrain
    cdef double[::1] _dStress_dK
    cdef double[::1] _dKLocal_dStrain
    cdef double[::1] _dKLocal_dK
    cdef double[::1] _dc_dK
    cdef double[::1] _d2c_dK2

    cdef double[:, ::1] _localTangent
    cdef double[::1] _localFlux
    cdef double[:, ::1] _operatorTimesTangent
    cdef double[::1] _displacementIncrements
    cdef double[::1] _nonlocalValues
    cdef double[::1] _nonlocalIncrements
    cdef double[::1] _laplacianTimesNonlocal
    cdef double[::1] _stressCoupling
    cdef double[::1] _yieldCoupling

    def __cinit__(
        self,
        str materialName,
        double[::1] materialProperties,
        bint planeStress,
        double[:, :, ::1] strainOperators,
        double[:, :, ::1] weightedTransposedOperators,
        double[::1] averageOperator,
        double[:, :, ::1] laplacianProducts,
        double[::1] materialPointVolumes,
        long[::1] displacementDofs,
        long[::1] nonlocalDofs,
        double[:, ::1] stateVars,
        double[:, ::1] stateVarsTemp,
        int nStateVarsOverhead,
    ):
        cdef string encodedName = materialName.encode("UTF-8")

        # Marmot keeps only the pointer to the properties, so the copy has to outlive the call
        self._materialProperties = np.ascontiguousarray(materialProperties, dtype=float).copy()

        self._material = new GradientEnhancedHypoElasticShim1(
            encodedName,
            &self._materialProperties[0],
            self._materialProperties.shape[0],
            1,
        )

        self._planeStress = planeStress

        self._B = strainOperators
        self._BTv = weightedTransposedOperators
        self._N = averageOperator
        self._GTG = laplacianProducts
        self._volumes = materialPointVolumes

        self._displacementDofs = displacementDofs
        self._nonlocalDofs = nonlocalDofs

        self._stateVars = stateVars
        self._stateVarsTemp = stateVarsTemp
        self._nStateVarsOverhead = nStateVarsOverhead

        self._nMaterialPoints = strainOperators.shape[0]
        self._nDisplacementDofs = displacementDofs.shape[0]
        self._nNonlocalDofs = nonlocalDofs.shape[0]

        self._dStrain = np.zeros(nVoigt)
        self._nonlocal = np.zeros(1)
        self._dNonlocal = np.zeros(1)
        self._stress = np.zeros(nVoigt)
        self._KLocal = np.zeros(1)
        self._c = np.zeros(1)
        self._elasticEnergyDensity = np.zeros(1)
        self._dissipation = np.zeros(1)
        self._dStress_dStrain = np.zeros(nVoigt * nVoigt)
        self._dStress_dK = np.zeros(nVoigt)
        self._dKLocal_dStrain = np.zeros(nVoigt)
        self._dKLocal_dK = np.zeros(1)
        self._dc_dK = np.zeros(1)
        self._d2c_dK2 = np.zeros(1)

        cdef int nTotal = self._nDisplacementDofs + self._nNonlocalDofs

        self._localTangent = np.zeros((nTotal, nTotal))
        self._localFlux = np.zeros(nTotal)
        self._operatorTimesTangent = np.zeros((self._nDisplacementDofs, nVoigt))
        self._displacementIncrements = np.zeros(self._nDisplacementDofs)
        self._nonlocalValues = np.zeros(self._nNonlocalDofs)
        self._nonlocalIncrements = np.zeros(self._nNonlocalDofs)
        self._laplacianTimesNonlocal = np.zeros(self._nNonlocalDofs)
        self._stressCoupling = np.zeros(self._nDisplacementDofs)
        self._yieldCoupling = np.zeros(self._nDisplacementDofs)

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
        cdef int nD = self._nDisplacementDofs
        cdef int nK = self._nNonlocalDofs

        cdef int p, i, j, k, l, v, w, dof
        cdef double volume, accumulated, c, KLocal, nonlocalValue, dcdK, dKLocaldK

        for i in range(nD):
            self._displacementIncrements[i] = dU[self._displacementDofs[i]]
        for k in range(nK):
            self._nonlocalValues[k] = U[self._nonlocalDofs[k]]
            self._nonlocalIncrements[k] = dU[self._nonlocalDofs[k]]

        for i in range(nD + nK):
            self._localFlux[i] = 0.0
            for j in range(nD + nK):
                self._localTangent[i, j] = 0.0

        for p in range(nMP):
            for i in range(self._stateVars.shape[1]):
                self._stateVarsTemp[p, i] = self._stateVars[p, i]

        try:
            with nogil:
                for p in range(nMP):
                    volume = self._volumes[p]

                    for v in range(nVoigt):
                        accumulated = 0.0
                        for j in range(nD):
                            accumulated += self._B[p, v, j] * self._displacementIncrements[j]
                        self._dStrain[v] = accumulated

                    accumulated = 0.0
                    for k in range(nK):
                        accumulated += self._N[k] * self._nonlocalValues[k]
                    self._nonlocal[0] = accumulated
                    nonlocalValue = accumulated

                    accumulated = 0.0
                    for k in range(nK):
                        accumulated += self._N[k] * self._nonlocalIncrements[k]
                    self._dNonlocal[0] = accumulated

                    # the stress and the local driving variable enter as those of the last
                    # increment, read from their state variable slots
                    for v in range(nVoigt):
                        self._stress[v] = self._stateVarsTemp[p, v]
                    self._KLocal[0] = self._stateVarsTemp[p, 2 * nVoigt]
                    self._c[0] = 0.0

                    self._material.computeStress(
                        &self._stress[0],
                        &self._KLocal[0],
                        &self._c[0],
                        &self._elasticEnergyDensity[0],
                        &self._dissipation[0],
                        &self._dStress_dStrain[0],
                        &self._dStress_dK[0],
                        &self._dKLocal_dStrain[0],
                        &self._dKLocal_dK[0],
                        &self._dc_dK[0],
                        &self._d2c_dK2[0],
                        &self._dStrain[0],
                        &self._nonlocal[0],
                        &self._dNonlocal[0],
                        &self._stateVarsTemp[p, self._nStateVarsOverhead],
                        time,
                        dTime,
                        self._planeStress,
                    )

                    c = self._c[0]
                    KLocal = self._KLocal[0]
                    dcdK = self._dc_dK[0]
                    dKLocaldK = self._dKLocal_dK[0]

                    for v in range(nVoigt):
                        self._stateVarsTemp[p, v] = self._stress[v]
                        self._stateVarsTemp[p, nVoigt + v] += self._dStrain[v]

                    # the layout is stress, strain, the *local* driving variable, then the
                    # nonlocal one -- in that order, see the stencil's initializeStencil
                    self._stateVarsTemp[p, 2 * nVoigt] = KLocal
                    self._stateVarsTemp[p, 2 * nVoigt + 1] = nonlocalValue

                    # G^T G K, needed by both the nonlocal residual and its tangent
                    for k in range(nK):
                        accumulated = 0.0
                        for l in range(nK):
                            accumulated += self._GTG[p, k, l] * self._nonlocalValues[l]
                        self._laplacianTimesNonlocal[k] = accumulated

                    # -- internal flux ------------------------------------------------------
                    for i in range(nD):
                        accumulated = 0.0
                        for v in range(nVoigt):
                            accumulated += self._BTv[p, i, v] * self._stress[v]
                        self._localFlux[i] += accumulated

                    for k in range(nK):
                        self._localFlux[nD + k] += (
                            self._N[k] * (nonlocalValue - KLocal) + c * self._laplacianTimesNonlocal[k]
                        ) * volume

                    # -- displacement block, in two steps ----------------------------------
                    for i in range(nD):
                        for v in range(nVoigt):
                            accumulated = 0.0
                            for w in range(nVoigt):
                                accumulated += self._BTv[p, i, w] * self._dStress_dStrain[w * nVoigt + v]
                            self._operatorTimesTangent[i, v] = accumulated

                    for i in range(nD):
                        for j in range(nD):
                            accumulated = 0.0
                            for v in range(nVoigt):
                                accumulated += self._operatorTimesTangent[i, v] * self._B[p, v, j]
                            self._localTangent[i, j] += accumulated

                    # -- the two coupling blocks -------------------------------------------
                    for i in range(nD):
                        accumulated = 0.0
                        for v in range(nVoigt):
                            accumulated += self._BTv[p, i, v] * self._dStress_dK[v]
                        self._stressCoupling[i] = accumulated

                        accumulated = 0.0
                        for v in range(nVoigt):
                            accumulated += self._dKLocal_dStrain[v] * self._B[p, v, i]
                        self._yieldCoupling[i] = accumulated

                    for i in range(nD):
                        for k in range(nK):
                            self._localTangent[i, nD + k] += self._stressCoupling[i] * self._N[k]

                    for k in range(nK):
                        for j in range(nD):
                            self._localTangent[nD + k, j] += (
                                -self._N[k] * self._yieldCoupling[j] * volume
                            )

                    # -- the nonlocal block ------------------------------------------------
                    for k in range(nK):
                        for l in range(nK):
                            self._localTangent[nD + k, nD + l] += (
                                self._N[k] * self._N[l] * (1.0 - dKLocaldK)
                                + c * self._GTG[p, k, l]
                                + self._laplacianTimesNonlocal[k] * dcdK * self._N[l]
                            ) * volume

                # -- scatter into the cell's global block ---------------------------------
                for i in range(nD + nK):
                    if i < nD:
                        dof = <int> self._displacementDofs[i]
                    else:
                        dof = <int> self._nonlocalDofs[i - nD]

                    P[dof] += self._localFlux[i]

                    for j in range(nD + nK):
                        if j < nD:
                            K[dof, <int> self._displacementDofs[j]] += self._localTangent[i, j]
                        else:
                            K[dof, <int> self._nonlocalDofs[j - nD]] += self._localTangent[i, j]

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
