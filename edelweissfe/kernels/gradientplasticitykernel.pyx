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
The compiled inner loop of a finite difference gradient plasticity cell.

Why this exists
---------------

A finite difference cell couples few degrees of freedom -- in two dimensions twelve
displacements and twelve plastic multipliers -- and its blocks are correspondingly small: a six
by six tangent, operators of six by twelve. At those sizes a numpy call costs far more in
dispatch than in arithmetic, so a Python kernel spends most of its time not computing. Measured
on a real localising run of the compressed panel example of EdelweissFD, 36.8 seconds of kernel
time contained only 6.8 seconds of material evaluation: the Marmot material is 18.4 percent of
the cost and the remaining 81.6 percent is overhead.

This kernel removes that overhead by doing the whole cell in C. It reaches Marmot through the
``double*`` interface of
:mod:`edelweissfe.materials.marmot._gradientplasticity`, whose declarations are already inside a
``nogil`` block, so the material evaluations and the surrounding algebra both run with the global
interpreter lock released. That matters twice over on a free threaded interpreter: it removes the
dispatch cost, and it removes the reference counting contention that limits how far a Python
kernel scales across threads.

What stays outside
------------------

Everything that is not the inner loop. The grid, the difference molecule with its ghost node
boundary treatment, the B-bar treatment of the volumetric strain, the degree of freedom
bookkeeping and the result access all remain in EdelweissFD's Python stencil, which passes the
finished operators in here once and then calls :meth:`computeKernels` per iteration. This class
knows nothing about grids.

The equations
-------------

Per material point ``p`` of the cell, with :math:`\\boldsymbol B_p` the strain operator,
:math:`\\boldsymbol L_p` the Laplacian coefficients over the molecule, :math:`V_p` the material
point volume and :math:`c_p` the index of the grid point the material point sits on:

.. math::
    \\boldsymbol P_u \\mathrel{+}= \\boldsymbol B_p^T \\boldsymbol \\sigma \\, V_p,
    \\qquad
    P_{\\lambda, c_p} \\mathrel{+}= f \\, V_p

and, writing :math:`\\boldsymbol e_{c}` for the unit vector of the centre grid point,

.. math::
    \\frac{\\partial \\boldsymbol \\sigma}{\\partial \\boldsymbol \\lambda}
      = \\frac{\\partial \\boldsymbol \\sigma}{\\partial \\nabla^2 \\lambda} \\boldsymbol L_p
      + \\frac{\\partial \\boldsymbol \\sigma}{\\partial \\lambda} \\boldsymbol e_{c_p}^T

so that the four coupling blocks follow as the products written out in :meth:`computeKernels`.
"""

cimport cython
from libcpp.string cimport string

import numpy as np

from edelweissfe.materials.marmot._gradientplasticity cimport (
    GradientPlasticityHypoElasticShim1,
)
from edelweissfe.materials.marmot._marmotmaterials cimport StateView

from edelweissfe.utils.exceptions import CutbackRequest


#: The number of Voigt components Marmot always works in, even in two dimensions.
cdef int nVoigt = 6


@cython.final
@cython.boundscheck(False)
@cython.wraparound(False)
cdef class GradientPlasticityKernel:
    """The compiled kernel of one gradient plasticity cell.

    One instance owns one Marmot material, mirroring the rule that no two cells may share a
    material instance -- the material carries the state variable storage it operates on as
    mutable state, and cells are distributed over threads. See
    :mod:`edelweissfd.materials.provider`.

    Parameters
    ----------
    materialName
        The Marmot material name, e.g. ``GRADIENTVONMISES``.
    materialProperties
        The material property vector.
    planeStress
        Whether the material should condense the out-of-plane components for plane stress.
    strainOperators
        The strain operators of the material points, shape ``(nMaterialPoints, 6, nDisplacementDofs)``.
    weightedTransposedOperators
        Their transposes, already multiplied by the material point volume, shape
        ``(nMaterialPoints, nDisplacementDofs, 6)``. Pre-scaling costs nothing here and saves a
        multiplication in every product the transpose appears in.
    laplacianCoefficients
        The Laplacian coefficients per material point over the molecule, shape
        ``(nMaterialPoints, nMultiplierDofs)``.
    materialPointVolumes
        The volume of each material point.
    materialPointNodes
        The index, within the molecule, of the grid point each material point sits on.
    displacementDofs
        The local indices of the displacement degrees of freedom.
    multiplierDofs
        The local indices of the plastic multiplier degrees of freedom.
    stateVars
        The accepted state variables, shape ``(nMaterialPoints, nStateVarsPerPoint)``.
    stateVarsTemp
        The trial state variables, same shape. The layout expected is stress, strain, the plastic
        multiplier, the yield function value and then the material's own variables.
    nStateVarsOverhead
        How many state variables the stencil keeps in front of the material's own ones.
    """

    cdef GradientPlasticityHypoElasticShim1* _material
    cdef double[::1] _materialProperties
    cdef bint _planeStress

    cdef int _nMaterialPoints
    cdef int _nDisplacementDofs
    cdef int _nMultiplierDofs
    cdef int _nStateVarsOverhead

    cdef double[:, :, ::1] _B
    cdef double[:, :, ::1] _BTv
    cdef double[:, ::1] _laplacians
    cdef double[::1] _volumes
    cdef long[::1] _centres

    cdef long[::1] _displacementDofs
    cdef long[::1] _multiplierDofs

    cdef double[:, ::1] _stateVars
    cdef double[:, ::1] _stateVarsTemp

    # per material point scratch, reused across evaluations
    cdef double[::1] _dStrain
    cdef double[::1] _dLambda
    cdef double[::1] _laplaceDLambda
    cdef double[::1] _yieldValue
    cdef double[::1] _dStress_dStrain
    cdef double[::1] _dStress_dLambda
    cdef double[::1] _dStress_dLaplacian
    cdef double[::1] _dF_dStrain
    cdef double[::1] _dF_dLambda
    cdef double[::1] _dF_dLaplacian
    cdef double[::1] _elasticEnergyDensity
    cdef double[::1] _dissipation

    # the cell contribution, in field grouped order
    cdef double[:, ::1] _localTangent
    cdef double[::1] _localFlux
    cdef double[:, ::1] _operatorTimesTangent
    cdef double[::1] _displacementIncrements
    cdef double[::1] _multiplierIncrements

    def __cinit__(
        self,
        str materialName,
        double[::1] materialProperties,
        bint planeStress,
        double[:, :, ::1] strainOperators,
        double[:, :, ::1] weightedTransposedOperators,
        double[:, ::1] laplacianCoefficients,
        double[::1] materialPointVolumes,
        long[::1] materialPointNodes,
        long[::1] displacementDofs,
        long[::1] multiplierDofs,
        double[:, ::1] stateVars,
        double[:, ::1] stateVarsTemp,
        int nStateVarsOverhead,
    ):
        cdef string encodedName = materialName.encode("UTF-8")

        # Marmot keeps only the pointer to the properties, so the copy has to outlive the call
        self._materialProperties = np.ascontiguousarray(materialProperties, dtype=float).copy()

        self._material = new GradientPlasticityHypoElasticShim1(
            encodedName,
            &self._materialProperties[0],
            self._materialProperties.shape[0],
            1,
        )

        self._planeStress = planeStress

        self._B = strainOperators
        self._BTv = weightedTransposedOperators
        self._laplacians = laplacianCoefficients
        self._volumes = materialPointVolumes
        self._centres = materialPointNodes

        self._displacementDofs = displacementDofs
        self._multiplierDofs = multiplierDofs

        self._stateVars = stateVars
        self._stateVarsTemp = stateVarsTemp
        self._nStateVarsOverhead = nStateVarsOverhead

        self._nMaterialPoints = strainOperators.shape[0]
        self._nDisplacementDofs = displacementDofs.shape[0]
        self._nMultiplierDofs = multiplierDofs.shape[0]

        self._dStrain = np.zeros(nVoigt)
        self._dLambda = np.zeros(1)
        self._laplaceDLambda = np.zeros(1)
        self._yieldValue = np.zeros(1)
        self._dStress_dStrain = np.zeros(nVoigt * nVoigt)
        self._dStress_dLambda = np.zeros(nVoigt)
        self._dStress_dLaplacian = np.zeros(nVoigt)
        self._dF_dStrain = np.zeros(nVoigt)
        self._dF_dLambda = np.zeros(1)
        self._dF_dLaplacian = np.zeros(1)
        self._elasticEnergyDensity = np.zeros(1)
        self._dissipation = np.zeros(1)

        self._localTangent = np.zeros(
            (self._nDisplacementDofs + self._nMultiplierDofs, self._nDisplacementDofs + self._nMultiplierDofs)
        )
        self._localFlux = np.zeros(self._nDisplacementDofs + self._nMultiplierDofs)
        self._operatorTimesTangent = np.zeros((self._nDisplacementDofs, nVoigt))
        self._displacementIncrements = np.zeros(self._nDisplacementDofs)
        self._multiplierIncrements = np.zeros(self._nMultiplierDofs)

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
            The cell's dense block of the global tangent. Any strides are accepted: the solvers
            hand over a Fortran ordered view, while the numerical tangent checker allocates its
            own C ordered array, and a kernel has no business demanding one or the other.
        P
            The cell's slice of the internal flux.
        U
            The total field values on the cell's molecule.
        dU
            Their increment.
        time
            The total time.
        dTime
            The time increment.
        """

        cdef int nMP = self._nMaterialPoints
        cdef int nD = self._nDisplacementDofs
        cdef int nM = self._nMultiplierDofs

        cdef int p, i, j, k, v, w, centre, dof
        cdef double volume, accumulated, dFdLap, dFdLam, yieldValue

        # gather the two fields, and reset the accumulators
        for i in range(nD):
            self._displacementIncrements[i] = dU[self._displacementDofs[i]]
        for k in range(nM):
            self._multiplierIncrements[k] = dU[self._multiplierDofs[k]]

        for i in range(nD + nM):
            self._localFlux[i] = 0.0
            for j in range(nD + nM):
                self._localTangent[i, j] = 0.0

        # the trial state starts from the last accepted one
        for p in range(nMP):
            for i in range(self._stateVars.shape[1]):
                self._stateVarsTemp[p, i] = self._stateVars[p, i]

        try:
            with nogil:
                for p in range(nMP):
                    centre = <int> self._centres[p]
                    volume = self._volumes[p]

                    for v in range(nVoigt):
                        accumulated = 0.0
                        for j in range(nD):
                            accumulated += self._B[p, v, j] * self._displacementIncrements[j]
                        self._dStrain[v] = accumulated

                    self._dLambda[0] = self._multiplierIncrements[centre]

                    accumulated = 0.0
                    for k in range(nM):
                        accumulated += self._laplacians[p, k] * self._multiplierIncrements[k]
                    self._laplaceDLambda[0] = accumulated

                    self._yieldValue[0] = 0.0

                    # the stress is read from and written to its state variable slot, which the
                    # reset above has restored to the last accepted value
                    self._material.computeStress(
                        &self._stateVarsTemp[p, 0],
                        &self._yieldValue[0],
                        &self._elasticEnergyDensity[0],
                        &self._dissipation[0],
                        &self._dStress_dStrain[0],
                        &self._dStress_dLambda[0],
                        &self._dStress_dLaplacian[0],
                        &self._dF_dStrain[0],
                        &self._dF_dLambda[0],
                        &self._dF_dLaplacian[0],
                        &self._dStrain[0],
                        &self._dLambda[0],
                        &self._laplaceDLambda[0],
                        &self._stateVarsTemp[p, self._nStateVarsOverhead],
                        time,
                        dTime,
                        self._planeStress,
                    )

                    yieldValue = self._yieldValue[0]

                    # accumulate the strain, and record the multiplier and the yield value
                    for v in range(nVoigt):
                        self._stateVarsTemp[p, nVoigt + v] += self._dStrain[v]

                    self._stateVarsTemp[p, 2 * nVoigt] = U[self._multiplierDofs[centre]]
                    self._stateVarsTemp[p, 2 * nVoigt + 1] = yieldValue

                    # -- internal flux ------------------------------------------------------
                    for i in range(nD):
                        accumulated = 0.0
                        for v in range(nVoigt):
                            accumulated += self._BTv[p, i, v] * self._stateVarsTemp[p, v]
                        self._localFlux[i] += accumulated

                    self._localFlux[nD + centre] += yieldValue * volume

                    # -- displacement block, in two steps rather than one triple loop -------
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

                    # -- coupling of the displacement to the plastic multiplier -------------
                    # through the Laplacian over the whole molecule, and directly at the centre
                    for i in range(nD):
                        accumulated = 0.0
                        for v in range(nVoigt):
                            accumulated += self._BTv[p, i, v] * self._dStress_dLaplacian[v]
                        for k in range(nM):
                            self._localTangent[i, nD + k] += accumulated * self._laplacians[p, k]

                        accumulated = 0.0
                        for v in range(nVoigt):
                            accumulated += self._BTv[p, i, v] * self._dStress_dLambda[v]
                        self._localTangent[i, nD + centre] += accumulated

                    # -- coupling of the yield condition to the displacement ----------------
                    for j in range(nD):
                        accumulated = 0.0
                        for v in range(nVoigt):
                            accumulated += self._dF_dStrain[v] * self._B[p, v, j]
                        self._localTangent[nD + centre, j] += accumulated * volume

                    # -- yield condition against the plastic multiplier --------------------
                    dFdLap = self._dF_dLaplacian[0]
                    dFdLam = self._dF_dLambda[0]

                    for k in range(nM):
                        self._localTangent[nD + centre, nD + k] += dFdLap * self._laplacians[p, k] * volume

                    self._localTangent[nD + centre, nD + centre] += dFdLam * volume

                # -- scatter into the cell's global block, undoing the field grouping -------
                for i in range(nD + nM):
                    if i < nD:
                        dof = <int> self._displacementDofs[i]
                    else:
                        dof = <int> self._multiplierDofs[i - nD]

                    P[dof] += self._localFlux[i]

                    for j in range(nD + nM):
                        if j < nD:
                            K[dof, <int> self._displacementDofs[j]] += self._localTangent[i, j]
                        else:
                            K[dof, <int> self._multiplierDofs[j - nD]] += self._localTangent[i, j]

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
