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
#
# Declarations of the Marmot material interfaces which EdelweissFE calls point-wise,
# i.e., without an element in between.
#
# This module declares the hypoelastic base class. The general gradient enhanced base
# class is templated on the *number* of nonlocal variables; since Cython cannot express
# non-type template parameters, it is reached through the C++ shim declared in
# ``_gradientenhanced.pxd``.

import numpy as np

cimport numpy as np
from libcpp.string cimport string


cdef extern from "Marmot/MarmotTypedefs.h" nogil:
    cdef cppclass Vector6d "Marmot::Vector6d":
        Vector6d() nogil
        Vector6d(double*) nogil
        double& operator()(int row)

    cdef cppclass Matrix6d "Marmot::Matrix6d":
        Matrix6d() nogil
        Matrix6d(double*) nogil
        double& operator()(int row, int col)

    cdef cppclass Vector3d "Marmot::Vector3d":
        Vector3d() nogil
        Vector3d(double*) nogil
        double& operator()(int row)

    cdef cppclass Matrix3d "Marmot::Matrix3d":
        Matrix3d() nogil
        Matrix3d(double*) nogil
        double& operator()(int row, int col)


cdef extern from "Marmot/MarmotUtils.h":
    cdef struct StateView:
        double *stateLocation
        int stateSize


cdef inline np.ndarray stateViewAsArray(StateView res):
    """Wrap a Marmot state view in a persistent numpy array, without copying.

    Shared by the three point-wise material wrappers (hypoelastic, gradient-enhanced,
    gradient-plasticity), whose ``getResult`` bodies are otherwise identical: this is a
    plain function, not a base class, so that a wrapper's ``_material`` pointer type stays
    whatever Marmot base class it actually is, and none of the three extensions gains a
    build dependency on either of the other two's C++ shims.
    """

    cdef double[::1] resultView = <double[:res.stateSize]> (res.stateLocation)

    return np.asarray(resultView)


cdef extern from "Marmot/MarmotMaterialHypoElastic.h":
    cdef cppclass MarmotMaterialHypoElastic nogil:

        StateView getStateView(const string& stateName, double* stateVars) except +

        void initializeYourself(double* stateVars, int nStateVars) except +

        void setCharacteristicElementLength(double length)

        int getNumberOfRequiredStateVars()

        double getDensity(const double* stateVars) except +

        struct state3D:
            Vector6d stress
            double elasticEnergyDensity
            double dissipation
            double* stateVars

        struct state2D:
            Vector3d stress
            double elasticEnergyDensity
            double dissipation
            double* stateVars

        struct state1D:
            double stress
            double elasticEnergyDensity
            double dissipation
            double* stateVars

        struct timeInfo:
            double time
            double dT

        void computeStress(state3D& state,
                           Matrix6d& dStress_dStrain,
                           const Vector6d& dStrain,
                           const timeInfo& timeInfo) except +

        void computePlaneStress(state2D& state,
                                Matrix3d& dStress_dStrain2D,
                                const Vector3d& dStrain2D,
                                const timeInfo& timeInfo) except +

        void computeUniaxialStress(state1D& state,
                                   double& dStress_dStrain1D,
                                   const double dStrain,
                                   const timeInfo& timeInfo) except +


cdef extern from "Marmot/MarmotMaterialHypoElasticFactory.h" namespace "MarmotLibrary" nogil:

    cdef cppclass MarmotMaterialHypoElasticFactory:
        @staticmethod
        MarmotMaterialHypoElastic* createMaterial(
                const string& materialName,
                const double* materialProperties,
                int nMaterialProperties,
                int materialNumber) except +
