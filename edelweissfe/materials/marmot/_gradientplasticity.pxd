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
# Declarations of the C++ shim which instantiates Marmot's templated gradient plasticity
# hypoelastic material base class. Kept separate from _marmotmaterials.pxd so that the
# hypoelastic extension does not have to include the gradient plasticity Marmot headers.

from libcpp.string cimport string

from edelweissfe.materials.marmot._marmotmaterials cimport StateView


cdef extern from "_gradientplasticityshim.h" namespace "EdelweissFE" nogil:

    cdef cppclass GradientPlasticityHypoElasticShim1 "EdelweissFE::GradientPlasticityHypoElasticShim1":

        GradientPlasticityHypoElasticShim1(const string& materialName,
                                           const double* materialProperties,
                                           int nMaterialProperties,
                                           int materialNumber) except +

        @staticmethod
        int getNumberOfYieldSurfaces()

        int getNumberOfRequiredStateVars()

        void initializeYourself(double* stateVars, int nStateVars) except +

        double getDensity(const double* stateVars) except +

        StateView getStateView(const string& stateName, double* stateVars) except +

        void computeStress(double* stress,
                           double* f,
                           double* elasticEnergyDensity,
                           double* dissipation,
                           double* dStress_dStrain,
                           double* dStress_dLambda,
                           double* dStress_dLaplacian,
                           double* dF_dStrain,
                           double* dF_dLambda,
                           double* dF_dLaplacian,
                           const double* dStrain,
                           const double* dLambda,
                           const double* laplaceDLambda,
                           double* stateVars,
                           double time,
                           double dT,
                           bint planeStress) except +
