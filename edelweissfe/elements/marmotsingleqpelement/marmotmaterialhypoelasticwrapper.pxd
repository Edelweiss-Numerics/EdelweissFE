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
# Created on Wed Aug 31 08:35:06 2022
# @author: matthias

from libcpp.string cimport string
from libcpp.vector cimport vector


cdef extern from "Marmot/MarmotTypedefs.h" nogil:
    cdef cppclass Vector6d "Marmot::Vector6d":
        Vector6d() nogil
        Vector6d(double*) nogil
        double& operator()(int row)

    cdef cppclass Matrix6d "Marmot::Matrix6d":
        Matrix6d() nogil
        Matrix6d(double*) nogil
        double& operator()(int row, int col)


cdef extern from "Marmot/MarmotMaterialHypoElasticFactory.h" namespace "MarmotLibrary" nogil:

    cdef cppclass MarmotMaterialHypoElasticFactory:
        @staticmethod
        MarmotMaterialHypoElastic* createMaterial(
                const string& materialName,
                const double* materialProperties,
                int nMaterialProperties,
                int materialNumber) except +IndexError

cdef extern from "Marmot/MarmotUtils.h":
    cdef struct StateView:
        double *stateLocation
        int stateSize

cdef extern from "Marmot/MarmotMaterialHypoElastic.h":
    cdef cppclass MarmotMaterialHypoElastic nogil:

        StateView getStateView(const string& stateName, double* stateVars) except +ValueError

        void initializeYourself(double* stateVars, int nStateVars) except +ValueError

        void setCharacteristicElementLength(double length)

        int getNumberOfRequiredStateVars()

        struct state3D:
            Vector6d stress
            double strainEnergyDensity
            double* stateVars

        struct timeInfo:
            double time
            double dT

        void computeStress(state3D& state,
                           Matrix6d& dStress_dStrain,
                           const Vector6d& dStrain,
                           const timeInfo& timeInfo) except +ValueError
