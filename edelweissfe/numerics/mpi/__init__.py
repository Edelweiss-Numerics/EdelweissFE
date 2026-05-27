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
MPI-based distributed parallelization infrastructure for EdelweissFE.

This package provides the components needed for distributed-memory
parallelization using MPI (via mpi4py) and PETSc (via petsc4py):

- :mod:`~edelweissfe.numerics.mpi.mpiutils`: MPI communicator utilities
- :mod:`~edelweissfe.numerics.mpi.partitioner`: Mesh/DOF partitioning
- :mod:`~edelweissfe.numerics.mpi.distributeddofmanager`: Distributed DOF management
- :mod:`~edelweissfe.numerics.mpi.petscsolver`: Distributed PETSc linear solver
"""
