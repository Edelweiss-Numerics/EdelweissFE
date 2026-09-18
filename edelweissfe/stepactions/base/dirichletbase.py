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
# Created on Mon Jan 23 13:03:09 2017

# @author: Matthias Neuner

from abc import abstractmethod

import numpy as np

from edelweissfe.stepactions.base.stepactionbase import StepActionBase
from edelweissfe.timesteppers.timestep import TimeStep


class DirichletBase(StepActionBase):
    """Base class for a Dirichlet (prescribed-value) boundary condition.

    A Dirichlet BC prescribes selected ``components`` of a ``field`` (e.g. the
    x- and z-component of the displacement) on every node of a node set
    ``nSet``. The nonlinear solver imposes it with the row-replacement method,
    for which it needs exactly two things from the BC:

        * which global DOFs are constrained  -> ``constrainedDofIndices``
        * by how much they must move this step -> ``getPrescribedIncrement()``
    """

    #: The field whose components are prescribed, e.g. "displacement".
    field: str
    #: The node set the boundary condition acts on.
    nSet: object
    #: Number of DOFs (components) the field has per node.
    fieldSize: int
    #: The global DOF indices constrained by this BC, in node-major order.
    #: Populated once per step by the solver (see
    #: :meth:`NonlinearSolverBase.locateConstrainedDofs`), because it depends on
    #: the DofManager's layout, which the boundary condition does not know itself.
    constrainedDofIndices: np.ndarray = None

    @property
    @abstractmethod
    def components(self) -> np.ndarray:
        """Column indices, within a node's field DOFs, that are prescribed."""

    @abstractmethod
    def getPrescribedIncrement(self, timeStep: TimeStep) -> np.ndarray:
        """The increment by which the prescribed DOFs must move in this time step.

        The returned array has shape (number of nodes in ``nSet``, number of
        prescribed ``components``), so that ``.flatten()`` yields the values in
        the same node-major order as :attr:`constrainedDofIndices`.
        """
