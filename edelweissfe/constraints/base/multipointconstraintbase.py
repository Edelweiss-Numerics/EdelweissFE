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

from abc import ABC, abstractmethod

from edelweissfe.models.femodel import FEModel


class MultiPointConstraintBase(ABC):
    """Base class for linear multi-point constraints (MPCs) enforced by degree-of-freedom
    elimination (master-slave condensation), Abaqus-style.

    Unlike :class:`~edelweissfe.constraints.base.constraintbase.ConstraintBase`, a multi-point
    constraint contributes nothing to the load vector or the system matrix, owns no degrees of
    freedom, and needs no VIJ block. Instead, it declares linear dependencies

    .. math::
        u_s = \\sum_a N_a \\, u_{m_a}

    between a slave degree of freedom :math:`u_s` and master degrees of freedom :math:`u_{m_a}`.
    The solver collects these records from all multi-point constraints in the model, condenses
    the slave degrees of freedom out of the equation system (implicit solvers), or folds slave
    masses/forces onto the masters and slaves the kinematics directly (explicit dynamics). See
    :class:`~edelweissfe.numerics.mpctransformation.MultiPointConstraintTransformation`.

    Multi-point constraints are defined via the ``*constraint`` keyword like ordinary constraints,
    but live in :attr:`~edelweissfe.models.femodel.FEModel.multiPointConstraints` and stay outside
    the DofManager and the constraint assembly loop entirely.
    """

    @abstractmethod
    def __init__(self, name: str, model: FEModel, **kwargs):
        """The multi-point constraint base class.

        Parameters
        ----------
        name
            The name of the constraint.
        model
            The model tree.
        kwargs
            Key value pairs of options.
        """

    @abstractmethod
    def getMultiPointConstraints(self, dofManager) -> list[tuple[int, list[tuple[int, float]]]]:
        """Return the linear dependency records of this constraint in global degree-of-freedom
        indices of the given DofManager.

        Parameters
        ----------
        dofManager
            The current :class:`~edelweissfe.numerics.dofmanager.DofManager`, used to resolve
            (node, field, component) to global degree-of-freedom indices.

        Returns
        -------
        list[tuple[int, list[tuple[int, float]]]]
            One record per slave degree of freedom:
            ``(slaveDofIndex, [(masterDofIndex, coefficient), ...])``.
        """
