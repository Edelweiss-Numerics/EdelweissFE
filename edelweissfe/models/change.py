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

"""What a model changed before an increment, as the topology update of a solver reports it, see :class:`Change`."""

from enum import IntEnum


class Change(IntEnum):
    """What a model changed before an increment, from the point of view of its discrete system, ordered by how much of
    the discrete system the change invalidates: ``NONE < LOCATIONS < NUMBERING < MESH``. A solver reacts on the
    largest change it is told of, e.g. ``change >= Change.NUMBERING``.

    The model classifies what its entities report, because it alone knows what they couple, and which scalar variables
    take part: ``NONE``, ``LOCATIONS`` and ``NUMBERING`` are the answer of
    :meth:`~edelweissfe.models.femodel.FEModel.updateConnectivity`. ``MESH`` is what the topology update reports,
    composed with that answer in
    :meth:`~edelweissfe.solvers.base.nonlinearsolverbase.NonlinearSolverBase.updateTopologyAndConnectivity`.
    The *numbering* of the degrees of freedom is the global index of every degree of freedom of a field or scalar
    variable, the *location* of an entity the global indices of its local degrees of freedom, and the *connectivity*
    the nodes an entity touches (for a contact constraint: the nodes it couples now).
    """

    #: Nothing changed, so the discrete system stands.
    NONE = 0
    #: An entity couples other nodes (e.g. a contact constraint paired anew), or takes part or not (a constraint
    #: within reach or not), without a change of the scalar variables: the numbering of the degrees of freedom
    #: stands, and the entities are located anew.
    LOCATIONS = 1
    #: The degrees of freedom to be numbered changed, i.e. the scalar variables of the active constraints (e.g. the
    #: Lagrange multipliers of a constraint that took part or does not any more), or, in a meshfree model, the nodes
    #: of the active domain.
    NUMBERING = 2
    #: The mesh changed, because a model modifier (e.g. h-adaptivity) created or removed nodes or elements, or because
    #: a mesh-dependent consumer (a tie or contact surface) caught up on such a change. The state of the model is not
    #: the state the solver carries, e.g. the solver re-equilibrates (``equilibrateAfterModelChange``).
    MESH = 3
