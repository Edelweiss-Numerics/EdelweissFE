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
from collections.abc import Sequence

import numpy as np

from edelweissfe.points.node import Node
from edelweissfe.variables.scalarvariable import ScalarVariable


class BaseNodeCouplingEntity(ABC):
    """
    This class represents the most basic entity which couples nodes (and their fields) in a numerical model.
    It is the protocol of the location of everything that has one in a discrete system
    (:class:`~edelweissfe.numerics.doflocations.DofLocations`): finite elements, constraints, and, in EdelweissMeshfree,
    cells, cell elements and particles. The Python entities derive from this class; the compiled wrappers of Marmot's
    elements, cells and particles (Cython extension types) cannot derive from a Python class, and implement the same
    protocol without deriving: they declare every public member of this class themselves, as a test of each
    repository checks.

    The entity information is used for locating the entity in the discrete system.
    Each entity is defined by a list of nodes it couples, and a list of fields per node, on which it actually operates;
    an entity may additionally own scalar variables (e.g. the Lagrange multipliers of a constraint), which follow its
    nodal degrees of freedom.

    The reason for a common base class is that not only finite elements couple nodes, but also constraints, cells (MPM)
    or particles (SPH, RKPM). Having a common base class allows for a unified treatment of these entities in the
    discrete system: the one protocol of the location of an entity (``nodes``, ``fields``, ``dofIndicesPermutation``,
    ``scalarVariables``). What an entity computes is the business of its kind. The inertia and the damping, for
    example, are reported by finite elements (:class:`~edelweissfe.elements.base.baseelement.BaseElement`), and which
    entities a solver asks for them, at the nodes, is a fact of the discrete system
    (:attr:`~edelweissfe.models.discretesystem.DiscreteSystem.elementsWithNodalInertia`).
    """

    #: Whether the connectivity of this entity has to be found anew again and again as the bodies move, because its
    #: ``updateConnectivity`` searches, whether or not its nodes change with the search. A contact constraint searches
    #: for the surface it touches and couples what it finds, the rigid-body surface contact finds its closest triangles
    #: anew but keeps the nodes it couples, the nodes of a meshfree constraint follow its particles, and the kernel
    #: functions of a particle reach other nodes as it moves. A solver that never updates the connectivity cannot
    #: evaluate such a constraint (``NEST`` refuses it), and an explicit solver updates it only every
    #: ``contact-update-frequency`` increments. The locations of an entity whose nodes change with the motion are made
    #: anew with every change of the connectivity, and what is counted over the entities is counted anew for it, while
    #: the entities with fixed nodes are counted once for a numbering. Declared True by every entity that searches in
    #: this way; an entity with fixed nodes and nothing to search, or one that is only refreshed after a change of the
    #: mesh (:class:`~edelweissfe.models.meshdependent.MeshDependent`), leaves it False.
    connectivityChangesWithMotion: bool = False

    #: The scalar variables this entity owns, whose degrees of freedom follow those of its nodes in the layout of its
    #: residual and tangent. Empty for most entities; a constraint that needs Lagrange multipliers gets them assigned
    #: (:meth:`~edelweissfe.constraints.base.constraintbase.ConstraintBase.assignAdditionalScalarVariables`).
    scalarVariables: Sequence[ScalarVariable] = ()

    @property
    @abstractmethod
    def nNodes(self) -> int:
        """The number of nodes this entity couples."""

    @property
    @abstractmethod
    def nodes(self) -> list[Node]:
        """The list of nodes this entity currently holds."""

    @property
    @abstractmethod
    def nDof(self) -> int:
        """The total number of degrees of freedom this entity has.
        Note that this information is redundant, as it results from the list of fields on nodes,
        but we keep this redundant information for the sake of performance."""

    @property
    @abstractmethod
    def fields(self) -> list[list[str]]:
        """The list of fields per node which this entity couples."""

    @property
    @abstractmethod
    def dofIndicesPermutation(self) -> np.ndarray | None:
        """If the entity computes residual vectors and stiffness matrices not nodewise, but e.g., fieldwise,
        this permutation pattern is used to aggregate all entries in order to resemble the defined fields nodewise.

        For performance reasons, this is computed once and stored for later use."""
