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
"""
The options shared by the integrated (surface-to-surface) penalty contact constraints, and the contact points of the
slave surface they choose.
"""

from dataclasses import dataclass

from edelweissfe.constraints.base.contactpoints import ContactPoints
from edelweissfe.constraints.base.contactpointsatslavenodes import (
    ContactPointsAtSlaveNodes,
)
from edelweissfe.constraints.base.contactpointsonslavesurface import (
    ContactPointsOnSlaveSurface,
)
from edelweissfe.models.femodel import FEModel
from edelweissfe.utils.schema import schemaField


@dataclass(frozen=True)
class SurfaceContactPenaltySchema:
    """The options of the slave side and the penalty law, shared by
    :mod:`~edelweissfe.constraints.surfacetodeformablesurfacepenalty` and
    :mod:`~edelweissfe.constraints.surfacetodiscreterigidbodypenalty`, which add their master side. The slave side
    is a :class:`~edelweissfe.constraints.base.contactpoints.ContactPoints`, built from these options by
    :func:`contactPointsOfTheSlaveSurface`.

    The update-type option is spelled ``type`` in the input file but the field is named
    ``contactType`` here -- a dataclass field literally called ``type`` would shadow the builtin,
    which this project's conventions avoid. ``penalty`` is an interface stiffness modulus per unit
    area, multiplied by the area each contact point stands for (its quadrature weight, or the tributary area of a
    node).
    """

    slaveSurface: str | None = schemaField(
        description="The element set of contact facet elements (Tria3ContactFacet/Line2ContactFacet) "
        "forming the slave surface; the contact points lie on these facets, by default at quadrature points. "
        "The facets must come from the surface element generator, which stamps their parent faces.",
        dtype=str,
        default=None,
        required=True,
    )
    penalty: float | None = schemaField(
        description="The numerical penalty value, an interface stiffness modulus per unit slave " "surface area.",
        dtype=float,
        default=None,
        required=True,
    )
    contactType: str = schemaField(
        description="The formulation type: 'linear' (linear force, constant stiffness with jump) "
        "or 'quadratic' (quadratic force, linear stiffness).",
        dtype=str,
        default="linear",
        optionName="type",
    )
    nQuadraturePoints: int = schemaField(
        description="The number of quadrature points per slave facet: 1, 3 or 6 for a Tria3 facet "
        "(3D), 1, 2 or 3 for a Line2 facet (2D). The default of 3 integrates the parent face's "
        "shape functions exactly over a facet, which is what makes the consistent nodal loads -- "
        "including the negative corner loads of a serendipity face -- come out exactly.",
        dtype=int,
        default=3,
    )
    searchDistance: float | None = schemaField(
        description="An optional broadphase distance for the per-increment candidate-facet search. "
        "If not given, no distance gate: every contact point is paired with the master (see the constraint for the "
        "rules of the pairing).",
        dtype=float,
        default=None,
    )
    contactPoints: str = schemaField(
        description="The contact points of the slave surface: 'quadraturePoints' (nQuadraturePoints quadrature "
        "points on each slave facet, distributed with the slave parent face's shape functions) or 'nodes' (the "
        "slave nodes, each standing for its tributary area).",
        dtype=str,
        default="quadraturePoints",
    )
    sliding: str = schemaField(
        description="The kinematic treatment of the contact geometry. Only 'small' (Abaqus-style "
        "small sliding: the closest-point projection -- master facet, parametric location and "
        "normal -- is frozen at every contact search from the last converged configuration, making the "
        "gap linear in the displacement DOFs) is implemented; 'finite' is rejected.",
        dtype=str,
        default="small",
    )


#: The kinds of contact points an input file can choose, by the value of the ``contactPoints`` option.
_contactPointsFromInput = {
    "quadraturepoints": lambda configuration, slaveSurface, name: ContactPointsOnSlaveSurface(
        slaveSurface, configuration.nQuadraturePoints, name
    ),
    "nodes": lambda configuration, slaveSurface, name: ContactPointsAtSlaveNodes(slaveSurface, name),
}


def contactPointsOfTheSlaveSurface(
    configuration: SurfaceContactPenaltySchema, model: FEModel, constraintName: str
) -> ContactPoints:
    """The contact points an input file chooses on the slave surface of a contact constraint: its quadrature points or
    its nodes (option ``contactPoints``). A kind of point of another library is given to a constraint in code, not
    chosen in an input file.

    Parameters
    ----------
    configuration
        The options of the constraint.
    model
        The model, holding the slave surface's element set.
    constraintName
        The name of the constraint, for error messages.

    Returns
    -------
    ContactPoints
        The contact points.
    """

    contactPointsOfSurface = _contactPointsFromInput.get(configuration.contactPoints.lower())
    if contactPointsOfSurface is None:
        raise ValueError(
            f"Constraint '{constraintName}': contactPoints '{configuration.contactPoints}' is not supported. Use "
            "'quadraturePoints' or 'nodes'."
        )
    return contactPointsOfSurface(configuration, model.elementSets[configuration.slaveSurface], constraintName)
