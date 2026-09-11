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
# Created on Tue Jan  10 19:10:42 2017

# @author: Matthias Neuner
"""
This module contains the central definition of
the known physics,
i.e.,

 * the available physical fields
 * their tensorial order
 * their default residual tolerances

If a new physical field should be added to EdelweissFE,
then this should happen (only) here.
"""

# field                  field type
phenomena = {
    "displacement": "vector",
    "rotation": "rotation vector",
    "micro rotation": "rotation vector",
    "thermal": "scalar",
    "nonlocal damage": "scalar",
    "nonlocal damage 2": "scalar",
    "concentration": "scalar",
    "chemical potential": "scalar",
    "strain symmetric": "symmetric tensor second order",
    "plastic multiplier": "scalar",
    "pressure": "scalar",
    "jacobi": "scalar",
}


# field                  kind of inertia
#
# The coefficient of a field's SECOND time derivative is not always a mass. A dynamic solver
# assembles one inertia vector over every field it carries -- that is the element interface, and
# rightly so, since the integrator divides by all of it alike -- but the moment it wants to report
# a momentum or an energy it has to know which entries may be added to which. Three answers, and
# the value recorded here is the ONLY place that question is answered:
#
#  * ``"mass"``                  -- m*v is a linear momentum and 0.5*m*v^2 is an energy. Summable
#                                   with every other mass field, in both balances.
#  * ``"rotational inertia"``    -- 0.5*I*w^2 is an energy, so it belongs in the energy balance,
#                                   but I*w is an ANGULAR momentum and must not be added to a
#                                   linear one. Adding the two is not caught by a dimension check:
#                                   in 3d both occupy three components and numpy would sum them
#                                   happily.
#  * ``"non-mechanical"``        -- neither. Typically a numerical regularisation: the non-local
#                                   micro-inertia that makes gradient-enhanced damage hyperbolic
#                                   is a time squared, so 0.5*m*v^2 there has the units of a
#                                   volume, not of an energy.
#
# A field with no inertia at all -- one integrated first order in time, or not integrated at all --
# is recorded as ``"non-mechanical"`` too: the question this answers is what its inertia WOULD
# mean, and for such a field the honest answer is that it has none to sum anywhere.
inertiaKind = {
    "displacement": "mass",
    "rotation": "rotational inertia",
    "micro rotation": "rotational inertia",
    "thermal": "non-mechanical",
    "nonlocal damage": "non-mechanical",
    "nonlocal damage 2": "non-mechanical",
    "concentration": "non-mechanical",
    "chemical potential": "non-mechanical",
    "strain symmetric": "non-mechanical",
    "plastic multiplier": "non-mechanical",
    "pressure": "non-mechanical",
    "jacobi": "non-mechanical",
}


# field                  tolerance
fieldCorrectionTolerance = {
    "displacement": 1e-7,
    "rotation": 1e-3,
    "micro rotation": 1e-8,
    "nonlocal damage": 1e-8,
    "nonlocal damage 2": 1e-8,
    "concentration": 1e-1,
    "chemical potential": 1e-1,
    "strain symmetric": 1e-7,
    "scalar variables": 1e-3,
    "plastic multiplier": 1e-8,
    "pressure": 1e-8,
    "jacobi": 1e-8,
}

fluxResidualTolerance = {
    "displacement": 1e-8,
    "rotation": 1e-4,
    "micro rotation": 1e-8,
    "nonlocal damage": 1e-8,
    "nonlocal damage 2": 1e-8,
    "concentration": 1e-1,
    "chemical potential": 1e-2,
    "strain symmetric": 1e-8,
    "scalar variables": 1e-8,
    "plastic multiplier": 1e-8,
    "pressure": 1e-8,
    "jacobi": 1e-8,
}

fluxResidualToleranceAlternative = {
    "displacement": 5e-3,
    "rotation": 5e-3,
    "micro rotation": 5e-3,
    "nonlocal damage": 5e-3,
    "nonlocal damage 2": 5e-3,
    "concentration": 5e-2,
    "chemical potential": 5e-2,
    "strain symmetric": 5e-3,
    "scalar variables": 1e-8,
    "plastic multiplier": 5e-3,
    "pressure": 5e-3,
    "jacobi": 5e-3,
}

# domain                 dimensions
domainMapping = {
    "1d": 1,
    "2d": 2,
    "3d": 3,
    "axisymmetric": 2,
}


def getInertiaKind(field: str) -> str:
    """The kind of inertia a field carries; see :data:`inertiaKind`.

    Parameters
    ----------
    field
        The name of the physical field.

    Returns
    -------
    str
        One of ``"mass"``, ``"rotational inertia"``, ``"non-mechanical"``.

    Raises
    ------
    NotImplementedError
        If the field is not registered here.
    """

    try:
        return inertiaKind[field]
    except KeyError:
        raise NotImplementedError(
            "Physical field {:} has no registered inertia kind. Add it to inertiaKind in "
            "edelweissfe/config/phenomena.py, alongside its entry in phenomena.".format(field)
        )


def carriesLinearMomentum(field: str) -> bool:
    """Whether a field's inertia times its rate is a linear momentum, i.e. whether the field may
    enter a linear-momentum balance and be summed with the other fields in it.

    Parameters
    ----------
    field
        The name of the physical field.

    Returns
    -------
    bool
        True only for a field whose inertia is a mass.
    """

    return getInertiaKind(field) == "mass"


def carriesKineticEnergy(field: str) -> bool:
    """Whether a field's ``0.5 * inertia * rate**2`` is a mechanical energy, i.e. whether the field
    may enter the energy balance and be summed with the other fields in it.

    Work done at a prescribed degree of freedom of such a field -- a force through a displacement,
    a moment through a rotation -- is an energy by the same token, which is why the external work
    is accumulated over exactly these fields as well.

    Parameters
    ----------
    field
        The name of the physical field.

    Returns
    -------
    bool
        True for a field whose inertia is a mass or a rotational inertia.
    """

    return getInertiaKind(field) in ("mass", "rotational inertia")


def getFieldSize(field, domainSize):
    fType = phenomena[field]
    if fType == "scalar":
        return 1
    if fType == "vector":
        return domainSize
    if fType == "rotation vector":
        if domainSize == 2:
            return 1
        elif domainSize == 3:
            return 3
    if fType == "symmetric tensor second order":
        # if domainSize == 2:
        #     return 3
        # elif domainSize == 3:
        return 6

    raise NotImplementedError("Invalid physical field {:} requested".format(field))
