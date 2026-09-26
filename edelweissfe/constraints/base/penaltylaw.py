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
The normal penalty force laws shared by the penalty contact constraints.

Sign convention: the gap :math:`g` is negative when the surfaces penetrate, and the normal force
:math:`f_n` carries the sign of :math:`g`, i.e. :math:`f_n \\le 0` in contact. The stiffness
:math:`\\mathrm{d}f_n/\\mathrm{d}g` is then positive for :math:`g < 0`.

``penaltyTimesArea`` is the penalty modulus times the area the contact point represents: a nodal
tributary area in node-based formulations, a quadrature weight in integrated ones.
"""

import numpy as np

#: The supported penalty force laws.
contactTypes = ("linear", "quadratic")


def validatedContactType(contactType: str) -> str:
    """Normalize a contact type option to lower case, and reject unsupported laws.

    Parameters
    ----------
    contactType
        The contact type as given in the input file.

    Returns
    -------
    str
        The lower-case contact type.
    """
    contactType = contactType.lower()
    if contactType not in contactTypes:
        raise ValueError(f"Constraint type '{contactType}' is not supported. Use 'linear' or 'quadratic'.")
    return contactType


def normalPenaltyForce(
    contactType: str, penaltyTimesArea: float | np.ndarray, gap: float | np.ndarray
) -> tuple[float | np.ndarray, float | np.ndarray]:
    """The normal penalty force and its derivative with respect to the gap, at a closed contact point.

    Works element-wise on arrays of points as well as on single points.

    - ``linear``: :math:`f_n = k A \\, g`, with constant stiffness :math:`k A`.
    - ``quadratic``: :math:`f_n = -\\tfrac{1}{2} k A \\, g^2`, with stiffness :math:`-k A \\, g`.

    Parameters
    ----------
    contactType
        'linear' or 'quadratic', see :func:`validatedContactType`.
    penaltyTimesArea
        The penalty modulus times the area of the contact point(s).
    gap
        The gap(s), negative in contact.

    Returns
    -------
    tuple
        The normal force(s) :math:`f_n` and the stiffness :math:`\\mathrm{d}f_n/\\mathrm{d}g`.
    """
    if contactType == "linear":
        normalForce = penaltyTimesArea * gap
        dNormalForce_dGap = penaltyTimesArea
    else:
        normalForce = -0.5 * penaltyTimesArea * gap**2
        dNormalForce_dGap = -penaltyTimesArea * gap
    return normalForce, dNormalForce_dGap
