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

import numpy as np


class VIJEntityBase:
    """Base class for entities that contribute to the VIJ (COO) system matrix.

    By default, the base implementation assumes that the entity contributes a full dense block of size
    ``nDof × nDof``.  If the nonzero pattern of the contribution is sparser, the entity should override
    :meth:`getVIJContributionSize` to return the actual number of nonzero entries, and override
    :meth:`initializeVIJContribution` to fill the corresponding ``I`` and ``J`` index arrays.
    """

    def __init__(self, nDof: int):
        """Initialize the VIJEntityBase."""
        self.nDof = nDof

    def getVIJContributionSize(self) -> int:
        """Return the number of entries this entity contributes to the VIJ (COO) system matrix.

        By default this is ``nDof**2``, which corresponds to a full dense block.

        Returns
        -------
        int
            Number of VIJ entries for this entity.
        """

        return self.nDof**2

    def initializeVIJContribution(self, idcs: np.ndarray, I_: np.ndarray, J_: np.ndarray, offset: int) -> None:
        """Fill the global ``I`` and ``J`` index arrays for this constraint's VIJ contribution.

        The default implementation writes a full dense ``nDof × nDof`` block, identical to
        the behaviour used for finite elements.
        Entities that override :meth:`getVIJContributionSize` to return a smaller value **must** also override
        this method so that exactly ``getVIJContributionSize()`` ``(I, J)`` pairs are
        written starting at ``offset``.

        Parameters
        ----------
        idcs
            Global DOF indices for this entity (length = ``nDof``).
        I
            The global row-index array of the VIJ triple (written in-place).
        J
            The global column-index array of the VIJ triple (written in-place).
        offset
            First position in ``I`` / ``J`` that belongs to this entity.
        """

        n = len(idcs)
        VIJLocations = np.tile(idcs, (n, 1))
        I_[offset : offset + n**2] = VIJLocations.flatten()
        J_[offset : offset + n**2] = VIJLocations.flatten("F")

    def shapeVIJContribution(self, flat_view: np.ndarray) -> np.ndarray:
        """Shape the flat VIJ values slice for this entity.

        By default, if the contribution is dense (size = nDof**2),
        it reshapes the flat view to a 2-D array of shape (nDof, nDof)
        using column-major (Fortran) order.
        """
        if self.getVIJContributionSize() == self.nDof**2:
            return flat_view.reshape((self.nDof, self.nDof), order="F")
        return flat_view
