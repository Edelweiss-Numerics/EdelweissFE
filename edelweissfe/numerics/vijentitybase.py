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
    :meth:`initializeVIJContribution` to fill the corresponding ``I`` and ``J`` index arrays.  Such an entity also
    overrides :attr:`couplesAllItsDofs` to return ``False``: that is how an assembly learns that the block is a list
    of the values of the entity's own ``(I, J)`` pairs rather than a dense matrix.
    """

    def __init__(self, nDof: int):
        """Initialize the VIJEntityBase."""
        self.nDof = nDof

    @property
    def couplesAllItsDofs(self) -> bool:
        """Whether the entity's block couples every one of its degrees of freedom with every other one.

        ``True`` (the default, every element) means the block is the dense ``nDof × nDof`` matrix, stored
        column-major, with the entity's global DOF indices as its rows and its columns. ``False`` means the entity
        declared a sparse pattern of its own in :meth:`initializeVIJContribution` (a constraint that couples only
        some of its degrees of freedom), and its block is the list of the values of these ``(I, J)`` pairs, in the
        order it declared them.

        The entity says which, because the number of entries cannot tell: a sparse pattern can happen to have
        exactly ``nDof**2`` entries. A rigid body contact with a single slave node in 3D couples the six degrees of
        freedom of the reference point with each other, the three of the node with each other, and the two groups
        both ways: 36 + 9 + 18 + 18 = 81 = 9², yet its entries are ordered block by block, not column by column.

        Returns
        -------
        bool
            ``True`` for a dense block, ``False`` for a pattern of the entity's own.
        """

        return True

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

        By default, if the contribution is dense (:attr:`couplesAllItsDofs`),
        it reshapes the flat view to a 2-D array of shape (nDof, nDof)
        using column-major (Fortran) order.
        """
        if self.couplesAllItsDofs:
            return flat_view.reshape((self.nDof, self.nDof), order="F")
        return flat_view
